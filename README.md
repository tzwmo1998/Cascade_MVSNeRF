# Cascade_MVSNeRF
This repository contains modified pytorch lightning implementation for the ICCV 2021 paper: [MVSNeRF: Fast Generalizable Radiance Field Reconstruction from Multi-View Stereo](https://arxiv.org/abs/2103.15595) and [Cascade Cost Volume for High-Resolution Multi-View Stereo
and Stereo Matching](https://arxiv.org/pdf/1912.06378.pdf). We replace the MVSNet in MVSNeRF with Cascade MVSNet. <br><br>


## Installation

#### Tested on Ubuntu 20.04 + Pytorch 1.10.1 + Pytorch Lignting 1.3.5

Install environment:
```
conda create -n casmvsnerf python=3.8
conda activate casmvsnerf
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install pytorch-lightning==1.3.5 imageio pillow scikit-image opencv-python configargparse lpips kornia warmup_scheduler matplotlib test-tube imageio-ffmpeg
pip install inplace-abn==1.1.0
```


## Training
Please see each subsection for training on different datasets. Available training datasets:

* [DTU](#dtu)

### DTU dataset

#### Data download

Download the preprocessed [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view)
and [Depth_raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip) from original [MVSNet repo](https://github.com/YoYo000/MVSNet)
and unzip.

#### Training general model

Run
```
CUDA_VISIBLE_DEVICES=0  python3 train_mvs_nerf_pl_ours.py \
   --expname $exp_name
   --num_epochs 6
   --use_viewdirs \
   --dataset_name dtu \
   --datadir $DTU_DIR \
   --pad 0\
   --batch_size 1024\
   --use_casmvs\
   --N_vis 6
```


*Important*: please always set batch_size to 1 when you are trining a genelize model, you can enlarge it when fine-tuning.

*Checkpoint*: a pre-trained mvsnerf checkpoint is included in `ckpts/mvsnerf-v0.tar` and a pre-trained casmvsnerf checkpoint is included in `ckpts/cas23.tar`. 

*Evaluation*: We also provide a rendering and quantity scipt  in `renderer.ipynb`, 




</details>

### Finetuning DTU
<details>
  <summary>Steps</summary>

```
CUDA_VISIBLE_DEVICES=0  python train_mvs_nerf_finetuning_pl.py  \
    --dataset_name dtu_ft --datadir /path/to/DTU/mvs_training/dtu/scan1 \
    --expname scan1-ft  --with_rgb_loss  --batch_size 1024  \
    --num_epochs 1 --imgScale_test 1.0   --pad 0 \
    --ckpt ./ckpts/cas23.tar --N_vis 1
```

</details>

## Rendering
After training or finetuning, you can render free-viewpoint videos
with the `renderer-video.ipynb`. 

![finetuned](https://github.com/wuyichia/Cascade_MVSNeRF/blob/3abe5a20fd7e51bc8e2629a41fbc9991cdf4963b/scan21-result-1e-scale1.gif)
![finetuned](https://github.com/wuyichia/Cascade_MVSNeRF/blob/34e95dd84a9d7e06af425064d38baa7e3370870d/scan21-result-40e-scale025.gif)
![finetuned](https://github.com/wuyichia/Cascade_MVSNeRF/blob/34e95dd84a9d7e06af425064d38baa7e3370870d/scan8-result-40e-scale025.gif)


Code apdapted from [**CasMVSNet_pl**](https://github.com/kwea123/CasMVSNet_pl), [**mvsnerf**](https://github.com/apchenstu/mvsnerf)


## Relevant Works
[**MVSNet: Depth Inference for Unstructured Multi-view Stereo (ECCV 2018)**](https://arxiv.org/abs/1804.02505)<br>
Yao Yao, Zixin Luo, Shiwei Li, Tian Fang, Long Quan

[**Cascade Cost Volume for High-Resolution Multi-View Stereo and Stereo Matching (CVPR 2020)**](https://arxiv.org/abs/1912.06378)<br>
Xiaodong Gu, Zhiwen Fan, Zuozhuo Dai, Siyu Zhu, Feitong Tan, Ping Tan

[**NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis (ECCV 2020)**](http://www.matthewtancik.com/nerf)<br>
Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng
