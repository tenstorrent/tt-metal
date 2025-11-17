# Pointpillars

## Platforms:
Wormhole (N150, N300)

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
   - To obtain the perf reports through profiler, please build with: `./build_metal.sh -p`

## Introduction:

PointPillars is a fast and accurate 3D object detection framework designed for point cloud data, commonly used in autonomous driving and robotics. It encodes point clouds into vertical “pillars” using PointNets and leverages efficient 2D CNNs for detection, achieving state-of-the-art performance while maintaining high inference speed.

## How to Run:

Use the following command to run the model with pre-trained weights :

```sh
pytest models/experimental/pointpillars/tests/test_ttnn_mvx_faster_rcnn.py
```

### Details:

The model picks up certain configs and weights from mmdetection3d/configs
/pointpillars/ pretrained model. We've used weights available in [repo](https://github.com/open-mmlab/mmdetection3d/tree/main/configs/pointpillars#nuscenes) under FPN.
Weights download link [here](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth)

Currently, the model uses dumped inputs from the original model implementation. Here are the instructions to extract the required inputs.
#### Original model setup
Create a python environment and activate it.
- python3 -m venv mmdet_env
- source mmdet_env/bin/activate
Install required packages
- pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
- pip install numpy scipy matplotlib tqdm opencv-python pyyaml numba tensorboard
- pip install mmengine==0.10.4
- pip install mmcv==2.1.0
- pip install mmdet==3.3.0
- pip install mmdet3d==1.4.0
Clone the repository
- git clone https://github.com/open-mmlab/mmdetection3d.git -b dev-1.x
- cd mmdetection3d
Command to run the demo:
- python demo/pcd_demo.py demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth --device cpu --out-dir outputs

To extract the inputs for test_reference_model and test_ttnn_mvx_faster_rcnn, add the below script and run the demo:
Add the code in demo/pcd_demo.py before inferencer is called in main function
```
from mmdet3d.models.detectors import MVXTwoStageDetector
import torch
def predict_wrapper(self, batch_inputs_dict, batch_data_samples, **kwargs):
   torch.save(batch_inputs_dict, "batch_inputs_dict_orig.pth")
   torch.save(batch_data_samples, "batch_data_samples_orig.pth")
   return self.__class__.predict_orig(self, batch_inputs_dict, batch_data_samples, **kwargs)
MVXTwoStageDetector.predict_orig = MVXTwoStageDetector.predict
MVXTwoStageDetector.predict = predict_wrapper
```
The .pth files saved can be moved to the required directory in tt-metal and model can be tested.

### TTNN Model:
- The entry point to `TtMVXFasterRCNN` model in `models/experimental/pointpillars/tt/ttnn_mvx_faster_rcnn.py`.
- Batch Size : `1` (Single Device).
