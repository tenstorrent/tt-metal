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
/pointpillars/ pretrained model. We've used weights available [here](https://github.com/open-mmlab/mmdetection3d/tree/main/configs/pointpillars#nuscenes) under FPN.

- The entry point to `TtMVXFasterRCNN` model in `models/experimental/pointpillars/tt/ttnn_mvx_faster_rcnn.py`.
- Batch Size : `1` (Single Device).
