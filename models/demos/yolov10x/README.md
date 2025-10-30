# YOLOV10x

## Platforms:
    Wormhole (n150, n300)

## Introduction:
Demo showcasing Yolov10x running on Wormhole - n150, n300 using ttnn.

YOLOv10x introduces a new approach to real-time object detection, addressing both the post-processing and model architecture deficiencies found in previous YOLO versions. By eliminating non-maximum suppression (NMS) and optimizing various model components, YOLOv10x achieves state-of-the-art performance with significantly reduced computational overhead. Extensive experiments demonstrate its superior accuracy-latency trade-offs across multiple model scales. We've used weights available [here](https://docs.ultralytics.com/models/yolov10x/#performance) under YOLOV10x.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run

Find yolov10x instructions for the following devices:

- Wormhole (n150, n300): [models/demos/wormhole/yolov10x](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/wormhole/yolov10x)

- Blackhole (p150): [models/demos/blackhole/yolov10x](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/blackhole/yolov10x)
