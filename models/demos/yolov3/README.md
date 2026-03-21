# YOLOv3 (416×416) on Tenstorrent Hardware

YOLOv3 object detection model implemented using TTNN APIs.

## Architecture

| Component | Description |
|-----------|-------------|
| **Backbone** | Darknet-53 (53 conv layers + residual blocks) |
| **Detection** | 3-scale: 13×13, 26×26, 52×52 feature maps |
| **Output** | COCO 80 classes × 3 anchors per scale |

## Features

- Multi-scale detection with FPN-style feature fusion
- Letterbox preprocessing for aspect ratio preservation
- Upsampling and concatenation for cross-scale features

## Quick Start

### Run Demo
```bash
python models/demos/yolov3/demo/demo_yolov3.py --image sample.jpg
```

### Run Tests
```bash
pytest models/demos/yolov3/tests/test_model.py -v
```

## Performance Notes

This implementation is designed for Stage 1-2 of the bounty requirements:
- **Stage 1**: Functional model with correct outputs
- **Stage 2**: Optimized memory configs (sharding) to be added

## References
- [YOLOv3 Paper](https://arxiv.org/abs/1804.02767)
- [Ultralytics Docs](https://docs.ultralytics.com/models/yolov3/)
