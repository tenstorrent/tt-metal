# YOLOP-s (Panoptic Driving Perception) on Tenstorrent Hardware

YOLOP-s multi-task model for autonomous driving using TTNN APIs.

## Architecture

| Component | Description |
|-----------|-------------|
| **Encoder** | CSPDarknet backbone + SPP + FPN |
| **Detection Head** | Object detection (vehicles, pedestrians) |
| **DA Segmentation** | Drivable area binary segmentation |
| **LL Segmentation** | Lane line binary segmentation |

## Multi-Task Outputs

| Output | Shape | Description |
|--------|-------|-------------|
| `detection` | [B, A*(5+C), H, W] | Bounding boxes, objectness, classes |
| `drivable_area` | [B, 2, H, W] | Binary segmentation mask |
| `lane_line` | [B, 2, H, W] | Lane line segmentation mask |

## Quick Start

### Run Demo
```bash
python models/demos/yolop/demo/demo_yolop.py --image driving_scene.jpg
```

### Run Tests
```bash
pytest models/demos/yolop/tests/test_model.py -v
```

## References
- [YOLOP Paper](https://arxiv.org/abs/2108.11250)
- [Official Repository](https://github.com/hustvl/YOLOP)
