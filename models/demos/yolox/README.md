# YOLOX on Tenstorrent Hardware

YOLOX object detection model implemented using TTNN APIs for Tenstorrent Wormhole devices.

## Model Architecture

| Component | Description |
|-----------|-------------|
| **Backbone** | CSPDarknet53 with cross-stage partial connections |
| **Neck** | FPN + PAN for multi-scale feature fusion |
| **Head** | Anchor-free decoupled head (cls/reg/obj branches) |

## Supported Variants

| Variant | Depth | Width | Params |
|---------|-------|-------|--------|
| YOLOX-Nano | 0.33 | 0.25 | 0.91M |
| YOLOX-Tiny | 0.33 | 0.375 | 5.06M |
| YOLOX-S | 0.33 | 0.50 | 9.0M |
| YOLOX-M | 0.67 | 0.75 | 25.3M |
| YOLOX-L | 1.0 | 1.0 | 54.2M |
| YOLOX-X | 1.33 | 1.25 | 99.1M |

## Quick Start

### Run Demo
```bash
python models/demos/yolox/demo/demo_yolox.py --image sample.jpg --model_variant yolox-s
```

### Run Tests
```bash
pytest models/demos/yolox/tests/test_model.py -v
```

## References

- [YOLOX Paper](https://arxiv.org/abs/2107.08430)
- [Official Repository](https://github.com/Megvii-BaseDetection/YOLOX)
