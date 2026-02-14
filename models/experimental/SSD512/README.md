# SSD512

### Platforms: Wormhole (n150)
### Supported Input Resolution:** `(512, 512)` = (Height, Width)

## Introduction
SSD512 (Single Shot MultiBox Detector) is a real-time object detection model that performs object detection in a single forward pass. The model uses a VGG backbone with additional extra layers and multibox heads to detect objects at multiple scales. SSD512 is implemented for 512x512 input images and can detect 21 object classes (VOC dataset format).

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## Setup

### Download Weights (Optional)
The demo currently uses random weights for testing. To use trained weights:
1. Download trained SSD512 weights (VOC format)
2. Load weights into the PyTorch model before transferring to TTNN


## Repository Layout
```
models/
└── experimental/
    └── SSD512/
        ├── resources/
        │   └── sample_input/
        │       └── 000007.jpg
        ├── reference/
        │   ├── ssd.py                    # TorchSSD (reference)
        │   ├── voc0712.py
        │   ├── config.py
        │   ├── box_utils.py
        │   ├── detection.py
        │   ├── prior_box.py
        │   └── l2norm.py
        ├── tt/
        │   ├── tt_ssd.py                 # TtSSD (TTNN)
        │   ├── tt_vgg_backbone.py
        │   ├── tt_extras_backbone.py
        │   ├── tt_l2norm.py
        │   ├── tt_multibox_heads.py
        │   └── utils.py
        ├── demo/
        │   ├── demo.py                   # CLI demo
        │   └── processing.py             # Pre/post-processing and visualization
        ├── tests/
        │   ├── pcc/
        │   │   ├── test_ssd512.py        # end-to-end pytest
        │   │   ├── test_vgg_backbone.py
        │   │   ├── test_extras_backbone.py
        │   │   └── test_multibox_heads.py
        │   └── perf/
        │       ├── test_ssd512_e2e_perf.py
        │       └── performant_infra.py   # SSD512PerformantTestInfra class
        ├── common.py                     # Common utilities and constants
        └── README.md
```

## Details

- The entry point to the TTNN SSD512 model is `TtSSD` in `models/experimental/SSD512/tt/tt_ssd.py`. The model uses random weights from the PyTorch reference implementation.
- Common utilities and constants are defined in `common.py`.
- Performance test infrastructure is encapsulated in `SSD512PerformantTestInfra` class in `tests/perf/performant_infra.py`.

## How to Run

### Run the Full Model Test
```bash
# From tt-metal root directory
pytest models/experimental/SSD512/tests/pcc/test_ssd512.py
```

### Performance
### Single Device (BS=1):
- Expected throughput: `66.5` FPS

### Run Device Performance Test
```bash
# Test full model performance
pytest models/experimental/SSD512/tests/perf/test_ssd512_e2e_perf.py
```

### Run the Demo
```bash
# Process a single image
python3 models/experimental/SSD512/demo/demo.py --input_image <path_to_image>

# With custom output path and detection parameters
python3 models/experimental/SSD512/demo/demo.py --input_image <path_to_image> --output_path <output_path> --conf_thresh 0.3 --max_detections 5
```

Example:
```bash
python3 models/experimental/SSD512/demo/demo.py --input_image models/experimental/SSD512/resources/sample_input/000007.jpg
```

### Demo Output Files

The demo generates output files for each processed image:
- `{image_name}_ttnn.jpg`: TTNN detection results with bounding boxes and labels.
- Output is saved to the same directory as the input image by default, or to the path specified by `--output_path`

## Configuration Notes
- Resolution: (H, W) = (512, 512) is supported end-to-end.
- Device: The demo opens a Wormhole device (default id typically 0). If you need to change it, adjust the DemoConfig or the device open call in the demo.
- Batch Size: Demo/tests are written for BS=1. For larger BS you’ll need to verify memory layouts and tile alignment.
