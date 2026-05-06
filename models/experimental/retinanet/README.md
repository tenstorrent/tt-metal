# RetinaNet

### Platforms: Wormhole (n150)
### Supported Input Resolution:** `(512, 512)` = (Height, Width)

## Introduction
RetinaNet employs a ResNet50 backbone with an FPN to extract multi-scale features from five pyramid levels. The regression head predicts bounding box coordinates for each anchor, while the classification head outputs class probabilities across all anchors. Together, these components enable efficient multi-scale object detection.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## Setup

### Download Weights (Optional)
The demo currently uses default pre-trained weights (RetinaNet_ResNet50_FPN_V2_Weight)


## Repository Layout

| Directory | Purpose |
|------------|----------|
| `tt/` | Core Tenstorrent native modules of **Retinanet** |
| `demo/` | Demo scripts and visualization |
| `resources/` | Sample images for testing |
| `tests/` | Validation(PCC) and Performance test scripts |


The `retinanet/` directory plugs into this structure, exposing inference, profiling, and test utilities consistent with other models in the repo.

## How to Run

### Run the Full Model Test
```bash
# From tt-metal root directory
pytest models/experimental/retinanet/tests/pcc/test_retinanet.py
```

This runs an end-to-end flow that:
- Loads the Torch reference from Torchvision
- Runs the TT-NN graph
- Post-processes outputs
- Compares results and validates PCC

Note: GroupNorm in the head is set to fall back to the PyTorch implementation because the TTNN version resulted in lower PCC (around 0.97 for the regression head and 0.88 for the classification head). With the fallback enabled, the PCC improves to approximately 0.99. To disable this behavior, set `export FALLBACK_ON_GROUPNORM=0`.

### Performance
### Run Device Performance Test
```bash
pytest models/experimental/retinanet/tests/perf/test_perf.py
```
- End-to-end performance: 35.75 FPS

### Run the Demo
```bash
python models/experimental/retinanet/demo/demo.py \
  --input <path/to/image.png> \
  --output <path/to/output_dir>
```
Note: Currently, the default input image directory path for the demo: models/experimental/retinanet/resources
Default directory path for output image: models/experimental/retinanet/resources/outputs/
The demo generates output files for each processed image:
- `{result}.jpg`: TTNN detection results with bounding boxes and labels

Default directory paths:
- Input images: `models/experimental/retinanet/resources`
- Output images: `models/experimental/retinanet/resources/outputs`

## Configuration Notes
- Resolution: (H, W) = (512, 512) is supported end-to-end.
- Device: The demo opens a Wormhole device (default id typically 0). If you need to change it, adjust the DemoConfig or the device open call in the demo.
- Batch Size: Demo/tests are written for BS=1. For larger BS you’ll need to verify memory layouts and tile alignment.
