# Depth Anything V2 Large on Tenstorrent

This directory contains an initial port of the Depth Anything V2 Large model to Tenstorrent's `ttnn` library.

## Introduction

Depth Anything V2 is a state-of-the-art monocular depth estimation model. This implementation uses the `ttnn` library to run on Tenstorrent hardware (Wormhole_B0).

## Requirements

- A Tenstorrent device (Wormhole_B0 or Blackhole)
- `tt-metal` environment installed
- Python dependencies:
  ```bash
  pip install torch transformers pillow opencv-python
  ```

## Running the Demo

The demo script downloads the model from Hugging Face and initializes it on the Tenstorrent device.

```bash
python models/demos/depth_anything_v2/demo/demo.py --model_id "depth-anything/Depth-Anything-V2-Large-hf"
```

## Running the Tests

To run the unit tests:

```bash
pytest models/demos/depth_anything_v2/tests/test_model.py
```

## Implementation Details

- **Backbone**: ViT Large (from DINOv2) ported to `ttnn`.
- **Neck & Head**: DPT structure implemented using `ttnn` operations.
- **Layout**: Optimized for `TILE_LAYOUT` where possible for efficient matmuls.
