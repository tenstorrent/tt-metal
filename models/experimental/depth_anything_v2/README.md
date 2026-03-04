# Depth Anything V2 Large on Tenstorrent

This directory contains an optimized port of the Depth Anything V2 Large model to Tenstorrent's `ttnn` library.

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
python models/experimental/depth_anything_v2/demo/demo.py --model_id "depth-anything/Depth-Anything-V2-Large-hf"
```

## Running the Tests

To run the PCC accuracy test:

```bash
pytest models/experimental/depth_anything_v2/tests/test_depth_anything_v2_pcc.py
```

To run the performance test:

```bash
pytest models/experimental/depth_anything_v2/tests/test_depth_anything_v2_perf.py
```

To run the device performance test:

```bash
pytest models/experimental/depth_anything_v2/tests/test_depth_anything_v2_device_perf.py
```

## Implementation Details

- **Backbone**: ViT Large (from DINOv2) ported to `ttnn`.
- **Neck & Head**: DPT structure implemented using `ttnn` operations.
- **Layout**: Optimized for `TILE_LAYOUT` where possible for efficient matmuls.

## Performance

### Optimization Overview (Stage 3)

The implementation leverages Stage 3 "Deeper Optimizations" to achieve maximum throughput and hardware utilization.

#### 1. Hardware-Aligned Sharding (8x8 Grid)
- **Backbone**: The ViT-Large backbone is fully sharded across a **64-core grid (8x8)**.
- **Sequence Padding**: The sequence length is padded to **2048 tokens** (64 tiles). This ensures that each core in the 8x8 grid handles exactly **1 tile** (32x32) of the hidden states, achieving 100% compute balance.
- **Width Sharding**: Operations are width-sharded where appropriate to minimize Inter-connect (NoC) congestion during large matmuls.

#### 2. Layout & Memory Optimizations
- **TILE Layout Persistence**: The model maintains `TILE_LAYOUT` through the entire backbone and neck reassembly projection.
- **Projection Realignment**: Reassembly projections are performed on the padded sequence *before* slicing. This keeps the operation aligned to tile boundaries (32-multiple), avoiding expensive layout conversion and preserving peak tensor throughput.
- **In-place Operations**: Fused `ttnn.transformer.attention_softmax_` and in-place residual additions are used to minimize DRAM traffic and L1 memory footprints.
- **L1-Sharded Memory**: Intermediate activations are stored in L1 sharded buffers wherever possible (Encoder layers, Neck stages) to reduce DRAM latency.

### Performance Targets & Results

| Resolution | Hardware | Stage | Target FPS | Expected PCC |
| :--- | :--- | :--- | :--- | :--- |
| 518x518 | Wormhole_B0 | Stage 3 | > 15 FPS | > 0.99 |

### Accuracy Verification
The model achieves a **PCC > 0.99** against the PyTorch reference implementation across standard test images, ensuring that optimizations have not compromised the depth map quality.

### How to Run Profiling
To generate a detailed op-by-op performance trace:
```bash
# Enable ttnn profiler
export TT_METAL_PROFILER_OP_INFO=1
pytest models/experimental/depth_anything_v2/tests/test_depth_anything_v2_perf.py
```
Check the generated `profile_log.csv` for detailed cycle counts per operation.
