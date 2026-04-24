# Depth Anything V2 Large on Tenstorrent

This directory contains a port of the [Depth Anything V2 Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf) model to Tenstorrent's `ttnn` library for Wormhole B0 hardware.

## Introduction

Depth Anything V2 is a state-of-the-art monocular depth estimation model built on a DINOv2 ViT-Large backbone with a DPT (Dense Prediction Transformer) neck and head. This implementation runs the full pipeline on Tenstorrent Wormhole B0 devices (N150/N300).

## Requirements

- A Tenstorrent device (Wormhole B0: N150 or N300)
- `tt-metal` environment installed
- Python dependencies:
  ```bash
  pip install torch transformers pillow
  ```

## File Layout

```
models/experimental/depth_anything_v2/
├── README.md                             # This file
├── tt/
│   ├── __init__.py
│   └── model_def.py                      # Core model: ViT backbone, DPT neck/head, weight preprocessor
├── demo/
│   ├── demo.py                           # Demo: load model, run inference, save depth map
│   └── validate.py                       # Validation: PCC vs PyTorch reference + FPS benchmark
└── tests/
    ├── test_depth_anything_v2_pcc.py      # PCC accuracy test (PCC > 0.99 vs HuggingFace)
    ├── test_depth_anything_v2_perf.py     # Inference timing with profiler + perf report
    └── test_depth_anything_v2_device_perf.py  # Device-level perf via run_device_perf
```

## Running the Demo

The demo script downloads the model from Hugging Face and initializes it on the Tenstorrent device.

```bash
python models/experimental/depth_anything_v2/demo/demo.py
```

To use a different model, edit the `model_id` parameter in the `run_demo()` call inside `demo.py`.

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

- **Backbone**: DINOv2 ViT-Large (24 transformer layers, 16 heads, hidden_size=1024) ported to `ttnn`.
- **Neck**: DPT Reassemble (4 scales with projection + spatial resampling) + DPT Fusion (top-down feature pyramid with residual conv blocks).
- **Head**: Two 3×3 convolutions with ReLU + 2× upsample + 1×1 depth projection.
- **Features extracted at**: Layers 5, 11, 17, 23 (matching HuggingFace config).

## Architecture

```
pixel_values (B, 3, 518, 518)
    │  Patch Embed (14×14 conv → flatten → project to 1024)
    │  + CLS token + Position Embeddings
    ▼
ViT-Large Encoder (24 layers)
    │  Extract features at layers {5, 11, 17, 23}
    ▼
DPT Reassemble × 4
    │  Linear projection → spatial reshape → resample
    ▼
DPT Fusion (top-down)
    │  Upsample + add + two residual conv blocks per level
    ▼
DPT Head
    │  Conv 256→256 → ReLU → Upsample 2× → Conv 256→128 → ReLU → Conv 128→1
    ▼
Depth Map (B, 1, H_out, W_out)
```

## Performance

### Current Status (Baseline / Stage 1)

The current implementation focuses on **correctness and baseline performance** using a DRAM-resident execution path.

#### Execution Model
- **Memory Strategy**: All activations stored in DRAM (`DRAM_MEMORY_CONFIG`). No L1 sharding.
- **Sequence Padding**: Sequence length padded to **1408 tokens** (44 tiles) to maintain tile alignment.
- **Attention**: Standard `ttnn.softmax` (not fused `attention_softmax_`). Residual connections use `ttnn.add` (not in-place).
- **QKV Fusion**: Q, K, V weights are pre-fused into a single (1024, 3072) matrix for efficient single-matmul projection.
- **Memory Cleanup**: Aggressive `ttnn.deallocate()` throughout the encoder to minimize DRAM footprint.

#### Layout Optimizations
- **TILE Layout**: Maintained through the entire backbone and neck reassembly projection for efficient matmuls.
- **Projection-before-slice**: Reassembly projections operate on the padded sequence before slicing to CLS-free patch tokens, keeping operations aligned to tile boundaries.
- **Conv weights**: `bfloat16` + `ROW_MAJOR_LAYOUT` (bfloat8_b requires TILE_LAYOUT which is invalid for conv kernels).
- **Attention weights**: `bfloat8_b` + `TILE_LAYOUT` for maximum Wormhole matrix engine throughput.

### Performance Targets

| Resolution | Hardware | Batch Size | Target FPS | Target PCC |
| :--- | :--- | :--- | :--- | :--- |
| 518×518 | Wormhole B0 (N150/N300) | 1 | ≥ 15 FPS | > 0.99 |

> **Note**: Actual measured FPS and PCC values will be recorded after hardware validation with `validate.py`.

### Future Optimizations (Stage 2/3)

The following optimizations are planned but **not yet implemented**:

- **L1-sharded activations**: Move intermediate tensors from DRAM to L1 sharded buffers.
- **8×8 grid sharding**: Shard the backbone across all 64 cores with 2048-token padding (1 tile per core).
- **Fused attention**: Replace `ttnn.softmax` with `ttnn.transformer.attention_softmax_` for in-place hardware acceleration.
- **In-place residual adds**: Use `ttnn.add_` to reduce DRAM traffic.
- **Trace + 2CQ execution**: Dual command queue with trace capture for pipelined inference.

### Accuracy Verification

The PCC test (`test_depth_anything_v2_pcc.py`) validates that the ttnn output achieves **PCC > 0.99** against the PyTorch HuggingFace reference across random test inputs.

### How to Run Profiling

To generate a detailed op-by-op performance trace:
```bash
# Enable ttnn profiler
export TT_METAL_PROFILER_OP_INFO=1
pytest models/experimental/depth_anything_v2/tests/test_depth_anything_v2_perf.py
```
Check the generated `profile_log.csv` for detailed cycle counts per operation.

### Hardware Validation

To run the full validation suite (PCC + FPS measurement):
```bash
python models/experimental/depth_anything_v2/demo/validate.py
```

This produces:
- Side-by-side depth map comparisons (input / PyTorch reference / ttnn)
- Per-image PCC scores
- FPS benchmark with warmup
- CSV summaries in the `validation/` directory
