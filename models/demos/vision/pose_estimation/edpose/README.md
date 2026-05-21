# ED-Pose on TT P150

End-to-end human pose estimation using [ED-Pose](https://github.com/IDEA-Research/ED-Pose) (DETR-based) ported to Tenstorrent P150 via TT-NN.

**Model**: ED-Pose Swin-L 5-scale, COCO val2017 AP = 75.8 (official)

## Architecture

ED-Pose is a fully end-to-end pose estimator built on Deformable DETR. No separate person detector or top-down cropping is needed — the model directly predicts bounding boxes and 17 COCO keypoints per person in a single forward pass.

```
Image (800x1216)
  │
  ▼
┌─────────────────────────────────┐
│  Swin-L Backbone (CPU)          │  4 stages → 4 feature maps
│  + Input Projection (1x1 conv)  │  + 1 extra level (stride-2 conv)
│  + Sine Position Encoding       │  → 5-scale features, ~81K tokens
└─────────────┬───────────────────┘
              │ src_flatten (1, 80997, 256)
              ▼
┌─────────────────────────────────┐
│  Deformable Encoder (ttnn)      │  6 layers x MSDeformAttn + FFN
│  Multi-Scale Deformable Attn    │  self-attn over all 81K tokens
└─────────────┬───────────────────┘
              │ memory (1, 80997, 256)
              ▼
┌─────────────────────────────────┐
│  Two-Stage Query Generation     │  Top-900 proposals from encoder
│  (CPU)                          │  output → initial queries + refs
└─────────────┬───────────────────┘
              │ tgt (1, 900, 256), refpoints (1, 900, 4)
              ▼
┌─────────────────────────────────┐
│  Deformable Decoder (ttnn)      │  6 layers: self-attn + cross-attn + FFN
│  - Layers 0-1: box detection    │  900 queries
│  - Layer 1→2: query expansion   │  900 → 100 groups × 18 = 1800 queries
│  - Layers 2-5: pose refinement  │  (1 box + 17 keypoint queries per group)
└─────────────┬───────────────────┘
              │ hs (6 layers), references (7 sets)
              ▼
┌─────────────────────────────────┐
│  Prediction Heads (CPU)         │  class, bbox, pose MLPs
│  + PostProcess                  │  → scores, boxes, 17 keypoints per person
└─────────────────────────────────┘
```

### Key Design Decisions

- **Backbone on CPU**: Swin-L window attention (12x12 windows = 144 tokens) produces many small matmuls that don't saturate the device. Measured host↔device transfer overhead (769ms) far exceeds device compute savings (69ms).
- **Encoder/Decoder on device**: MSDeformAttn, FFN, and LayerNorm run on TT device via ttnn. Grid sample uses `ttnn.grid_sample` with precomputed bilinear weights.
- **Two-stage on CPU**: Dynamic indexing (topk, gather) required for proposal selection.
- **Decoder query expansion on CPU**: At layer 2 boundary, 900 box queries expand to 1800 (100 groups × 18), involving dynamic selection that stays on host.
- **bfloat16 compute**: All device operations run in bfloat16.

### Module Files

| File | Description |
|------|-------------|
| `common/tt/ttnn_swin_backbone.py` | Swin-L backbone with optional `torch.compile` |
| `common/tt/ttnn_edpose_backbone.py` | Backbone + input projection + position encoding |
| `common/tt/ttnn_ms_deform_attn.py` | Multi-Scale Deformable Attention (ttnn) |
| `common/tt/ttnn_deformable_encoder.py` | 6-layer deformable encoder (ttnn) |
| `common/tt/ttnn_deformable_decoder.py` | 6-layer deformable decoder (ttnn) |
| `common/tt/ttnn_edpose_decoder_wrapper.py` | Decoder + query expansion + iterative refinement |
| `common/tt/ttnn_edpose_two_stage.py` | Two-stage query generation from encoder output |
| `common/reference/edpose_reference.py` | CPU-only reference implementation |

## Prerequisites

### ED-Pose Repository and Weights

```bash
# Clone ED-Pose (with PyTorch grid_sample fallback patch)
cd ~/ttwork
git clone https://github.com/IDEA-Research/ED-Pose.git

# Download Swin-L 5-scale checkpoint
mkdir -p ED-Pose/weights
# Place edpose_swinl_5scale_coco.pth in ED-Pose/weights/
```

### COCO Dataset (for evaluation)

```bash
mkdir -p ~/datasets/coco
# Download val2017 images and annotations:
#   - val2017/*.jpg
#   - annotations/person_keypoints_val2017.json
```

### Environment Variables

```bash
export TT_METAL_HOME=/home/yito/ttwork/tt-metal
export PYTHONPATH=$TT_METAL_HOME
export EDPOSE_ROOT=~/ttwork/ED-Pose
```

### Python Dependencies

```bash
pip install pycocotools opencv-python
```

## Quick Start: Single-Image Demo

```bash
# Run on a specific image
python models/demos/vision/pose_estimation/edpose/demo.py \
  --image /path/to/image.jpg

# Run on first COCO val2017 image (default)
python models/demos/vision/pose_estimation/edpose/demo.py

# Adjust detection threshold
python models/demos/vision/pose_estimation/edpose/demo.py \
  --image /path/to/image.jpg --score-threshold 0.5
```

Output: prints per-phase timing, detected persons with bounding boxes, and 17 COCO keypoints with confidence scores.

### Demo with Pose Overlay (saves image)

```bash
python models/demos/vision/pose_estimation/edpose/common/tests/run_demo_with_overlay.py
```

Draws bounding boxes, skeleton lines, and keypoint dots on the image. Output saved to `/home/yito/edpose_demo_output.jpg`. Edit `INPUT_IMAGE` and `OUTPUT_PATH` constants in the script to change paths.

## COCO Evaluation

### TT Device Evaluation (encoder + decoder on device)

```bash
# Quick test (100 images)
python models/demos/vision/pose_estimation/edpose/common/tests/run_coco_eval.py \
  --coco-dir /home/yito/datasets/coco --max-images 100

# Full evaluation (~5000 images, ~5h)
python models/demos/vision/pose_estimation/edpose/common/tests/run_coco_eval.py \
  --coco-dir /home/yito/datasets/coco

# Save raw results to JSON
python models/demos/vision/pose_estimation/edpose/common/tests/run_coco_eval.py \
  --coco-dir /home/yito/datasets/coco --max-images 100 --output results.json
```

Images are sorted by padded size by default to minimize JIT recompilations. Use `--no-sort-by-size` to disable.

### CPU Reference Evaluation (fp32 baseline)

```bash
python models/demos/vision/pose_estimation/edpose/common/tests/run_cpu_ref_eval.py \
  --coco-dir /home/yito/datasets/coco --max-images 10
```

Runs the entire pipeline in fp32 on CPU to establish a ground-truth AP baseline (validates bfloat16 accuracy loss).

## Pipeline Validation

### E2E Pipeline Test

```bash
python models/demos/vision/pose_estimation/edpose/common/tests/run_e2e_pipeline_test.py
```

Runs backbone → encoder(ttnn) → two-stage → decoder(CPU) → prediction heads with synthetic input. Prints per-phase timing and top detection scores. Uses CPU decoder for validation; the demo uses ttnn decoder.

## Profiling

### E2E Latency Profile

```bash
python models/demos/vision/pose_estimation/edpose/common/tests/profile_e2e.py
```

Runs the full pipeline 3 times (1 cold + 2 warm) with synthetic 800×1216 input. Reports per-phase breakdown:

```
  Backbone (CPU):    XXXXms
  Encoder  (ttnn):   XXXXms
  Two-stage (CPU):   XXXXms
  Decoder  (ttnn):   XXXXms
  Heads    (CPU):    XXXXms
  Total:             XXXXms
```

### Backbone (Swin-L) Profile

```bash
python models/demos/vision/pose_estimation/edpose/common/tests/profile_backbone.py
```

Profiles Swin-L internals: PatchEmbed, per-stage timing, input projection, position encoding, and block-level timing within Stage 2 (the dominant stage with 18 blocks).

### Swin-L with torch.compile Profile

```bash
python models/demos/vision/pose_estimation/edpose/common/tests/profile_swin_backbone.py
```

Profiles `TTSwinBackbone` which wraps Swin-L with optional `torch.compile(mode="reduce-overhead")`.

### Decoder Per-Op Profile

```bash
python models/demos/vision/pose_estimation/edpose/common/tests/profile_decoder.py
```

Instruments each sub-operation within decoder layers with device synchronization barriers:
- Self-attention: QKV linear, split_heads, matmul, scale, mask, softmax, merge, out_proj
- Cross-attention (MSDeformAttn) total
- FFN: linear, relu, add, layer_norm
- Host↔device transfers: from_torch, to_torch

Runs a warm-up pass first (JIT compilation), then a profiled pass + 3 overall decoder timing runs.

### Cross-Attention (MSDeformAttn) Profile

```bash
# Decoder cross-attention at 900/1800-query scale
python models/demos/vision/pose_estimation/edpose/common/tests/profile_cross_attn.py

# Encoder self-attention at 81K-query scale
python models/demos/vision/pose_estimation/edpose/common/tests/profile_encoder_msda.py
```

Fine-grained sub-op timing within multi-scale deformable attention: value projection, offset computation, grid sample per level, attention weight softmax, weighted aggregation, output projection.

### Two-Stage Query Generation Profile

```bash
python models/demos/vision/pose_estimation/edpose/common/tt/tests/profile_two_stage.py
```

Profiles the CPU-side two-stage proposal generation: enc_output projection, class scoring, topk selection, box proposal generation.

## Unit Tests

```bash
# Multi-Scale Deformable Attention correctness
python models/demos/vision/pose_estimation/edpose/common/tests/test_ms_deform_attn.py

# Swin-L backbone output validation
python models/demos/vision/pose_estimation/edpose/common/tests/test_swin_backbone.py

# Grid sample compatibility (ttnn vs PyTorch)
python models/demos/vision/pose_estimation/edpose/common/tests/test_grid_sample_compat.py

# Encoder PCC (Pearson Correlation Coefficient) vs CPU reference
python models/demos/vision/pose_estimation/edpose/common/tt/tests/test_encoder_pcc.py

# Grid precompute validation
python models/demos/vision/pose_estimation/edpose/common/tt/tests/test_python_grid_precompute.py
```

## Additional Test Scripts

| Script | Description |
|--------|-------------|
| `run_backbone_test.py` | Validate backbone output shapes and values |
| `run_backbone_encoder_test.py` | Backbone + encoder integration test |
| `run_full_encoder_test.py` | Full 6-layer encoder correctness test |
| `run_encoder_decoder_test.py` | Encoder + decoder integration test |
| `run_encoder_integration_test.py` | Encoder integration with two-stage |
| `run_grid_sample_test.py` | Grid sample operation test |
| `run_ms_deform_attn_test.py` | MSDeformAttn integration test |
| `run_debug_bbox.py` | Debug bounding box predictions |
| `run_debug_proposal_overlap.py` | Debug two-stage proposal overlap |
| `bench_cpu_backbone.py` | CPU backbone throughput benchmark |

## Directory Structure

```
edpose/
├── README.md                           # This file
├── demo.py                             # Single-image E2E inference demo
├── __init__.py
├── common/
│   ├── __init__.py
│   ├── reference/
│   │   ├── __init__.py
│   │   └── edpose_reference.py         # CPU-only reference model
│   ├── tt/
│   │   ├── __init__.py
│   │   ├── ttnn_swin_backbone.py       # Swin-L backbone (CPU, optional torch.compile)
│   │   ├── ttnn_edpose_backbone.py     # Backbone + input_proj + pos encoding
│   │   ├── ttnn_ms_deform_attn.py      # Multi-Scale Deformable Attention (ttnn)
│   │   ├── ttnn_deformable_encoder.py  # Deformable encoder (ttnn)
│   │   ├── ttnn_deformable_decoder.py  # Deformable decoder layers (ttnn)
│   │   ├── ttnn_edpose_decoder_wrapper.py  # Decoder + query expansion + heads
│   │   ├── ttnn_edpose_two_stage.py    # Two-stage query generation (CPU)
│   │   └── tests/
│   │       ├── profile_two_stage.py
│   │       ├── test_encoder_pcc.py
│   │       └── test_python_grid_precompute.py
│   └── tests/
│       ├── run_inference_demo.py       # Single-image demo (original)
│       ├── run_demo_with_overlay.py    # Demo with pose overlay output
│       ├── run_e2e_pipeline_test.py    # Full pipeline validation
│       ├── run_coco_eval.py            # COCO val2017 AP evaluation
│       ├── run_cpu_ref_eval.py         # CPU fp32 reference evaluation
│       ├── profile_e2e.py              # E2E latency profiling
│       ├── profile_backbone.py         # Swin-L stage profiling
│       ├── profile_swin_backbone.py    # TTSwinBackbone profiling
│       ├── profile_decoder.py          # Decoder per-op profiling
│       ├── profile_cross_attn.py       # Cross-attention profiling
│       ├── profile_cross_attn_v2.py    # Cross-attention profiling (v2)
│       ├── profile_encoder_msda.py     # Encoder MSDeformAttn profiling
│       ├── run_backbone_test.py
│       ├── run_backbone_encoder_test.py
│       ├── run_full_encoder_test.py
│       ├── run_encoder_decoder_test.py
│       ├── run_encoder_integration_test.py
│       ├── run_grid_sample_test.py
│       ├── run_ms_deform_attn_test.py
│       ├── run_debug_bbox.py
│       ├── run_debug_proposal_overlap.py
│       ├── bench_cpu_backbone.py
│       ├── test_ms_deform_attn.py
│       ├── test_swin_backbone.py
│       └── test_grid_sample_compat.py
```
