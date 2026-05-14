# Depth-Anything-V2-Large TTNN Bring-Up

## Overview

This directory contains the Tenstorrent TTNN implementation of [Depth-Anything-V2-Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large) for monocular depth estimation on Wormhole N300 hardware.

**Model**: Depth-Anything-V2-Large (DINOv2 ViT-L/14 backbone + DPT decoder head)
**Task**: Monocular depth estimation from single RGB images
**Input**: 518×518×3 RGB images
**Output**: Single-channel depth map (518×518)

## Architecture

```
Input Image (518×518×3)
    │
    ▼
Patch Embedding (Conv2d: 3→1024, k=14, s=14)
    │  37×37 = 1369 patches + 1 CLS token = 1370 tokens
    ▼
Positional Encoding (+ CLS token)
    │
    ▼
ViT-L/14 Encoder (24 transformer blocks)
    │  - LayerNorm → Multi-Head Self-Attention (16 heads) → LayerScale → Residual
    │  - LayerNorm → MLP (1024→4096→1024) → LayerScale → Residual
    │  Extract intermediate features at layers [4, 11, 17, 23]
    ▼
DPT Decoder Head
    │  - 4× 1×1 Conv projections (1024 → [256, 512, 1024, 1024])
    │  - Resize layers (ConvTranspose2d / Identity / Conv2d stride=2)
    │  - FeatureFusionBlocks with ResidualConvUnits
    │  - Output convolutions → depth map
    ▼
Depth Map (518×518)
```

## Directory Structure

```
models/demos/wormhole/depth_anything_v2/
├── __init__.py
├── tt/
│   ├── __init__.py
│   ├── depth_anything_v2_config.py    # Model configuration
│   └── ttnn_depth_anything_v2.py      # TTNN implementation
├── demo/
│   ├── __init__.py
│   └── demo_depth_anything_v2_inference.py  # Demo script
├── tests/
│   ├── __init__.py
│   └── test_depth_anything_v2.py      # Tests
└── README.md
```

## Setup

### Prerequisites

1. Tenstorrent Wormhole N300 (or N150) hardware
2. tt-metal built and installed
3. Python dependencies:
   ```bash
   pip install torch torchvision opencv-python loguru
   ```

### Model Weights

Download from HuggingFace:
```bash
# Option 1: Using Depth-Anything-V2 official repo
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
# Download ViT-L checkpoint from their instructions

# Option 2: Using HuggingFace transformers
pip install transformers
# Model weights are auto-downloaded on first run
```

## Usage

### Running the Demo

```bash
# Run with default test image
python models/demos/wormhole/depth_anything_v2/demo/demo_depth_anything_v2_inference.py

# Run with custom image
python models/demos/wormhole/depth_anything_v2/demo/demo_depth_anything_v2_inference.py \
    --image path/to/image.jpg \
    --output depth_output.png

# Skip PyTorch reference comparison
python models/demos/wormhole/depth_anything_v2/demo/demo_depth_anything_v2_inference.py \
    --no-reference
```

### Running Tests

```bash
# Run all tests
pytest models/demos/wormhole/depth_anything_v2/tests/ -v

# Run specific test
pytest models/demos/wormhole/depth_anything_v2/tests/test_depth_anything_v2.py::test_depth_anything_v2_inference -v

# Run throughput benchmark
pytest models/demos/wormhole/depth_anything_v2/tests/test_depth_anything_v2.py::test_depth_anything_v2_throughput -v
```

## Implementation Details

### Stage 1: Bring-Up (Current)

- ✅ Full ViT-L/14 backbone (24 transformer blocks) implemented in TTNN
- ✅ Patch embedding via unfold + linear
- ✅ Multi-head self-attention with QKV projection
- ✅ MLP with GELU activation
- ✅ LayerScale support (init_values=1.0)
- ✅ Intermediate feature extraction at layers [4, 11, 17, 23]
- ✅ DPT decoder head (CPU fallback for Stage 1)
- ✅ Weight preprocessing and device loading
- ✅ Sub-module unit tests (LayerNorm, MLP, Attention)
- ✅ End-to-end inference test with PCC validation
- ✅ Throughput benchmarking

### Stage 2: Basic Optimizations (Planned)

- [ ] Migrate DPT decoder head to TTNN
- [ ] Optimal sharded/interleaved memory configs for ViT encoder
- [ ] Efficient sharding strategy for patch embedding and transformer blocks
- [ ] Fuse simple ops (GELU+LayerNorm, attention softmax)
- [ ] Store intermediate activations in L1
- [ ] Use TT library of fused ops for attention and MLP blocks
- [ ] Optimize DPT decoder head

### Stage 3: Deeper Optimization (Planned)

- [ ] Maximize core utilization
- [ ] Efficient multi-head attention with optimal head sharding
- [ ] Optimized patch embedding and position encoding
- [ ] Minimize tensor reshaping and transpositions in decoder
- [ ] Efficient upsampling in DPT head
- [ ] Document advanced tuning and known limitations

## Key Model Parameters

| Parameter | Value |
|-----------|-------|
| Image Size | 518×518 |
| Patch Size | 14×14 |
| Patches | 37×37 = 1369 |
| Sequence Length | 1370 (patches + CLS) |
| Embed Dim | 1024 |
| Num Heads | 16 |
| Head Dim | 64 |
| Num Layers | 24 |
| MLP Hidden Dim | 4096 |
| LayerScale Init | 1.0 |
| DPT Features | 256 |
| DPT Out Channels | [256, 512, 1024, 1024] |
| Intermediate Layers | [4, 11, 17, 23] |

## Performance Targets

| Metric | Target | Stage 1 |
|--------|--------|---------|
| Throughput | ≥ 15 FPS | TBD |
| Accuracy (PCC) | > 0.99 | ✅ |
| Resolution | 518×518 | ✅ |

## References

- [Depth-Anything-V2 Paper](https://arxiv.org/abs/2406.09414)
- [Depth-Anything-V2 GitHub](https://github.com/DepthAnything/Depth-Anything-V2)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [TTNN Model Bring-up Tech Report](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/ttnn/TTNN-model-bringup.md)
- [ViT TTNN Documentation](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/ViT-TTNN/vit.md)
