# OWL-ViT (Open-World Localization Vision Transformer)

## Platforms
- Wormhole (N300)

## Introduction

This demo implements **OWL-ViT** (Vision Transformer for Open-World Localization) using TTNN APIs for zero-shot text-conditioned object detection on Tenstorrent hardware.

OWL-ViT is a powerful model that can detect objects based on natural language queries without category-specific training. It combines:
- **ViT-B/32 Image Encoder**: Processes images into patch embeddings
- **CLIP Text Encoder**: Encodes text queries into embeddings
- **Detection Heads**: Predicts bounding boxes and region-text similarity scores

Reference: [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230) (Minderer et al., 2022)

HuggingFace Model: [google/owlvit-base-patch32](https://huggingface.co/google/owlvit-base-patch32)

## Prerequisites

1. **TT-Metalium / TT-NN Installation**
   - Follow the installation guide at [INSTALLING.md](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

2. **HuggingFace Authentication**
   ```bash
   huggingface-cli login
   # OR
   export HF_TOKEN=<your_token>
   ```

3. **Python Dependencies**
   ```bash
   pip install transformers pillow requests
   ```

## Model Architecture

### Vision Encoder (ViT-B/32)
| Parameter | Value |
|-----------|-------|
| Hidden Size | 768 |
| Attention Heads | 12 |
| Encoder Layers | 12 |
| Intermediate Size | 3072 |
| Image Size | 768×768 |
| Patch Size | 32×32 |
| Num Patches | 576 (24×24) |

### Text Encoder (CLIP)
| Parameter | Value |
|-----------|-------|
| Hidden Size | 512 |
| Attention Heads | 8 |
| Encoder Layers | 12 |
| Vocabulary Size | 49,408 |
| Max Position Embeddings | 16 |

### Detection Heads
| Component | Description |
|-----------|-------------|
| Box Head | 3-layer MLP → 4 outputs (cx, cy, w, h) |
| Class Head | Projects patches to query space, computes similarity |

### 1. Run PyTorch Reference Demo (CPU)
Validate the reference implementation and generate baseline outputs:
```bash
python models/demos/wormhole/owl_vit/demo/demo_owl_vit_pytorch.py
```
Output saved to `demo/outputs/detection_result_pytorch.png`.

### 2. Run Demo (on TT Hardware)
Run the optimized TTNN implementation on Tenstorrent hardware:

```bash
# Run with default image (cats)
python models/demos/wormhole/owl_vit/demo/demo_owl_vit_inference.py

# Run with custom image and queries
python models/demos/wormhole/owl_vit/demo/demo_owl_vit_inference.py \
  --image "http://images.cocodataset.org/val2017/000000000285.jpg" \
  --queries "a bear" "video game" "grass" \
  --output "bear_detection.png"
```

### 3. Run Tests
Run the comprehensive test suite (unit tests and end-to-end PCC validation):

```bash
# Run all tests
pytest models/demos/wormhole/owl_vit/tests/ -v

# Run specific end-to-end detection test
pytest models/demos/wormhole/owl_vit/tests/test_end_to_end.py -v
```

## Directory Structure

```
models/demos/wormhole/owl_vit/
├── README.md                           # This file
├── __init__.py                         # Package exports
├── reference/
│   └── torch_owl_vit.py               # PyTorch reference implementation
├── tt/
│   └── ttnn_owl_vit.py                # TTNN implementation (modules)
├── tests/
│   ├── test_ttnn_owl_vit.py           # Unit tests
│   └── test_end_to_end.py             # Full detection pipeline + PCC tests
└── demo/
    ├── demo_owl_vit_inference.py      # TTNN inference demo (on device)
    ├── demo_owl_vit_pytorch.py        # PyTorch reference demo (on CPU)
    └── outputs/                        # Demo output images
```

## Implementation Details

### Optimization 1: LoFi Math & Fused Operations

1. **LoFi Math Fidelity**: Compute kernel config with `MathFidelity.LoFi` for faster matmul operations
2. **Fused QKV**: Query, Key, Value projections are fused into a single matrix multiplication
3. **Native TTNN Operations**: Full pipeline implemented using native ttnn ops
4. **GELU Fusion**: Feed-forward GELU activation is applied after linear operations
5. **Aggressive Deallocation**: Intermediate tensors are deallocated immediately after use

### Optimization 2: L1 Memory & Core Sharding

1. **L1 Memory Config**: Activations stored in L1 for faster access vs DRAM
2. **Full Core Grid**: Linear operations use `core_grid=(7, 8)` for 56-core utilization
3. **Sharded Vision Encoder**: `run_vision_encoder_layer_sharded()` for optimized performance

### Optimization 3: Vision Encoder FlashAttention

1. **SDPA Kernel**: Replaced manual implementation with `ttnn.transformer.scaled_dot_product_attention`
2. **Operation Fusion**: Fuses QK^T + Scale + Softmax + AV into a single kernel
3. **Optimized Layouts**: Avoids costly transpositions for K matrix

### Optimization 4: Text Encoder FlashAttention

1. **Text SDPA**: Extended FlashAttention optimization to Text Encoder
2. **L1 Sharding**: Text encoder layers now use L1 memory and optimized core grid
3. **Dynamic Padding**: Implemented input padding to satisfy SDPA chunk alignments

### Performance Results

Profiling results on N300 (Wormhole B0, 56 cores):

| Metric | DRAM Interleaved | L1 Sharded | Vision SDPA | Full Opt (Text+Vision) | Speedup |
|--------|------------------|------------|-------------|------------------------|---------|
| Device Time | 18.57 ms | 16.96 ms | 10.16 ms | **8.92 ms** | **2.08x** |
| Matmul Time | 11.34 ms | 10.80 ms | 5.03 ms | 3.73 ms | 3.04x |
| Attn Time | ~6.00 ms | ~5.80 ms | 0.80 ms | **1.03 ms** | **5.8x** |

### Optimization Configuration

```python
from models.demos.wormhole.owl_vit.tt.ttnn_owl_vit import (
    OwlViTTTNNConfig,
    run_vision_encoder_layer_sharded,  # Optimized version
)

# Default: LoFi enabled for performance
config = OwlViTTTNNConfig(use_lofi=True)
compute_kernel_config = config.get_compute_kernel_config()

# Use sharded layer for better core utilization
hidden = run_vision_encoder_layer_sharded(
    hidden, layer_params, config, device, compute_kernel_config
)
```



## Validation Criteria

- [x] Model implementation using TTNN APIs
- [x] Runs on N300 hardware
- [x] Vision encoder produces valid output
- [x] Full end-to-end detection with text encoder and heads
- [x] Output verifiable (region-text similarity scores, visualization)

## Current Status

**Vision Encoder**: Running on TT hardware
- All 12 transformer encoder layers execute on device
- Output shape: [1, 577, 768] matches PyTorch

**Text Encoder**: Running on TT hardware
- All 12 transformer encoder layers execute on device
- Embeddings computed via ttnn.embedding
- Output shape: [batch, 16, 512]

**Detection Heads**: Running on TT hardware
- Box head produces 576 box predictions [batch, 576, 4]
- Class head computes region-text similarity [batch, 576, num_queries]

**End-to-End Pipeline**:
- All components run on TT device
- Objects detected with high confidence scores
- Full detection pipeline functional

## Known Limitations

1. **Batch Size**: Currently optimized for batch size 1
2. **Image Size**: Fixed at 768×768 (OWL-ViT native resolution)

## References

- [OWL-ViT Paper](https://arxiv.org/abs/2205.06230)
- [HuggingFace Model Card](https://huggingface.co/google/owlvit-base-patch32)
- [CLIP Encoder Implementation](../../experimental/tt_dit/encoders/clip/model_clip.py)
