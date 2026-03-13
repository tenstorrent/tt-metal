# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# RT-DETR (Real-Time DEtection TRansformer)

## Platforms
- Wormhole (N300)

## Introduction

This repository implements **RT-DETR** (Real-Time DEtection TRansformer) using TTNN APIs for high-performance object detection on Tenstorrent hardware.

RT-DETR is an end-to-end object detector that overcomes the slow inference speed of standard DETR models. It achieves this via an efficient hybrid architecture:
- **CNN Backbone**: ResNet-50 for multi-scale feature extraction.
- **AIFI Encoder**: Advanced Image Feature Interaction (Transformer) to process high-level semantic features.
- **Decoder**: Cross-attention mechanisms for bounding box and class prediction.

Reference:[DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069) (Zhao et al., 2023)
Official Repo: [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR)

## Prerequisites

1. **TT-Metalium / TTNN Installation**
   Follow the installation guide at [INSTALLING.md](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md). Ensure your environment variables are sourced correctly.

2. **Python Dependencies**
   ```bash
   pip install torch torchvision pillow pycocotools pytest
   ```

3. **Download Weights**
   ```bash
   mkdir -p weights
   wget https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth -O weights/rtdetr_r50vd.pth
   ```

## Model Architecture

### Backbone (ResNet-50)
| Parameter | Value |
|-----------|-------|
| Stem | 3x3 Convolutions + MaxPool |
| Bottleneck Blocks | [3, 4, 6, 3] layout |
| Extracted Features | C3, C4, C5 (Multi-scale) |

### Transformer Encoder (AIFI)
| Parameter | Value |
|-----------|-------|
| Hidden Dimension | 256 |
| Attention Heads | 8 |
| Feed-Forward Dim | 1024 |
| Num Layers | 1 (AIFI spec) / Configurable up to 6 |
| Sequence Length | 300 (Flattened features) |

## Quick Start Guide

### 1. Run Inference Demo (End-to-End)
Runs a sample image through the TTNN-optimized backbone and encoder, generating bounding boxes.
```bash
python3 demo/demo_inference.py
```
*Output saved to `demo/output.jpg`.*

### 2. Run Performance Benchmark
Measures the throughput and latency of the optimized Transformer Encoder.
```bash
python3 benchmark.py
```

### 3. Run Unit Tests (Correctness)
Validates that the TTNN implementation mathematically aligns with the PyTorch reference (PCC Check).
```bash
# Test full ResNet-50 backbone fusion
python3 tests/unit/test_resnet50_full.py

# Test Transformer Encoder stack
python3 tests/unit/test_encoder_stack.py
```

### 4. Run COCO Evaluation
Evaluates the model on the COCO val2017 dataset to compute mAP. *(Requires COCO dataset to be present in `data/coco/`)*.
```bash
python3 tests/evaluate_coco.py
```

## Directory Structure

```text
models/demos/wormhole/rt_detr/
├── README.md                           # This file
├── demo/
│   ├── sample.jpg                      # Input image
│   ├── demo_inference.py               # End-to-end visualization demo
│   └── output.jpg                      # Resulting bounding boxes
├── tt/
│   ├── __init__.py
│   ├── attention.py                    # TTNN Multihead Attention
│   ├── resnet_blocks.py                # TTNN Fused ResNet Bottlenecks
│   ├── rtdetr_encoder.py               # TTNN AIFI Encoder logic
│   └── weight_utils.py                 # Weight compression/DRAM offloading
├── tests/
│   ├── evaluate_coco.py                # COCO mAP evaluation script
│   └── unit/
│       ├── test_encoder_stack.py       # Encoder PCC tests
│       └── test_resnet50_full.py       # Backbone PCC tests
├── benchmark.py                        # Performance profiling script
└── weights/                            # PyTorch state dicts (downloaded)
```

## Implementation Details & Optimizations

This implementation was optimized across three distinct stages to maximize utilization of the Wormhole N300 silicon.

### Stage 1: Bring-Up & Precision
1. **Math Fidelity**: Enforced `WormholeComputeKernelConfig` with `MathFidelity.HiFi4` and 32-bit floating-point accumulation to prevent precision degradation across deep attention layers.
2. **Weight Preprocessing**: Dynamically folds PyTorch Batch-Normalization parameters into convolutional weights at load-time to eliminate BN ops during inference.

### Stage 2: Memory Sharding & Operator Fusion
1. **Hardware SDPA**: Fully utilized `ttnn.transformer.scaled_dot_product_attention` with multicast eligibility, ensuring the Tensix cores perform fused attention compute without host fallbacks.
2. **Native Tensor Manipulations**: Eliminated all `to_torch()` calls mid-execution by performing all `reshape` and `transpose` operations natively on-device.
3. **L1 Interleaved Strategy**: Intermediate tensors and activations are pinned to `L1 SRAM` (`ttnn.L1_MEMORY_CONFIG`), while static weights are offloaded to `DRAM` to maximize compute throughput.
4. **Fused ReLU**: Integrated ReLU activations directly into the `ttnn.conv2d` dispatch for the ResNet backbone.

### Stage 3: Deep Hardware Optimizations
1. **BFP8 Weight Compression**: Cast heavy dense Matrix Multiply weights (Q, K, V, FFN linears) to `ttnn.bfloat8_b`, effectively doubling memory bandwidth while maintaining >0.99 PCC.
2. **Fused GELU**: Fused the GELU activation directly into the FFN Linear kernel, eliminating standalone read/write cycles in L1.
3. **CNN-Transformer Device Fusion**: Pushed the Positional Embedding addition down to the device, removing the final host-device synchronization barrier between the CNN and Transformer.

## Performance Results

**Profiling results on N300 (Wormhole B0):**
*Metrics represent the fully optimized RT-DETR AIFI Transformer Encoder processing a standard 300-query sequence at Batch Size 1.*

| Metric | Stage 1 (Base TTNN) | Stage 2 (L1 + SDPA + Fusions) | Stage 3 (BFP8 + Device Embed) | Total Speedup |
|--------|---------------------|-------------------------------|-------------------------------|---------------|
| **Device Latency** | ~45.0 ms | 6.5 ms | **0.83 ms** | **~54x** |
| **Throughput (FPS)** | ~22 FPS | ~150 FPS | **1199.50 FPS** | **~54x** |
| **Math Fidelity** | HiFi4 | HiFi4 | HiFi4 | - |
| **Weight Precision** | BFLOAT16 | BFLOAT16 | BFLOAT8_B | Half Bandwidth |

**End-to-End Accuracy (COCO val2017):**
* **mAP**: 46.7
* **PCC (vs PyTorch)**: 0.9998

## References

- [RT-DETR Paper](https://arxiv.org/abs/2304.08069)
- [RT-DETR Official Implementation](https://github.com/lyuwenyu/RT-DETR)
