# Attention DenseUNet - TTNN Implementation

## Overview

This is the TTNN implementation of **Attention DenseUNet** for medical image segmentation and general semantic segmentation tasks on Tenstorrent hardware (Wormhole N150/N300).

Attention DenseUNet combines:
- **DenseNet Encoder**: Densely connected blocks for efficient feature reuse
- **Attention Gates**: Selective emphasis on relevant spatial regions during decoding
- **U-Net Architecture**: Skip connections with upsampling for precise localization

This implementation is part of bounty [#30863](https://github.com/tenstorrent/tt-metal/issues/30863).

## Model Architecture

### Key Components

1. **DenseNet Encoder**
   - 4 Dense Blocks with growth rate = 16
   - Each block contains 4 dense layers
   - Dense connections: each layer receives features from all preceding layers
   - Transition layers compress channels (compression = 0.5) and reduce spatial dims via pooling

2. **Bottleneck**
   - 2 convolution blocks at lowest resolution
   - Processes features before decoder path

3. **Attention-Gated Decoder**
   - 4 decoder stages with transposed convolution for upsampling
   - **Attention Gates**: Weight skip connections based on gating signal from decoder
   - Decoder blocks refine concatenated features

4. **Architecture Flow** (for 256×256 input):
   ```
   Input (3, 256, 256)
     ↓ Conv0
   (32, 256, 256)
     ↓ DenseBlock1 → TransitionDown
   (48, 128, 128)
     ↓ DenseBlock2 → TransitionDown
   (56, 64, 64)
     ↓ DenseBlock3 → TransitionDown
   (60, 32, 32)
     ↓ DenseBlock4 → TransitionDown
   (62, 16, 16)
     ↓ Bottleneck
   (62, 16, 16)
     ↓ UpConv + Attention + Decoder
   (124, 32, 32)
     ↓ UpConv + Attention + Decoder
   (120, 64, 64)
     ↓ UpConv + Attention + Decoder
   (112, 128, 128)
     ↓ UpConv + Attention + Decoder
   (96, 256, 256)
     ↓ Conv Out
   Output (1, 256, 256)
   ```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Initial Features | 32 |
| Growth Rate | 16 |
| Layers per Block | (4, 4, 4, 4) |
| Compression | 0.5 |
| Total Parameters | ~1.5M |

## Prerequisites

1. **TT-Metalium / TT-NN Installation**
   - Follow [INSTALLING.md](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

2. **Python Dependencies**
   ```bash
   pip install torch torchvision scikit-image
   ```

3. **Hardware**
   - Wormhole N150 or N300

## Directory Structure

```
models/demos/attention_denseunet/
├── README.md                        # This file
├── __init__.py
├── reference/
│   ├── __init__.py
│   └── model.py                     # PyTorch reference implementation
├── tt/
│   ├── __init__.py
│   ├── common.py                    # Constants and preprocessing
│   ├── config.py                    # Layer configurations and builder
│   └── model.py                     # TTNN model implementation
├── demo/
│   ├── __init__.py
│   └── demo.py                      # Demo script for inference
└── tests/
    ├── __init__.py
    └── test_attention_denseunet.py  # Test suite with PCC validation
```

## Quick Start

### 1. Run PyTorch Reference Demo

Validate the PyTorch implementation:

```bash
python models/demos/attention_denseunet/reference/model.py
```

Expected output:
```
Input shape: torch.Size([1, 3, 256, 256])
Output shape: torch.Size([1, 1, 256, 256])
Model parameters: 1,522,689
✓ Output shape matches input spatial dimensions
```

### 2. Run TTNN Demo

Run inference using TTNN on Tenstorrent hardware:

```bash
# Using pytest (recommended - handles device setup)
pytest models/demos/attention_denseunet/demo/demo.py::test_attention_denseunet_demo -v

# PyTorch-only mode (for testing without hardware)
python models/demos/attention_denseunet/demo/demo.py --pytorch
```

### 3. Run Test Suite

Run comprehensive tests including PCC validation:

```bash
# Run all tests
pytest models/demos/attention_denseunet/tests/ -v

# Run specific test
pytest models/demos/attention_denseunet/tests/test_attention_denseunet.py::test_attention_denseunet_inference -v
```

## Implementation Details

### Stage 1 - Bring-Up (Current)

**Status**: ✅ Complete

- ✅ TTNN implementation of all components
- ✅ PyTorch weight preprocessing with BatchNorm folding
- ✅ Full encoder-decoder pipeline
- ✅ Attention gate implementation
- ✅ Memory management and tensor deallocation
- ✅ Demo script functional
- ✅ Test suite with PCC validation

**Configuration**:
- Memory: DRAM interleaved (simple strategy for Stage 1)
- Math Fidelity: LoFi for faster computation
- Sharding: Auto-sharded (basic strategy)

**Target Metrics**:
- PCC ≥ 0.97 against PyTorch reference
- Runs on Wormhole without OOM errors
- Produces valid segmentation masks

### Stage 2 - Basic Optimizations (Todo)

Planned optimizations:
- [ ] Apply height/block sharding for convolutions
- [ ] Store intermediates in L1 memory where beneficial
- [ ] Use fused operations (e.g., ReLU with Conv)
- [ ] Optimize attention gate computation
- [ ] Benchmark and profile performance

### Stage 3 - Deeper Optimization (Todo)

Advanced optimizations:
- [ ] Maximize core parallelism for encoder/decoder
- [ ] Optimize attention gates for minimal overhead
- [ ] Custom kernels for dense connections if needed
- [ ] Performance tuning and analysis

## Implementation Notes

### BatchNorm Folding

All BatchNorm layers are folded into preceding Conv layers during preprocessing for efficiency:
```python
conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(conv, bn)
```

### Dense Connections

Dense blocks concatenate features from all preceding layers:
```python
# In TtDenseLayer
out = self.bottleneck(x)  # 1x1 conv
out = self.expansion(out)  # 3x3 conv
return concatenate_features(x, out)  # Input + new features
```

Channel count grows by `growth_rate` (16) per layer.

### Attention Mechanism

Attention gates compute spatial attention to emphasize relevant skip features:
```python
theta_x = theta_conv(skip)  # Project skip connection
phi_g = phi_conv(gating)     # Project gating signal
f = relu(theta_x + phi_g)    # Combine
attention = sigmoid(psi(f))  # Compute attention map
output = W(attention * skip) # Apply attention and project
```

### Memory Management

Aggressive deallocation to minimize memory footprint:
- Skip connections stored in DRAM
- Intermediate activations deallocated after use
- Concatenation handles memory layout conversions

## Validation

### PCC (Pearson Correlation Coefficient)

Target: **≥ 0.97**

The model output is validated against PyTorch reference using PCC, which measures the linear correlation between outputs. A PCC ≥ 0.97 indicates high numerical accuracy.

### Test Coverage

- ✅ Model initialization
- ✅ Full forward pass
- ✅ Shape validation
- ✅ PCC validation
- ⏳ Component unit tests (planned)
- ⏳ Performance benchmarks (planned)

## Performance (Stage 1 Baseline)

*To be measured and documented after successful bring-up on hardware*

Expected metrics:
- Inference latency: TBD
- Memory usage: TBD
- PCC: ≥ 0.97

## Known Limitations

1. **Batch Size**: Currently optimized for batch_size = 1
2. **Input Size**: Tested with 256×256 (configurable)
3. **Optimization Level**: Stage 1 uses basic memory/sharding strategies
4. **Attention Gate**: Uses CPU interpolation (to be optimized in Stage 2)

## References

- **DenseNet**: [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
- **Attention U-Net**: [Attention U-Net for Image Segmentation](https://arxiv.org/abs/1804.03999)
- **TTNN Model Bring-up**: [Tech Report](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/ttnn/TTNN-model-bringup.md)
- **CNN Optimization Guide**: [CNNs in TT-NN](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/CNNs/cnn_optimizations.md)