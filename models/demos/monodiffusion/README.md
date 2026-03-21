# MonoDiffusion - Monocular Depth Estimation with Diffusion Models

## Overview

MonoDiffusion is a state-of-the-art self-supervised monocular depth estimation model that leverages diffusion models for depth prediction. This implementation brings MonoDiffusion to Tenstorrent hardware using TTNN APIs.

### Key Features
- **Diffusion-based depth estimation**: Conditional diffusion for high-quality depth generation
- **Coarse-to-fine processing**: Progressive refinement for faster inference
- **Uncertainty estimation**: Provides depth uncertainty crucial for safety-critical applications
- **Optimized for TT hardware**: Leverages sharding, memory optimization, and operation fusion

### Architecture Components
1. **Encoder**: ResNet-like feature extraction network
2. **Diffusion U-Net**: Conditional denoising network with timestep embedding
3. **Decoder**: Multi-scale depth refinement with skip connections
4. **Uncertainty Head**: Generates uncertainty estimates

## Performance Targets

### Stage 1 - Bring-Up
- ✅ Model runs on N150/N300 hardware without errors
- ✅ Produces valid depth maps with uncertainty estimates
- ✅ Baseline throughput: 10 FPS at 640x192 resolution
- ✅ Accuracy: PCC > 0.99 vs PyTorch reference

### Stage 2 - Basic Optimizations
- Optimal sharded/interleaved memory configs
- Operation fusion (Conv+BatchNorm+ReLU)
- L1 memory utilization for intermediate activations

### Stage 3 - Deeper Optimizations
- Maximize core utilization
- Optimized diffusion sampling strategy
- Minimize tensor manipulation overheads
- Target: 10+ FPS at 640x192 or higher resolutions

## Getting Started

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Download model weights
bash weights_download.sh
```

### Quick Demo
```bash
# Run single image inference
pytest models/demos/monodiffusion/demo/demo.py::test_monodiffusion_demo_single_image

# Run performance benchmarks
pytest models/demos/monodiffusion/tests/test_monodiffusion_perf.py
```

## Input/Output Specifications

- **Input**: RGB images (640x192 for KITTI, 1024x320 for higher resolution)
- **Output**:
  - Depth map (same resolution as input)
  - Uncertainty map (same resolution as input)
- **Precision**: BFloat16 for activations and weights

## Model Details

- **Base Architecture**: Lite-Mono with diffusion-based depth prediction
- **Inference Steps**: 20 denoising iterations (configurable)
- **Target Dataset**: KITTI, Make3D, DIML

## References

- [MonoDiffusion Paper (TCSVT 2024)](https://arxiv.org/abs/2311.16495)
- [MonoDiffusion Official Repository](https://github.com/ShuweiShao/MonoDiffusion)
- [Lite-Mono](https://github.com/noahzn/Lite-Mono)
- [TT-NN CNN Optimization Guide](../../tech_reports/CNNs/cnn_optimizations.md)
