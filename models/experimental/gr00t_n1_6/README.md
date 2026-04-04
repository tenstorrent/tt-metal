# GR00T N1.6-3B for Tenstorrent

GR00T N1.6 (NVIDIA Isaac GR00T N1.6-3B) is a vision-language-action (VLA) model
for humanoid and robotic manipulation. This implementation runs on Tenstorrent
Blackhole (p150a) using TTNN.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          GR00T N1.6-3B                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     BACKBONE (Eagle-Block2A)                     │    │
│  │                                                                  │    │
│  │  ┌───────────┐    ┌─────────────┐    ┌───────────────────────┐  │    │
│  │  │  Images   │    │  SigLIP2    │    │  Pixel Shuffle + MLP  │  │    │
│  │  │ (224x224) │───►│  27 layers  │───►│  Connector            │──┼──┐ │
│  │  │           │    │  1152 dim   │    │  4608 → 2048          │  │  │ │
│  │  └───────────┘    └─────────────┘    └───────────────────────┘  │  │ │
│  │                                                                  │  │ │
│  │  ┌───────────┐    ┌──────────────────────────────────────────┐  │  │ │
│  │  │ Language  │    │  Qwen3-1.7B (28 layers, 2048 dim)       │  │  │ │
│  │  │ Tokens    │───►│  Select layer 16 → backbone_features     │──┼──┤ │
│  │  └───────────┘    └──────────────────────────────────────────┘  │  │ │
│  └──────────────────────────────────────────────────────────────────┘  │ │
│                                                                        │ │
│  ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │                     ACTION HEAD                                  │   │ │
│  │                                                                  │   │ │
│  │  ┌────────┐   ┌────────────────────┐                            │   │ │
│  │  │ State  │──►│ StateEncoder[emb]  │──┐                         │   │ │
│  │  │ (128)  │   │ 128→1024→1536     │  │                         │   │ │
│  │  └────────┘   └────────────────────┘  │                         │   │ │
│  │                                       ▼                         │   │ │
│  │  ┌────────┐   ┌────────────────────┐  ┌─────────────────────┐  │   │ │
│  │  │ Noisy  │──►│ ActionEncoder[emb] │──►│                     │  │   │ │
│  │  │Actions │   │ + TimestepEncoder  │  │  AlternateVLDiT     │◄─┼───┘ │
│  │  │(50,128)│   │ 128→1536          │  │  32 layers, 1536d   │  │     │
│  │  └────────┘   └────────────────────┘  │  32 heads × 48 dim  │  │     │
│  │                                       │                     │  │     │
│  │                                       │  Even: cross-attn   │  │     │
│  │                                       │  Odd:  self-attn    │  │     │
│  │                                       └──────────┬──────────┘  │     │
│  │                                                  │             │     │
│  │                               ┌──────────────────▼──────────┐  │     │
│  │                               │ ActionDecoder[emb]          │  │     │
│  │                               │ 1024→1024→128               │  │     │
│  │                               └──────────────────┬──────────┘  │     │
│  └───────────────────────────────────────────────────┼─────────┘  │     │
│                                                      │             │
│  ┌───────────────────────────────────────────────────▼──────────┐  │
│  │              FLOW MATCHING (4 Euler steps)                    │  │
│  │              actions += dt * predicted_velocity               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│                    ┌───────────────────┐                            │
│                    │   Action Output   │                            │
│                    │  [batch, 50, 128] │                            │
│                    └───────────────────┘                            │
└─────────────────────────────────────────────────────────────────────┘
```

**Key architectural details:**
- **AlternateVLDiT**: Even blocks cross-attend to backbone features (Q from actions, K/V from 2048-dim backbone). Odd blocks do self-attention only.
- **Flow Matching**: Iterative denoising from Gaussian noise to actions over 4 Euler steps.
- **Embodiment-Specific**: Up to 32 robot embodiments with separate state/action encode/decode MLPs.
- **AdaLN**: Adaptive Layer Normalization conditioned on diffusion timestep.

## Directory Structure

```
gr00t_n1_6/
├── README.md                              # This file
├── conftest.py                            # Pytest fixtures
├── run_env.sh                             # Environment setup script
├── common/                                # Shared configs and utilities
│   ├── configs.py                         # Model configurations
│   └── weight_loader.py                   # HuggingFace weight loading
├── reference/                             # PyTorch reference implementation
│   └── torch_groot_n16.py                 # CPU reference for PCC validation
├── tt/                                    # TTNN implementation
│   ├── ttnn_common.py                     # Common TTNN utilities
│   ├── ttnn_siglip2.py                    # SigLIP2 vision encoder (27L)
│   ├── ttnn_dit.py                        # AlternateVLDiT action head (32L)
│   ├── ttnn_dit_optimized.py              # Optimized on-device DiT attention
│   ├── ttnn_embodiment.py                 # Per-embodiment MLPs
│   └── ttnn_groot_n16_model.py            # Main model assembly
└── tests/
    ├── download_weights.py                # Download weights from HuggingFace
    ├── pcc/                               # PCC (accuracy) tests
    │   ├── test_pcc_siglip2.py            # SigLIP2 PCC validation
    │   ├── test_pcc_dit.py                # DiT PCC validation
    │   └── test_pcc_embodiment.py         # Embodiment MLP PCC validation
    ├── perf/                              # Performance benchmarks
    │   └── test_perf_e2e.py               # End-to-end latency benchmark
    └── demo/                              # Demo scripts
        ├── run_demo.py                    # Interactive demo with sample inputs
        ├── sample_images/                 # Sample robot images
        └── visualize_actions.py           # Visualize predicted actions
```

## Quick Start

### 1. Environment Setup

```bash
# Set required environment variables
export TT_METAL_HOME=/path/to/pi0/tt-metal
export ARCH_NAME=blackhole

# Run from the pi0 tt-metal directory (where TTNN is built)
cd $TT_METAL_HOME
```

### 2. Download Weights

Weights are automatically downloaded from HuggingFace on first use:
- Model: [nvidia/GR00T-N1.6-3B](https://huggingface.co/nvidia/GR00T-N1.6-3B)
- Size: ~6.5 GB (2 safetensors shards)

```bash
# Or download manually:
python models/experimental/gr00t_n1_6/tests/download_weights.py
```

### 3. Run PCC Tests

```bash
# All PCC tests
python -m pytest models/experimental/gr00t_n1_6/tests/pcc/ -svv

# Individual components
python -m pytest models/experimental/gr00t_n1_6/tests/pcc/test_pcc_siglip2.py -svv
python -m pytest models/experimental/gr00t_n1_6/tests/pcc/test_pcc_dit.py -svv
```

### 4. Run Performance Benchmark

```bash
python -m pytest models/experimental/gr00t_n1_6/tests/perf/test_perf_e2e.py -svv
```

### 5. Run Demo

```bash
python models/experimental/gr00t_n1_6/tests/demo/run_demo.py
```

## Performance

Tested on Tenstorrent Blackhole p150a (single chip):

| Component | Latency | Notes |
|-----------|---------|-------|
| Vision Encoder (SigLIP2) | ~14ms | 27-layer, on-device attention |
| Pixel Shuffle + Connector | ~4ms | LayerNorm + 3-layer MLP |
| Flow Matching (4 steps) | ~87ms | 32-layer DiT per step |
| **Total E2E** | **~105ms** | **~9.5 Hz** |

### Comparison with NVIDIA Benchmarks

| Device | E2E Latency | Hz |
|--------|-------------|-----|
| RTX 5090 | 37ms | 27.3 |
| H100 | 38ms | 26.3 |
| RTX 4090 | 44ms | 22.8 |
| **Blackhole p150a** | **105ms** | **9.5** |
| Jetson Thor | 105ms | 9.5 |
| DGX Spark | 89ms | 11.2 |

Note: Blackhole numbers exclude the Qwen3 backbone (not yet ported to TTNN).

## PCC Verification Results

Validated against upstream [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) reference:

| Component | PCC | Threshold | Status |
|-----------|-----|-----------|--------|
| SigLIP2 Vision Encoder | 0.966 | >= 0.95 | PASS |
| Pixel Shuffle Connector | 0.999 | >= 0.98 | PASS |
| Embodiment State Encoder | 0.999 | >= 0.99 | PASS |
| AlternateVLDiT (32 layers) | 0.969 | >= 0.95 | PASS |

## Model Specifications

| Component | Details |
|-----------|---------|
| Vision Encoder | SigLIP2 (27 layers, 1152 hidden, 16 heads, 14x14 patches) |
| Language Model | Qwen3-1.7B (28 layers, 2048 hidden, 16/8 GQA heads) |
| Action Head | AlternateVLDiT (32 layers, 1536 inner dim, 32x48 heads) |
| Action Dimension | 128 (padded) |
| Action Horizon | 50 timesteps |
| State Dimension | 128 (padded) |
| Denoising Steps | 4 (Euler integration) |
| Embodiments | Up to 32 |
| Total Parameters | ~3.3B |

## References

- [GR00T N1.6 Model Card](https://huggingface.co/nvidia/GR00T-N1.6-3B)
- [Isaac-GR00T Repository](https://github.com/NVIDIA/Isaac-GR00T)
- [GR00T N1 Paper (arXiv:2503.14734)](https://arxiv.org/abs/2503.14734)
- [NVIDIA GR00T N1.6 Research Page](https://research.nvidia.com/labs/gear/gr00t-n1_6/)

## License

SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
