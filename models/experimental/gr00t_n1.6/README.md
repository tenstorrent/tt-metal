# GR00T N1.6-3B on Tenstorrent Blackhole

NVIDIA Isaac GR00T N1.6-3B is a vision-language-action (VLA) model for humanoid and
robotic manipulation. This implementation runs fully on Tenstorrent Blackhole (p150a)
using TTNN, including the Qwen3-1.7B language backbone.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          GR00T N1.6-3B                                   │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────── BACKBONE (Eagle-Block2A) ──────────────┐  │
│  │                                                                    │  │
│  │  ┌───────────┐   ┌───────────────┐   ┌─────────────────────────┐  │  │
│  │  │  Images   │   │  SigLIP2      │   │  Pixel Shuffle + MLP    │  │  │
│  │  │ (224×224) │──►│  27 layers    │──►│  Connector              │──┼─┐│
│  │  │           │   │  1152 dim     │   │  4608 → 2048            │  │ ││
│  │  └───────────┘   └───────────────┘   └─────────────────────────┘  │ ││
│  │                                                                    │ ││
│  │  ┌───────────┐   ┌────────────────────────────────────────────┐   │ ││
│  │  │ Language  │   │  Qwen3-1.7B (first 16 of 28 layers)       │   │ ││
│  │  │  Tokens   │──►│  2048 hidden dim, 16/8 GQA heads          │───┼─┤│
│  │  └───────────┘   │  Image features spliced at token positions │   │ ││
│  │                  └────────────────────────────────────────────┘   │ ││
│  └────────────────────────────────────────────────────────────────────┘ ││
│                                                                         ││
│  ┌────────────────────────── ACTION HEAD ──────────────────────────────┐ ││
│  │                                                                     │ ││
│  │  ┌────────┐  ┌──────────────────┐                                  │ ││
│  │  │ State  │─►│ StateEncoder     │──┐                               │ ││
│  │  │ (128)  │  │ 128→1024→1536    │  │                               │ ││
│  │  └────────┘  └──────────────────┘  ▼                               │ ││
│  │                                  ┌─────────────────────────────┐   │ ││
│  │  ┌────────┐  ┌──────────────────┐│                             │   │ ││
│  │  │ Noisy  │─►│ ActionEncoder    ││  AlternateVLDiT             │◄──┼─┘│
│  │  │Actions │  │ + TimestepEncoder││  32 layers · 1536 dim       │   │  │
│  │  │(50,128)│  │ 128→1536         ││  32 heads × 48 dim          │   │  │
│  │  └────────┘  └──────────────────┘│  Even: cross-attn to VL     │   │  │
│  │                                  │  Odd:  self-attn only        │   │  │
│  │                                  └──────────────┬──────────────┘   │  │
│  │                                                 │                  │  │
│  │                                  ┌──────────────▼──────────────┐   │  │
│  │                                  │ ActionDecoder 1024→1024→128 │   │  │
│  │                                  └──────────────┬──────────────┘   │  │
│  └─────────────────────────────────────────────────┼──────────────────┘  │
│                                                    │                     │
│  ┌─────────────────────────────────────────────────▼──────────────────┐  │
│  │              Flow Matching — 4 Euler steps                         │  │
│  │              actions += dt × predicted_velocity                    │  │
│  └─────────────────────────────────────────────────┬──────────────────┘  │
│                                                    ▼                     │
│                                        ┌───────────────────┐             │
│                                        │  Action Output    │             │
│                                        │  [batch, 50, 128] │             │
│                                        └───────────────────┘             │
└──────────────────────────────────────────────────────────────────────────┘
```

**Key design points:**
- **AlternateVLDiT**: even blocks cross-attend to backbone features (Q from actions, K/V from 2048-dim VL output); odd blocks do self-attention only.
- **Flow Matching**: 4-step Euler denoising from Gaussian noise to action trajectory.
- **Embodiment-specific MLPs**: up to 32 robot embodiments, each with separate state/action encode/decode heads.
- **AdaLN**: adaptive layer norm conditioned on the diffusion timestep.
- **Qwen3 backbone fully on-device**: first 16 transformer layers run on Blackhole; QK-norm and RoPE use CPU-assist (small tensors, negligible overhead).

## Directory Structure

```
gr00t_n1.6/
├── Dockerfile                             # Blackhole container build
├── README.md                              # This file
├── conftest.py                            # Pytest session fixtures
├── run_env.sh                             # Environment setup script
├── common/
│   ├── configs.py                         # Dataclass configs for all sub-models
│   └── weight_loader.py                   # HuggingFace safetensors weight loading
├── reference/
│   └── torch_groot_n16.py                 # Pure-PyTorch reference (for PCC checks)
├── tt/
│   ├── ttnn_common.py                     # Shared TTNN helpers (to_tt_tensor, etc.)
│   ├── ttnn_siglip2.py                    # SigLIP2 vision encoder (27 layers)
│   ├── ttnn_dit.py                        # AlternateVLDiT action head (32 layers)
│   ├── ttnn_dit_optimized.py              # On-device fused DiT attention kernel
│   ├── ttnn_embodiment.py                 # Per-embodiment state/action MLPs
│   ├── ttnn_qwen3.py                      # Qwen3-1.7B backbone (16 layers on-device)
│   └── ttnn_groot_n16_model.py            # Top-level model assembly and forward pass
└── tests/
    ├── download_weights.py                # Download weights from HuggingFace Hub
    ├── pcc/
    │   ├── test_pcc_siglip2.py            # SigLIP2 PCC vs. reference
    │   ├── test_pcc_dit.py                # DiT PCC vs. reference
    │   └── test_pcc_embodiment.py         # Embodiment MLP PCC vs. reference
    ├── perf/
    │   └── test_perf_e2e.py               # End-to-end latency benchmark
    └── demo/
        ├── run_demo.py                    # Interactive inference demo
        └── visualize_actions.py           # Plot predicted action trajectories
```

## Quick Start

### 1. Environment

```bash
source models/experimental/gr00t_n1.6/run_env.sh
cd /home/ttuser/experiments/gr00t_n16/tt-metal
```

### 2. Download Weights

Weights are auto-downloaded from HuggingFace on first use (~6.5 GB):
- **Model**: [nvidia/GR00T-N1.6-3B](https://huggingface.co/nvidia/GR00T-N1.6-3B)

```bash
python models/experimental/gr00t_n1.6/tests/download_weights.py
```

### 3. PCC Tests (accuracy validation)

```bash
# All PCC tests
python -m pytest models/experimental/gr00t_n1.6/tests/pcc/ -svv

# Individual components
python -m pytest models/experimental/gr00t_n1.6/tests/pcc/test_pcc_siglip2.py -svv
python -m pytest models/experimental/gr00t_n1.6/tests/pcc/test_pcc_dit.py -svv
python -m pytest models/experimental/gr00t_n1.6/tests/pcc/test_pcc_embodiment.py -svv
```

### 4. Performance Benchmark

```bash
python -m pytest models/experimental/gr00t_n1.6/tests/perf/test_perf_e2e.py -svv
```

### 5. Demo

```bash
python models/experimental/gr00t_n1.6/tests/demo/run_demo.py
```

### 6. Docker (optional)

```bash
# Build
docker build -t gr00t-n16-bh -f models/experimental/gr00t_n1.6/Dockerfile .

# Run (bind Blackhole device 2)
docker run --rm -it \
  --device /dev/tenstorrent/2 \
  -v /dev/hugepages:/dev/hugepages \
  -v $(pwd):/workspace \
  gr00t-n16-bh
```

## PCC Results

Validated against [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) upstream reference:

| Component | PCC | Threshold | Status |
|-----------|-----|-----------|--------|
| SigLIP2 Vision Encoder (27L) | 0.966 | ≥ 0.95 | PASS |
| Pixel Shuffle + Connector | 0.999 | ≥ 0.98 | PASS |
| Embodiment State Encoder | 0.999 | ≥ 0.99 | PASS |
| AlternateVLDiT (32L) | 0.969 | ≥ 0.95 | PASS |

## Performance

Measured on Tenstorrent Blackhole p150a (single chip):

| Component | Latency |
|-----------|---------|
| SigLIP2 vision encoder | ~14 ms |
| Pixel shuffle + connector | ~4 ms |
| Qwen3 backbone (16L, QK-norm/RoPE CPU-assist) | TBD |
| Flow matching — 4 DiT steps | ~87 ms |
| **Total E2E** | **~105 ms** (excl. Qwen3 backbone) |

## Model Specifications

| Component | Details |
|-----------|---------|
| Vision encoder | SigLIP2 — 27L, 1152 hidden, 16 heads, 14×14 patches |
| Language backbone | Qwen3-1.7B — 28L total, first 16L used, 2048 hidden, 16/8 GQA |
| Action head | AlternateVLDiT — 32L, 1536 inner dim, 32×48 attn heads |
| Action dimension | 128 (padded) |
| Action horizon | 50 timesteps |
| State dimension | 128 (padded) |
| Denoising steps | 4 (Euler) |
| Max embodiments | 32 |
| Total parameters | ~3.3 B |

## References

- [GR00T N1.6 Model Card](https://huggingface.co/nvidia/GR00T-N1.6-3B)
- [Isaac-GR00T Repository](https://github.com/NVIDIA/Isaac-GR00T)
- [GR00T N1 Paper — arXiv:2503.14734](https://arxiv.org/abs/2503.14734)
- [NVIDIA GR00T N1.6 Research Page](https://research.nvidia.com/labs/gear/gr00t-n1_6/)

## License

SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC  
SPDX-License-Identifier: Apache-2.0
