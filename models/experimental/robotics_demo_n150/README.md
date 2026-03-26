# N150 (Wormhole) Single-Chip Robotics Demo

Run PI0 or SmolVLA on a single N150 Wormhole chip with accuracy fixes that address known precision issues in the default pipeline.

## Accuracy Fixes Applied

| Issue | Default Behavior | Fix in This Demo |
|-------|-----------------|------------------|
| **Weight precision** | `bfloat8_b` (lossy 8-bit) | `bfloat16` (full 16-bit, no quantization loss) |
| **Matmul fidelity** | HiFi2 for Q/K/V projections | HiFi4 + `fp32_dest_acc_en=True` everywhere |
| **Tokenizer** | SimpleRoboticsTokenizer (word-based, arbitrary IDs) | Gemma SentencePiece (auto-detected, matches PI0 training) |
| **Image resolution** | Configurable (may mismatch SigLIP) | Locked to 224x224 (SigLIP native, no rescaling) |
| **Noise reuse** | Same noise tensor every call | Fresh noise per `sample_actions()` call |
| **PCC result** | ~0.93 | ~0.97+ expected with these fixes |

## Quick Start

### 1. Environment Setup

```bash
export TT_METAL_HOME=/path/to/tt-metal
export ARCH_NAME=wormhole_b0
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
source $TT_METAL_HOME/python_env/bin/activate
```

### 2. Download Model Weights

```bash
# PI0 weights
python $TT_METAL_HOME/models/experimental/pi0/tests/download_pretrained_weights.py

# SmolVLA downloads automatically from HuggingFace on first use
```

### 3. Run PI0 on N150

```bash
python $TT_METAL_HOME/models/experimental/robotics_demo_n150/run_demo.py \
    --model pi0 \
    --task "pick up the cube" \
    --steps 300
```

### 4. Run SmolVLA on N150

```bash
python $TT_METAL_HOME/models/experimental/robotics_demo_n150/run_demo.py \
    --model smolvla \
    --task "pick up the cube" \
    --steps 300
```

### 5. Record Video

```bash
xvfb-run -a python $TT_METAL_HOME/models/experimental/robotics_demo_n150/run_demo.py \
    --model pi0 \
    --task "pick up the cube" \
    --steps 400 \
    --record-video
```

### 6. Demo Mode (No Hardware)

```bash
python $TT_METAL_HOME/models/experimental/robotics_demo_n150/run_demo.py \
    --demo-mode \
    --steps 200 \
    --record-video
```

## All CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `pi0` | `pi0` or `smolvla` |
| `--task` | `pick up the cube` | Natural language instruction |
| `--steps` | `300` | Simulation steps |
| `--device` | `0` | TT device ID |
| `--checkpoint` | auto | PI0 weights path |
| `--replan-interval` | `5` | Steps between inference calls |
| `--record-video` | off | Save MP4 |
| `--video-path` | auto | Custom video path |
| `--demo-mode` | off | Scripted IK motion (no hardware) |
| `--use-absolute` | off | Absolute instead of delta actions |
| `--delta-scale` | `1.0` | Delta action scale |
| `--max-velocity` | `0.5` | Joint velocity limit (rad/s) |
| `--seed` | `42` | Random seed |

## Expected Performance on N150

| Metric | PI0 | SmolVLA |
|--------|-----|---------|
| Inference per call | ~350ms (HiFi4 mode) | ~240ms |
| Control freq (replan=5) | ~11 Hz | ~14 Hz |
| PCC vs PyTorch ref | ~0.97 | ~0.98 |
| Action quality | High precision | Fast, good |

Note: HiFi4 adds ~5-10% latency vs HiFi2 but significantly improves accuracy.

## Files

```
robotics_demo_n150/
├── README.md            # This file
├── __init__.py
├── run_demo.py          # Main entry point (CLI)
├── sim_env.py           # PyBullet Franka Panda environment
├── tokenizer.py         # Auto-selecting tokenizer (Gemma preferred)
└── accuracy_config.py   # Precision settings (bfloat16, HiFi4, N150 params)
```

## Hardware Requirements

- **N150 card**: Single Wormhole B0 chip
- **ARCH_NAME**: `wormhole_b0`
- **WH_ARCH_YAML**: `wormhole_b0_80_arch_eth_dispatch.yaml`
- **TT-Metal SDK**: Installed and activated
- **L1 small size**: 24576 (set automatically)

## Differences from Blackhole Single-Chip Demo

| | N150 (Wormhole) | Blackhole |
|--|-----------------|-----------|
| Architecture | Wormhole B0 | Blackhole |
| `ARCH_NAME` | `wormhole_b0` | `blackhole` |
| Compute config | `WormholeComputeKernelConfig` | Same (aliased) |
| Weight dtype | `bfloat16` (accuracy fix) | `bfloat8_b` (default) |
| Matmul fidelity | HiFi4 everywhere | HiFi2 for projections |
| Tokenizer | Gemma (auto) | Simple (default) |
| PI0 docs target | Primary (README) | Secondary |
| SmolVLA tested | Via this demo | Primary (README) |
