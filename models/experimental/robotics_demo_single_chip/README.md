# Single-Chip Robotics Demo

A simple, self-contained demo running one VLA model on one Tenstorrent chip. The user picks either **PI0** or **SmolVLA** and watches a Franka Panda robot attempt the given task in real-time PyBullet simulation.

## Quick Start

```bash
# PI0 (default)
python models/experimental/robotics_demo_single_chip/run_demo.py \
    --model pi0 --task "pick up the cube" --steps 300

# SmolVLA
python models/experimental/robotics_demo_single_chip/run_demo.py \
    --model smolvla --task "pick up the cube" --steps 300

# Record video
xvfb-run -a python models/experimental/robotics_demo_single_chip/run_demo.py \
    --model pi0 --record-video --steps 400

# Demo mode (no TT hardware -- scripted IK motion)
python models/experimental/robotics_demo_single_chip/run_demo.py \
    --demo-mode --steps 200 --record-video
```

## Models

| Model | Params | Inference | Notes |
|-------|--------|-----------|-------|
| **PI0** | 2.3B | ~330ms | Higher precision, flow-matching denoiser with 10 Euler steps |
| **SmolVLA** | 450M | ~229ms | Faster, compact, flow-matching with 10 steps |

## Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `--model` | `pi0` | `pi0` or `smolvla` |
| `--task` | `pick up the cube` | Natural language instruction |
| `--steps` | `300` | Simulation steps |
| `--device` | `0` | TT device ID |
| `--replan-interval` | `5` | Steps between model calls |
| `--record-video` | off | Save MP4 file |
| `--demo-mode` | off | Scripted motion (no hardware) |
| `--use-absolute` | off | Absolute instead of delta actions |
| `--delta-scale` | `1.0` | Scale for delta actions |
| `--max-velocity` | `0.5` | Joint velocity limit (rad/s) |
| `--seed` | `42` | Random seed |

## Files

```
robotics_demo_single_chip/
├── README.md       # This file
├── __init__.py
├── run_demo.py     # Main entry point (CLI)
├── sim_env.py      # PyBullet Franka Panda environment
└── tokenizer.py    # Simple word-based tokenizer for PI0
```

## Prerequisites

- One Tenstorrent chip (any: Wormhole, Blackhole)
- TT-Metal SDK (`TT_METAL_HOME` set, `python_env` activated)
- `pip install pybullet numpy torch`
- PI0 weights downloaded (for `--model pi0`)
- HuggingFace access (for `--model smolvla`, auto-downloads)

## How It Works

1. A PyBullet Franka Panda robot is loaded with a red cube on the table.
2. The chosen model (PI0 or SmolVLA) is loaded onto device 0.
3. Each control step:
   - Two camera images (front + side) are captured.
   - Every `replan_interval` steps, the model runs inference.
   - Between replans, buffered actions from the 50-step horizon are used.
   - The first 7 action dimensions are applied to the robot's joints.
4. Console prints EE position, distance-to-target, and timing every 50 steps.
5. If `--record-video`, an MP4 is saved with model name, metrics, and position overlaid.
