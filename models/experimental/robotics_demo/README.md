# Tenstorrent Robotics Intelligence Demo Suite

A live, customer-facing robotic simulation demo that runs two state-of-the-art Vision-Language-Action (VLA) models -- **PI0** and **SmolVLA** -- on a Tenstorrent Quiet Box with **4 Blackhole chips**. The demo showcases data parallelism, multi-model comparison, and an innovative ensemble pipeline through a real-time Streamlit dashboard.

---

## What It Does

| Scenario | Description | Chips Used |
|----------|-------------|------------|
| **S1: Data-Parallel PI0** | 4 Franka Panda robots, each on its own chip, solving different tasks simultaneously | 4 |
| **S2: PI0 vs SmolVLA** | Side-by-side comparison of both models on the same task in mirrored environments | 2 + 2 |
| **S3: Ensemble Pipeline** | SmolVLA (fast coarse planner) + PI0 (precise controller) with action fusion | 1 + 1 |
| **S4: Scaling Benchmark** | Throughput measurement from 1 to 4 chips with scaling efficiency charts | 1-4 |

Each scenario runs as a **live real-time demo** with:
- PyBullet physics simulation of Franka Panda robots
- Multi-camera RGB observations fed directly to TT hardware
- Real-time video composition (quad-view or side-by-side)
- Live performance metrics (latency, FPS, distance-to-target)
- Optional video recording for offline review

---

## Quick Start

### 1. First-Time Setup

```bash
cd $TT_METAL_HOME
./models/experimental/robotics_demo/run_demo.sh --setup
```

This installs Python dependencies, optionally downloads the Gemma tokenizer, and checks for PI0 model weights.

### 2. Smoke Test (No TT Hardware Required)

```bash
./models/experimental/robotics_demo/run_demo.sh --test
```

Validates that PyBullet environments, video composition, and metrics collection all work without Tenstorrent hardware.

### 3. Launch the Live Dashboard

```bash
./models/experimental/robotics_demo/run_demo.sh
```

Opens a Streamlit web interface at `http://localhost:8501` where you can select scenarios, configure parameters, and watch the demo live.

### 4. Run Scenarios via CLI (Headless)

```bash
# Scenario 1: 4 robots on 4 chips
./models/experimental/robotics_demo/run_demo.sh --cli 1

# Scenario 2: PI0 vs SmolVLA
./models/experimental/robotics_demo/run_demo.sh --cli 2

# Scenario 3: Ensemble pipeline
./models/experimental/robotics_demo/run_demo.sh --cli 3

# Scenario 4: Scaling benchmark
./models/experimental/robotics_demo/run_demo.sh --cli 4
```

---

## Prerequisites

### Hardware
- Tenstorrent Quiet Box with 4 Blackhole chips (P150 or equivalent)
- Chips accessible via `ttnn.get_device_ids()`

### Software
- TT-Metal SDK installed and configured (`TT_METAL_HOME` set)
- Python 3.10+ with the tt-metal virtual environment activated

### Model Weights
```bash
# PI0 weights (required for Scenarios 1-4)
python models/experimental/pi0/tests/download_pretrained_weights.py

# SmolVLA weights (auto-downloaded from HuggingFace on first use)
# Checkpoint: lerobot/smolvla_base
```

### Python Dependencies
```bash
pip install pybullet numpy torch imageio opencv-python-headless \
            streamlit matplotlib Pillow safetensors transformers
```

---

## Architecture

```
                    +-----------------------+
                    |  Streamlit Dashboard  |
                    |  (streamlit_app.py)   |
                    +----------+------------+
                               |
                    +----------v------------+
                    |  Demo Orchestrator     |
                    |  (demo_orchestrator.py)|
                    +---+-----+-----+---+---+
                        |     |     |   |
               +--------+  +-+--+  |   +--------+
               |           |     |  |            |
          Scenario 1  Scenario 2 | Scenario 3  Scenario 4
          (DP PI0)    (Compare)  | (Ensemble)  (Benchmark)
               |           |     |     |            |
       +-------v---+ +----v--+  | +---v--------+   |
       |DataParallel| |DP PI0 |  | |Ensemble    |   |
       |PI0 (x4)   | |DP Smol|  | |Pipeline    |   |
       +-----------++ +-------+  | +------------+   |
               |           |     |     |            |
       +-------v-----------v-----v-----v------------v--+
       |          Multi-Environment (multi_env.py)      |
       |  N independent PyBullet Franka Panda simulations|
       +----+------+------+------+---------------------+
            |      |      |      |
       +----v--+---v--+---v--+---v--+
       | BH 0  | BH 1 | BH 2 | BH 3 |   Blackhole Chips
       +-------+------+------+------+
```

---

## File Structure

```
robotics_demo/
├── README.md                    # This file
├── run_demo.sh                  # Launch script (setup/test/cli/dashboard)
├── streamlit_app.py             # Live web dashboard
├── demo_orchestrator.py         # Top-level scenario controller
├── multi_env.py                 # Multi-instance PyBullet environments
├── data_parallel_pi0.py         # Data-parallel PI0 across MeshDevice submeshes
├── data_parallel_smolvla.py     # Data-parallel SmolVLA across MeshDevice submeshes
├── ensemble_pipeline.py         # Concurrent PI0+SmolVLA with 3 fusion strategies
├── benchmark.py                 # Throughput scaling measurement + chart generation
├── video_composer.py            # Quad-view / side-by-side frame composition
├── metrics.py                   # Thread-safe real-time metrics collection
├── tokenizer_setup.py           # Gemma tokenizer pre-download helper
└── __init__.py
```

---

## Models

### PI0 (Physical Intelligence Zero)
- **Architecture**: SigLIP (27 layers) + Gemma 2B VLM + Gemma 300M Action Expert
- **Action generation**: Flow matching with 10 Euler denoising steps
- **Output**: 50-step action horizon, 32 action dimensions
- **Latency on Blackhole**: ~330ms per inference
- **Path**: `models/experimental/pi0/`

### SmolVLA (Small Vision-Language-Action)
- **Architecture**: SigLIP ViT (12 layers) + VLM (16 layers) + Expert (16 layers)
- **Action generation**: Flow matching with 10 steps
- **Output**: 50-step action horizon, configurable action dimensions
- **Latency on Blackhole**: ~229ms per inference
- **Path**: `models/experimental/smolvla/`

---

## Fusion Strategies (Scenario 3)

| Strategy | Description | Best For |
|----------|-------------|----------|
| **Weighted Average** | `alpha * PI0 + (1-alpha) * SmolVLA` | General-purpose blending |
| **Temporal Blend** | SmolVLA for near-future (fast reflexes), PI0 for later (precision) | Tasks needing both speed and accuracy |
| **Confidence Gate** | Per-timestep selection based on action variance | When models disagree |

---

## Performance Expectations

| Configuration | Control Frequency | Notes |
|--------------|------------------|-------|
| 1 chip, PI0 | ~3 FPS | Single inference per step |
| 1 chip, PI0 + buffering (replan=5) | ~12 Hz | 80% of steps use buffered actions |
| 4 chips, PI0 (data-parallel) | ~12 FPS aggregate | Near-linear scaling |
| 1 chip, SmolVLA | ~4.5 FPS | Faster model, lower latency |
| Ensemble (PI0 + SmolVLA) | ~3 FPS (wall) | Concurrent inference overlaps latency |

---

## Roadmap to Real Hardware

This simulation demo is Phase 1 of a two-phase deployment plan:

1. **Phase 1 (Current)**: Simulation-validated inference pipeline on Tenstorrent hardware
2. **Phase 2 (Next)**: Deploy to physical Franka Panda robot with real cameras, replacing PyBullet with hardware I/O drivers

The `multi_env.py` abstraction is designed so that swapping PyBullet for real hardware requires only changing the observation capture and action application methods.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `TT_METAL_HOME not set` | `export TT_METAL_HOME=/path/to/tt-metal` |
| `Checkpoint not found` | Run `python models/experimental/pi0/tests/download_pretrained_weights.py` |
| `PyBullet not installed` | `pip install pybullet` |
| `No TT devices detected` | Verify Tenstorrent drivers and run `tt-smi` |
| Robot takes wrong trajectory | Run `python models/experimental/pi0/tests/demo/test_tokenization.py` |
| Camera views look wrong | Run `python models/experimental/pi0/tests/demo/visualize_cameras.py` |

---

## License

SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
