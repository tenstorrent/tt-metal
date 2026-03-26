# N150 Robotics Demo -- Test Commands

**Machine**: N150 Wormhole (single chip, PCI 05:00.0, device 0)
**Date**: March 2026
**Goal**: Verify PI0 and SmolVLA run live on N150 with correct robot motion

---

## Prerequisites

| Requirement | Status | Path |
|-------------|--------|------|
| N150 hardware | Detected | PCI `05:00.0`, kernel driver `tenstorrent` |
| TTNN runtime | Working | `/home/ubuntu/backup/tt-metal/python_env/bin/python3` |
| TT-Metal source | Available | `/home/ubuntu/agent/agentic/tt-metal` |
| Demo code | Available | `/home/ubuntu/robotics/tt-metal/models/experimental/robotics_demo_n150/` |
| PyBullet | Installed | In venv |
| Gemma tokenizer | Cached | Auto-detected |
| SmolVLA weights | HuggingFace | Auto-downloads on first run |
| PI0 weights | **Needs download** | See Test 3 below |

---

## Test 1: Demo Mode (No Hardware -- Sanity Check)

**Purpose**: Verify PyBullet simulation, IK motion, and video pipeline work before using TT hardware.

```bash
cd /home/ubuntu/robotics/tt-metal
bash models/experimental/robotics_demo_n150/run_on_n150.sh demo
```

**Expected output**:
```
======================================================================
  N150 Robotics Demo
  ARCH: wormhole_b0 | Device: Wormhole N150
======================================================================

Running scripted IK demo (no TT hardware)...

[1/3] Initializing PyBullet environment...
  Robot: Franka Panda 7-DOF
  Cube: [0.5, 0.0, 0.025]
  Image: 224x224 (SigLIP native, no rescaling)
  Tokenizer: Gemma (SentencePiece)

[2/3] Demo mode -- scripted IK motion (no TT hardware)

[3/3] Ready.
  Step    0 | EE=[0.43,0.00,0.64] | Dist: 0.619m | Inf: 0ms | 757 Hz
  Step   50 | EE=[0.21,-0.07,0.15] | Dist: 0.321m | Inf: 0ms | 753 Hz
  Step  100 | EE=[0.18,-0.00,0.34] | Dist: 0.447m | Inf: 0ms | 784 Hz
  ...
  Done!
  Final distance:  0.447m
  Video: n150_pi0_YYYYMMDD_HHMMSS.mp4
```

**Pass criteria**:
- [ ] No errors or crashes
- [ ] Distance decreases from ~0.6m to <0.5m
- [ ] Video file created (if `--record-video` used)
- [ ] Tokenizer shows "Gemma (SentencePiece)"

**Duration**: ~30 seconds

---

## Test 2: SmolVLA Live on N150

**Purpose**: Run SmolVLA model inference on the N150 chip in closed-loop control.

```bash
cd /home/ubuntu/robotics/tt-metal
bash models/experimental/robotics_demo_n150/run_on_n150.sh smolvla
```

**Expected output**:
```
======================================================================
  N150 Robotics Demo
  ARCH: wormhole_b0 | Device: Wormhole N150
======================================================================

Running SmolVLA on N150...

[1/3] Initializing PyBullet environment...
  Robot: Franka Panda 7-DOF
  Cube: [0.5, 0.0, 0.025]

[2/3] Loading SMOLVLA on N150 device 0...
  SmolVLA loaded (lerobot/smolvla_base)

[3/3] Ready.
  Model:       SMOLVLA
  Task:        pick up the cube

  Warming up (JIT compilation)...
    Warmup 1/2: ~2000ms
    Warmup 2/2: ~250ms
  Warmup complete.

  Step    0 | EE=[0.09,0.00,0.82] | Dist: 0.896m | Inf: 240ms |  4.3 Hz
  Step   50 | EE=[0.35,0.01,0.25] | Dist: 0.312m | Inf: 231ms | 14.2 Hz
  Step  100 | EE=[0.48,0.00,0.12] | Dist: 0.092m | Inf: 229ms | 14.5 Hz
  Step  150 | EE=[0.49,0.01,0.08] | Dist: 0.058m | Inf: 228ms | 14.8 Hz
  ...
  Done!
  Model:           SMOLVLA on N150 Wormhole
  Avg inference:   229 +/- 5 ms
  Control freq:    14.5 Hz
  Final distance:  <0.1m
```

**Pass criteria**:
- [ ] Model loads successfully onto device 0
- [ ] Warmup completes (first call is slow due to JIT, second should be ~250ms)
- [ ] Inference latency settles around ~229ms per call
- [ ] Control frequency reaches ~14 Hz with replan_interval=5
- [ ] Distance to cube decreases over time (0.9m -> <0.1m)
- [ ] Robot moves visibly toward the cube (not random motion)
- [ ] No device errors or hangs
- [ ] Clean shutdown (device closed)

**Duration**: ~2 minutes (includes first-run HuggingFace download of ~450MB)

---

## Test 2b: SmolVLA with Video Recording

**Purpose**: Same as Test 2 but saves an MP4 for review.

```bash
cd /home/ubuntu/robotics/tt-metal
bash models/experimental/robotics_demo_n150/run_on_n150.sh smolvla --record
```

**Additional pass criteria**:
- [ ] MP4 file created in current directory
- [ ] Video shows robot arm moving toward cube
- [ ] Metrics overlay visible (model name, inference time, distance)

---

## Test 3: Download PI0 Weights

**Purpose**: Set up PI0 model weights for Test 4.

```bash
cd /home/ubuntu/robotics/tt-metal
bash models/experimental/robotics_demo_n150/run_on_n150.sh setup
```

**If automatic download fails**, manual steps:
1. Open: https://drive.google.com/drive/folders/1qfY0EBGh_-6Zz-omKPQW6nBcc1Cp2_WN
2. Download `model.safetensors` and `config.json`
3. Place them in: `/home/ubuntu/agent/agentic/tt-metal/models/experimental/pi0/weights/pi0_base/`

**Verify**:
```bash
ls -lh /home/ubuntu/agent/agentic/tt-metal/models/experimental/pi0/weights/pi0_base/
```

**Expected**:
```
model.safetensors   (~4.6 GB)
config.json         (~1 KB)
```

---

## Test 4: PI0 Live on N150

**Purpose**: Run PI0 model inference on the N150 chip in closed-loop control.

**Prerequisite**: Test 3 completed (PI0 weights downloaded).

```bash
cd /home/ubuntu/robotics/tt-metal
bash models/experimental/robotics_demo_n150/run_on_n150.sh pi0
```

**Expected output**:
```
======================================================================
  N150 Robotics Demo
  ARCH: wormhole_b0 | Device: Wormhole N150
======================================================================

Running PI0 on N150...

[1/3] Initializing PyBullet environment...
  Robot: Franka Panda 7-DOF

[2/3] Loading PI0 on N150 device 0...
  PI0 loaded (accuracy mode: bfloat16 weights, HiFi4)

[3/3] Ready.
  Model:       PI0
  Task:        pick up the cube

  Warming up (JIT compilation)...
    Warmup 1/2: ~5000ms
    Warmup 2/2: ~350ms
  Warmup complete.

  Step    0 | EE=[0.09,0.00,0.82] | Dist: 0.896m | Inf: 350ms |  2.9 Hz
  Step   50 | EE=[0.32,0.01,0.30] | Dist: 0.350m | Inf: 332ms | 11.7 Hz
  Step  100 | EE=[0.46,0.00,0.15] | Dist: 0.130m | Inf: 330ms | 11.9 Hz
  Step  150 | EE=[0.48,0.01,0.10] | Dist: 0.080m | Inf: 331ms | 11.8 Hz
  ...
  Done!
  Model:           PI0 on N150 Wormhole
  Accuracy mode:   bfloat16 weights, HiFi4, Gemma tokenizer
  Avg inference:   331 +/- 7 ms
  Control freq:    11.8 Hz
  Final distance:  <0.1m
```

**Pass criteria**:
- [ ] Model loads (this takes ~30 seconds for weight loading)
- [ ] Warmup completes (first call ~5s due to JIT, second ~350ms)
- [ ] Inference latency settles around ~330ms per call
- [ ] Control frequency reaches ~12 Hz with replan_interval=5
- [ ] Distance to cube decreases over time (0.9m -> <0.1m)
- [ ] Robot moves visibly toward the cube
- [ ] Accuracy mode confirmed in output: "bfloat16 weights, HiFi4, Gemma tokenizer"
- [ ] No device errors or hangs
- [ ] Clean shutdown

**Duration**: ~3 minutes (includes weight loading)

---

## Test 4b: PI0 with Video Recording

```bash
cd /home/ubuntu/robotics/tt-metal
bash models/experimental/robotics_demo_n150/run_on_n150.sh pi0 --record
```

**Additional pass criteria**:
- [ ] MP4 file created
- [ ] Video shows robot reaching toward cube
- [ ] "PI0 on N150" label visible in video overlay

---

## Test 5: Custom Task Instructions

**Purpose**: Verify the model responds differently to different tasks.

```bash
cd /home/ubuntu/robotics/tt-metal

# Set up environment
PYBIN=/home/ubuntu/backup/tt-metal/python_env/bin/python3
export TT_METAL_HOME=/home/ubuntu/agent/agentic/tt-metal
export PYTHONPATH="$TT_METAL_HOME:/home/ubuntu/robotics/tt-metal"
export ARCH_NAME=wormhole_b0
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml

# Run with different tasks
$PYBIN models/experimental/robotics_demo_n150/run_demo.py \
    --model smolvla --task "push the block right" --steps 200

$PYBIN models/experimental/robotics_demo_n150/run_demo.py \
    --model smolvla --task "lift the object" --steps 200
```

**Pass criteria**:
- [ ] Different tasks produce different robot motions
- [ ] No crashes when changing task text

---

## Test 6: Parameter Tuning

**Purpose**: Verify configurable parameters work correctly.

```bash
PYBIN=/home/ubuntu/backup/tt-metal/python_env/bin/python3
export TT_METAL_HOME=/home/ubuntu/agent/agentic/tt-metal
export PYTHONPATH="$TT_METAL_HOME:/home/ubuntu/robotics/tt-metal"
export ARCH_NAME=wormhole_b0
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml

# Higher replan interval (smoother but less reactive)
$PYBIN models/experimental/robotics_demo_n150/run_demo.py \
    --model smolvla --replan-interval 10 --steps 200

# Lower replan interval (more reactive but slower)
$PYBIN models/experimental/robotics_demo_n150/run_demo.py \
    --model smolvla --replan-interval 1 --steps 100

# Different seed (different initial noise -> different trajectory)
$PYBIN models/experimental/robotics_demo_n150/run_demo.py \
    --model smolvla --seed 123 --steps 200
```

**Pass criteria**:
- [ ] `replan-interval 10` shows higher Hz (~20+ Hz) but same inference latency
- [ ] `replan-interval 1` shows lower Hz (~4.5 Hz) with inference every step
- [ ] Different seeds produce different final distances

---

## Quick Reference: Environment Setup

If running commands manually (not through `run_on_n150.sh`):

```bash
export TT_METAL_HOME=/home/ubuntu/agent/agentic/tt-metal
export PYTHONPATH="$TT_METAL_HOME:/home/ubuntu/robotics/tt-metal"
export ARCH_NAME=wormhole_b0
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
PYBIN=/home/ubuntu/backup/tt-metal/python_env/bin/python3

# Then run any command with $PYBIN instead of python3
$PYBIN models/experimental/robotics_demo_n150/run_demo.py --model smolvla --steps 300
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `module 'ttnn' has no attribute 'bfloat16'` | Wrong Python binary | Use `/home/ubuntu/backup/tt-metal/python_env/bin/python3` |
| `Checkpoint not found` | PI0 weights missing | Run `./run_on_n150.sh setup` |
| `TTNN device query timed out` | Device busy or driver issue | Run `tt-smi -r 0` to reset, wait 10s, retry |
| `pybullet not found` | Not installed in venv | `$PYBIN -m pip install pybullet` |
| Robot doesn't move | `replan_interval` too high or inference failing | Check console for errors, try `--demo-mode` first |
| `No module named 'transformers'` | SmolVLA needs transformers | `$PYBIN -m pip install transformers` |
| Very slow first inference (~5s) | Normal JIT compilation | Second inference will be fast (~230-330ms) |

---

## Test Execution Order

| # | Test | Time | Hardware | Status |
|---|------|------|----------|--------|
| 1 | Demo mode (sanity) | 30s | None | |
| 2 | SmolVLA live | 2 min | N150 | |
| 2b | SmolVLA + video | 2 min | N150 | |
| 3 | PI0 weight download | 5 min | None | |
| 4 | PI0 live | 3 min | N150 | |
| 4b | PI0 + video | 3 min | N150 | |
| 5 | Custom tasks | 3 min | N150 | |
| 6 | Parameter tuning | 5 min | N150 | |
| **Total** | | **~23 min** | | |
