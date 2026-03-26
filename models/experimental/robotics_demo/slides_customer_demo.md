---
marp: true
theme: uncover
paginate: true
backgroundColor: #1a1a2e
color: #dfe6e9
style: |
  h1, h2 { color: #00cec9; }
  h3 { color: #a29bfe; }
  table { font-size: 0.8em; }
  strong { color: #fdcb6e; }
  code { background: #16213e; }
  section { font-family: 'Inter', 'Helvetica Neue', sans-serif; }
---

<!-- Slide 1: Title -->

# Tenstorrent Robotics Intelligence Suite

### Live Multi-Model Robotic Control on Blackhole AI Accelerators

**4 Blackhole Chips | 2 VLA Models | Real-Time Simulation**

---

<!-- Slide 2: The Challenge -->

# The Challenge

Autonomous robotic manipulation demands:

- **Vision**: Understanding the scene from camera images
- **Language**: Following human instructions ("pick up the cube")
- **Action**: Generating precise motor commands at high frequency
- **Scale**: Running multiple robots simultaneously

**Vision-Language-Action (VLA) models** solve all four -- but they need specialized hardware to run in real time.

---

<!-- Slide 3: What We Are Showing Today -->

# What We Are Showing Today

A **live, real-time** robotic simulation demo running on Tenstorrent hardware.

| Capability | Detail |
|-----------|--------|
| **Models** | PI0 (2.3B params) + SmolVLA (450M params) |
| **Hardware** | 4x Blackhole chips in a Quiet Box |
| **Simulation** | Franka Panda 7-DOF arms in PyBullet physics |
| **Interface** | Live Streamlit dashboard with real-time video + metrics |
| **Scenarios** | 4 demonstration modes showing different capabilities |

Everything runs **live** -- no pre-recorded video.

---

<!-- Slide 4: The Models -->

# Two VLA Models, Two Strengths

### PI0 (Physical Intelligence Zero)
- 2.3 billion parameters
- SigLIP vision (27 layers) + Gemma 2B language + 300M action expert
- Flow matching denoising (10 steps)
- **~330ms inference** -- highest quality actions

### SmolVLA (Small VLA)
- 450 million parameters
- SigLIP ViT (12 layers) + VLM (16 layers) + Expert (16 layers)
- Flow matching denoising (10 steps)
- **~229ms inference** -- faster responses, compact footprint

---

<!-- Slide 5: Scenario 1 -->

# Scenario 1: Data-Parallel PI0

### 4 Robots, 4 Chips, 4 Tasks -- Simultaneously

```
  Chip 0          Chip 1          Chip 2          Chip 3
 ┌────────┐     ┌────────┐     ┌────────┐     ┌────────┐
 │ PI0    │     │ PI0    │     │ PI0    │     │ PI0    │
 │ "pick  │     │ "push  │     │ "lift  │     │ "reach │
 │  cube" │     │  block"│     │ object"│     │ target"│
 └───┬────┘     └───┬────┘     └───┬────┘     └───┬────┘
     │              │              │              │
 ┌───▼────┐     ┌───▼────┐     ┌───▼────┐     ┌───▼────┐
 │ Panda  │     │ Panda  │     │ Panda  │     │ Panda  │
 │ Robot  │     │ Robot  │     │ Robot  │     │ Robot  │
 └────────┘     └────────┘     └────────┘     └────────┘
```

**Message**: 4x throughput with **near-linear scaling**. Double your chips, double your robots.

---

<!-- Slide 6: Scenario 1 Live Demo -->

# Scenario 1: Live Demo

**What you will see:**

- Quad-view video: 4 robots moving simultaneously in 2x2 grid
- Each robot working on a different colored cube
- Per-chip inference latency displayed in real-time
- Aggregate throughput: **~12 FPS** across 4 chips

**Key metric to watch:** Scaling efficiency (expect >95%)

---

<!-- Slide 7: Scenario 2 -->

# Scenario 2: PI0 vs SmolVLA -- Head to Head

### Same Task, Different Models, Real-Time Comparison

```
        PI0 (Chip 0-1)        |       SmolVLA (Chip 2-3)
  ┌──────────────────────┐    |    ┌──────────────────────┐
  │  Franka Panda        │    |    │  Franka Panda        │
  │  "pick up the cube"  │    |    │  "pick up the cube"  │
  │                      │    |    │                      │
  │  Inf: 330ms          │    |    │  Inf: 229ms          │
  │  Freq: 3.0 Hz        │    |    │  Freq: 4.5 Hz        │
  └──────────────────────┘    |    └──────────────────────┘
```

**Message**: Flexible hardware allocation. Run **heterogeneous models** on the same cluster and compare in real time.

---

<!-- Slide 8: Scenario 2 Comparison -->

# Model Comparison Results

| Metric | PI0 | SmolVLA |
|--------|-----|---------|
| **Parameters** | 2.3B | 450M |
| **Inference latency** | ~330ms | ~229ms |
| **Action quality** | Higher precision | Good for fast tasks |
| **Control frequency** | 3.0 Hz (buffered: 12 Hz) | 4.5 Hz (buffered: 15 Hz) |
| **Action horizon** | 50 steps x 32 dims | 50 steps x 32 dims |

Both models run the **full inference pipeline on-chip**: vision encoding, language processing, and action denoising.

---

<!-- Slide 9: Scenario 3 -->

# Scenario 3: Ensemble Pipeline

### The Innovation: Combine Fast + Precise

```
  Camera Obs ──┬──► SmolVLA (Chip 0)  ──► Coarse Actions ──┐
               │         ~229ms                              │
               │                                     ┌──────▼──────┐
               │                                     │   Action     │
               │                                     │   Fusion     │──► Robot
               │                                     │   Layer      │
               └──► PI0 (Chip 1)      ──► Refined ──┘──────────────┘
                        ~330ms            Actions
```

SmolVLA provides **"quick reflexes"** while PI0 provides **"careful planning"**.

Wall-clock time: **~330ms** (both run concurrently, not sequentially).

---

<!-- Slide 10: Fusion Strategies -->

# Three Fusion Strategies

### 1. Weighted Average
`fused = 0.6 * PI0 + 0.4 * SmolVLA`
Best for general-purpose blending.

### 2. Temporal Blend
SmolVLA for **immediate actions** (fast reflexes), PI0 for **future actions** (precise planning).
Smooth sigmoid crossover.

### 3. Confidence Gate
Per-timestep: pick whichever model is **more confident** (lower action variance).
Automatic model selection.

---

<!-- Slide 11: Scenario 4 -->

# Scenario 4: Scaling Benchmark

### Provable, Near-Linear Throughput Scaling

```
   FPS
   12 ─┤                              ████
      │                        ████  ████
    9 ─┤                  ████  ████  ████
      │            ████  ████  ████  ████
    6 ─┤      ████  ████  ████  ████  ████
      │ ████  ████  ████  ████  ████  ████
    3 ─┤ ████  ████  ████  ████  ████  ████
      │ ████  ████  ████  ████  ████  ████
    0 ─┼──────┼──────┼──────┼──────┼──────
         1       2       3       4    Chips
```

**Message**: Predictable scaling. **Budget your hardware, predict your throughput.**

---

<!-- Slide 12: The Dashboard -->

# Live Dashboard

The Streamlit interface provides:

- **Scenario selector**: Switch between all 4 demos instantly
- **Configuration panel**: Task prompts, re-plan interval, fusion strategy
- **Live video feed**: Quad-view or side-by-side, updated at ~7 FPS
- **Real-time metrics**: Per-chip latency, control frequency, distance tracking
- **Metric cards**: At-a-glance performance per chip
- **Video recording**: Every run saves MP4 for offline review

---

<!-- Slide 13: Path to Real Hardware -->

# From Simulation to Reality

### Phase 1: Simulation (Today)
- Full inference pipeline validated on Tenstorrent hardware
- Physics-accurate robot simulation (Franka Panda in PyBullet)
- All models running on real Blackhole chips -- only the robot is simulated

### Phase 2: Physical Robot (Next)
- Replace PyBullet with real Franka Panda arm + cameras
- Same inference code, same chip allocation, same dashboard
- `multi_env.py` abstraction: swap `capture_observations()` for hardware drivers

**The demo you see today IS the production inference pipeline.**

---

<!-- Slide 14: Why Tenstorrent -->

# Why Tenstorrent for Robotics

| Advantage | Detail |
|-----------|--------|
| **Deterministic latency** | Consistent ~330ms inference, no GPU scheduling jitter |
| **Linear scaling** | Add chips, add robots -- predictable throughput |
| **Multi-model flexibility** | Run different models on different chips simultaneously |
| **Low power** | Quiet Box form factor, suitable for edge deployment |
| **Pipeline innovation** | Ensemble multi-model inference not feasible on single-GPU |

---

<!-- Slide 15: Summary -->

# Summary

Today we demonstrated:

1. **Data parallelism**: 4 robots on 4 chips with >95% scaling efficiency
2. **Model comparison**: PI0 vs SmolVLA running simultaneously, same hardware
3. **Ensemble intelligence**: Two models cooperating for superior control
4. **Predictable scaling**: Linear throughput growth from 1 to 4 chips

### Next Steps
- Deploy on physical Franka Panda robot
- Expand to additional VLA models (OpenVLA, RT-2)
- Scale to 8+ chip configurations (Galaxy)

---

<!-- Slide 16: Thank You -->

# Thank You

### Questions?

**Live demo available now** -- choose any scenario from the dashboard.

```
./models/experimental/robotics_demo/run_demo.sh
```

Tenstorrent Robotics Intelligence Suite v0.1
Powered by Blackhole AI Accelerators
