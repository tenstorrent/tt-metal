# PI0.5 (`pi0_5`) — Tenstorrent

End-to-end TTNN implementation of the **π₀.₅** (PI0.5) vision-language-action policy on Blackhole, with a PyTorch reference, LIBERO simulator integration, and a real-weights trace+2CQ perf on a 1×8 Blackhole mesh — measured against the upstream `pi05_libero` checkpoint (10-action chunks, 5 denoise steps, 2/3 cameras).

The supported multi-chip path is the **1×8 single-mesh pipeline** (`pipeline_1x8.py`):
SigLIP DP + prefill TP=8 + replicated denoise on one 1×8 Blackhole mesh.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                             PI0.5 Model                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────┐   ┌──────────────────────────┐│
│  │         PREFIX EMBEDDING            │   │    SUFFIX EMBEDDING      ││
│  │                                     │   │                          ││
│  │  ┌───────────┐   ┌───────────────┐  │   │           ┌────────┐     ││
│  │  │  Images   │   │   Language    │  │   │           │ Noisy  │     ││
│  │  │  (224x224)│   │ (200 tokens,  │  │   │           │Actions │     ││
│  │  │  (3 views)│   │  task prompt) │  │   │           │(10, 32)│     ││
│  │  └─────┬─────┘   └───────┬───────┘  │   │           └───┬────┘     ││
│  │        │                 │          │   │               │          ││
│  │        ▼                 │          │   │               ▼          ││
│  │  ┌───────────┐           │          │   │      ┌────────────────┐  ││
│  │  │  SigLIP   │           │          │   │      │ action_in_proj │  ││
│  │  │  Vision   │           │          │   │      │  (32 → 1024)   │  ││
│  │  │  Tower    │           │          │   │      └────────┬───────┘  ││
│  │  │(27 blocks)│           │          │   │               │          ││
│  │  └─────┬─────┘           │          │   │               │          ││
│  │        │                 │          │   │  ┌──────────┐ │          ││
│  │        ▼                 │          │   │  │ sincos(t)│ │          ││
│  │  ┌───────────┐           │          │   │  └─────┬────┘ │          ││
│  │  │Projector  │           │          │   │        ▼      │          ││
│  │  │(1152→2048)│           │          │   │  ┌──────────┐ │          ││
│  │  └─────┬─────┘           │          │   │  │ time_mlp │ │          ││
│  │        │                 │          │   │  │ in→silu→ │ │          ││
│  │        ▼                 ▼          │   │  │ out→silu │ │          ││
│  │  ┌───────────────────────────────┐  │   │  └─────┬────┘ │          ││
│  │  │  Image Embeds + Lang Embeds   │  │   │        │      │          ││
│  │  │  (Gemma 2B embedding)         │  │   │  adarms_cond  │          ││
│  │  └───────────────┬───────────────┘  │   └────────┼──────┼──────────┘│
│  │                  │                  │            │      │           │
│  └──────────────────┼──────────────────┘            │      │           │
│                     │                               │      │           │
│                     ▼                               │      ▼           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │               DUAL-EXPERT TRANSFORMER (18 layers)                │  │
│  │  ┌────────────────────────┐    ┌────────────────────────┐        │  │
│  │  │     Gemma 2B VLM       │    │   Gemma 300M Expert    │        │  │
│  │  │   (processes prefix)   │    │ ★ AdaRMSNorm variant ★ │        │  │
│  │  │   Plain RMSNorm        │    │                        │        │  │
│  │  │                        │    │   adaRMS Dense per     │        │  │
│  │  │  Q_vlm ──┐             │    │   layer:               │        │  │
│  │  │  K_vlm ──┼─► SHARED ◄──┼────┼─ scale, shift, gate    │        │  │
│  │  │  V_vlm ──┘   ATTN      │    │   ← from adarms_cond   │        │  │
│  │  │                        │    │                        │        │  │
│  │  │  MLP_vlm               │    │   normed = RMS(x)      │        │  │
│  │  │                        │    │   out = normed*(1+s)+b │        │  │
│  │  │                        │    │   residual ← gate * .  │        │  │
│  │  └────────────────────────┘    └────────────────────────┘        │  │
│  └────────────────────────────────────┬─────────────────────────────┘  │
│                                       │                                │
│                                       ▼                                │
│                       ┌──────────────────────────────┐                 │
│                       │     FLOW MATCHING DENOISER   │                 │
│                       │     (5 denoising steps)      │                 │
│                       │                              │                 │
│                       │  for t in [1.0 → 0.0]:       │                 │
│                       │    v_t = action_out_proj(    │                 │
│                       │           expert_out)        │                 │
│                       │    x_t ← x_t + dt · v_t      │                 │
│                       └──────────────┬───────────────┘                 │
│                                      │                                 │
│                                      ▼                                 │
│                        ┌──────────────────────────┐                    │
│                        │      Action Output       │                    │
│                        │     [batch=1, 10, 32]    │                    │
│                        └──────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key architectural details:**
- **Shared Attention**: VLM and Expert share K,V tensors (concatenated), but have separate Q and MLPs (same as PI0)
- **AdaRMSNorm in Expert**: each layer reads `(scale, shift, gate)` from a Dense projection of `adarms_cond` (time-derived). `out = RMSNorm(x)·(1+scale) + shift`; residual is gated.
- **Language prompt**: the default `pi05_libero` checkpoint (`discrete_state_input=False`) feeds the task description only — there is no separate state token in the suffix and no state projection weight in either expert. (A checkpoint trained with `discrete_state_input=True` instead discretizes the 8-dim robot state into 256 bins appended to the prompt; select it at rollout with `--state-in-prompt true`.)
- **Flow Matching**: same Euler integration as PI0; **10** denoising steps (openpi training default) from N(0,I) → actions. The perf-tuned path uses **5** (validated PCC-equal on LIBERO, ~half the device time).
- **Dual Experts**: VLM (2B) processes images + language; Expert (300M, adaRMS) processes only the action tokens.

---

## Directory layout

```
pi0_5/
├── common/
│   ├── configs.py            # GemmaConfig, SigLIPConfig, PaliGemmaConfig, ...
│   ├── weight_loader.py      # PI0WeightLoader, categorize_weights, Pi0_5WeightLoader
│   ├── checkpoint_meta.py    # action_horizon_from_checkpoint (reads config.json)
│   ├── prod_env.py           # apply_production_env_defaults() loader
│   └── pi05_production.env   # validated production perf-flag defaults
├── reference/                    # PyTorch reference (PCC + libero pytorch backend)
│   ├── torch_denoise.py
│   ├── torch_gemma.py        # GemmaAttn/MLP/Block + AdaRMSGemmaBlock
│   ├── torch_paligemma.py    # PaliGemmaBackbone + Pi0_5PaliGemmaBackbone
│   ├── torch_prefix.py
│   ├── torch_siglip.py
│   ├── torch_suffix.py       # SuffixEmbedding + Pi0_5SuffixEmbedding
│   └── torch_pi0_5_model.py  # Pi0_5Model
├── tt/                       # TTNN implementation
│   ├── ttnn_{common,gemma,paligemma,prefix,siglip,suffix}.py
│   ├── ttnn_pi0_5_model.py   # Pi0_5ModelTTNN (single-chip)
│   └── tt_bh_glx/            # 1×8 single-mesh pipeline (see below)
│       ├── pipeline_1x8.py   #   Pi0_5GLX1x8Pipeline (supported multi-chip path)
│       ├── heads.py          #   _PrefillHead / _DenoiseHead (shared, standalone)
│       ├── stage_prefill_tp8.py, *_slice.py, mesh_setup.py
├── libero_sim/               # LIBERO simulator rollout (see libero_sim/README.md)
│   └── libero_rollout.py     #   checkpoint → policy → success rate / videos
├── demo/                     # real-robot deployment (see demo/README.md)
│   ├── policy.py             #   build_policy() + headless Pi0_5LiberoAdapter re-export
│   ├── robot_runtime.py      #   RobotInterface / run_realrobot / MockRobot
│   ├── demo_realrobot.py     #   in-process CLI demo (log-only; --enable-motion)
│   └── policy_server.py      #   remote policy server (HTTP/JSON)
├── tests/
│   ├── pcc/                  # Reference-vs-spec correctness
│   └── perf/                 # Latency / throughput on Blackhole
└── weights/                  # checkpoints (not tracked; see weights/README.md)
    └── download_pi05_libero.py  # download + prepare + verify the upstream checkpoint
```

---

## Prerequisites

Build tt-metal from this branch and set up the Python env:

```bash
git clone -b tt_bh_glx_pi05_8_chips https://github.com/tenstorrent/tt-metal.git --recurse-submodules
cd tt-metal
./install_dependencies.sh
./build_metal.sh
./create_venv.sh
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
```

**Resetting / selecting devices** (Blackhole Galaxy = 32 chips, numbered 0–31):

```bash
tt-smi -r                    # reset all devices  (Galaxy: `tt-smi -glx_reset` also works)
tt-smi -r 0,1                # reset only specific device(s) by number (0..31)
TT_VISIBLE_DEVICES=0,1 <cmd> # run on a specific device subset only
```

The 1×8 tests/pipeline need 8 chips on one healthy tray — e.g.
`TT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15`. Reset first if mesh-open reports a
fabric router-sync / ethernet-handshake timeout.

---

## Quickstart

Set once (get the checkpoint via [`weights/download_pi05_libero.py`](weights/README.md)):

```bash
export PYTHONPATH=$PWD TT_METAL_HOME=$PWD PI05_CHECKPOINT_DIR=/path/to/pi05_libero_upstream
```

### TTNN — e2e trace + 2CQ perf (Blackhole)

The 1×8 test uses the visible devices (pin a subset with `TT_VISIBLE_DEVICES` on a
shared box). Two knobs vary the workload (defaults from `pi05_production.env`):

- `PI0_NUM_CAMERAS` — 2 or 3 cameras (3 = training spec; 2 also set `PI0_VLM_CHUNK_SIZE=768`).
- `PI05_NUM_DENOISE_STEPS` — denoise steps (5 = perf-tuned, 10 = training default).

```bash
source models/experimental/pi0_5/common/pi05_production.env   # perf flags (tests auto-apply too)

PI0_NUM_CAMERAS=3 PI05_NUM_DENOISE_STEPS=5 \
  python_env/bin/pytest -sq models/experimental/pi0_5/tests/perf/test_perf_tt_bh_glx_1x8_e2e_trace_2cq.py
```

### LIBERO sim — task-success rollout

Full setup + all flags in [`libero_sim/README.md`](libero_sim/README.md). Prefix the
LIBERO env vars (`PI0_TOKENIZER_PATH`, `LIBERO_REPO_PATH`, `MUJOCO_GL=osmesa`) and pick
`--backend ttnn` (single chip) or `ttnn_1x8` (1×8 mesh; pin devices with `TT_VISIBLE_DEVICES` if needed):

```bash
ROLL="models/experimental/pi0_5/libero_sim/libero_rollout.py --checkpoint $PI05_CHECKPOINT_DIR \
  --backend ttnn --steps-sweep 5 --replan-steps 5 \
  --suites libero_spatial libero_object libero_goal libero_10 --task-range 0 9"

# Quick check — 40 episodes (1 init/task × 10 tasks × 4 suites)
python_env/bin/python -u $ROLL --num-episodes 1

# Full run — 400 episodes (10 init/task × 10 tasks × 4 suites)
python_env/bin/python -u $ROLL --num-episodes 10
```

---

## Tests

Skipped automatically if the checkpoint (`$PI05_CHECKPOINT_DIR`) is missing. All
commands assume:

```bash
export PYTHONPATH=$PWD TT_METAL_HOME=$PWD PI05_CHECKPOINT_DIR=/path/to/pi05_libero_upstream
```

### PCC tests

Compare TTNN vs the PyTorch reference on real upstream weights, gated at **PCC ≥ 0.99**
(8 BH chips, 1×8 mesh):

```bash
python_env/bin/pytest -sq models/experimental/pi0_5/tests/pcc/test_pcc_tt_bh_glx_1x8.py
# On by default (runs a slow CPU torch reference); set PI05_E2E_PCC=0 to skip.
```

Results (upstream pi05_libero, ≥ 0.99 bar):

| stage | PCC |
|---|---|
| vision | 0.9997 |
| prefill | 0.9946 |
| e2e | 0.9965 |

### Perf tests

Trace + 2CQ (the canonical "fast" path). Vary the workload with two env vars
(defaults from `pi05_production.env`):

- `PI0_NUM_CAMERAS` — `2` or `3` cameras (`3` = training spec; `2` also set `PI0_VLM_CHUNK_SIZE=768`).
- `PI05_NUM_DENOISE_STEPS` — denoise steps (`5` = perf-tuned, `10` = training default).

Run (8 BH chips, 1×8 mesh):
```bash
PI0_NUM_CAMERAS=3 PI05_NUM_DENOISE_STEPS=5 \
  python_env/bin/pytest -sq models/experimental/pi0_5/tests/perf/test_perf_tt_bh_glx_1x8_e2e_trace_2cq.py
```

Results (trace + 2CQ, N=5 — per-chunk ms):

| | 2 cameras | 3 cameras |
|---|---|---|
| **8 BH chips (1×8)** | 29.0 ms | 31.2 ms |

---

## LIBERO simulator rollout

End-to-end benchmark on the four LIBERO suites (`libero_spatial`, `libero_object`,
`libero_goal`, `libero_10`), 10 tasks each.

### One-time setup

```bash
export PI05_SIM=$HOME/pi05_sim        # any writable dir

# 1. PaliGemma tokenizer (~4 MB, public)
curl -L -o $PI05_SIM/paligemma_tokenizer.model \
  https://storage.googleapis.com/big_vision/paligemma_tokenizer.model

# 2. Checkpoint: upstream openpi pi05_libero (torch/safetensors). Downloads +
#    fills config.json / norm_stats + verifies via the loader. Gated repo →
#    `huggingface-cli login` first. See weights/README.md.
huggingface-cli login
python_env/bin/python models/experimental/pi0_5/weights/download_pi05_libero.py \
  --out $PI05_SIM/pi05_libero_upstream
export PI05_CHECKPOINT_DIR=$PI05_SIM/pi05_libero_upstream

# 3. LIBERO from source (the PyPI package is broken)
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git $PI05_SIM/libero_repo

# 4. System packages for headless MuJoCo render
sudo apt install -y libosmesa6 libegl1-mesa xvfb ffmpeg

# 5. Python deps into python_env (uv-managed). robosuite 1.4.0 is REQUIRED
#    (1.5.x breaks libero 0.1.0's import path); do NOT pin numpy (<2 downgrades ttnn).
export VIRTUAL_ENV=$PWD/python_env
uv pip install "robosuite==1.4.0" mujoco bddl easydict cloudpickle gym imageio-ffmpeg
uv pip install --no-deps -e $PI05_SIM/libero_repo
```

### Run

```bash
source models/experimental/pi0_5/common/pi05_production.env      # perf flags + checkpoint path

PI0_TOKENIZER_PATH=$PI05_SIM/paligemma_tokenizer.model \
LIBERO_REPO_PATH=$PI05_SIM/libero_repo \
MUJOCO_GL=osmesa TT_METAL_HOME=$PWD PYTHONPATH=$PWD:$PI05_SIM/libero_repo \
python_env/bin/python -u models/experimental/pi0_5/libero_sim/libero_rollout.py \
  --checkpoint $PI05_CHECKPOINT_DIR \
  --suites libero_spatial libero_object libero_goal libero_10 \
  --task-range 0 9 --num-episodes 1 --steps-sweep 5 \
  --backend ttnn --replan-steps 5
# → 40 episodes (1 init/task × 4 suites). Add --video-dir <dir> for per-episode mp4s;
#   scale up with --num-episodes N (up to 50 canonical inits/task).
```

Machine-specific env vars (not in `pi05_production.env`): `PI0_TOKENIZER_PATH`,
`LIBERO_REPO_PATH`, `MUJOCO_GL=osmesa`, and `TT_METAL_CACHE` if `$HOME/.cache` is a
dangling symlink.

Key flags: `--backend {ttnn | ttnn_1x8 | pytorch}` · `--steps-sweep 5` (our path) ·
`--action-horizon 10` / `--state-in-prompt false` (upstream defaults) ·
`--replan-steps 5` · `--num-episodes` · `--suites` · `--task-range` · `--max-steps`
(per-suite defaults: spatial=220, object=280, goal=300, libero_10=520).

### LIBERO success rate (upstream pi05_libero, 100 episodes/suite × 4 suites, N=5)

Measured on the 1×8 mesh (trace+2CQ):

| suite | success |
|---|---|
| libero_spatial | 99/100 |
| libero_object | 98/100 |
| libero_goal | 95/100 |
| libero_10 | 94/100 |
| **GRAND TOTAL** | **386/400 (96.5%)** |


---

## Real-robot demo

Drive a real robot with the same TT policy — the LIBERO simulator swapped for live
cameras + a real arm. The policy (`Pi0_5LiberoAdapter.predict_chunk`) runs unchanged;
only the obs source + action sink change. See [`demo/README.md`](demo/README.md).

```bash
# No-hardware smoke on the 1×8 mesh (MockRobot, log-only)
export PI05_CHECKPOINT_DIR=/home/tt-admin/pi05_cache/pi05_libero_upstream
export PI0_TOKENIZER_PATH=/home/tt-admin/pi05_cache/tokenizer/paligemma_tokenizer.model
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD TT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
python_env/bin/python models/experimental/pi0_5/demo/demo_realrobot.py \
  --checkpoint $PI05_CHECKPOINT_DIR --backend ttnn_1x8 --task "pick up the black bowl" --steps 40
```

- `demo/policy.py` — `build_policy()` (opens device/mesh, loads checkpoint) + headless
  `Pi0_5LiberoAdapter` re-export (no LIBERO/MuJoCo needed).
- `demo/robot_runtime.py` — `RobotInterface` (the hardware seam) + `run_realrobot` loop + `MockRobot`.
- `demo/demo_realrobot.py` — in-process CLI (log-only by default; `--enable-motion` to move the arm).
- `demo/policy_server.py` — remote policy server (HTTP/JSON) for a robot on a separate machine.

To run on hardware: implement `RobotInterface` for your cameras + arm and swap it for
`MockRobot` (see the Next steps in `demo/README.md`).

---

## Dtype mapping

Weights and matmul activations are **`bfloat8_b`** across SigLIP, the VLM, and the
action expert; activation **outputs are `bfloat16`** (the bf8-output flips were
reverted after an 800-episode LIBERO sweep showed a 1–2 pp regression — weights stay
bf8). The **KV cache stays `bfloat16`** (hot-read path). Compute kernels are HiFi2
with `fp32_dest_acc_en=True` for SigLIP/SDPA and `False` for the Gemma matmuls +
sharded LN. Live code:
`tt/{ttnn_siglip,ttnn_paligemma,ttnn_gemma,ttnn_suffix,ttnn_pi0_5_model}.py`.

| Stage | Weights / biases | Matmul output | Notes |
|---|---|---|---|
| Inputs | — | images/state/x_t `bf16`; lang tokens `uint32` | `adarms_cond` precomputed `bf16` |
| SigLIP · 27 layers | `bf8_b` | attn + MLP `bf8_b` | patch-conv weight `bf16`; `fp32_dest_acc_en=True` |
| VLM Gemma-2B · 18 | `bf8_b` | `bf16` | KV cache + biases `bf16`; `fp32_dest_acc_en=False` |
| Expert Gemma-300M · 18 | `bf8_b` | `bf16` | adaRMS modulation `bf16`; sharded RMSNorm |
| Suffix · 4 linears | `bf8_b` | `bf16` | sincos(t) `fp32`→`bf16` |
| Denoise loop | — | x_t `bf16` | intentionally `bf16` (opt-in fp32 via `PI0_DENOISE_FP32=1`) |

> More bf8 is not strictly better — always re-run a LIBERO sweep (not just PCC)

---

## License

SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
SPDX-License-Identifier: Apache-2.0
