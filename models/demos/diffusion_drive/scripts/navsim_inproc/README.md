# NavSim PDM evaluation with the TTNN model (single-env, in-process)

This is the **default** path for evaluating the DiffusionDrive TTNN model under
the NavSim PDM harness. It runs the model **in the same process** as
`run_pdm_score.py`, so it needs **no socket bridge and no second process** —
unlike `../navsim_bridge/` (kept as a fallback, see the bottom of this file).

```
 run_pdm_score.py  (navsim310, Python 3.10)
   └─ DiffusionDriveTtnnInprocAgent.forward ──► TtnnDiffusionDriveModel (on Wormhole)
        (TransfuserFeatureBuilder, in-process)      returns trajectory (1,8,3)
```

## Requirements (assumed already provisioned)

Two repos cooperate — `tt-metal` supplies the **model**, the NavSim devkit supplies
the **scorer + data**. "Single env" means one Python **process/interpreter**, not one
repo:

- **`/root/tt/tt-metal`** — built so `ttnn` imports (provides the model + this agent).
- **`/root/02/DiffusionDrive`** — the NavSim devkit: the `run_pdm_score.py` harness,
  the `navsim` package, and the hydra configs. **Required and cannot be moved** —
  `navsim` is *editable*-installed into `navsim310` pointing here (`import navsim`
  loads `/root/02/DiffusionDrive/navsim/__init__.py`).
- **`navsim310` conda env** (Python 3.10) — the navsim stack plus, after the one-time
  fix below, `ttnn`.
- **Data** — OpenScene sensors (`/root/02/DiffusionDrive/download`), maps
  (`/root/02/dataset/maps`), PDM metric cache (`/root/02/exp`), the checkpoint
  (`/root/02/weights/diffusiondrive_navsim_88p1_PDMS.pth`) and plan anchors
  (`/root/02/resnet34/kmeans_navsim_traj_20.npy`).

You run `run_pdm_score.py` *from* the devkit; it imports the TTNN model from
`tt-metal` via `PYTHONPATH`. PDM is NavSim's metric, so the devkit is intrinsic.

## Why this works now

The bridge existed for two reasons: (a) navsim was Python 3.9 while `ttnn` is
Python 3.10, and (b) a single Wormhole must be arbitrated across workers.

(a) is solved by the **`navsim310` conda env** (Python 3.10) — the
materialisation of the upstream DiffusionDrive `python310` branch's dependency
modernisation (`torch==2.0.1+cpu`, `numpy==1.23.4`, hydra/ray/lightning,
`diffusers==0.27.2`, `einops==0.8.2`; CPU-only). It has the full navsim stack
**and** imports `ttnn`, so both run in one process.

(b) is handled by running the eval **single-process** (see step 3). The model
forward serialises on the one device either way, so this costs nothing versus
the bridge.

## Prerequisite — make `ttnn` importable in `navsim310`

`navsim310` has the navsim stack but not the tt-metal Python wiring. Two one-time
fixes (no torch/numpy change needed):

1. **Path.** The real `ttnn` package lives at `/root/tt/tt-metal/ttnn/ttnn/`, so
   the *inner-package parent* must be on `PYTHONPATH` — just `/root/tt/tt-metal`
   resolves to an empty namespace package (`ttnn.open_device` missing). Every
   command below uses:
   ```bash
   export TTNN_PP=/root/tt/tt-metal/ttnn:/root/tt/tt-metal:/root/tt/tt-metal/tools
   ```
2. **Deps.** Install ttnn's pure-Python deps that navsim310 lacks, pinned to the
   tt-metal venv (`--no-deps` so navsim's numpy/matplotlib pins are untouched):
   ```bash
   /root/02/miniconda3/envs/navsim310/bin/pip install --no-deps \
     loguru==0.6.0 graphviz==0.21 seaborn==0.13.2 ml_dtypes==0.5.4
   ```
   (ttnn declares `numpy>=1.24.4`, but it imports and runs fine on navsim310's
   pinned `1.23.4` — leave numpy alone.)

Verify: `PYTHONPATH=$TTNN_PP python -c "import ttnn; print(ttnn.open_device)"`.

## 0. One-time: validate parity (recommended before a full run)

Confirm the on-device stack still matches the PyTorch reference **under torch
2.0.1** (navsim310) — the one thing the env change could perturb:

```bash
conda activate navsim310
export TTNN_PP=/root/tt/tt-metal/ttnn:/root/tt/tt-metal:/root/tt/tt-metal/tools
PYTHONPATH=$TTNN_PP:/root/02/DiffusionDrive \
  python /root/tt/tt-metal/models/demos/diffusion_drive/scripts/navsim_inproc/check_parity.py \
    --checkpoint /root/02/weights/diffusiondrive_navsim_88p1_PDMS.pth \
    --anchors    /root/02/resnet34/kmeans_navsim_traj_20.npy
# expect: trajectory PCC ~0.9998, "RESULT: OK"
```

## 1. Install the agent into navsim (navsim310 env)

```bash
conda activate navsim310
BR=/root/tt/tt-metal/models/demos/diffusion_drive/scripts/navsim_inproc
export TTNN_PP=/root/tt/tt-metal/ttnn:/root/tt/tt-metal:/root/tt/tt-metal/tools
# (a) agent class on PYTHONPATH (imported as a bare top-level module), plus tt-metal ttnn
export PYTHONPATH=$BR:$TTNN_PP:$NAVSIM_DEVKIT_ROOT
# (b) hydra config into the navsim agent config dir
cp $BR/diffusiondrive_ttnn_inproc_agent.yaml \
   $NAVSIM_DEVKIT_ROOT/navsim/planning/script/config/common/agent/
```

## 2. Run the PDM eval (navsim310 env)

```bash
export HYDRA_FULL_ERROR=1
export NAVSIM_DEVKIT_ROOT=/root/02/DiffusionDrive
export NAVSIM_EXP_ROOT=/root/02/exp
export OPENSCENE_DATA_ROOT=/root/02/DiffusionDrive/download
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT=/root/02/dataset/maps
export PYTHONPATH=$BR:$TTNN_PP:$NAVSIM_DEVKIT_ROOT

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
    train_test_split=navtest \
    agent=diffusiondrive_ttnn_inproc_agent \
    worker=sequential \
    experiment_name=diffusiondrive_ttnn_inproc_eval
```

The final `Final average score of valid results: …` line is the PDM score to
compare against the upstream **88.04** (and the bridge run's ~0.8786). Cap a quick
run with `train_test_split.scene_filter.max_scenes=N`.

## 3. Device arbitration — pick the worker carefully

One Wormhole = one device handle; the TTNN device context is **thread-affine**;
and `run_pdm_score` instantiates a **fresh agent + `initialize()` per worker
chunk** (not one shared instance).

- **`worker=sequential`** (validated default): one chunk → one agent → opens +
  uses the device on that thread. The proven path — parity, smoke, full navtest.
- **`worker=single_machine_thread_pool`** (**fastest**): the N worker chunks build
  features in parallel and funnel every forward to a **process-wide singleton**
  owning the device on one thread. This overlaps the CPU-bound feature-building
  with the serialized device forward. Measured on a 2000-scene navtest: **237 s vs
  377 s sequential = 1.59×** (118.5 vs 188.5 ms/scene), PDM **0.8678** (matches
  sequential 0.8669/0.8676 → correct).
  ```bash
  python …/run_pdm_score.py … worker=single_machine_thread_pool …
  ```
  The agent selects this path **automatically** when run under the thread-pool
  worker (no env var) — it detects it's on a pool thread, not the main thread.
  The funnel is mandatory here: per-chunk agents would otherwise each
  `open_device(0)` and touch the thread-affine context from many threads
  (→ "context_id invalid" FATAL). It exits via `os._exit(0)` after the results CSV
  is written, to skip tt_metal's main-thread `MetalContext` teardown which SIGABRTs
  on the thread-affine cluster — results are unaffected.
- **`worker=ray_distributed`**: **do not use** — each ray worker is a separate
  process and would try to `ttnn.open_device(0)` concurrently → conflict.

For a quick smoke run, cap scenes with `train_test_split.scene_filter.max_scenes=5`.

## Notes / caveats

- **bf16 accuracy.** Trajectory PCC ≈ 0.9998 vs fp32 (parity check). bf16 has been
  shown to hold the PDM near 88.04 in the bridge run (~0.8786); the in-process
  path is numerically identical (same on-device graph), only the transport differs.
- **torch 2.0.1.** The model now runs under navsim310's torch 2.0.1 (the bridge
  server ran it under the venv's 2.11). The parity check (step 0) is the gate; if
  an API gap appears, bump navsim310's torch toward ttnn's build target and re-pin.
- **DDIM noise** is drawn fresh (unseeded) per forward, matching the upstream eval.
- **Trace capture (on by default).** The agent captures the consolidated
  backbone-loop trace once in `initialize()` and replays it per scene via
  `execute_compiled()`. Set `DD_TRACE=0` to disable (eager `__call__`, for A/B).
  Measured on a 2000-scene navtest A/B (sequential, navsim310): PDM unchanged
  (eager 0.8669 vs traced 0.8676), wall **394→377 s = 1.045×** (−8.5 ms/scene).
  The win is modest because the **in-process per-scene cost is dominated by
  navsim CPU** (feature building + PDM scoring), not the model forward — the
  forward itself is ~1.34× faster but is a minority of per-scene wall, so more
  model-side tracing buys little here. The **bigger eval lever is the
  `worker=single_machine_thread_pool` funnel (§3)** — it overlaps that navsim CPU
  with the device forward for **~1.59×** (and stacks with the trace). Removing the socket (vs the
  bridge) is a separate <1% effect.

## Fallback: the cross-process bridge

If `ttnn` cannot be imported in the navsim env (e.g. a binary/source skew), use
the original two-process bridge in `../navsim_bridge/` (a `ttnn_pdm_server.py` in
the tt-metal venv + a socket agent in the navsim env). Same model, same build
chain; only the transport differs.
