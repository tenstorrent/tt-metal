# NavSim PDM evaluation with the TTNN model (single-env, in-process)

This is the **default** path for evaluating the DiffusionDrive TTNN model under
the NavSim PDM harness. It runs the model **in the same process** as
`run_pdm_score.py`, so it needs **no socket bridge and no second process** —
unlike `../navsim_bridge/` (kept as a fallback, see the bottom of this file).

```
 run_pdm_score.py  (navsim, Python 3.10)
   └─ DiffusionDriveTtnnInprocAgent.forward ──► TtnnDiffusionDriveModel (on Wormhole)
        (TransfuserFeatureBuilder, in-process)      returns trajectory (1,8,3)
```

## Requirements (assumed already provisioned)

Two repos cooperate — `tt-metal` supplies the **model**, the NavSim devkit supplies
the **scorer + data**. "Single env" means one Python **process/interpreter**, not one
repo:

All machine-specific paths come from environment variables — **export the
[§9.1 "Eval environment" block from the demo README](../../README.md#9-navsim-pdm-evaluation)
first** (`$TT_METAL_HOME`, `$DD_DATA_ROOT`, `$DD_CHECKPOINT_PATH`,
`$DD_ANCHOR_PATH`, `$NAVSIM_*`, `$OPENSCENE_DATA_ROOT`, `$NUPLAN_*`). The
reference defaults assume everything is staged under `$DD_DATA_ROOT` (default
`/mnt/diffusion-drive`). The eval runs in the `navsim` conda env.

- **`$TT_METAL_HOME`** — built so `ttnn` imports (provides the model + this agent).
- **`$NAVSIM_DEVKIT_ROOT`** (default `$DD_DATA_ROOT/DiffusionDrive`) — the NavSim
  devkit: the `run_pdm_score.py` harness, the `navsim` package, and the hydra
  configs. **Required and cannot be moved** — `navsim` is *editable*-installed
  into the `navsim` env pointing here (`import navsim` loads
  `$NAVSIM_DEVKIT_ROOT/navsim/__init__.py`).
- **`navsim` conda env** (Python 3.10) — the navsim stack plus, after the
  one-time fix below, `ttnn`.
- **Data** — OpenScene sensors (`$OPENSCENE_DATA_ROOT`), maps (`$NUPLAN_MAPS_ROOT`),
  PDM metric cache (`$NAVSIM_EXP_ROOT/metric_cache` — see demo README §9.3), the
  checkpoint (`$DD_CHECKPOINT_PATH`) and plan anchors (`$DD_ANCHOR_PATH`).

You run `run_pdm_score.py` *from* the devkit; it imports the TTNN model from
`tt-metal` via `PYTHONPATH`. PDM is NavSim's metric, so the devkit is intrinsic.

## Why this works now

The bridge existed for two reasons: (a) navsim was Python 3.9 while `ttnn` is
Python 3.10, and (b) a single Wormhole must be arbitrated across workers.

(a) is solved by the **`navsim` conda env** (Python 3.10) — the
materialisation of the upstream DiffusionDrive `python310` branch's dependency
modernisation (`torch==2.0.1+cpu`, `numpy==1.23.4`, hydra/ray/lightning,
`diffusers==0.27.2`, `einops==0.8.2`; CPU-only). It has the full navsim stack
**and** imports `ttnn`, so both run in one process.

(b) is handled by running the eval **single-process** (see step 3). The model
forward serialises on the one device either way, so this costs nothing versus
the bridge.

## Prerequisite — make `ttnn` importable in `navsim`

`navsim` has the navsim stack but not the tt-metal Python wiring. Two one-time
fixes (no torch/numpy change needed):

1. **Path.** The real `ttnn` package lives at `$TT_METAL_HOME/ttnn/ttnn/`, so
   the *inner-package parent* must be on `PYTHONPATH` — just `$TT_METAL_HOME`
   resolves to an empty namespace package (`ttnn.open_device` missing). Every
   command below uses:
   ```bash
   export TTNN_PP=$TT_METAL_HOME/ttnn:$TT_METAL_HOME:$TT_METAL_HOME/tools
   ```
2. **Deps.** Install ttnn's pure-Python deps that the navsim env lacks, pinned to
   the tt-metal venv (`--no-deps` so navsim's numpy/matplotlib pins are untouched).
   Run pip **inside the activated env** (machine-specific — don't hard-code a
   `…/miniconda3/envs/<name>/bin/pip` path):
   ```bash
   conda activate navsim
   pip install --no-deps loguru==0.6.0 graphviz==0.21 seaborn==0.13.2 ml_dtypes==0.5.4
   ```
   (ttnn declares `numpy>=1.24.4`, but it imports and runs fine on the navsim
   env's pinned `1.23.4` — leave numpy alone.)

Verify: `PYTHONPATH=$TTNN_PP python -c "import ttnn; print(ttnn.open_device)"`.

## 0. One-time: validate parity (recommended before a full run)

Confirm the on-device stack still matches the PyTorch reference **under torch
2.0.1** (navsim) — the one thing the env change could perturb:

```bash
conda activate navsim
export TTNN_PP=$TT_METAL_HOME/ttnn:$TT_METAL_HOME:$TT_METAL_HOME/tools
PYTHONPATH=$TTNN_PP:$NAVSIM_DEVKIT_ROOT \
  python $TT_METAL_HOME/models/demos/diffusion_drive/scripts/navsim_inproc/check_parity.py \
    --checkpoint "$DD_CHECKPOINT_PATH" \
    --anchors    "$DD_ANCHOR_PATH"
# expect: trajectory PCC ~0.9998, "RESULT: OK"
# (--checkpoint/--anchors default to $DD_CHECKPOINT_PATH/$DD_ANCHOR_PATH if exported)
```

## 1. Install the agent into the navsim env

```bash
conda activate navsim
BR=$TT_METAL_HOME/models/demos/diffusion_drive/scripts/navsim_inproc
export TTNN_PP=$TT_METAL_HOME/ttnn:$TT_METAL_HOME:$TT_METAL_HOME/tools
# (a) agent class on PYTHONPATH (imported as a bare top-level module), plus tt-metal ttnn
export PYTHONPATH=$BR:$TTNN_PP:$NAVSIM_DEVKIT_ROOT
# (b) hydra config into the navsim agent config dir
cp $BR/diffusiondrive_ttnn_inproc_agent.yaml \
   $NAVSIM_DEVKIT_ROOT/navsim/planning/script/config/common/agent/
```

## 2. Run the PDM eval (navsim env)

```bash
export HYDRA_FULL_ERROR=1
# NavSim env vars — canonical block is demo README §9.1 (these derive from
# $DD_DATA_ROOT, default /mnt/diffusion-drive). Repeated here so this recipe is
# standalone; each respects an already-exported value:
export DD_DATA_ROOT="${DD_DATA_ROOT:-/mnt/diffusion-drive}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$DD_DATA_ROOT/DiffusionDrive}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$DD_DATA_ROOT/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$NAVSIM_DEVKIT_ROOT/download}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$DD_DATA_ROOT/dataset/maps}"
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
  with the serialized device forward. Measured — **full navtest (12146 scenes):
  25m53s, PDM 0.878922** (clean exit 0; matches the sequential baseline 0.878902
  → correct at scale), **~2.1× faster** than the eager-sequential full run
  (~54 min). Same-conditions 2000-scene A/B: **237 s vs 377 s sequential-traced =
  1.59×** (118.5 vs 188.5 ms/scene).
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
- **torch 2.0.1.** The model now runs under navsim's torch 2.0.1 (the bridge
  server ran it under the venv's 2.11). The parity check (step 0) is the gate; if
  an API gap appears, bump navsim's torch toward ttnn's build target and re-pin.
- **DDIM noise** is drawn fresh (unseeded) per forward, matching the upstream eval.
- **Trace capture (on by default).** The agent captures the consolidated
  backbone-loop trace once in `initialize()` and replays it per scene via
  `execute_compiled()`. Set `DD_TRACE=0` to disable (eager `__call__`, for A/B).
  Measured on a 2000-scene navtest A/B (sequential, navsim): PDM unchanged
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
