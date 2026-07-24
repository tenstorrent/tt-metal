# NavSim PDM evaluation with the TTNN model (cross-process bridge)

> **This is the fallback path.** The default is the in-process agent in
> [`../navsim_inproc/`](../navsim_inproc/README.md) (single env, no socket/server).
> Use this bridge only when `ttnn` cannot be imported in the navsim env.

The NavSim eval harness (`run_pdm_score.py`) runs in the **`navsim` conda env
(Python 3.9)**; TTNN is only importable in the **tt-metal venv (Python 3.10)**.
They can't share the compiled `ttnn` wheel, so inference is delegated over a
Unix-domain socket:

```
 run_pdm_score.py (conda 3.9)                 ttnn_pdm_server.py (venv 3.10)
   └─ DiffusionDriveTtnnAgent.forward ──socket──► TtnnDiffusionDriveModel (full on-device)
        (builds camera/lidar/status)   ◄──────────  returns trajectory (8,3)
```

Status: the bridge is **validated end-to-end** (conda agent → venv server →
trajectory).  The server now builds the **fully on-device** stack
(`build_stage2 → 3 → 3_4 → 3_5 → 3_6 → 3_7` — every weight-bearing op on TTNN),
validated at production resolution (256×1024 / 256×256, real 88.04 checkpoint)
at **traj PCC 0.9998** / scores 0.9985 vs the PyTorch reference.  A full PDM run
has **not** been executed here (needs the 3.1 GB PDM metric cache — build it once
per the demo README [§9.3](../../README.md#9-navsim-pdm-evaluation) — plus ~12 k
scenarios); recipe below.

## 1. Start the TTNN server (tt-metal venv)

```bash
# Paths come from the demo README §9.1 "Eval environment" block
# ($TT_METAL_HOME, $DD_CHECKPOINT_PATH, $DD_ANCHOR_PATH, $TTNN_DD_SOCKET).
source "$TT_METAL_HOME/python_env/bin/activate"
export PYTHONPATH="$TT_METAL_HOME"
python "$TT_METAL_HOME/models/demos/diffusion_drive/scripts/ttnn_pdm_server.py" \
    --checkpoint "$DD_CHECKPOINT_PATH" \
    --anchors    "$DD_ANCHOR_PATH" \
    --socket     "${TTNN_DD_SOCKET:-/tmp/ttnn_dd.sock}"
# wait for "[ttnn_pdm_server] listening on …/ttnn_dd.sock"
# (--checkpoint/--anchors/--socket default to $DD_CHECKPOINT_PATH/$DD_ANCHOR_PATH/$TTNN_DD_SOCKET)
```

## 2. Install the bridge agent into navsim (conda env)

```bash
BR=$TT_METAL_HOME/models/demos/diffusion_drive/scripts/navsim_bridge
# (a) agent class on PYTHONPATH
export PYTHONPATH=$BR:$NAVSIM_DEVKIT_ROOT
# (b) hydra config into the navsim agent config dir
cp $BR/diffusiondrive_ttnn_agent.yaml \
   $NAVSIM_DEVKIT_ROOT/navsim/planning/script/config/common/agent/
```

## 3. Run the PDM eval (conda env)

```bash
conda activate navsim            # the bridge's Python-3.9 navsim env (machine-specific name)
export HYDRA_FULL_ERROR=1
# NavSim env vars — canonical block is demo README §9.1 (derive from $DD_DATA_ROOT,
# default /mnt/diffusion-drive); each respects an already-exported value:
export DD_DATA_ROOT="${DD_DATA_ROOT:-/mnt/diffusion-drive}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$DD_DATA_ROOT/DiffusionDrive}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$DD_DATA_ROOT/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$NAVSIM_DEVKIT_ROOT/download}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$DD_DATA_ROOT/dataset/maps}"
export PYTHONPATH=$TT_METAL_HOME/models/demos/diffusion_drive/scripts/navsim_bridge:$NAVSIM_DEVKIT_ROOT
export TTNN_DD_SOCKET="${TTNN_DD_SOCKET:-/tmp/ttnn_dd.sock}"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
    train_test_split=navtest \
    agent=diffusiondrive_ttnn_agent \
    worker=ray_distributed \
    experiment_name=diffusiondrive_ttnn_eval
```

The final `Final average score of valid results: …` line is the PDM score to
compare against the upstream **88.04**.

## Notes / caveats

- **Single device, serialized inference.** All ray workers share one TTNN
  server + one Wormhole device, so forwards serialize.  Correctness is
  unaffected; wall-clock will differ from the CPU-torch run.  If ray contention
  is a problem, use `worker=single_machine_thread_pool` or reduce workers.
- **bf16 accuracy.** Per-forward scores PCC ≈ 0.996 vs fp32.  Whether that holds
  the PDM near 88.04 across all scenarios is exactly what this run answers.  If
  it drifts, raise fidelity in the server (fp32 accumulation / HiFi math) rather
  than changing the agent.
- **DDIM noise.** The server draws fresh DDIM noise per forward (unseeded),
  matching the upstream eval.  Set a seed in the server only for A/B debugging.
- **Smaller dry run first:** `train_test_split=navtest` with a few-log subset, or
  `worker=sequential`, to confirm the pipeline before the full ~12 k scenarios.
