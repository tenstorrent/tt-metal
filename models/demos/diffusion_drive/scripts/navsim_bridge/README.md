# NavSim PDM evaluation with the TTNN model (cross-process bridge)

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
has **not** been executed here (needs the 3.1 GB metric cache + ~12 k
scenarios); recipe below.

## 1. Start the TTNN server (tt-metal venv)

```bash
source /root/tt/tt-metal/python_env/bin/activate
export PYTHONPATH=/root/tt/tt-metal
python /root/tt/tt-metal/models/demos/diffusion_drive/scripts/ttnn_pdm_server.py \
    --checkpoint /root/02/weights/diffusiondrive_navsim_88p1_PDMS.pth \
    --anchors    /root/02/resnet34/kmeans_navsim_traj_20.npy \
    --socket     /tmp/ttnn_dd.sock
# wait for "[ttnn_pdm_server] listening on /tmp/ttnn_dd.sock"
```

## 2. Install the bridge agent into navsim (conda env)

```bash
BR=/root/tt/tt-metal/models/demos/diffusion_drive/scripts/navsim_bridge
# (a) agent class on PYTHONPATH
export PYTHONPATH=$BR:$NAVSIM_DEVKIT_ROOT
# (b) hydra config into the navsim agent config dir
cp $BR/diffusiondrive_ttnn_agent.yaml \
   $NAVSIM_DEVKIT_ROOT/navsim/planning/script/config/common/agent/
```

## 3. Run the PDM eval (conda env)

```bash
conda activate navsim
export HYDRA_FULL_ERROR=1
export NAVSIM_DEVKIT_ROOT=/root/02/DiffusionDrive
export NAVSIM_EXP_ROOT=/root/02/exp
export OPENSCENE_DATA_ROOT=/root/02/DiffusionDrive/download
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT=/root/02/dataset/maps
export PYTHONPATH=/root/tt/tt-metal/models/demos/diffusion_drive/scripts/navsim_bridge:$NAVSIM_DEVKIT_ROOT
export TTNN_DD_SOCKET=/tmp/ttnn_dd.sock

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
