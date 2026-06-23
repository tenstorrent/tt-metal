# moe_compute fused-combine deadlock — reproduction & investigation

Investigation of the `ttnn.experimental.moe_compute` fused-combine hang on a Wormhole (4,8)
galaxy (issue #47523). This documents **what was done, how to reproduce it, and the findings**.

- Root-cause writeup: **`FINDINGS.md`**
- Live tt-triage analysis (signal, callstacks, mesh grid, axis comparison): **`logs/TRIAGE_SUMMARY.md`**
- Curated log/triage bundle: **`logs/to_send/`**

## TL;DR
- `moe_compute` full mode (fused `selective_reduce_combine`) on mesh (4,8) / `cluster_axis=0`
  **deadlocks**; the `synchronize_device` after the combine never returns.
- Every hang parks at the same line:
  `selective_reduce_combine/device/kernels/dataflow/writer.cpp:352`
  (`noc_semaphore_wait(semaphore_ptr, replicate_group_devices)`), the sync-core ring barrier
  after `fabric_multicast_bidirectional_atomic_inc_ring_1d<...replicate_axis...>`.
- Same **4 devices** every run (16/20/24/28) = **column 0 of the mesh, all 4 rows** = the
  col-0 "tail" of each replicate ring (see `logs/TRIAGE_SUMMARY.md` for the mesh grid).
- `cluster_axis=1` (8-device col dispatch ring) runs the same fused combine **to completion**.
- Input-independent (seeds 1234, 4242), reset-independent (survives full container +
  `glx_reset_auto`).

## Environment
- Runs **inside** the container; device work via `python3` in the container venv, resets/
  `docker restart` from the host.
- Runtime tt-metal: commit `68e82deb155ec50633cbb505d33a5a014cf678e2` (short `68e82deb155`,
  dated 2026-05-29; `.so` compiled 2026-06-23). This is the model's runtime build.
- Config: mesh (4,8), COL dispatch (`DispatchCoreAxis.COL`), `FABRIC_1D_RING`, GLM-4.7 MoE
  dims (160 experts, 5/dev, H=5120, N=1536, K=8, bf4 experts).
- `.env.sh` sets `TT_METAL_HOME`/`TT_METAL_RUNTIME_ROOT`/`PYTHONPATH`/`ARCH_NAME` and
  `USE_TORCH_XLA=0 ACCELERATE_USE_XLA=false`. Edit the paths for your checkout.

## The harness: `moe_compute_smoke.py`
Standalone (random weights, no HF load, ~2 min). Env knobs:

| env | effect |
|---|---|
| `SMOKE_CLUSTER_AXIS=0\|1` | dispatch ring: 0 = 4-device rows (hangs), 1 = 8-device cols (passes). Derives ring size, tokens/dev, replicate group, epilogue axis. |
| `SMOKE_PINPOINT=1` | `synchronize_device` immediately after the fused combine (isolates the hang to the combine, not the epilogue). |
| `SMOKE_SEED=<int>` | deterministic inputs (consistency check). |
| `SMOKE_COMBINE_TOPO=ring\|linear` | force the fused combine topology (default = fabric default = Ring). |
| `SMOKE_COMPUTE_ONLY=1` | matmul only, no combine (control: passes). |
| `SMOKE_CO_COMBINE=1` | compute_only + standalone combine on its real outputs (control). |

## How to reproduce
Reset the galaxy with `tt-smi -glx_reset_auto` before each run. `tt-smi -r` alone is
insufficient (intermittent `topology_mapper` "target node not mapped" at `open_mesh_device`).

```bash
# inside the container, after: source moe_compute_combine_deadlock_repro/.env.sh
# 1) axis-0 fused combine -> HANG (sync never returns)
SMOKE_CLUSTER_AXIS=0 SMOKE_PINPOINT=1 SMOKE_SEED=4242 python3 moe_compute_smoke.py
# 2) axis-1 fused combine -> PASS (combine drains + full epilogue)
SMOKE_CLUSTER_AXIS=1 SMOKE_PINPOINT=1 SMOKE_SEED=4242 python3 moe_compute_smoke.py
# 3) capture the hang: run (1) WITHOUT a timeout in the background, then in parallel:
python3 $TT_METAL_RUNTIME_ROOT/tools/tt-triage.py --skip-version-check
python3 $TT_METAL_RUNTIME_ROOT/tools/tt-triage.py --skip-version-check --run=dump_callstacks --all-cores -vv
# 4) mesh enumeration (device-id grid by row/col)
python3 probe_mesh.py
```

`tt-triage` must run while the **host process is still alive** (Inspector reads the live
runtime) — do not `timeout`-kill the smoke before triage attaches.

## Orchestration scripts (host-side)
Each resets, runs the smoke (kept alive on hang), runs tt-triage in parallel, extracts the
stuck signature, then kills only its own smoke PID.

| script | purpose |
|---|---|
| `live_hang_triage.sh` | single axis-0 hang + full triage (default + `--all-cores -vv`). |
| `axis_compare.sh` | axis-0 (hang+triage) then container-restart/reset then axis-1. |
| `axis0_fullreset.sh` | axis-0 after **container restart + glx_reset_auto** (reset-independence control). |
| `axis1_only.sh` | axis-1 only (expects drain/pass). |
| `topo_test.sh` | axis-0 with explicit `TOPO=ring\|linear`. |
| `run_probe.sh` | reset + `probe_mesh.py`. |
| `cycle_seed_triage.sh` | multi-seed cycle with per-hang triage signatures. |
| `ccl_health.py`, `standalone_combine_driver.py` | controls: bare CCLs / standalone combine (pass). |

## Notes / caveats
- Each hang leaves a wedged (D-state) process; do a container restart + `glx_reset_auto`
  before the next run for a clean slate.
- Holding global batch=64 couples ring size with tokens/dev (axis0=16, axis1=8); a
  tokens/dev-matched control is the one remaining isolation experiment.
- Explicit `SMOKE_COMBINE_TOPO=linear` does NOT hang here — it errors at program build
  (`selective_reduce_combine_program_factory.cpp:77`, mux-L1 overlap), a separate failure mode.
- `dump_op_mesh`/`device_info` error on this `tt-exalens` (`get_tray_id`), so the mesh grid
  comes from `probe_mesh.py` instead.
