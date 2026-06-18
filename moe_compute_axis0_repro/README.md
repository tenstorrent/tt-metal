# moe_compute axis-0 all_gather deadlock — investigation artifacts

Investigation of a device hang seen when integrating `ttnn.experimental.moe_compute`
(cluster_axis=0) into a GLM-4.7 decode model on a WH galaxy (mesh (4,8), FABRIC_1D_RING,
DispatchCoreAxis.COL). The "hang" is actually a **tripped device ASSERT in the
`all_gather_async` writer kernel** during a `cluster_axis=0` all_gather — see `FINDINGS.md`.

## Files
- `FINDINGS.md` — full write-up: smoking-gun watcher assert, code locations, refuted
  hypotheses, ruled-out fixes, related issues, suggested next steps.
- `repro_moe_compute_axis0_deadlock.py` — standalone harness (random GLM-shaped weights, no
  HF load). NOTE: this **does not reproduce** the assert in isolation — moe_compute + a
  cluster_axis=0 all_gather is fine on its own. It documents the bisection (every probe
  passes), proving the fault is a full-model interaction, not the standalone op.

## Run the standalone harness (galaxy, inside the tt-metal venv)
```
export TT_METAL_RUNTIME_ROOT=<tt-metal>; export PYTHONPATH=$TT_METAL_RUNTIME_ROOT/ttnn:$TT_METAL_RUNTIME_ROOT/tools/
cd $TT_METAL_RUNTIME_ROOT/moe_compute_axis0_repro
python repro_moe_compute_axis0_deadlock.py --probe sync       # passes
python repro_moe_compute_axis0_deadlock.py --probe axis0      # passes
python repro_moe_compute_axis0_deadlock.py --probe model_seq  # passes
python repro_moe_compute_axis0_deadlock.py --probe model_seq --warmup 3  # passes (prior axis-0/1 CCL traffic)
```
All probes pass — the standalone harness does NOT reproduce the assert. See FINDINGS.md
("NOT reproducible in isolation"): the trigger needs the full forward (attention KV-cache
point-to-point + large persistent KV allocations / core-allocation pressure), not the
moe_compute↔CCL sequence the harness models.
Reset the galaxy between runs if anything hangs: `tt-smi -glx_reset_auto` (host).

## Full-model repro (the one that actually hangs)
ttnn-models repo, branch `mvasiljevic/glm-4.7-perf-tuning` (commit f220417),
`zai-org/GLM-4.7/model/graph_0`:
```
GLM_CHECK_PCC=1 python main.py                          # hangs at lm_head cluster_axis=0 all_gather
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DUMP_ALL=1 python main.py   # aborts on the tripped assert
```
