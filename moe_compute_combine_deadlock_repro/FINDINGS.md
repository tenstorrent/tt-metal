# moe_compute fused-combine deadlock on WH galaxy (4,8), cluster_axis=0 — repro + isolation

**TL;DR for codeowners (gajanan-choudhary / amorrison):** `ttnn.experimental.moe_compute`
in **Full mode** (fused `selective_reduce_combine`) **deadlocks** on a (4,8) WH galaxy,
`cluster_axis=0` (4-device dispatch ring × 8 replicated cols), GLM-4.7 MoE dims. But the
**standalone `selective_reduce_combine` op passes** on the *same* config (both Ring and
Linear), and `moe_compute(compute_only)` (matmul, no combine) passes, and bare CCLs pass.
**⇒ the bug is in moe_compute's FUSED integration of the combine (core placement / mux /
matmul→combine handoff), not the combine op, not the topology, not the device.**

## Config
mesh (4,8); `cluster_axis=0` → 4 dispatch devices, 8 replicated cols; 160 experts,
experts_per_device=5, H=5120, N(moe_intermediate)=1536, K=8, tokens_per_device=16,
COL dispatch (`DispatchCoreAxis.COL`), `FABRIC_1D_RING`, bf4 experts.

## Build / version
Smoke repro run on build-tree tt-metal HEAD `027d0f97b19`. **The deadlock is
version-independent**: the full GLM-4.7 model hangs in the fused combine on base
`dev20260519`, latest-main `6008ff55566`, AND `027d0f97b19` — it is NOT a recent regression.
Run with `USE_TORCH_XLA=0 ACCELERATE_USE_XLA=false` (the rebuilt tt-metal is ABI-incompatible
with the venv's torch_xla; the repro doesn't need it).

## The signal (two views; host pinpoint is authoritative)
- **Host:** a `ttnn.synchronize_device` placed immediately after the fused `moe_compute`
  **never returns** (process `timeout` → exit 124). The sync after `all_to_all_dispatch_metadata`
  returns fine ⇒ the hang is inside the fused combine, not dispatch.
  *(Without that sync, the host async-enqueues the epilogue and races ahead — the apparent
  "hang at the epilogue reduce_scatter" is an illusion; the device is already stuck in the combine.)*
- **Device (watcher, `TT_METAL_WATCHER=10`):** `Device 0 core(1,0) BRISC tripped assert on
  line 260, kernel all_gather_async/.../minimal_default_writer.cpp; Last waypoint NSMD,CRBW,W,W,W`
  — a core parked in a NOC-semaphore wait (a ring-barrier atomic-inc that never arrives).
  (Watcher op-attribution is unreliable under async; the pinpoint above is authoritative.)

## Isolation matrix (all on the SAME (4,8)/cluster_axis=0/COL/FABRIC_1D_RING device)
| Test | Result |
|---|---|
| `ccl_health.py` — standalone CCLs (reduce_scatter/all_gather, axis0/axis1, Ring+Linear), NO moe_compute | **PASS** (exit 0) |
| `moe_compute_smoke.py SMOKE_COMPUTE_ONLY=1` — device + dispatch_metadata + moe_compute **matmul** + sync | **PASS** (exit 0) |
| `moe_compute_smoke.py SMOKE_PINPOINT=1` — fused combine, Ring (fabric default) | **HANG** (exit 124) |
| `moe_compute_smoke.py SMOKE_PINPOINT=1 SMOKE_COMBINE_TOPO=linear` — fused combine, explicit Linear | **HANG** (exit 124) |
| `standalone_combine_driver.py COMBINE_TOPO=ring` — standalone `selective_reduce_combine` | **PASS** (exit 0) |
| `standalone_combine_driver.py COMBINE_TOPO=linear` — standalone `selective_reduce_combine` | **PASS** (exit 0) |

The standalone driver runs the codeowners' own `run_combine_test`
(`models/demos/deepseek_v3/tests/test_combine_tg.py`), parametrized for exactly
(4,8)/batch64/cluster_axis0/COL/FABRIC_1D_RING/worker`((0,0),(3,3))`/mux`((4,0),(5,7))`/
token×data parallel 4×4 — i.e. the same 4-device ring the fused path deadlocks on.

## Conclusion / where to look
The `selective_reduce_combine` op (and its ring barrier
`fabric_multicast_bidirectional_atomic_inc_ring_1d`) works standalone on this config with
**both** topologies. The fused `moe_compute` deadlocks using that same op internally. The
differences are in **moe_compute's fused setup of the combine**:
- core placement (`moe_core_placement.cpp` — log: "selected tilize cores 4, combine cores
  16, matmul cores 12") vs the standalone test's explicit worker `((0,0),(3,3))`;
- mux core range (smoke used `((3,0),(4,7))`; the passing standalone test uses `((4,0),(5,7))`);
- the internal matmul-output → combine `dense_input` handoff (slots: combine consumes
  moe_compute outputs [4]=matmul_output, [1]=activations, [2]=token_maps, [0]=token_counts);
- topology resolution: fused path resolves from the fabric default (FABRIC_1D_RING→Ring) at
  `moe_compute_device_operation.cpp:482-494`; passing `topology=Linear` explicitly did NOT help.

**Version-independent — NOT a recent regression.** The full-model fused-combine hang occurs
on base `dev20260519`, latest-main `6008ff55566`, and `027d0f97b19` alike. (Separately: an
isolated *smoke* of the fused combine passed on a pre-`027d0f97b19` build but hangs on the
current one; unconfirmed — a `.so`-only downgrade is ABI-incompatible with the current python
tree. This is secondary; the underlying full-model deadlock predates any recent core-placement
change.) So the suspect is moe_compute's fused combine setup in general, not a specific PR.

## Workaround status (honest)
`moe_compute(compute_only=True)` (matmul only; `cluster_axis=None`, no combine) passes, and
the standalone `selective_reduce_combine` passes on **synthetic** inputs (deepseek test). BUT
**chaining them is NOT a drop-in.** Feeding `compute_only`'s actual outputs (slots
0/1/2/4 = token_counts/activation/token_maps/matmul) into the standalone op
(`moe_compute_smoke.py SMOKE_CO_COMBINE=1`, with `get_moe_combine_cores` for worker cores)
**fails validation**:
```
compute_only outs: counts (70,8) act (1,768) maps (5,260) matmul (70,2,32,5120)
TT_FATAL selective_reduce_combine_device_operation.cpp:64:
  "token activations tensor expected to have aligned 2*experts_per_device+1 elements per token"
  (activations_stride_elm == expected_activations_stride_elm)
```
So the public `moe_compute(compute_only)` outputs are NOT in the exact metadata layout the
standalone combine expects (its `dense_activations`/metadata tensor must have stride
`2*experts_per_device+1` per token) — the matmul→combine handoff inside the fused op does
something the public outputs don't expose. **A clean `compute_only` + standalone-combine
workaround therefore needs codeowner guidance on the exact tensor mapping/format** (or a
supported path). It has NOT been made to work end-to-end (smoke or full model).

What the GLM-4.7 model actually ships today: `compute_only` matmul + a Ring `all_gather`+`sum`
**proxy** combine (runs end-to-end, but PCC NOT validated) — NOT `selective_reduce_combine`.

## Open question for codeowners
Given the standalone `selective_reduce_combine` works (both topologies) but the fused
`moe_compute` combine deadlocks on this exact (4,8)/cluster_axis=0 config: (a) what is the
intended supported way to run `compute_only` → standalone combine (the exact output→input
tensor mapping/format)? and (b) why does the fused integration deadlock where the standalone
op does not — in moe_compute's combine/mux core placement, the matmul→combine handoff, or the
fabric-default topology resolution? (version-independent; reproduces on base / latest-main / 027d0f97b19)

## How to run
```bash
# inside the build env (TT_METAL_RUNTIME_ROOT on PYTHONPATH), USE_TORCH_XLA=0 ACCELERATE_USE_XLA=false
# 1) device + CCL health (no moe_compute) — PASS
python3 moe_compute_combine_deadlock_repro/ccl_health.py
# 2) moe_compute matmul only — PASS
SMOKE_COMPUTE_ONLY=1 python3 moe_compute_combine_deadlock_repro/moe_compute_smoke.py
# 3) fused combine — HANG (exit 124). add SMOKE_COMBINE_TOPO=linear: still HANG
SMOKE_PINPOINT=1 timeout 360 python3 moe_compute_combine_deadlock_repro/moe_compute_smoke.py
# 4) standalone selective_reduce_combine — PASS (both topologies)
USE_TORUS_MODE=1 COMBINE_TOPO=ring   python3 moe_compute_combine_deadlock_repro/standalone_combine_driver.py
USE_TORUS_MODE=1 COMBINE_TOPO=linear python3 moe_compute_combine_deadlock_repro/standalone_combine_driver.py
```
Reset the galaxy between hanging runs (`tt-smi -glx_reset_auto`). `standalone_combine_driver.py`
needs `pip install pytz` (transitive import of `models.perf.benchmarking_utils`).
The smoke uses random weights / no HF load (~2 min); it is fully standalone (ttnn only).
