# AGMM Task 1 — parent-op baselines (complete)

Execution-plan reference: `tools/mm_sweep/REGIME_A_AGMM_EXECUTION_PLAN.md`, Task 1.

## Environment
- Branch/commit: `cglagovich/regime-a-bh` @ `de7df778f44` (merged origin/main `714f6714429` + integration fixes); submodules `umd 4ed96fb`, `tracy 1171005`.
- Machine: `bh-glx-120-b10u08` (Blackhole Galaxy, 32 chips), ubuntu-22.04-dev container; mesh descriptor `single_bh_galaxy_torus_xy_graph_descriptor.textproto`.
- Fabric: **mux v2** (PR #48675, `*FabricMuxV2Smoke*` 12/12). AG: `FABRIC_1D_RING`, **4 KiB** packets, `num_links=2`, `num_workers_per_link=6`, `num_buffers_per_channel=2`, `chunks_per_sync=16`, topology `Ring`.
- Harness: `models/tt_dit/tests/models/wan2_2/test_agmm_regime_a_baseline.py` (reuses conftest `mesh_device` fixture). Raw JSON: `/data/cglagovich/agmm_baseline.json`.
- Timing: **trace capture + host wall**, median of 10 timed iters (2 warmup dropped) — same basis for `T_mm` and `T_ag`. (Host-wall includes dispatch, so these exceed the device-profiler kernel spans in `REGIME_A_GLX_MERGED_PERF_REPORT.md`; that report remains the kernel-span single-chip characterization.)

## Reference commands (step 1)
| ref | result |
|---|---|
| single-chip `regime_a_matmul` (correctness + cache) | ✅ 111 passed |
| BH 4x8 fused AGMM `unit/fused/check` | ✅ 1 passed, PCC 0.9999875 |
| nightly `all_gather_minimal_matmul_addcmul` | ❌ 2 failed — pre-existing L1 overflow in `minimal_matmul` (unrelated; accepted) |
| `*FabricMuxV2Smoke*` gtests | ✅ 12/12 |

## Baseline table (host-wall trace µs; T_mm = single-chip regime_a config=None; T_ag = standalone all_gather)
`in0[M,K]` K-sharded across D on the cluster axis, `all_gather_async(dim=3)` → full `in0`; `T_unfused = T_mm + T_ag`; fused ideal = `max(T_mm, T_ag)`. All PCC = 1.0000 (T_mm vs torch; T_ag gather vs torch).

| shape | category | T_mm | T_ag D4 | T_ag D8 | ideal=max(T_mm,T_ag) | T_unfused D4 | T_unfused D8 | unfused/ideal (D4) |
|---|---|---|---|---|---|---|---|---|
| 32x6144x3072 | flagship | 134.2 | 89.8 | 64.3 | 134.2 | 224.0 | 198.4 | 1.67× |
| 32x6144x6144 | wide-N (in1-bound) | 243.7 | 66.6 | 63.5 | 243.7 | 310.2 | 307.2 | 1.27× |
| 256x6144x768 | narrow-N (fabric-sensitive) | 126.5 | 83.9 | 99.3 | 126.5 | 210.4 | 225.8 | 1.66× |
| 128x2048x512 | shallow-K (early stall) | 74.4 | 71.4 | 91.5 | 74.4 | 145.8 | 165.9 | 1.96× |
| 32x15360x768 | deep-K | 132.9 | 64.7 | 72.0 | 132.9 | 197.6 | 204.9 | 1.49× |

Observations:
- In every case `T_mm > T_ag`, so the fused **ideal ≈ T_mm** (compute/in1-read dominates); the AG time is the overhead a perfectly-overlapped fused op could hide.
- Potential fused speedup vs unfused (D4): **1.27×–1.96×**; largest on shallow-K/small shapes where AG is a big fraction of T_mm, smallest on wide-N in1-bound (T_ag ≪ T_mm).
- D8 vs D4: AG time is similar or slightly higher at D8 for small shapes (more ring hops / latency-bound), lower where bandwidth-bound.

## Finding: multi-device ring-distance is available (Task-2 unblocker)
Earlier, `regime_a_matmul` could not build on a D>1 mesh (its PARETO ring optimizer calls the unit-mesh-only overload of `get_worker_noc_hop_distance`). Searching the codebase (`tt_metal/api/tt-metalium/experimental/device.hpp`) shows **two overloads**:
1. `get_worker_noc_hop_distance(IDevice*, src, dst, noc)` — asserts `num_devices()==1` on a MeshDevice.
2. **`get_worker_noc_hop_distance(MeshDevice*, MeshCoordinate, src, dst, noc)`** — resolves a specific device by `mesh_coord.to_linear_index(...)`, **no unit-mesh assertion**; header note: *"NOC distances may vary depending on the target device due to harvesting."*

So the **new fused op (Task 2) can compute per-device ring orders on a multi-device mesh** using overload 2 (with a representative or per-device `MeshCoordinate`), correctly accounting for each chip's harvesting. **No metalium fix and no modification to `regime_a` is required.** The Task-1 baseline did not need multi-device `regime_a`: `T_unfused = T_ag + T_mm` where `T_mm` is per-device single-chip `regime_a`, measured on a unit mesh in the same basis as `T_ag`.

## Gate
- Both parents correct (regime_a 111 passed + cache; all_gather gather-PCC 1.0 D4/D8). ✅
- Cached replay correct (regime_a cache tests). ✅
- Profiler usable (device-profiler corpus report). ✅
- ≥1 D=2/D=4 and one full-4x8 measurement, reproducible: D=4 and D=8 measured; within-launch 10-iter median spreads recorded in JSON (`spread_pct`). ✅
