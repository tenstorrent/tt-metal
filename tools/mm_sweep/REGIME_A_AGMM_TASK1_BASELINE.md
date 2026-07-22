# AGMM Task 1 — parent-op baselines (partial; blocked on multi-device regime_a)

Execution-plan reference: `tools/mm_sweep/REGIME_A_AGMM_EXECUTION_PLAN.md`, Task 1.

## Environment
- Branch/commit: `cglagovich/regime-a-bh` @ `de7df778f44` (merged origin/main `714f6714429` + integration fixes); submodules `umd 4ed96fb`, `tracy 1171005`.
- Machine: `bh-glx-120-b10u08` (Blackhole Galaxy, 32 chips), ubuntu-22.04-dev container; mesh descriptor `tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto`.
- Fabric: **mux v2** present (PR #48675, verified via `*FabricMuxV2Smoke*` gtests, 12/12). AG uses `FABRIC_1D_RING`, `max_packet_payload_size_bytes=4096` (4 KiB), `num_links=2`, `num_workers_per_link=6`, `num_buffers_per_channel=2`, `chunks_per_sync=16`, topology `Ring`.
- Harness: `models/tt_dit/tests/models/wan2_2/test_agmm_regime_a_baseline.py` (reuses the conftest `mesh_device` fixture). Raw JSON: `/data/cglagovich/agmm_baseline.json`.
- Measurement: trace capture + host wall, median of 10 timed iters (2 warmup dropped).

## Reference commands (Task 1 step 1) — status
| ref | result |
|---|---|
| single-chip `regime_a_matmul` (correctness + cache replay) | ✅ 111 passed |
| BH 4x8 fused AGMM `unit/fused/check` | ✅ 1 passed, PCC 0.9999875 |
| nightly `all_gather_minimal_matmul_addcmul` | ❌ 2 failed — **pre-existing** L1 CB overflow (~13 KB over) in `minimal_matmul` addcmul path; unrelated to regime_a; accepted by orchestrator |
| `*FabricMuxV2Smoke*` gtests | ✅ 12/12 |

## Standalone all-gather (T_ag) — measured, correct
`in0[M,K]` K-sharded across D devices on the cluster axis; `all_gather_async(dim=3)` → full `in0` on every device. Gather correctness vs torch = PCC 1.0.

| shape | D | cluster_axis | links | workers | gather PCC | **T_ag (µs)** |
|---|---|---|---|---|---|---|
| 32x6144x3072 | 4 | 0 (submesh 4×1) | 2 | 6 | 1.0000 | **70.2** |
| 32x6144x3072 | 8 | 1 (submesh 1×8) | 2 | 6 | 1.0000 | **63.9** |

## T_mm (single-chip regime_a, config=None Picker v3, GLX merged)
From `REGIME_A_GLX_MERGED_PERF_REPORT.md` / `regime_a_glx_merged_perf.json`:
- 32x6144x3072, cfg (1,3,1,4,6): **T_mm = 91.7 µs** (417.9 GB/s eff, device-profiler kernel span).

## Partial baseline table (flagship 32x6144x3072)
| D | T_mm (µs) | T_ag (µs) | max(T_mm,T_ag) = fused ideal | T_unfused = T_mm+T_ag | unfused/ideal |
|---|---|---|---|---|---|
| 4 | 91.7 | 70.2 | **91.7** | 161.9 | 1.77× |
| 8 | 91.7 | 63.9 | **91.7** | 155.6 | 1.70× |

Interpretation: for this Mt=1 shape, `T_ag` is a large fraction of `T_mm` (small-tensor AG is latency/overhead-bound, not the roofline `(M/N)(D-1)/D` bandwidth ratio), so a perfectly-overlapped fused op could approach ~1.7–1.8× over unfused.

## BLOCKER / key finding (decision requested)
The literal Task-1 unfused sequence `all_gather_async → regime_a_matmul` and a multi-device `T_mm` are **not measurable** as specified, because:

- **`regime_a_matmul` cannot build on a D>1 (non-unit) MeshDevice.** Its program factory's default **PARETO ring-order optimizer** calls `tt::tt_metal::experimental::Device::get_worker_noc_hop_distance(...)`, which asserts `mesh->num_devices() == 1` (`tt_metal/impl/device/experimental/device.cpp:19`). On a gathered multi-device tensor it hits `TT_FATAL: get_worker_noc_hop_distance() is only supported on unit MeshDevice.`
- The only ring order that skips that query is the internal **`DIAG_RING_BANK`** diagnostic (bank order [0..7]), which is **not exposed to Python** (no `regime_a_matmul_diag` binding on the public module), so it can't be selected from a test.

Consequences:
- `T_mm` above is measured single-chip (unit mesh) via device profiler; `T_ag` is measured multi-device via trace host-wall. **Different measurement bases** — `T_unfused`/max here are indicative, not a same-basis A/B.
- This limitation is **central to Task 2+**: the fused op must build its program per-device on a multi-device mesh, so it cannot use the unit-mesh-only PARETO query as-is. It needs a multi-device-safe ring-distance (e.g., query a per-device single-chip view, precompute a hop-distance table, or env-gate to bank order for early bring-up).

### Options for the orchestrator
1. **Expose bank-order / env-gate the ring optimizer** to skip `get_worker_noc_hop_distance` on non-unit meshes (small change): enables a true traced multi-device `T_mm` and unfused `T_ag→T_mm` baseline (bank-order ring; note it's not PARETO production).
2. **Accept `T_unfused = T_ag + T_mm(unit-mesh)`** analytically for Task 1 and treat multi-device regime_a as the Task-2 design driver.
3. **Fix `get_worker_noc_hop_distance` for multi-device meshes** at the metalium level (larger, benefits the production PARETO path multi-device).
