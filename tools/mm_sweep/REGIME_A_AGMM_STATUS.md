# Regime-A fused all-gather matmul — overall status & handoff

Op: `ttnn.experimental.all_gather_regime_a_matmul_async`. Branch: `cglagovich/regime-a-bh` (worktree
`resilient-marinating-piglet`). This tracks the plan in `REGIME_A_AGMM_EXECUTION_PLAN.md`.

## Done and verified on hardware (single BH galaxy, container `trusting_borg`)

- **Task 1** — parent-op baselines (`REGIME_A_AGMM_TASK1_BASELINE.md`).
- **Task 2** — scaffold + pure host plan; D=1 delegates to `regime_a_matmul` (`test_...` offline plan tests).
- **Task 3 — DONE.** Phase-A DRAM-staged fused op:
  - MeshWorkload / `create_at`-per-coordinate device op; D inferred from the K-shard ratio.
  - Per-device DRAM gather buffer (output slot 1, mesh tensor → uniform address).
  - Fabric **mux v2** injector: reads local in0 shard, writes local gather slice, unicasts to all D-1 other
    devices (ring forward hops 1..D-1; naive all-to-all Phase-A reference, no receiver forwarding).
  - **Streaming (default), LOCAL-FIRST per-shard:** 2*D GlobalSemaphores (D `shard_ready` + D `shard_landed`).
    The injector fans out its OWN shard's readiness first, then each remote shard as it lands (arrival order).
    The copied regime_a in0 reader (`AGMM_STREAM`) gates each read on `shard_ready[kg/Kt_local]`, so device d's
    local Pk band computes without waiting for remote shard 0. `TT_AGMM_FULL_GATHER=1` (hashed attribute) is the
    same-binary no-overlap diagnostic (reader waits all D shards).
  - Production placement restored: IN1_NEAR M-slave placement + PARETO in0-ring ordering via the multi-device
    hop-distance overload, so the fused compute engine matches single-chip placement.
  - Cache-safe: program hash covers mem-configs/dtype/compute-config/transport/diagnostic; the override
    relocates all GlobalSemaphore addresses on replay (verified with a fresh-sems 2nd iteration).
  - Validated: bf16/INTERLEAVED bf16 output, D in [2,8], ring for D>2, num_links==workers==1, >= 2*D sems;
    persistent_output_buffer/barrier_semaphore/compute-config-override rejected or documented.
  - regime_a single-chip kernels UNTOUCHED (copied the in0 reader; reused in1_reader/compute by path).
  - **Tests (19 passed):** offline plan (D=1/2/4/8); D=1 == regime_a; D=2 (linear) / D=4,D=8 (ring) streaming,
    fresh program + cached replay with a FRESH sem set; D=4 config coverage (auto-picker, Ns=2, Sm=2). PCC ≥ 0.999.
- **Task 4 — DONE** (`REGIME_A_AGMM_TASK4_REPORT.md`): per-RISC device profiler + watcher both usable via the
  documented workarounds; overlap A/B + DRAM-staging characterization measured. (Re-measured after the F3/F5
  fixes — see the report.)

## Known scope limits / deferred

- **Linear topology is D=2 only**; D>2 requires ring (one forward mux, hops 1..D-1). Bidirectional-linear for
  D>2 is a follow-up (2 muxes).
- **No epilogues / tails yet** (bf16, tile-aligned K divisible by D, no bias/activation/addcmul/chunks) — Task 9.
- Gathered-A is always materialized to DRAM (Phase A). Direct-L1 (Phase B) = Tasks 5-6.

## Profiler/watcher: usable with workarounds (Task 4 done); Tasks 5-8/10 remain

Both mechanisms run on this container with the setup in the `agmm-profiler-watcher-blocked` memory:
- **Watcher:** `TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1` → D=4 test passes watcher-clean (ETH cores
  run the fabric router, not our kernels; Tensix kernels fully watched). Without DISABLE_ETH the fabric-router
  program overflows the ACTIVE_ETH kernel-config buffer.
- **Profiler:** `TT_METAL_DEVICE_PROFILER=1` + pre-created tracy dirs + `TT_METAL_HOME`=worktree + a CLEAN
  submesh teardown (reset stall group / clear+remove sub-device manager / close submesh then parent) →
  `ReadMeshDeviceProfilerResults` flushes `profile_log_device.csv`. `tools/mm_sweep/agmm_profile.py` parses it.
- **Host wall is overhead-bound** (~1250 µs vs ~88 µs single-chip compute) — use profiler total-device-span.

Task 4 measured (see `REGIME_A_AGMM_TASK4_REPORT.md`): at Mt=1 corpus shapes compute span > gather span so the
gather is mostly hidden in both modes; streaming wins ~2% on narrow-N (32x15360x768). Fused compute span
(~143 µs) >> single-chip (~88 µs) → DRAM staging adds a 2nd in0 read → Phase-B direct-L1 ceiling ≈ ~55 µs.

**Next (Tasks 5-8/10):** implement Phase-B direct remote-L1 streaming (Task 5), A/B it against staged DRAM
(Task 6) using `agmm_profile.py`, then placement sweeps (7-8) and the fabric-aware picker (10).

## Build/run notes (critical)

- After `ninja -C build_Release ttnn`, **redeploy** or nothing changes at runtime:
  `cp -p build_Release/ttnn/_ttnncpp.so build_Release/lib/_ttnncpp.so && cp -p build_Release/ttnn/_ttnn.so ttnn/ttnn/_ttnn.so`
  (see `agmm-build-deploy-gotcha` memory). Header-only edits: `touch` the op `.cpp` to force the Unity TU.
- Tests: open the FULL mesh with a bare fabric config (no conftest fixture; no watcher) and carve a (1,D)
  submesh (see `agmm-multidevice-fabric-bringup` memory).
- Run: `python -m pytest tests/ttnn/unit_tests/operations/matmul/test_all_gather_regime_a_matmul_async.py -q`
