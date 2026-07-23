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
    devices (ring forward hops 1..D-1), per-shard readiness via `shard_landed[e]` GlobalSemaphores.
  - **Streaming (default):** monotonic `gather_ready` advanced in global-K order; the copied regime_a in0
    reader (`AGMM_STREAM`) gates each in0 read on `gather_ready > kg/Kt_local` → matmul overlaps the gather.
  - **Full-gather diagnostic:** same binary, `TT_AGMM_FULL_GATHER=1` → reader waits `gather_ready==D` (no
    overlap A/B).
  - Cache-safe: all coordination via D+1 GlobalSemaphores reset per launch (fresh+cached both correct).
  - regime_a single-chip kernels UNTOUCHED (copied the in0 reader; reused in1_reader/compute by path).
  - **Tests (19 passed):** offline plan (D=1/2/4/8); D=1 == regime_a; D=2 (linear) / D=4,D=8 (ring)
    full-gather streaming, fresh+cached; D=4 config coverage (auto-picker, Ns=2, Sm=2). PCC ≥ 0.999.
  - "More chunks than slots" is inherent: every in0 tile is a packet through a 2-slot mux channel.
- **Task 4 — PARTIAL** (`REGIME_A_AGMM_TASK4_REPORT.md`): host-wall A/B (overhead-bound, can't resolve
  overlap); per-RISC profiler + watcher **blocked in this container** (see below). Harnesses committed.

## Known scope limits / deferred

- **Linear topology is D=2 only**; D>2 requires ring (one forward mux, hops 1..D-1). Bidirectional-linear for
  D>2 is a follow-up (2 muxes).
- **No epilogues / tails yet** (bf16, tile-aligned K divisible by D, no bias/activation/addcmul/chunks) — Task 9.
- Gathered-A is always materialized to DRAM (Phase A). Direct-L1 (Phase B) = Tasks 5-6.

## Environment blockers (why Tasks 4-full, 5-8, 10 can't finish here)

Per-RISC overlap evidence and the direct-L1 ceiling estimate that the plan gates Tasks 5-8/10 on **cannot be
captured in this container** — see `agmm-profiler-watcher-blocked` memory:
- Watcher overflows the ACTIVE_ETH kernel-config buffer under fabric.
- `ttnn.ReadDeviceProfiler(mesh)` writes no `profile_log_device.csv` for a MeshDevice; tracy targets an
  unwritable `build/profiler/build_wasm`.
- Host wall is overhead-bound at corpus shapes (fused ~1250 µs vs ~88 µs single-chip compute).

**Resume on a profiler/watcher-capable bare-metal galaxy or CI:** run `tools/mm_sweep/agmm_profile.py`
(per-RISC overlap already coded) and the watcher, then proceed to Task 5 (direct-L1) with the measured ceiling.

## Build/run notes (critical)

- After `ninja -C build_Release ttnn`, **redeploy** or nothing changes at runtime:
  `cp -p build_Release/ttnn/_ttnncpp.so build_Release/lib/_ttnncpp.so && cp -p build_Release/ttnn/_ttnn.so ttnn/ttnn/_ttnn.so`
  (see `agmm-build-deploy-gotcha` memory). Header-only edits: `touch` the op `.cpp` to force the Unity TU.
- Tests: open the FULL mesh with a bare fabric config (no conftest fixture; no watcher) and carve a (1,D)
  submesh (see `agmm-multidevice-fabric-bringup` memory).
- Run: `python -m pytest tests/ttnn/unit_tests/operations/matmul/test_all_gather_regime_a_matmul_async.py -q`
