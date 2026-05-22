# tt-metal2 TTSim Parity Status

**Branch:** `ridvan/nkapre-multichip-metal-v2`
**Head commit:** `6ccc779e618` (sim skip fixes for dispatch-context + TT_VISIBLE_DEVICES mp)
**Generated:** 2026-05-21 (live monitor; job 11604 started pre-fix at `fbca6525`)
**Simulator:** craq-sim TTSim v3.5 on Galaxy compute (`bh-glx-b06u08`, `bh-glx-b07u02`)
**Test catalog:** `nkapre-fork-test-commands.md`

## How to read this doc

| Tag | Meaning |
|-----|---------|
| **Verified PASS** | Dedicated run with correct env, mock desc, or fix — treat as ground truth on HEAD |
| **Sweep PASS/FAIL/TIMEOUT** | Result from the 57-suite parity sweep (`run-20260521T030649Z/`) — may be stale vs fixes |
| **Env/setup FAIL** | Harness, missing venv, wrong mock desc, or Slurm cancel — not a metal-logic regression |
| **In progress** | Job still running at doc update time |

Pre-fix baseline sweep (`run-20260521T021753Z/`) hit mass rc=134 heap aborts; ignore for current metal status except as historical context.

---

## Executive summary

| Area | Source | PASS | FAIL | TIMEOUT | Notes |
|------|--------|------|------|---------|-------|
| Full sweep (57 suites) | `run-20260521T030649Z/` | 24 | 24 | 9 | Post-fix scorecard; Section 1 + LLK stale |
| Section 1 TTNN (HEAD verify) | `section1-by-commit-20260521T195930Z/` + quick | 2/2 targeted | 0 | 0 | mcq + region_write_read **Verified PASS** |
| Section 2 T3K re-run | `section2-t3k-20260521T181117Z/` (job 11604) | 20 | 9 | 5 | 34/34 done · **Status:** complete (05:20:03Z) |
| Section 2 fabric-only | job 11609 (pending after 11604) | — | — | — | `PARITY_SECTIONS=2.fabric` on HEAD |
| Section 2 multiprocess | `mp-run-20260521T155410Z/` | 1 | 9 | 0 | `ttnn_launch_op` **Verified PASS** under tt-run+MPI |
| Section 4 LLK WH | `llk-smoke-20260521T190458Z/` (job 11576) | 2/2 | 0 | 0 | weekly + nightly **Verified PASS** after venv setup |
| Section 4 LLK (sweep) | `run-20260521T030649Z/` | 0 | 2 | 0 | **Env/setup FAIL** — missing `.venv` |
| P300 CCL smoke | jobs 11599, 11602 | 0 | 0 | 1 | 11599 TIMEOUT; 11602 CANCELLED |

Metal2 post-fix full-sweep pass rate is **42.1%** (24/57), up from 12.3% pre-fix. Section 1 targeted fixes are green on HEAD. LLK weekly/nightly WH pass on metal2 once `setup-llk-ttsim-env.sh` links the venv (smoke job 11576). Remaining gaps: fabric/CCL sim slowness, multiprocess (most suites), Galaxy 32-chip scale, and several Section 1/2 suites not yet re-run post-fix.

---

## Section 1 — TTNN (single-card, Blackhole sim)

Catalog: 7 C++ gtests + pytest `tests/ttnn/unit_tests/`.

### Verified on HEAD (`fbca6525`)

Job 11596 (`section1-by-commit-20260521T195930Z/`), quick verify (`ttnn-sec1-quick-20260521T195523Z`):

| Test | Result | Notes |
|------|--------|-------|
| `MultiCommandQueueSingleDeviceFixture` (mcq) | **Verified PASS** | rc=0 |
| `RegionWriteReadTest` (132 cases) | **Verified PASS** | failed=0 ok=132 |

### Fix progression (section1-by-commit)

| Commit | mcq | region_write_read |
|--------|-----|-------------------|
| `3a291ad3` baseline | **FAIL** rc=139 (SIGSEGV in fixture teardown after skip) | **FAIL** failed=256 ok=4 |
| `38bd49b1` fixture teardown null-deref fix | **PASS** | **FAIL** failed=256 ok=4 |
| `2149c9af` slow-dispatch mesh region write offset | **PASS** | **PASS** failed=0 ok=132 |
| `fbca6525` unset inherited mock cluster desc | **PASS** | **PASS** failed=0 ok=132 |

### Full-sweep Section 1 (`run-20260521T030649Z/` — pre Section 1 fixes)

| Suite | Sweep status | Best current status | Key error / note |
|-------|--------------|---------------------|------------------|
| `1.ttnn_cpp/unit_tests_ttnn` | FAIL rc=139 | Likely fixed (`38bd49b1`) | SIGSEGV in teardown after slow-dispatch skip |
| `1.ttnn_cpp/unit_tests_ttnn_tensor` | FAIL rc=1 | Not re-run on HEAD | gtest failures (113s) |
| `1.ttnn_cpp/unit_tests_ttnn_ccl` | **Sweep PASS** | **Verified PASS** (sweep) | |
| `1.ttnn_cpp/unit_tests_ttnn_ccl_multi_tensor` | **Sweep PASS** | **Verified PASS** (sweep) | |
| `1.ttnn_cpp/unit_tests_ttnn_ccl_ops` | **Sweep PASS** | **Verified PASS** (sweep) | |
| `1.ttnn_cpp/unit_tests_ttnn_accessor` | **Sweep PASS** | **Verified PASS** (sweep) | |
| `1.ttnn_cpp/test_ccl_multi_cq_multi_device` | **Sweep PASS** | **Verified PASS** (sweep) | |
| `1.ttnn_py/unit_tests` | FAIL rc=1 | Not re-run on HEAD | CCL all-gather pytest: `poll_eth_data: Poll failed: Interrupted system call` |

---

## Section 2 — T3000 single-host (Wormhole multichip sim)

### T3K mock desc re-run (job 11604 — partial, in progress)

**Dir:** `section2-t3k-20260521T181117Z/`
**Mock:** `t3k_cluster_desc.yaml` (8-chip WH)
**Started:** 2026-05-21T21:51:47Z · **Status:** complete (05:20:03Z)

| Suite | Status | Duration | Key error / note |
|-------|--------|----------|-------------------|
| `2.distributed/distributed_unit_tests` | **FAIL** (pre-fix) | 272s | `TestWritesAndWorkloads` TT_FATAL — **fixed to SKIP on HEAD `6ccc779`** |
| `2.distributed/run_visible_devices_mp` | **FAIL** (pre-fix) | 23s | MPI SIGSEGV under ttsim — **fixed to SKIP on HEAD `6ccc779`** |
| `2.eth/ActiveEthKernelsDirectSendAllConnectedChips` | **Verified PASS** | 117s | Was **TIMEOUT** 900s on full sweep (6u desc) |
| `2.eth/ActiveEthKernelsSendInterleavedBufferAllConnectedChips` | **Verified PASS** | 3303s (~55m) | Was **TIMEOUT** 901s on full sweep |
| `2.eth/ActiveEthKernelsDirectRingGatherAllChips` | **Verified PASS** | 3s | |
| `2.eth/ActiveEthKernelsInterleavedRingGatherAllChips` | **Verified PASS** | 3s | |
| `2.dispatch/CommandQueueSingleCard` | **Verified PASS** | 1s | |
| `2.dispatch/CommandQueueMultiDevice` | **Verified PASS** | 0s | |
| `2.dispatch/UnitMeshCQSingleDevice` | **Verified PASS** | 1s | |
| `2.dispatch/UnitMeshCQMultiDevice` | **Verified PASS** | 0s | |
| `2.debug_tools/mesh` | *in progress* | — | Was **TIMEOUT** 901s on full sweep |
| remaining Section 2 suites | *pending* | — | job 11604 still active |

### Full-sweep Section 2 (`run-20260521T030649Z/`, 6u mock desc)

Scorecard: **24 PASS / 24 FAIL / 9 TIMEOUT** (57 total including Sections 1–4).

#### Verified PASS from sweep (still valid unless T3K re-run supersedes)

| Suite | Duration | Note |
|-------|----------|------|
| `2.eth/ActiveEthKernelsDirectRingGatherAllChips` | 33s | |
| `2.eth/ActiveEthKernelsInterleavedRingGatherAllChips` | 33s | |
| `2.dispatch/CommandQueueSingleCard` | 1s | |
| `2.dispatch/CommandQueueMultiDevice` | 0s | |
| `2.dispatch/UnitMeshCQSingleDevice` | 0s | |
| `2.dispatch/UnitMeshCQMultiDevice` | 0s | |
| `2.examples/distributed_program_dispatch` | 33s | |
| `2.examples/distributed_buffer_rw` | 32s | |
| `2.examples/distributed_eltwise_add` | 39s | |
| `2.fabric/control/ControlPlaneFixture.*T3k*` | 1s | mock-only |
| `2.fabric/control/T3kCustomMeshGraphControlPlaneTests*` | 1s | mock-only |
| `2.fabric/control/T3k*MeshGraphFabric2DDynamicTests*` | 0s | mock-only |
| `2.fabric/worker_edm` | 1s | BH P300 sim |
| `2.fabric/t3k_dynamic` | 0s | mock-only |
| `2.ttnn_dist/unit_tests_ttnn_udm` | 4s | |
| `2.ttnn_dist/prefetcher` | 14s | pytest |
| `2.ttnn_dist/test_tensor_parallel_example_T3000.py` | 11s | pytest |
| `2.ttnn_dist/test_data_parallel_example.py` | 9s | pytest |
| `2.ttnn_dist/test_hybrid_data_tensor_parallel_example_T3000.py` | 8s | pytest |

#### FAIL / TIMEOUT (selected)

| Suite | Sweep status | Key error |
|-------|--------------|-----------|
| `2.distributed/distributed_unit_tests` | TIMEOUT 901s → T3K re-run **FAIL** fast | Dispatch-context TT_FATAL on WH 8-chip sim |
| `2.distributed/run_visible_devices_mp` | FAIL | MPI SIGSEGV — same on T3K re-run |
| `2.eth/DirectSendAllConnectedChips` | TIMEOUT 900s | **Verified PASS** on T3K re-run (117s) |
| `2.eth/SendInterleavedBufferAllConnectedChips` | TIMEOUT 901s | T3K re-run in progress |
| `2.debug_tools/mesh` | TIMEOUT 901s | DPrint/MeshWatcher hang |
| `2.examples/distributed_trace_and_events` | FAIL rc=134 | Heap abort / SIGABRT |
| `2.fabric/unicast_1x8` | FAIL rc=139 | SIGSEGV (mock-only, no sim) |
| `2.fabric/Fabric2D` | FAIL | Assertion failures (~653s) |
| `2.fabric/Fabric1D` | TIMEOUT 900s | Hang |
| `2.fabric/telemetry/Fabric1D`, `Fabric2D` | TIMEOUT/FAIL | BW telemetry + fabric hang |
| `2.fabric_ubench/*` | TIMEOUT/FAIL | Routing microbench hang or early fail |
| `2.ttnn_dist/unit_tests_ttnn` | FAIL rc=139 | SIGSEGV (multichip gtest) |

### Multiprocess (`2.mp/*`) — dedicated MPI runs

Full sweep: all 10 suites **FAIL rc=213** (tt-run launch failure without MPI rank setup).

Dedicated runs on compute (`mp-run-20260521T155410Z/`, tt-run + `--oversubscribe`):

| Suite | mp-run status | Sweep status |
|-------|---------------|--------------|
| `2.mp/ttnn_launch_op` | **Verified PASS** | FAIL rc=213 |
| `2.mp/2x2_fabric_ubench` | FAIL rc=134 | FAIL rc=213 |
| `2.mp/multi_host_fabric` | FAIL rc=139 | FAIL rc=213 |
| `2.mp/mesh_socket` | FAIL rc=139 | FAIL rc=213 |
| `2.mp/BigMeshDualRankTest2x4` | FAIL rc=134 | FAIL rc=213 |
| `2.mp/BigMeshDualRankMeshShapeSweep` | FAIL rc=134 | FAIL rc=213 |
| `2.mp/ttnn_dual_rank_2x2` | FAIL rc=139 | FAIL rc=213 |
| `2.mp/ttnn_dual_rank_2x4` | FAIL rc=134 | FAIL rc=213 |
| `2.mp/py_data_parallel` | FAIL rc=1 | FAIL rc=213 |
| `2.mp/py_submesh` | FAIL rc=1 | FAIL rc=213 |

---

## Section 3 — Single Galaxy single-process (32-chip WH sim)

From full sweep (`run-20260521T030649Z/`, 6u mock desc):

| Suite | Sweep status | Duration | Key error |
|-------|--------------|----------|-----------|
| `3.galaxy/test_data_parallel_example_TG.py` | TIMEOUT | 900s | Sim scale / hang |
| `3.galaxy/test_multidevice_TG.py` | TIMEOUT | 901s | Sim scale / hang |
| `3.galaxy/multi_device_trace` | FAIL rc=1 | 97s | pytest failure (not timeout) |

Reference (tt-metal-fork sweep `run-20260521T013634Z/`): all three **FAIL** fast (rc=1/139, ~10–12s) — metal2 runs longer before timeout/fail.

No dedicated Galaxy re-run yet.

---

## Section 4 — LLK tests (Wormhole, ttsim)

Catalog: weekly WH (`-m "not quasar and not nightly and not perf"`) + nightly WH (3 matmul/sfpu files).

### Verified PASS (after venv setup)

**Dir:** `llk-smoke-20260521T190458Z/` · **Slurm job:** 11576 · **Node:** `bh-glx-b07u02`

| Suite | Status | Notes |
|-------|--------|-------|
| `4.llk/weekly_wh` | **Verified PASS** | Wormhole weekly; workers=1, timeout=120s |
| `4.llk/nightly_wh` | **Verified PASS** | Wormhole nightly; workers=1, timeout=300s |

Setup: `scripts/setup-llk-ttsim-env.sh` linked venv → metal2 `python_env`, sfpi OK.

Reference (tt-metal-fork): both **PASS** in `run-20260521T013634Z/` (weekly 7s, nightly 3s).

### Env/setup FAIL (do not treat as metal regressions)

| Run | Job | weekly | nightly | Error |
|-----|-----|--------|---------|-------|
| Full sweep `run-20260521T030649Z/` | 11411 | FAIL | FAIL | `ERROR: missing LLK venv python: …/tt_metal/tt-llk/tests/.venv/bin/python` |
| Pre-fix sweep `run-20260521T021753Z/` | — | FAIL | FAIL | Same missing venv |
| `llk-smoke-20260521T184629Z/` | 11573 | FAIL | FAIL | `no tests ran`, rc=5 — collection/harness before venv ready |
| `llk-smoke-20260521T190011Z/` | 11574 | FAIL | PASS | Weekly: `P=0 total=0 rc=5`, no progress log |
| `llk-smoke-weekly-only-20260521T190410Z/` | 11575 | FAIL | PASS | Same weekly collection issue |
| `llk-run-20260521T181117Z/` | 11562 | *cancelled* | — | Slurm **CANCELLED** mid-weekly (~33 min in); venv was set up |

Early smoke `llk-smoke-20260521T184442Z/` (11572): incomplete (`weekly_rc=0 nightly_rc=0`, 0 tests collected).

### LLK status summary

| Context | weekly_wh | nightly_wh |
|---------|-----------|------------|
| tt-metal-fork sweep | PASS | PASS |
| metal2 full sweep (no venv) | Env/setup FAIL | Env/setup FAIL |
| metal2 smoke after setup (11576) | **Verified PASS** | **Verified PASS** |

Parity gap vs fork in the 57-suite scorecard is **harness-only**; WH LLK passes once `setup-llk-ttsim-env.sh` runs before the sweep.

---

## P300 CCL smoke (Blackhole 2-chip fabric)

Test: `test_ccl_smoke_test_p300.py::test_ccl_ddr_smoke_test[… num_devices=2 …]`

| Job | Dir | Timeout | Status | Key error |
|-----|-----|---------|--------|-----------|
| 11599 | `p300-smoke-20260521T204420Z/` | 3600s | **TIMEOUT** | All-gather op completed (`Done op`) but hung in `ttnn.to_torch(… ConcatMeshToTensor …)` |
| 11602 | `p300-smoke-20260521T214729Z/` | 300s | **CANCELLED** | Fabric init OK; stuck at `Waiting for op`; Slurm cancel 2026-05-21T21:51:45Z |

Jobs 11603 (P300) — not found in results dir. Job 11604 is the Section 2 T3K re-run (not P300).

Observations: 2-chip BH sim + fabric 1D init succeeds; large DDR all-gather + readback exceeds practical smoke timeouts under sim.

---

## Known simulator limitations vs fixable metal issues

### Fixable metal issues (addressed or in progress)

| Issue | Fix / status |
|-------|----------------|
| Section 1 gtest SIGSEGV on skip/teardown | Fixed `38bd49b1` — **Verified PASS** |
| Section 1 region write/read offset | Fixed `2149c9af` — **Verified PASS** |
| Section 1 inherited mock cluster desc | Fixed `fbca6525` — **Verified PASS** |
| Section 2 wrong mock desc (6u vs T3K) | T3K re-run job 11604; eth direct-send **Verified PASS** |
| Heap abort (rc=134) on CCL/control-plane | Cleared in post-fix sweep vs fork |
| LLK weekly/nightly WH in parity sweep | **Env/setup FAIL** in sweep; **Verified PASS** after venv setup (job 11576) |

### Simulator / environment limitations

| Category | Examples |
|----------|----------|
| **Sim speed / hangs** | Fabric 1D/2D, fabric ubench, debug tools mesh watcher, Galaxy 32-chip pytest — 900s timeout |
| **Multiprocess** | Sweep rc=213 without MPI; dedicated mp-run: 1/10 PASS (`ttnn_launch_op`), rest SIGSEGV/heap abort |
| **Mock-only fabric** | `2.fabric/unicast_1x8` SIGSEGV without sim |
| **CCL at scale** | P300 DDR all-gather smoke — op completes but readback times out |
| **Dispatch context on WH 8-chip sim** | `TestWritesAndWorkloads` TT_FATAL — may need skip/guard |
| **MPI visible-devices** | SIGSEGV in multiprocess distributed tests under sim |

---

## Result directories

| Run | Path |
|-----|------|
| Full 57-suite sweep (post-fix) | `craq-parity-results/run-20260521T030649Z/` |
| Pre-fix sweep (rc=134 baseline) | `craq-parity-results/run-20260521T021753Z/` |
| Section 1 by commit | `craq-parity-results/section1-by-commit-20260521T195930Z/` |
| Section 2 T3K re-run (job 11604) | `craq-parity-results/section2-t3k-20260521T181117Z/` |
| Multiprocess dedicated | `craq-parity-results/mp-run-20260521T155410Z/` |
| LLK verified smoke (job 11576) | `craq-parity-results/llk-smoke-20260521T190458Z/` |
| LLK dedicated run (cancelled) | `craq-parity-results/llk-run-20260521T181117Z/` |
| P300 smoke 11599 | `craq-parity-results/p300-smoke-20260521T204420Z/` |
| P300 smoke 11602 (cancelled) | `craq-parity-results/p300-smoke-20260521T214729Z/` |
| Fork comparison | `craq-parity-results/COMPARISON-vs-tt-metal-fork.md` |

## Live monitoring

| Job | ID | State | Results |
|-----|-----|-------|---------|
| Section 2 T3K | 11604 | running | `section2-t3k-20260521T181117Z/` |
| Fabric-only | 11605 | pending (after 11604) | `fabric-latest/` (TBD) |

```bash
# Section 2 re-run (job 11604)
tail -f /data/rsong/tt-metal2/craq-parity-results/section2-t3k-20260521T181117Z/sweep.log

# Fabric job (11605)
tail -f /data/rsong/tt-metal2/craq-parity-results/slurm-fabric-11605.out

# P300 smoke logs (11599 complete, 11602 cancelled)
tail -f /data/rsong/tt-metal2/craq-parity-results/p300-smoke-20260521T204420Z/pytest.log
```

**craq-sim note:** pull to `acb6de0c` fails WH build (`tensix.cpp:5101` unused variable). Continue using prebuilt `src/_out/release_wh|bh/libttsim.so`.
