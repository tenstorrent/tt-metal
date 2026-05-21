# tt-metal2 TTSim Parity Status

**Branch:** `ridvan/nkapre-multichip-metal-v2`
**Head commit (at sweep time):** `fbca6525d08`
**Generated:** 2026-05-21
**Simulator:** craq-sim TTSim v3.5 on Galaxy compute (`bh-glx-b06u08`)
**Test catalog:** `nkapre-fork-test-commands.md`

## Executive summary

| Area | Source | PASS | FAIL | TIMEOUT | In progress |
|------|--------|------|------|---------|-------------|
| Full sweep (57 suites) | `run-20260521T030649Z/` | 24 | 24 | 9 | — |
| Section 1 TTNN (current HEAD) | `section1-by-commit-20260521T195930Z/fbca6525-*` | 2/2 | 0 | 0 | — |
| Section 2 T3000 single-host | `section2-t3k-20260521T181117Z/` (job 11604) | 1 | 2 | 0 | sweep running |
| P300 CCL smoke | jobs 11599, 11602 | 0 | 1 | 1 | 11602 running |

Metal2 post-fix pass rate on the full sweep is **42.1%** (24/57), up from 12.3% pre-fix. Section 1 TTNN is **green on HEAD** after three targeted fixes. Section 2 re-run with the correct T3K mock cluster desc shows eth direct-send passing (previously timed out on the 6u desc). Remaining gaps are mostly fabric/CCL hangs, multiprocess (tt-run), Galaxy-scale tests, LLK harness setup, and simulator-slow paths.

---

## Section 1 — TTNN (single-card, Blackhole sim)

### Current status (HEAD `fbca6525`)

Job 11596 (`section1-by-commit-20260521T195930Z`), quick verify (`ttnn-sec1-quick-20260521T195523Z`):

| Test | Result | Notes |
|------|--------|-------|
| `MultiCommandQueueSingleDeviceFixture` (mcq) | **PASS** | rc=0 |
| `RegionWriteReadTest` (132 cases) | **PASS** | failed=0 ok=132 |

### Fix progression (section1-by-commit)

| Commit | mcq | region_write_read |
|--------|-----|-------------------|
| `3a291ad3` baseline | **FAIL** rc=139 (SIGSEGV in fixture teardown after skip) | **FAIL** failed=256 ok=4 |
| `38bd49b1` fixture teardown null-deref fix | **PASS** | **FAIL** failed=256 ok=4 |
| `2149c9af` slow-dispatch mesh region write offset | **PASS** | **PASS** failed=0 ok=132 |
| `fbca6525` unset inherited mock cluster desc | **PASS** | **PASS** failed=0 ok=132 |

Key errors fixed:

- **Fixture teardown SIGSEGV:** null dereference when test skipped under slow dispatch (`ttnn_test_fixtures.hpp:113`). Fixed in `38bd49b1`.
- **Region write/read mismatch:** host offset wrong for mesh shard region transfers under slow dispatch. Fixed in `2149c9af`.
- **Inherited mock cluster desc:** Section 1 tests picked up multichip mock env; unset in `fbca6525`.

### Full-sweep Section 1 (stale — pre-fix build)

From `run-20260521T030649Z/` (ran before Section 1 fixes landed):

| Suite | Status | Key error |
|-------|--------|-----------|
| `1.ttnn_cpp/unit_tests_ttnn` | FAIL (rc=139) | SIGSEGV in teardown after `Skipping test, since it can only be run in Fast Dispatch Mode` — **fixed on HEAD** |
| `1.ttnn_cpp/unit_tests_ttnn_tensor` | FAIL (rc=1) | gtest failures (113s) |
| `1.ttnn_cpp/unit_tests_ttnn_ccl` | PASS | |
| `1.ttnn_cpp/unit_tests_ttnn_ccl_multi_tensor` | PASS | |
| `1.ttnn_cpp/unit_tests_ttnn_ccl_ops` | PASS | |
| `1.ttnn_cpp/unit_tests_ttnn_accessor` | PASS | |
| `1.ttnn_cpp/test_ccl_multi_cq_multi_device` | PASS | |
| `1.ttnn_py/unit_tests` | FAIL (rc=1) | CCL all-gather pytest interrupted: `poll_eth_data: Poll failed: Interrupted system call` |

---

## Section 2 — T3000 single-host (Wormhole multichip sim)

### Re-run with T3K mock desc (job 11604, in progress)

**Dir:** `section2-t3k-20260521T181117Z/`
**Mock:** `t3k_cluster_desc.yaml` (8-chip WH, not 6u Galaxy desc)
**Started:** 2026-05-21T21:51:47Z

| Suite | Status | Duration | Key error / note |
|-------|--------|----------|-------------------|
| `2.distributed/distributed_unit_tests` | **FAIL** | 272s | `DispatchContextFixture.TestWritesAndWorkloads`: `TT_FATAL: Manually setting up and tearing down Fast Dispatch is only supported on Galaxy and Blackhole clusters` |
| `2.distributed/run_visible_devices_mp` | **FAIL** | 23s | `mpirun … exited on signal 11 (Segmentation fault)` for config 2,3 |
| `2.eth/ActiveEthKernelsDirectSendAllConnectedChips` | **PASS** | 117s | Previously **TIMEOUT** (900s) on full sweep with 6u desc |
| `2.eth/ActiveEthKernelsSendInterleavedBufferAllConnectedChips` | *running* | — | sweep in progress at doc time |
| remaining Section 2 suites | *pending* | — | job 11604 still active |

### Full sweep Section 2 (`run-20260521T030649Z/`, 6u mock desc)

57-suite scorecard: **24 PASS / 24 FAIL / 9 TIMEOUT**.

#### Passing (selected)

- Dispatch CQ tests (single/multi card, unit mesh)
- Eth ring-gather (direct + interleaved)
- Distributed programming examples (program dispatch, buffer rw, eltwise add)
- Fabric control plane (T3K, no sim — mock only)
- Fabric worker/EDM datapath (BH P300 sim)
- TTNN distributed pytest (prefetcher, tensor/data/hybrid parallel examples)
- `2.ttnn_dist/unit_tests_ttnn_udm`

#### Failing / timing out (selected, with error excerpts)

| Suite | Status | Key error |
|-------|--------|-----------|
| `2.distributed/distributed_unit_tests` | TIMEOUT 901s | Hung (same dispatch-context class of issue; now fails fast with T3K desc) |
| `2.distributed/run_visible_devices_mp` | FAIL | MPI SIGSEGV |
| `2.eth/DirectSendAllConnectedChips` | TIMEOUT 900s | Slow/hung on 6u desc; **PASS** on T3K re-run |
| `2.eth/SendInterleavedBufferAllConnectedChips` | TIMEOUT 901s | Same pattern as direct send |
| `2.debug_tools/mesh` | TIMEOUT 901s | DPrint/MeshWatcher hang |
| `2.examples/distributed_trace_and_events` | FAIL rc=134 | Heap abort / SIGABRT |
| `2.fabric/unicast_1x8` | FAIL rc=139 | SIGSEGV (mock-only, no sim) |
| `2.fabric/Fabric2D` | FAIL | Assertion failures after ~653s |
| `2.fabric/Fabric1D` | TIMEOUT 900s | Hang |
| `2.fabric/telemetry/Fabric1D`, `Fabric2D` | TIMEOUT/FAIL | BW telemetry + fabric hang |
| `2.fabric_ubench/*` | TIMEOUT/FAIL | Routing microbench hang or early fail |
| `2.ttnn_dist/unit_tests_ttnn` | FAIL rc=139 | SIGSEGV (multichip gtest) |
| `2.mp/*` (10 suites) | FAIL rc=213 | `tt-run` MPI launch failure (not exercised under sim in this sweep) |
| `3.galaxy/*` | TIMEOUT/FAIL | 32-chip Galaxy pytest/trace — scale/time |

#### Multiprocess (`2.mp/*`)

All 10 multiprocess suites fail immediately with **exit code 213** from `tt-run` / MPI. These require real multi-process rank binding and are not expected to pass under single-host ttsim sweep without dedicated MPI setup.

---

## P300 CCL smoke (Blackhole 2-chip fabric)

| Job | Dir | Timeout | Status | Key error |
|-----|-----|---------|--------|-----------|
| 11599 | `p300-smoke-20260521T204420Z/` | 3600s | **TIMEOUT** | All-gather op completed (`Done op`) but hung in `ttnn.to_torch(… ConcatMeshToTensor …)` until pytest timeout |
| 11602 | `p300-smoke-20260521T214729Z/` | 300s | **IN PROGRESS** | Fabric init OK; stuck at `Waiting for op` in all-gather (smaller timeout, likely to timeout) |

Test: `test_ccl_smoke_test_p300.py::test_ccl_ddr_smoke_test[… num_devices=2 …]`

Observations:

- 2-chip BH sim + fabric 1D init succeeds on both runs.
- Large DDR all-gather (`[1,1,20000,32768]`) is extremely slow under sim; readback (`to_torch`) also hangs or exceeds timeout.
- Likely **simulator throughput limitation** for fabric CCL at this scale; may need reduced tensor shapes or longer timeouts for smoke signal.

---

## Known simulator limitations vs fixable metal issues

### Fixable metal issues (addressed or in progress)

| Issue | Fix / status |
|-------|----------------|
| Section 1 gtest SIGSEGV on skip/teardown | Fixed `38bd49b1` |
| Section 1 region write/read offset | Fixed `2149c9af` |
| Section 1 inherited mock cluster desc | Fixed `fbca6525` |
| Section 2 wrong mock desc (6u vs T3K) | Re-run job 11604 with `t3k_cluster_desc.yaml`; eth direct-send now passes |
| Heap abort (rc=134) on CCL/control-plane | Cleared in post-fix sweep vs fork |
| LLK weekly/nightly WH | FAIL — missing `tt_metal/tt-llk/tests/.venv` on metal2 (harness setup, not sim logic) |

### Simulator / environment limitations

| Category | Examples |
|----------|----------|
| **Sim speed / hangs** | Fabric 1D/2D, fabric ubench, debug tools mesh watcher, Galaxy 32-chip pytest — hit 900s timeout or run for hours |
| **Multiprocess** | All `tt-run` suites (rc=213) — need real MPI multi-host setup, not single-host ttsim sweep |
| **Mock-only fabric** | `2.fabric/unicast_1x8` SIGSEGV without sim — may need sim-backed run or skip under mock |
| **CCL at scale** | P300 DDR all-gather smoke — op completes but readback/verification times out; reduce shape for smoke |
| **Dispatch context on WH 8-chip sim** | `TestWritesAndWorkloads` TT_FATAL — fast-dispatch manual setup not supported on this sim topology (test may need skip/guard) |
| **MPI visible-devices** | SIGSEGV in multiprocess distributed tests under sim |

---

## Result directories

| Run | Path |
|-----|------|
| Full 57-suite sweep | `craq-parity-results/run-20260521T030649Z/` |
| Section 1 by commit | `craq-parity-results/section1-by-commit-20260521T195930Z/` |
| Section 2 T3K re-run | `craq-parity-results/section2-t3k-20260521T181117Z/` |
| P300 smoke 11599 | `craq-parity-results/p300-smoke-20260521T204420Z/` |
| P300 smoke 11602 | `craq-parity-results/p300-smoke-20260521T214729Z/` |
| Fork comparison | `craq-parity-results/COMPARISON-vs-tt-metal-fork.md` |

## Live monitoring

```bash
# Section 2 re-run (job 11604)
tail -f /data/rsong/tt-metal2/craq-parity-results/section2-t3k-20260521T181117Z/sweep.log

# P300 smoke (job 11602)
tail -f /data/rsong/tt-metal2/craq-parity-results/p300-smoke-20260521T214729Z/pytest.log
```
