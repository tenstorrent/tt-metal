<!--
AI-IP SW milestones and tests, cumulative through the Nov-13 release (v0.82.0).

Sources, in order of authority:

  1. The emulator release runs on tt-metal a74e6aca315 (v0.82.0-rc2), 2026-11-04.
     a74e6aca315 has the same code as the v0.82.0 release commit bfc3f70fd49:
     the two later commits (PRs 65908, 65903) change CI workflow files only.
     Results are the per-test JSON blocks of the GitHub checks "Quasar Release"
     (126695051504, tt-umd-simulators pipeline 1233855), "Horizon Release"
     (126695036074, pipeline 1233853), "Trinity Release" (126695067166,
     pipeline 1233857) and "RTL Sim CI test" (126694988291). Copies are in
     releases/0.82.0/evidence/ (internal; not for publication).
     Run selection rule (user, 2026-11-16): use the runs where most tests
     passed, even if they ran a few commits before the release commit. One
     entry is credited from a nearby run: Horizon test_linear.py failed in
     pipeline 1233853 and passed on the Horizon emulator in nightly pipeline
     1229129 (v0.82.0-dev20261103, 74f0213fd40, 63 commits before the v0.82.0
     main cut). Nightly runs on main use shorter lists, all of whose tests are
     in the full list; they add no other passes.
     The release runs on bfc3f70fd49 itself were still running on 2026-11-16
     22:26 UTC (checks 133534280446, 133534251324, 133534311996).

  2. Confluence space AIIP, page 2664202491 ("AI IP SW Milestones and Tests"),
     version 5 of 2026-10-16 (unchanged on 2026-11-16). It has the v0.80.0
     results; rows for Jul-15 to Oct-15 come from it and the 0.80.0 document.

  3. Jira epic AIIPSW-18 (Nov-13 Release) and its 11 child tickets, read
     2026-11-16. Owner runs: AIIPSW-85 and AIIPSW-86 on main 6f4477578a1
     (2026-10-30), AIIPSW-27 on main 3d23c7655d7 (2026-10-27); both commits are
     in v0.82.0. AIIPSW-82 and AIIPSW-83 were closed by the owner on 2026-10-27
     as not needed for the Horizon design.

  4. The tt-umd-horizon "Release Horizon" run 35572470887 of 2026-09-21
     (AIIPSW-15), unchanged from 0.80.0.

Editorial rules for this document (customer-facing):
  - A blank Status cell means no passing test is recorded for the requirement.
    The row is kept so every requirement is listed.
  - Only final results are shown. Transient failures are not mentioned.
  - Internal process state (pipeline backlog, tooling gaps, ticket churn,
    development emulator tags) is not mentioned.

Failures are reported at issue level (RELEASE Jira tickets, status read
2026-11-16; fix status from the release owner's table of 2026-11-07).
Left out on purpose (user, 2026-11-16: exclude items that are not part of this
release; some failing tests are not part of Trinity):
  - Trinity watcher and device print entries (RELEASE-104, RELEASE-122,
    RELEASE-131), Trinity multi-CQ trace (RELEASE-105) and Trinity profiler
    (RELEASE-106): debug tools, multi-CQ dispatch and the profiler are not
    Trinity deliverables.
  - Llama 3.2 1B graph_ops on all three emulators (RELEASE-119, done): the
    emulator CI artifact lacked model_tracer/, so the entry ran 0 tests. Not a
    tt-metal defect.
  - Quasar profiler entries at 2x3_DISPATCH (test_noc_event_profiler,
    test_custom_cycle_count, test_full_buffer): not run in the v0.82.0 runs
    (section not_run = 3). test_custom_cycle_count and test_full_buffer passed
    at 2x3_DISPATCH in every v0.81.x stable run. No ticket.
  - RELEASE-108 (emulator farm timeouts) and the non-Quasar RELEASE tickets.
Still to confirm: Quasar yolo_ops fails as one entry in rc1 and rc2 and has no
RELEASE ticket; it is listed as under investigation.

Decisions left open for the release owner (2026-11-16):
  - Whether every row of the Known issues table stays in the customer
    document (emulation failure review of 2026-11-02).
  - Trinity: the full-list run passed 86 of 128 entries; with the exclusions
    above, the Trinity known issues are RELEASE-103, 107, 111, 120 and 121.
  - AIIPSW-78 vs AIIPSW-80: the Horizon IP-partition jobs are credited to both;
    the per-build split between the two tickets is not recorded in Jira.
  - AIIPSW-53 stays blank: its TTNN op suite passes, but no LLK-level test is
    recorded. AIIPSW-57 stays owner-verified.

This document is cumulative by design: it lists the tests for every AI-IP
milestone to date, not only the ones new in this release.
-->

# AI IP SW Milestones and Tests

Cumulative test inventory for the AI-IP software deliverables, covering the
Jul-15, Aug-15, Sep-15, Oct-15 and Nov-13 milestones. Results are from the
v0.82.0 release runs of 2026-11-04 unless a row says otherwise.

**Known issues: 7 open issues affect this release.** Five affect Trinity, two
affect Horizon and two affect Quasar (some affect more than one program). Six
requirements are affected. See [Known issues](#known-issues).

## How to read this document

**Programs.** Quasar is the base IP. Horizon and Trinity are program-specific
work on that IP.

**Platforms.** The release pipeline runs the Quasar test lists on three
emulators:

- **Quasar emulator.** The runtime, data-movement, debug-tool, profiler,
  ResNet50 and model-op entries run at `2x3_DISPATCH` (two compute columns, with
  fast dispatch on dedicated dispatch cores). The LLK entries run at `1x3` (one
  compute column).
- **Horizon emulator.** The same lists with the Horizon register map. The
  pipeline also opens four IP-partitioned Horizon builds (`gasket`, `whole`,
  `split`, `flat`). For each build it runs the UMD partition suite and three
  tt-metal tests on every device of the build.
- **Trinity emulator.** The same lists as the Quasar emulator, with the Trinity
  register map and the four-row math configuration.

The Quasar RTL simulator (VCS) runs the four `1x3` rows of
`quasar_sim_regresion_tests.yaml`.

**Feature status.**

| Status | Meaning |
| --- | --- |
| ✅ Verified in release pipeline | An automated release run passed every test of the feature. |
| ⚠️ Verified, with an open failure | The feature passes its tests in a release run or an owner run, but at least one of its tests fails in the release pipeline. See Known issues. |
| 🟡 Owner-verified | The feature owner ran the tests on the emulator. |
| 🔴 Not passing yet | The feature's tests run in the release pipeline and do not pass yet. See Known issues. |

**Per-test results.** ✅ passed in the release run · ❌ failed · ⏭️ skipped on
purpose.

**LLK tests.** The LLK pytest suites under
`tt_metal/tt-llk/tests/python_tests/quasar/` are run by their owners on the
emulator. The `unit_tests_llk` gtest entries (SFPU unary, binary and ternary,
typecast, tilize and untilize, transpose, reduce, binary compute, matmul, copy
block, cumsum and the `LLKQuasarMeshDeviceSingleCardFixture` suite) run in the
release pipeline: 13 entries on Quasar and Trinity, 10 on Horizon.

## Coverage summary

| Milestone | Release | Requirements | Verified by a release pipeline | Owner-verified | Not passing yet | With a known issue |
| --- | --- | --- | --- | --- | --- | --- |
| Jul-15 | v0.74.0 | 3 | 2 (AIIPSW-2, 9) | 1 (AIIPSW-3) | 0 | 1 (AIIPSW-9) |
| Aug-15 | v0.76.0 | 8 | 4 (AIIPSW-4, 6, 13; AIIPSW-15 in tt-umd-horizon CI) | 3 (AIIPSW-8, 14; FPU/tilize 32-bit) | 0 | 0 |
| Sep-15 | v0.78.0 | 8 | 2 (AIIPSW-16, 55) | 2 (AIIPSW-52, 54) | 1 (AIIPSW-26) | 2 (AIIPSW-16, 26) |
| Oct-15 | v0.80.0 | 4 | 2 (AIIPSW-33, 56) | 2 (AIIPSW-7, 57) | 0 | 1 (AIIPSW-33) |
| Nov-13 | v0.82.0 | 11 | 4 (AIIPSW-78, 80, 85, 128) | 2 (AIIPSW-27, 86) | 1 (AIIPSW-35) | 2 (AIIPSW-35, 86) |
| **Total** | | **34** | **14** | **10** | **2** | **6** |

"With a known issue" counts the requirements in the other columns that a known
issue affects (status ⚠️ or 🔴). The Aug-15 count includes one unticketed
requirement (FPU / tilize with 32-bit dest mode), as in the earlier documents.
Two Nov-13 requirements (AIIPSW-82 and AIIPSW-83, Fabric) are not needed for the
Horizon design and have no tests.

## The v0.82.0 release run

| | |
| --- | --- |
| Release commit | `bfc3f70fd49af2cbdc7d8263ad288daaf6eed049` (v0.82.0). The release runs below ran on `a74e6aca315`, which has the same code; the two later commits change CI workflow files only. |
| Quasar Release | tt-umd-simulators pipeline 1233855, GitHub check 126695051504. All 89 gtest entries at `2x3_DISPATCH` and all 19 ResNet50 entries passed. Emulator tag `release_Quasar_A0_1.0.0_v2026.W42.0`. |
| Horizon Release | pipeline 1233853, check 126695036074: all runtime, LLK, profiler and Qwen3-VL entries passed, and all 8 IP-partition jobs passed. `test_linear.py` is the result of the Horizon nightly run of 2026-11-03 (pipeline 1229129). |
| Trinity Release | pipeline 1233857, check 126695067166. The run uses the full Quasar list. The results for the Trinity requirements are in the Milestones table; debug-tool, profiler and multi-CQ entries in the list are not Trinity deliverables. |
| RTL Sim CI test | check 126694988291: **4 of 4** Quasar `1x3` rows passed on VCS. |

### Known issues

| Issue | Impact | Program | Requirement | Ticket | Status |
| --- | --- | --- | --- | --- | --- |
| The YOLO op suite hangs in its first conv2d case on the Trinity emulator. | YOLO does not run on Trinity yet. The first 21 YOLO op tests (add, clone, concat) pass. | Trinity | AIIPSW-35 | RELEASE-121 | Under investigation |
| ResNet50 op tests hang on the Trinity emulator. | ResNet50 does not run on Trinity yet. | Trinity | AIIPSW-26 | RELEASE-107 | Fix in progress |
| The ResNet50 end-to-end test gives a wrong top-1 result, and the result changes from run to run. It passes on Quasar. | ResNet50 end-to-end classification results on Horizon and Trinity are not reliable. All ResNet50 op tests pass on Horizon. | Horizon, Trinity | AIIPSW-9, AIIPSW-26 | RELEASE-120 | Fix in review |
| DM firmware: a NoC transaction ID left by an earlier kernel can make later plain NoC reads end before the data is in L1 (seen in `QuasarTransposeWHInt32`). | On Trinity, a kernel with plain NoC reads that runs after a kernel that uses NoC transaction IDs can read data that is not complete. | Trinity | AIIPSW-33 | RELEASE-111 | Fix in review |
| The `LLKQuasarMeshDeviceSingleCardFixture` suite hangs in `TensixComputeSfpuReduceColumn`. | The SFPU column reduce does not complete on Trinity. The other 11 Trinity LLK entries pass. | Trinity | AIIPSW-33 | RELEASE-103 | Fix in review |
| ResNet50 `test_linear` can stop on a TRISC1 hardware fault in the matmul compute kernel. It fails in some runs and passes in others. | The ResNet50 fully connected layer can stop the device on Quasar and Horizon. | Quasar, Horizon | AIIPSW-16, AIIPSW-9 | RELEASE-118 | Fixed after v0.82.0 |
| The YOLO op suite does not pass as one entry with fast dispatch on the Quasar emulator. | The YOLO ops pass in the owner's run (AIIPSW-86) but not yet in the release pipeline. The cause is not known yet. | Quasar | AIIPSW-86 |  | Under investigation |

## Milestones

| **Milestone** | **Requirement (ticket)** | **Program** | **Status** | **Tests and v0.82.0 results** | **Owner** |
| --- | --- | --- | --- | --- | --- |
| Jul-15 | **Runtime: Fast dispatch (FD) support** (AIIPSW-2) | Quasar | ✅ Verified in release pipeline | Three `unit_tests_legacy` entries. `*DmLoopback*`: Quasar ✅ `1x3`, ✅ `2x3_DISPATCH`; Horizon ✅; Trinity ✅; RTL sim ✅ `1x3`. `*MultiDmAddTwoInts*`: Quasar ✅ `2x3_DISPATCH`; Horizon ✅; Trinity ✅. `*QuasarComputeKernelMultipleThread*`: Quasar ✅ `2x3_DISPATCH`; Horizon ✅; Trinity ✅; RTL sim ✅ `1x3`. | Kevin Stevens |
| Jul-15 | **LLK: Compute API calling YOLO LLK APIs** (AIIPSW-3) | Quasar | 🟡 Owner-verified | Five LLK pytest suites, owner-run on the emulator: `test_eltwise_unary_sfpu_quasar.py`, `test_eltwise_binary_sfpu_quasar.py`, `test_topk_quasar.py`, `test_unpack_tilize_quasar.py`, `test_reduce_quasar.py` (kernels under `tt_metal/tt-llk/tests/sources/quasar/`). The TTNN YOLO op suite is under AIIPSW-86. | Filip Vranic |
| Jul-15 | **LLK: ResNet LLK APIs passing on the Horizon emulator** (AIIPSW-9) | Horizon | ⚠️ Verified, with an open failure | The 19 ResNet50 rows on the Horizon emulator. 18 op-test files ✅: `test_add`, `test_avg_pool2d`, `test_conv2d_layer3_downsample_split`, `test_conv2d_layer_conv2_modelcfg`, `test_global_avgpool`, `test_linear` (Horizon nightly run of 2026-11-03; can fail, known issue RELEASE-118), `test_max_pool2d_1x3`, `test_max_pool2d_strided_reduce`, `test_padded_slice`, `test_reallocate`, `test_reshape`, `test_reshape_tiled`, `test_sharded_to_interleaved`, `test_slice_write`, `test_tilize`, `test_to_layout`, `test_to_memory_config`, `test_untilize_with_unpadding`. `test_resnet50_e2e` ❌ (known issue RELEASE-120). | Filip Vranic |
| Aug-15 | **TTNN / Kernel Ops: ResNet kernel ops** (AIIPSW-4) | Quasar | ✅ Verified in release pipeline | 11 op-test files at `2x3_DISPATCH` on the Quasar emulator, whole file each: `test_add`, `test_padded_slice`, `test_reallocate`, `test_reshape`, `test_reshape_tiled`, `test_sharded_to_interleaved`, `test_slice_write`, `test_tilize`, `test_to_layout`, `test_to_memory_config`, `test_untilize_with_unpadding`, all ✅. The same 11 files ✅ on Horizon. | Borys Bradel |
| Aug-15 | **Runtime: Fast dispatch for the dispatch engine** (AIIPSW-6) | Quasar | ✅ Verified in release pipeline | All 89 gtest entries at `2x3_DISPATCH` on the Quasar emulator ✅: 37 in `unit_tests_legacy` (single-thread and multi-kernel compute, globals and TLS, mesh buffers in L1 and DRAM, sub-devices, trace replay, events, non-blocking enqueue, multi-node Bmm, pack-relu, `QuasarFdsFixture*`, multi-CQ), 28 in `unit_tests_api` (CRTA, run arguments, scoped-lock cache, DFB implicit sync and overrides, L1 banks under ATT), 16 in `unit_tests_debug_tools` (watcher and DPRINT), 6 in `unit_tests_data_movement` (L1 and L2 cache ops, iDMA, cache write, address generator, im2col), `unit_tests_dispatch` `*QuasarDispatchSInstantiatedAndRunning*` and `unit_tests_device` `*QuasarIoWindowReadWrite*`. The 79 runtime gtest entries of the Horizon list all ✅. `*TensixSingleCoreDirectDramReaderDatacopyWriter` also ✅ on the RTL sim at `1x3`. | Kevin Stevens |
| Aug-15 | **LLK: Quasar INT8 support** (AIIPSW-8) | Quasar | 🟡 Owner-verified | `test_eltwise_unary_sfpu_quasar.py`, `test_eltwise_binary_sfpu_quasar.py` (INT8 `add_int` / `mul_int`, PR 48208) and `test_reduce_quasar.py` (INT8 to INT32 reduce, PR 49390), owner-run. | Filip Vranic |
| Aug-15 | **LLK: Quant / requant / dequant kernels** (AIIPSW-14) | Quasar | 🟡 Owner-verified | `test_eltwise_binary_sfpu_quasar.py`, quant / requant / dequant family (PR 48208), owner-run. | Filip Vranic |
| Aug-15 | **LLK: FPU / tilize with 32-bit dest mode** (no ticket) | Quasar | 🟡 Owner-verified | Six LLK suites extended from 16-bit to 32-bit dest, owner-run: `test_unpack_tilize_quasar.py`, `test_pack_untilize_quasar.py`, `test_reduce_quasar.py`, `test_transpose_dest_quasar.py`, `test_matmul_quasar.py`, `test_pack_l1_acc_quasar.py`. | Filip Vranic |
| Aug-15 | **LLK: PDL LLK features** (AIIPSW-12) | Quasar |  | No Quasar test recorded. PDL tests exist in `models/experimental/panoptic_deeplab/tests/` without a Quasar variant. | Filip Vranic |
| Aug-15 | **Runtime: Profiler debug tool support** (AIIPSW-13) | Quasar | ✅ Verified in release pipeline | `tests/tt_metal/tools/profiler/test_device_profiler.py`: `::test_custom_cycle_count_slow_dispatch` ✅ `1x3` on Quasar, Horizon and Trinity; `::test_custom_cycle_count` and `::test_full_buffer` ✅ on Horizon. On the Quasar emulator these two fast-dispatch tests have no v0.82.0 result; they passed at `2x3_DISPATCH` in every v0.81.x release run. `unit_tests_data_movement --gtest_filter=*QuasarCacheWrite*` ✅ `2x3_DISPATCH` on Quasar, Horizon and Trinity. The 14 LLK performance suites `perf_*_quasar.py` are owner-run. | Kevin Stevens |
| Aug-15 | **UMD: Higher-level layer support (integration with the Metal layer)** (AIIPSW-15) | Horizon | ✅ Verified in release pipeline (tt-umd-horizon) | The "Release Horizon" workflow of the `tenstorrent/tt-umd-horizon` repository, run 35572470887 on 2026-09-21: 6 of 6 passed: `test_axi_device`, `test_axi_protocol`, `test_horizon_cluster`, `test_horizon_coordinate_translator`, `test_horizon_dma`, `test_horizon_tt_device`. tt-umd-horizon is released separately. | Nenad Bunčić |
| Sep-15 | **TTNN / Kernel Ops: ResNet conv2d, pool and linear** (AIIPSW-16) | Quasar | ⚠️ Verified, with an open failure | Seven op-test files at `2x3_DISPATCH` on the Quasar emulator, all ✅ in the release run: `test_linear`, `test_avg_pool2d`, `test_global_avgpool`, `test_conv2d_layer3_downsample_split`, `test_conv2d_layer_conv2_modelcfg`, `test_max_pool2d_1x3`, `test_max_pool2d_strided_reduce`. `test_linear` can fail on a hardware fault in some runs (known issue RELEASE-118). The same seven ✅ on Horizon. `test_resnet50_e2e.py::test_resnet50_e2e[pretrained-device_params0]` ✅ `2x3_DISPATCH` on Quasar (Horizon result under AIIPSW-9). | Borys Bradel |
| Sep-15 | **Debug tools: TT-Exalens call stack and step-by-step debugging** (AIIPSW-52) | Quasar | 🟡 Owner-verified | `rocket_step_test.py` and `rocket_callstack_test/rocket_callstack_test.py` of `tenstorrent/tt-exalens`, owner-run on Quasar simulation. Validated against tt-exalens 0.3.31. | Aleksandar Đorđević |
| Sep-15 | **Debug tools: Dynamic visualizer / NPE support** (AIIPSW-54) | Quasar | 🟡 Owner-verified | SoC-descriptor dump: `test_device_profiler.py::test_noc_event_profiler`, which runs in CI on Wormhole and Blackhole silicon and ✅ on the Horizon emulator in the release run. Visualizer compatibility was checked by hand against ttnn-visualizer 0.101.0. | Sohaib Nadeem, Denis Kartashevsky |
| Sep-15 | **LLK: Qwen3-VL-2B LLK features** (AIIPSW-53) | Quasar |  | No LLK-level test recorded. The TTNN op suite `models/experimental/ops/quasar/tests/qwen3_vl_ops/` passes in the release run on all three emulators (AIIPSW-85). | Filip Vranic |
| Sep-15 | **Runtime: Port basic runtime functionality from Quasar to Trinity** (AIIPSW-55) | Trinity | ✅ Verified in release pipeline | On the Trinity emulator: `*SingleDmL1Write*`, `*DmLoopback*` and `*MeshDeviceSingleCardFixture.Bmm` ✅ at `1x3` and `2x3_DISPATCH`. | Erik Tomusk |
| Sep-15 | **TTNN / Kernel Ops: Port ResNet kernel ops from Quasar to Trinity** (AIIPSW-26) | Trinity | 🔴 Not passing yet | The 19 ResNet50 entries run on the Trinity emulator in the release pipeline. The op tests hang (known issue RELEASE-107) and the end-to-end test gives a wrong result (known issue RELEASE-120). | Borys Bradel |
| Sep-15 | **LLK: YOLO LLK API INT8 on Trinity** (AIIPSW-32) | Trinity |  | No test recorded for this release. | Filip Vranic |
| Sep-15 | **Debug tools: TT-Exalens on Trinity** (AIIPSW-30) | Trinity |  | TT-Exalens needs no Trinity-specific features: its Tensix and Rocket core access, already supported for Quasar, covers Trinity. The AIIPSW-52 tests apply. | Aleksandar Đorđević |
| Oct-15 | **TTNN / Kernel Ops: Llama 3.2 1B kernel ops** (AIIPSW-7) | Quasar | 🟡 Owner-verified | 36 graph-op test files in `models/experimental/llama32_1b_quasar/tests/graph_ops/`, owner-run one file at a time with slow dispatch and forced JIT; each passes or is skipped on purpose (0.80.0 document). `tests/demos/llama32_1b/test_llama_e2e.py` passed on main commit `9d423e4bb6f` on 2026-10-02. The same tests passed on the Trinity emulator on main commit `3d23c7655d7` (AIIPSW-27). | Borys Bradel |
| Oct-15 | **LLK: Port LLK from Quasar to Trinity (4-row math)** (AIIPSW-33) | Trinity | ⚠️ Verified, with an open failure | The 13 `unit_tests_llk` entries on the Trinity emulator: 11 ✅, 1 ⏭️ (`TensixComputeReduceH` group), 1 ❌ (`LLKQuasarMeshDeviceSingleCardFixture*`, known issues RELEASE-103 and RELEASE-111). Seven LLK pytest suites, owner-run for v0.80.0 with `QUASAR_ARCH_VARIANT=quasar_4row`: `test_eltwise_binary_sfpu_quasar.py`, `test_matmul_quasar.py`, `test_matmul_custom_no_mop_quasar.py`, `test_reduce_quasar.py`, `test_unary_broadcast_quasar.py`, `perf_reduce_quasar.py`, `test_quasar_stall_wrapper_codegen.py`. At v0.82.0 the four-row headers are in `tt_metal/tt-llk/tt_llk_quasar/arch/trinity/`, and tt-metal selects them from the Trinity IP variant. | Filip Vranic |
| Oct-15 | **Runtime: ATT / flat address, use cases UA and UB** (AIIPSW-56) | Horizon | ✅ Verified in release pipeline | The Horizon IP-partition jobs: `*SingleDmL1Write*`, `*DmLoopback*` and `*MeshDeviceSingleCardFixture.Bmm` ✅ on every device of the `gasket`, `whole`, `split` and `flat` builds. `unit_tests_api --gtest_filter=QuasarMeshDeviceSingleCardFixture.L1BanksAreInRowMajorOrderWithAtt` ✅ `2x3_DISPATCH` on Quasar. `unit_tests_noc --gtest_filter='QuasarAtt*'` (47 tests) and `unit_tests_api --gtest_filter='SimulatorBuildDir.*'` are owner-run. | Nenad Bunčić |
| Oct-15 | **Runtime: Overlay registers** (AIIPSW-57) | Horizon, Trinity | 🟡 Owner-verified | Every Horizon and Trinity release entry runs with that program's register map. On Horizon, all runtime, LLK and profiler entries passed. On Trinity, the runtime, data-movement and API entries for the Trinity requirements passed (AIIPSW-55). Cache-controller register tests and FDS tests are owner-run on both. | Arik Yaacob |
| Nov-13 | **TTNN / Kernel Ops: Qwen3-VL-2B kernel ops** (AIIPSW-85) | Quasar | ✅ Verified in release pipeline | `models/experimental/ops/quasar/tests/qwen3_vl_ops/` as one entry ✅ `2x3_DISPATCH` on the Quasar emulator (also ✅ on Horizon and Trinity). Owner run on main `6f4477578a1`: 71 passed, 8 skipped. The end-to-end demo needs more DRAM than the emulator models; see the test detail below. | Borys Bradel |
| Nov-13 | **TTNN / Kernel Ops: YOLO kernel ops** (AIIPSW-86) | Quasar | ⚠️ Verified, with an open failure | `models/experimental/ops/quasar/tests/yolo_ops/` owner-run on the emulator on main `6f4477578a1`: 137 passed, 5 skipped. In the release pipeline the suite runs as one entry with fast dispatch on the Quasar emulator and does not pass yet (known issue). | Borys Bradel |
| Nov-13 | **TTNN / Kernel Ops: Port Llama kernel ops from Quasar to Trinity** (AIIPSW-27) | Trinity | 🟡 Owner-verified | All `graph_ops` tests and the single-layer end-to-end test, owner-run on main `3d23c7655d7` on 2026-10-27. | Borys Bradel |
| Nov-13 | **TTNN / Kernel Ops: Port YOLO kernel ops from Quasar to Trinity** (AIIPSW-35) | Trinity | 🔴 Not passing yet | `models/experimental/ops/quasar/tests/yolo_ops/` runs on the Trinity emulator in the release pipeline. Its first 21 tests (add, clone, concat) pass, then it hangs in its first conv2d case (known issue RELEASE-121). | Borys Bradel |
| Nov-13 | **TTNN / Kernel Ops: Port Qwen3-VL-2B kernel ops from Quasar to Horizon** (AIIPSW-128) | Horizon | ✅ Verified in release pipeline | `models/experimental/ops/quasar/tests/qwen3_vl_ops/` as one entry ✅ on the Horizon emulator. | Borys Bradel |
| Nov-13 | **LLK: Quasar OFT LLK features** (AIIPSW-100) | Quasar |  | OFT op tests are in `models/experimental/ops/quasar/tests/oft_ops/` (PRs 60669, 61658). No result recorded for this release. | Filip Vranic |
| Nov-13 | **Runtime: ATT / flat address, use cases UD and UE** (AIIPSW-78) | Horizon | ✅ Verified in release pipeline | The eight Horizon IP-partition jobs, all ✅: UMD's `RtlSimIpPartitionTest.*` suite on the `gasket`, `whole`, `split` and `flat` builds, and `*SingleDmL1Write*`, `*DmLoopback*` and `*MeshDeviceSingleCardFixture.Bmm` on every device of each build (`col0` and `col1`, or the whole chip). | Nenad Bunčić |
| Nov-13 | **Runtime: D3 verification of multiple programs across partitions** (AIIPSW-80) | Horizon | ✅ Verified in release pipeline | The same eight Horizon IP-partition jobs, all ✅: each partition opens as its own device and runs its own programs. | Erik Tomusk |
| Nov-13 | **Debug tools: tt-npe on Trinity** (AIIPSW-39) | Trinity |  | tt-npe models Quasar-family devices from a device-model YAML file (tt-npe PRs 137, 139, 141). No test recorded for this release. |  |
| Nov-13 | **Fabric: Fabric port to DM cores** (AIIPSW-82) | Horizon |  | Not needed for Horizon. The NoC-to-AXI blocks and their ATTs translate addresses and route across modules, so no DM core is needed as an intermediary. | Sean Nijjar |
| Nov-13 | **Fabric: Fabric PCIe hop** (AIIPSW-83) | Horizon |  | Not needed for Horizon. The Horizon module-to-module datapath has no PCIe hop. | Sean Nijjar |

### Platform tests not tied to one requirement

The `extra` and `kept` entries of the release lists support every feature and
map to no single requirement. All passed on the Quasar emulator (9 entries),
the Horizon emulator (4 entries) and the Trinity emulator (6 entries): RISC-V
atomics, interleaved and unaligned L1 buffers, DRAM bounds on the emulator,
the `QuasarMeshDeviceSingleCardFixture` LLK suite, `*SingleDmL1Write*`,
`*DmLoopback*`, `*MeshDeviceSingleCardFixture.Bmm` and
`test_custom_cycle_count_slow_dispatch` at `1x3`; and on Quasar at
`2x3_DISPATCH`, `DmStatefulLoopback`, `DmWorkerMulticast`, three watcher
sanitizer and ring-buffer tests and `L1BanksAreInRowMajorOrderWithAtt`.

## Nov-13 test detail

### AIIPSW-85 — Qwen3-VL-2B kernel ops

```
pytest models/experimental/ops/quasar/tests/qwen3_vl_ops/ -m emulator
```

On main commit `6f4477578a1`: 71 passed, 8 skipped. In the release pipeline the
directory runs as one entry on all three emulators and passes.

The end-to-end demo runs a cut-down model:

```
export MESH_DEVICE=N150 HF_MODEL=Qwen/Qwen3-VL-4B-Instruct
export TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 TT_METAL_DISABLE_SFPLOADMACRO=1
pytest models/experimental/ops/quasar/qwen3_vl/demo/demo.py --timeout=21600
```

The model needs more DRAM than the emulator models. With
`TT_METAL_SIMULATOR_DRAM_UNCAPPED=1` the demo runs, but the DRAM addresses wrap
and the output does not match the reference (low PCC). PR 63013 records the
details.

### AIIPSW-86 — YOLO kernel ops

```
pytest models/experimental/ops/quasar/tests/yolo_ops/ -m emulator
```

On main commit `6f4477578a1`: 137 passed, 5 skipped.

### AIIPSW-27 — Llama kernel ops on Trinity

```
MESH_DEVICE=N150 pytest models/experimental/llama32_1b_quasar/tests/graph_ops/
```

The single-layer, single-step end-to-end test needs the model weights on the
host:

```
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TT_METAL_SLOW_DISPATCH_MODE=1 LLAMA32_1B_TF_MAX_DECODE_STEPS=1 LLAMA32_1B_DEMO_NUM_LAYERS=1 MESH_DEVICE=N150 \
pytest models/experimental/llama32_1b_quasar/tests/demos/llama32_1b/test_llama_e2e.py
```

Both passed on main commit `3d23c7655d7` on 2026-10-27.

### AIIPSW-78 and AIIPSW-80 — ATT, flat addresses and partitions on Horizon

The release pipeline opens four IP-partitioned Horizon builds. For each build
it runs UMD's `RtlSimIpPartitionTest.*` suite, then runs `*SingleDmL1Write*`,
`*DmLoopback*` and `*MeshDeviceSingleCardFixture.Bmm` in one process per
device: `col0` and `col1` for the `gasket`, `split` and `flat` builds, and the
whole chip for the `whole` build. All eight jobs passed. The UMD side is in
tt-umd commit `7b508db`, which tt-metal v0.82.0 pins (see the release notes,
section 1).

## Notes on this release

- **Seven known issues.** Five affect Trinity: YOLO and ResNet50 do not run on
  Trinity yet, and two LLK and firmware issues are in review. On Quasar and
  Horizon, the ResNet50 end-to-end result on Horizon and an intermittent fault
  in `test_linear` remain, and the YOLO op suite does not pass in the Quasar
  release pipeline yet. See Known issues.
- **All fast dispatch runtime tests are green on Quasar.** All 89 gtest entries
  at `2x3_DISPATCH` passed on the Quasar emulator, and all 79 runtime gtest
  entries passed on Horizon.
- **Trinity runs in the release pipeline.** The full Quasar test list now runs
  on the Trinity emulator with the Trinity register map and four-row math. The
  basic runtime port (AIIPSW-55) and 11 of the 13 LLK entries pass; the Trinity
  known issues are listed above.
- **Horizon IP partitions run in the release pipeline.** Four partitioned
  Horizon builds open each partition as its own device, and all eight jobs
  passed.
- **LLK coverage in the release pipeline.** The `unit_tests_llk` gtest entries
  run on all three emulators. On Quasar 12 of 13 passed and 1 is skipped on
  purpose; on Horizon all 10 passed.
- **Qwen3-VL ops pass on all three emulators.** The `qwen3_vl_ops` suite
  passed on Quasar, Horizon and Trinity.
- **AIIPSW-15 is verified in the tt-umd-horizon repository.** Its "Release
  Horizon" workflow passed all six suites on 2026-09-21. tt-umd-horizon is a
  separate repository released on its own cadence.
- **Debug-tool deliverables ship outside tt-metal.** TT-Exalens,
  TTNN-Visualizer and TT-NPE are released as separate packages.
