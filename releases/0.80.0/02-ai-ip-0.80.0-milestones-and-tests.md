<!--
AI-IP SW milestones and tests, cumulative through the Oct-15 release (v0.80.0).

Sources, in order of authority:

  1. Confluence space AIIP, page 2664202491 ("AI IP SW Milestones and Tests"),
     rebuilt as one cumulative per-test table on 2026-10-14 (version 3) and
     last modified 2026-10-16 (version 5), author Sapna Fun Bahadur Khatri.
     Retrieved 2026-10-16. This document summarises its 348 test rows per
     requirement; the page keeps the full row list.

  2. The v0.80.0 release evidence artifact (release-test-evidence.md,
     quasar-test-results.json, horizon-test-results.json, rtl-sim-results.json,
     tests-by-feature.csv): the Quasar Release, Horizon Release and RTL Sim CI
     test runs on the release commit 978f93b56e36 (= v0.80.0-rc6 = v0.80.0-rc7
     = v0.80.0), started 2026-10-13 16:29 UTC. Read from GitLab and GitHub on
     2026-10-14. Results here are the final result of each entry on that build.

  3. The tt-umd-horizon results file (schema horizon-test-results/v1) from
     that repository's "Release Horizon" workflow, run 35572470887 of
     2026-09-21 (AIIPSW-15).

  4. Jira: epic AIIPSW-17 (Oct-15 Release) and its four child tickets
     AIIPSW-7, AIIPSW-33, AIIPSW-56 and AIIPSW-57. The AIIPSW-7 commands are
     from the owner's ticket comments of 2026-09-29 and 2026-10-02.

Editorial rules for this document (customer-facing):
  - A blank Status cell means no test is recorded for the requirement in the
    sources. The row is kept so every requirement is listed.
  - Only final results on the release build are shown. Transient failures that
    passed again on the same build are not mentioned.
  - Internal process state (pipeline backlog, tooling gaps, ticket churn) is
    not mentioned. Owner runs are reported as results, with the commit where
    one was given.

Maintainer notes, not for the rendered document:
  - The release commit is on the release branch, not on main. Owner runs on
    later main commits (for example the AIIPSW-7 end-to-end pass on
    9d423e4bb6f) are owner evidence, not release-commit evidence.
  - The GitHub checks on the release commit show the first attempt of each
    run only, so their summaries differ from the totals here.
  - The automated release report maps test_resnet50_e2e to AIIPSW-4 and
    QuasarCacheWrite to AIIPSW-6; the Confluence page puts them under AIIPSW-16
    and AIIPSW-13. This document follows Confluence.
  - AIIPSW-15 is recorded from source 3; the Confluence page predates it.

This document is cumulative by design: it lists the tests for every AI-IP
milestone to date, not only the ones new in this release.
-->

# AI IP SW Milestones and Tests

Cumulative test inventory for the AI-IP software deliverables, covering the
Jul-15, Aug-15, Sep-15 and Oct-15 milestones. Results are from the v0.80.0
release run of 2026-10-13 unless a row says otherwise.

## How to read this document

**Programs.** Quasar is the base IP. Horizon and Trinity are program-specific
work on that IP.

**Platforms.** The Quasar emulator runs the Quasar test lists at three
configs: `1x3` (one compute column), `2x3` (two compute columns) and
`2x3_DISPATCH` (`2x3` with fast dispatch on dedicated dispatch cores). The
Horizon emulator runs the same lists except the `2x3_DISPATCH` entries and the
multi-CQ test. The Quasar RTL simulator (VCS) runs the four `1x3` rows of
`quasar_sim_regresion_tests.yaml`. Trinity results are owner runs on the
Trinity emulator.

**Feature status.**

| Status | Meaning |
| --- | --- |
| ✅ Verified in release pipeline | An automated run on the release commit passed every test of the feature. |
| ⚠️ Verified, with an open failure | As above, but one test failed and is under investigation. |
| 🟡 Owner-verified | The feature owner ran the tests on the emulator. |

**Per-test results.** ✅ passed in the release run · ❌ failed · ⏭️ skipped on
purpose.

**LLK tests.** The LLK pytest suites under
`tt_metal/tt-llk/tests/python_tests/quasar/` are run by their owners on the
emulator. Twelve `unit_tests_llk` gtest entries join the Quasar release list
from the next release (PR 59926, merged 2026-10-14).

## Coverage summary

| Milestone | Release | Requirements | Verified by a release pipeline | Owner-verified |
| --- | --- | --- | --- | --- |
| Jul-15 | v0.74.0 | 3 | 2 (AIIPSW-2; AIIPSW-9 with an open failure) | 1 (AIIPSW-3) |
| Aug-15 | v0.76.0 | 8 | 4 (AIIPSW-4, 6, 13; AIIPSW-15 in tt-umd-horizon CI) | 3 (AIIPSW-8, 14; FPU/tilize 32-bit) |
| Sep-15 | v0.78.0 | 8 | 1 (AIIPSW-16) | 3 (AIIPSW-52, 54, 55) |
| Oct-15 | v0.80.0 | 4 | 0 | 4 (AIIPSW-7, 33, 56, 57) |
| **Total** | | **23** | **7** | **11** |

The Aug-15 count includes one unticketed requirement (FPU / tilize with 32-bit
dest mode), as in the 0.76.0 and 0.78.0 documents. The Sep-15 count includes
the four Trinity requirements (AIIPSW-26, 30, 32, 55), which the 0.78.0
document did not list. All four Oct-15 requirements are owner-verified on the
Quasar, Horizon and Trinity emulators.

## The v0.80.0 release run

| | |
| --- | --- |
| Release commit | `978f93b56e36b68318d11ecf7b101085a81d718e` (v0.80.0-rc6, rc7 and the v0.80.0 tag; GitHub release published 2026-10-14 12:25 UTC) |
| Quasar Release | tt-umd-simulators pipelines 1169420 and 1170341, GitHub check 115412654258: **112 of 112** entries passed, 213 individual tests. Emulator tag `release_Quasar_A0_1.0.0_v2026.W39.0`. |
| Horizon Release | pipelines 1169418 and 1170342, check 115412629415: **61 of 62** entries passed, 115 individual tests. The one failure is the ResNet50 end-to-end test, below. |
| RTL Sim CI test | pipeline 1169416, check 115412590721: **4 of 4** Quasar `1x3` rows passed on VCS. One Wormhole and one Blackhole versim LLK test also passed; they are not Quasar and map to no requirement. |

### Open failure

One entry failed on the release build. It is under investigation by the
ResNet owners.

| Platform | Entry | Result |
| --- | --- | --- |
| Horizon | `[2x3] models/demos/vision/classification/resnet50/quasar/tests/test_resnet50_e2e.py::test_resnet50_e2e[pretrained-device_params0]` | Top-1 mismatch: device class 549 and 783 in two runs, golden 610. Final logits PCC 0.9905 and 0.9949. The first op with a different output is op011, `layer1_module2.conv2`. The same test passes on the Quasar emulator. |

## Milestones

| **Milestone** | **Requirement (ticket)** | **Program** | **Status** | **Tests and v0.80.0 results** | **Owner** |
| --- | --- | --- | --- | --- | --- |
| Jul-15 | **Runtime: Fast dispatch (FD) support** (AIIPSW-2) | Quasar | ✅ Verified in release pipeline | Three `unit_tests_legacy` entries. `*DmLoopback*`: Quasar ✅ `1x3`, ✅ `2x3_DISPATCH`; Horizon ✅; RTL sim ✅ `1x3`. `*MultiDmAddTwoInts*`: Quasar ✅ `2x3`, ✅ `2x3_DISPATCH`; Horizon ✅. `*QuasarComputeKernelMultipleThread*`: Quasar ✅ `2x3_DISPATCH`; RTL sim ✅ `1x3`. | Kevin Stevens |
| Jul-15 | **LLK: Compute API calling YOLO LLK APIs** (AIIPSW-3) | Quasar | 🟡 Owner-verified | Five LLK pytest suites, owner-run on the emulator: `test_eltwise_unary_sfpu_quasar.py`, `test_eltwise_binary_sfpu_quasar.py`, `test_topk_quasar.py`, `test_unpack_tilize_quasar.py`, `test_reduce_quasar.py` (kernels under `tt_metal/tt-llk/tests/sources/quasar/`). The repository also holds 23 TTNN YOLO op tests under `models/experimental/ops/quasar/tests/yolo_ops/`, outside the release run. | Filip Vranic |
| Jul-15 | **LLK: ResNet LLK APIs passing on the Horizon emulator** (AIIPSW-9) | Horizon | ⚠️ Verified, with an open failure | The 19 ResNet50 rows of `quasar_local_tests.yaml` on the Horizon emulator. 18 op-test files ✅: `test_add`, `test_avg_pool2d`, `test_conv2d_layer3_downsample_split`, `test_conv2d_layer_conv2_modelcfg`, `test_global_avgpool`, `test_linear`, `test_max_pool2d_1x3`, `test_max_pool2d_strided_reduce`, `test_padded_slice`, `test_reallocate`, `test_reshape`, `test_reshape_tiled`, `test_sharded_to_interleaved`, `test_slice_write`, `test_tilize`, `test_to_layout`, `test_to_memory_config`, `test_untilize_with_unpadding`. `test_resnet50_e2e` ❌ (top-1 mismatch; see the open failure above). | Filip Vranic |
| Aug-15 | **TTNN / Kernel Ops: ResNet kernel ops** (AIIPSW-4) | Quasar | ✅ Verified in release pipeline | 11 op-test files at `2x3` on the Quasar emulator, whole file each: `test_add`, `test_padded_slice`, `test_reallocate`, `test_reshape`, `test_reshape_tiled`, `test_sharded_to_interleaved`, `test_slice_write`, `test_tilize`, `test_to_layout`, `test_to_memory_config`, `test_untilize_with_unpadding`, all ✅. The same 11 files ✅ on Horizon. Five more op files are available in the same directory outside the release run: `test_tilize_width_quasar`, `test_fold`, `test_fold_c8`, `test_fold_transpose`, `test_reduce_sum_mean`. | Borys Bradel |
| Aug-15 | **Runtime: Fast dispatch for the dispatch engine** (AIIPSW-6) | Quasar | ✅ Verified in release pipeline | 43 gtest entries on the Quasar emulator, all ✅ at `2x3_DISPATCH`: 32 in `unit_tests_legacy` (single-thread and multi-kernel compute, globals and TLS, mesh buffers in L1 and DRAM, sub-devices, trace replay, events, non-blocking enqueue, multi-node Bmm, pack-relu, `QuasarFdsFixture*` without `DispatchEngineConcurrentEngines`); 6 in `unit_tests_api` (`*QuasarCRTASharedL1Address*`, `*QuasarCRTAUniqueL1Addresses*`, `*TensixSingleCoreDirectDramReaderDatacopyWriter`, `*ScopedLockCache*`, `*QuasarMergeProgramRunArgs`, `*QuasarUpdateProgramRunArgs`); 3 in `unit_tests_data_movement` (`*QuasarL1DCacheOps*`, `*QuasarL2CacheOps*`, `*QuasarIdmaOps*`); `unit_tests_debug_tools` `*WatcherAssertTests*/DispatchDM2`; `unit_tests_dispatch` `*QuasarDispatchSInstantiatedAndRunning*` (also ✅ `2x3` and on Horizon). `*TensixSingleCoreDirectDramReaderDatacopyWriter` also ✅ on the RTL sim at `1x3`. `*SingleDmL1Write*` also ✅ at `1x3` and `2x3`; `*MeshDeviceSingleCardFixture.Bmm` also ✅ at `1x3` and on the RTL sim; both ✅ on Horizon. | Kevin Stevens |
| Aug-15 | **LLK: Quasar INT8 support** (AIIPSW-8) | Quasar | 🟡 Owner-verified | `test_eltwise_unary_sfpu_quasar.py`, `test_eltwise_binary_sfpu_quasar.py` (INT8 `add_int` / `mul_int`, PR 48208) and `test_reduce_quasar.py` (INT8 to INT32 reduce, PR 49390), owner-run. | Filip Vranic |
| Aug-15 | **LLK: Quant / requant / dequant kernels** (AIIPSW-14) | Quasar | 🟡 Owner-verified | `test_eltwise_binary_sfpu_quasar.py`, quant / requant / dequant family (PR 48208), owner-run. | Filip Vranic |
| Aug-15 | **LLK: FPU / tilize with 32-bit dest mode** (no ticket) | Quasar | 🟡 Owner-verified | Six LLK suites extended from 16-bit to 32-bit dest, owner-run: `test_unpack_tilize_quasar.py`, `test_pack_untilize_quasar.py`, `test_reduce_quasar.py`, `test_transpose_dest_quasar.py`, `test_matmul_quasar.py`, `test_pack_l1_acc_quasar.py` (for example 32-bit dest TilizeA_B, PR 49579). | Filip Vranic |
| Aug-15 | **LLK: PDL LLK features** (AIIPSW-12) | Quasar |  | No Quasar test recorded. INT8 support (AIIPSW-8) was brought forward ahead of the PDL features. PDL tests exist in `models/experimental/panoptic_deeplab/tests/` without a Quasar variant. | Filip Vranic |
| Aug-15 | **Runtime: Profiler debug tool support** (AIIPSW-13) | Quasar | ✅ Verified in release pipeline | `tests/tt_metal/tools/profiler/test_device_profiler.py`: `::test_custom_cycle_count_slow_dispatch` ✅ `1x3` (Horizon ✅), `::test_custom_cycle_count` ✅ `2x3_DISPATCH`, `::test_full_buffer` ✅ `2x3_DISPATCH`. `unit_tests_data_movement --gtest_filter=*QuasarCacheWrite*` ✅ `2x3_DISPATCH`. The 14 LLK performance suites `perf_*_quasar.py` (eltwise binary and its broadcast and reuse-dest variants, unary broadcast and datacopy, pack, pack_l1_acc, pack_untilize, unpack_tilize, unpack_unary_operand, reduce, transpose_dest, unary and binary SFPU) are owner-run. | Kevin Stevens |
| Aug-15 | **UMD: Higher-level layer support (integration with the Metal layer)** (AIIPSW-15) | Horizon | ✅ Verified in release pipeline (tt-umd-horizon) | The "Release Horizon" workflow of the `tenstorrent/tt-umd-horizon` repository, run 35572470887 on 2026-09-21 at that repository's commit `31f18e075e8`: 6 of 6 passed: `test_axi_device`, `test_axi_protocol`, `test_horizon_cluster`, `test_horizon_coordinate_translator`, `test_horizon_dma`, `test_horizon_tt_device`. tt-umd-horizon is released separately, so the run is not tied to the tt-metal release commit. | Nenad Bunčić |
| Sep-15 | **TTNN / Kernel Ops: ResNet conv2d, pool and linear** (AIIPSW-16) | Quasar | ✅ Verified in release pipeline | Seven op-test files at `2x3` on the Quasar emulator, all ✅: `test_linear`, `test_avg_pool2d`, `test_global_avgpool`, `test_conv2d_layer3_downsample_split`, `test_conv2d_layer_conv2_modelcfg`, `test_max_pool2d_1x3`, `test_max_pool2d_strided_reduce`. The same seven ✅ on Horizon. `test_resnet50_e2e.py::test_resnet50_e2e[pretrained-device_params0]` ✅ `2x3` on Quasar, ❌ on Horizon (counted under AIIPSW-9). The ops directory holds 48 test files at v0.80.0; further conv2d, max-pool, K-spill, model-level and debug-probe tests in the same tree are available outside the release run. The run recipe is in the 0.78.0 document and unchanged. | Borys Bradel |
| Sep-15 | **Debug tools: TT-Exalens call stack and step-by-step debugging** (AIIPSW-52) | Quasar | 🟡 Owner-verified | `rocket_step_test.py` and `rocket_callstack_test/rocket_callstack_test.py` on branch `adjordjevic/release_testing` of `tenstorrent/tt-exalens`, owner-run on Quasar simulation. Validated against tt-exalens 0.3.31. | Aleksandar Đorđević |
| Sep-15 | **Debug tools: Dynamic visualizer / NPE support** (AIIPSW-54) | Quasar | 🟡 Owner-verified | SoC-descriptor dump: `pytest tests/tt_metal/tools/profiler/test_device_profiler.py::test_noc_event_profiler`, which runs in CI on Wormhole and Blackhole silicon. Visualizer compatibility with the new SoC-descriptor information in the timeline file was checked by hand. Validated against ttnn-visualizer 0.101.0. | Sohaib Nadeem, Denis Kartashevsky |
| Sep-15 | **LLK: Qwen3-VL-2B LLK features** (AIIPSW-53) | Quasar |  | No LLK test recorded for this feature. A TTNN op suite for Qwen3-VL is in the v0.80.0 tree at `models/experimental/ops/quasar/tests/qwen3_vl_ops/`, outside the release run. | Filip Vranic |
| Sep-15 | **Runtime: Port basic runtime functionality from Quasar to Trinity** (AIIPSW-55) | Trinity | 🟡 Owner-verified | Three `unit_tests_legacy` tests, owner-run on the Trinity emulator (2026-09-02): `QuasarMeshDeviceSingleCardFixture.SingleDmL1Write`, `QuasarMeshDeviceSingleCardFixture.DmLoopback`, `AnyDispatchMeshDeviceSingleCardFixture.Bmm`. | Erik Tomusk |
| Sep-15 | **TTNN / Kernel Ops: Port ResNet kernel ops from Quasar to Trinity** (AIIPSW-26) | Trinity |  | ResNet was reported passing on the Trinity emulator on 2026-10-09. Test details to follow. | Borys Bradel |
| Sep-15 | **LLK: YOLO LLK API INT8 on Trinity** (AIIPSW-32) | Trinity |  | No test recorded for this release. | Filip Vranic |
| Sep-15 | **Debug tools: TT-Exalens on Trinity** (AIIPSW-30) | Trinity |  | TT-Exalens needs no Trinity-specific features: its Tensix and Rocket core access, already supported for Quasar, covers Trinity. The AIIPSW-52 tests apply. | Aleksandar Đorđević |
| Oct-15 | **TTNN / Kernel Ops: Llama 3.2 1B kernel ops** (AIIPSW-7) | Quasar | 🟡 Owner-verified | 36 graph-op test files in `models/experimental/llama32_1b_quasar/tests/graph_ops/`, owner-run one file at a time with slow dispatch and forced JIT; each passes or is skipped on purpose. Ten of them also ran on the Quasar emulator at `2x3` on 2026-10-02 and 2026-10-03 (7 passed, 3 skipped on purpose). `tests/demos/llama32_1b/test_llama_e2e.py` passed on main commit `9d423e4bb6f` on 2026-10-02 (1 h 36 min; model weights required). Seven module suites and 45 isolated op tests in the same tree are available outside the release run. Commands are given below. | Borys Bradel |
| Oct-15 | **LLK: Port LLK from Quasar to Trinity (4-row math)** (AIIPSW-33) | Trinity | 🟡 Owner-verified | Seven LLK suites, owner-run on the Quasar and Trinity emulators with `QUASAR_ARCH_VARIANT=quasar_4row`: `test_eltwise_binary_sfpu_quasar.py`, `test_matmul_quasar.py`, `test_matmul_custom_no_mop_quasar.py`, `test_reduce_quasar.py`, `test_unary_broadcast_quasar.py`, `perf_reduce_quasar.py`, `test_quasar_stall_wrapper_codegen.py`. The full LLK list on the Trinity emulator will follow. | Filip Vranic |
| Oct-15 | **Runtime: ATT / flat address, use cases UA and UB** (AIIPSW-56) | Horizon | 🟡 Owner-verified | `unit_tests_noc --gtest_filter='QuasarAtt*'`: 47 tests in `tests/tt_metal/tt_metal/noc/test_quasar_att_{address,config,image}.cpp`, including the 4 `QuasarAttHorizonConfig` tests; `unit_tests_api --gtest_filter='SimulatorBuildDir.*'` (the 2 tests of `tests/tt_metal/tt_metal/api/test_simulator_target_devices.cpp`). On the Horizon emulator with the split build: `*SingleDmL1Write*`, `*DmLoopback*` and `*MeshDeviceSingleCardFixture.Bmm` over the whole grid and each column of the split, host and flat addresses, ATT on and off. All owner-run. | Nenad Bunčić |
| Oct-15 | **Runtime: Overlay registers** (AIIPSW-57) | Horizon, Trinity | 🟡 Owner-verified | Horizon: every Horizon release entry runs with the Horizon register map (`TT_METAL_QUASAR_VARIANT=horizon`); 61 of 62 passed in the release pipeline (the open failure is the ResNet50 end-to-end test). `*SingleDmL1Write*` with that variant also owner-run. Trinity: cache-controller register tests, FDS tests, and fast dispatch with `TT_METAL_QUASAR_VARIANT=trinity` (falls back to NOC go signalling), owner-run. Trinity test details to follow. | Arik Yaacob |

### Platform tests not tied to one requirement

37 entries on the Quasar and Horizon emulators support every feature and map
to no single requirement. All passed:
`unit_tests_legacy --gtest_filter=*QuasarMultiCQMeshDeviceSingleCardFixture*`
(✅ `2x3`, not run on Horizon); 14 `unit_tests_debug_tools` entries at `1x3`
(watcher asserts, stack usage, waypoints, tile-counter log, out-of-bounds
argument detection, and the three `DevicePrintOutputFixture.PrintConcurrent*`
tests; ✅ on both platforms); 19 `unit_tests_api` DFB implicit-sync and
override tests (18 at `1x3`, one at `2x3`; ✅ on both); `unit_tests_data_movement`
`*QuasarAddrgenOps*` and `*QuasarIm2ColOps*` (✅ `1x3`, both platforms);
`unit_tests_device` `*SimulatorFixture*QuasarIoWindowReadWrite*` (✅ `1x3`,
both platforms).

## Oct-15 test detail

### AIIPSW-7 — Llama 3.2 1B kernel ops

Run each graph-op file on its own. Every file either passes or is skipped on
purpose.

```
TT_METAL_SLOW_DISPATCH_MODE=1 MESH_DEVICE=N150 TT_METAL_FORCE_JIT_COMPILE=1 pytest models/experimental/llama32_1b_quasar/tests/graph_ops/<file>
```

The 36 files: `test_add.py`, `test_concat.py`, `test_concat_small_grid.py`,
`test_embedding.py`, `test_embedding_small_grid.py`,
`test_interleaved_to_sharded.py`, `test_linear.py`, `test_linear_small_grid.py`,
`test_minimal_matmul.py`, `test_multiply.py`, `test_nlp_concat_heads.py`,
`test_nlp_concat_heads_decode.py`, `test_nlp_concat_heads_decode_small_grid.py`,
`test_nlp_create_qkv_heads.py`, `test_nlp_create_qkv_heads_decode.py`,
`test_paged_fill_cache.py`, `test_paged_scaled_dot_product_attention_decode.py`,
`test_paged_scaled_dot_product_attention_decode_small_grid.py`,
`test_paged_update_cache.py`, `test_parse_argument.py`, `test_reshape.py`,
`test_rms_norm.py`, `test_rotary_embedding_llama.py`,
`test_scaled_dot_product_attention.py`,
`test_scaled_dot_product_attention_small_grid.py`,
`test_sharded_to_interleaved.py`, `test_sharded_to_interleaved_small_grid.py`,
`test_slice.py`, `test_tensor_getitem.py`, `test_to_device.py`,
`test_to_memory_config.py`, `test_transpose.py`, `test_typecast.py`,
`test_unsqueeze_to_4D.py`, `test_untilize.py`, `test_untilize_small_grid.py`.

Ten of these files also ran on the Quasar emulator at `2x3` (tt-umd-simulators
pipelines 1139952 on 2026-10-02 and 1142305 on 2026-10-03, tt-metal commits
`a269610ed1d6` and `70222e907f97`). Passed: `test_embedding_small_grid`,
`test_interleaved_to_sharded`, `test_to_device`, `test_transpose`,
`test_nlp_concat_heads_decode_small_grid`, `test_nlp_create_qkv_heads_decode`,
`test_paged_update_cache`. Skipped on purpose: `test_concat`,
`test_nlp_concat_heads_decode`, `test_paged_scaled_dot_product_attention_decode`.

The end-to-end model test, one layer and one decode step, needs the model
weights on the host:

```
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LLAMA32_1B_TF_MAX_DECODE_STEPS=1 LLAMA32_1B_DEMO_NUM_LAYERS=1 MESH_DEVICE=N150 \
pytest models/experimental/llama32_1b_quasar/tests/demos/llama32_1b/test_llama_e2e.py
```

It passed on main commit `9d423e4bb6f` on 2026-10-02 in 1 h 36 min
(`test_llama_e2e[performance-N150]`).

### AIIPSW-33 — LLK port from Quasar to Trinity

Trinity uses the Quasar LLK with the four-row math variant. Select it with:

```
export QUASAR_ARCH_VARIANT=quasar_4row
```

The variable name stays `QUASAR_ARCH_VARIANT` for Trinity. It selects the
headers under `tt_metal/tt-llk/tt_llk_quasar/arch/quasar_4row/`, which are in
the v0.80.0 tree. The seven suites listed in the table were run by the owner
on the Quasar and Trinity emulators. The full LLK list on Trinity will follow.

### AIIPSW-56 — ATT / flat address on Horizon

```
./build/test/tt_metal/unit_tests_noc --gtest_filter='QuasarAtt*'
./build/test/tt_metal/unit_tests_api --gtest_filter='SimulatorBuildDir.*'
```

The first command runs 47 tests in three files under
`tests/tt_metal/tt_metal/noc/`. The second runs the two simulator-build tests
(`CPU_SingleChipBuildOpensChipZero`, `CPU_PartitionedBuildLeavesTheDevicesToUmd`).
The Horizon matrix of `*SingleDmL1Write*`, `*DmLoopback*` and
`*MeshDeviceSingleCardFixture.Bmm` (whole grid and each column of the split,
host and flat addresses, ATT on and off) was run by the owner on the split
build of the Horizon emulator.

### AIIPSW-57 — Overlay registers

On Horizon the evidence is the Horizon Release run itself: every entry runs
with `TT_METAL_QUASAR_VARIANT=horizon`, and 61 of 62 passed. On Trinity the
owner ran the cache-controller register tests, the FDS tests and fast dispatch
with `TT_METAL_QUASAR_VARIANT=trinity`. Trinity test details will follow.

## Notes on this release

- **All fast dispatch tests are green.** All 49 `2x3_DISPATCH` entries (47
  gtest entries and 2 profiler tests) and the 5 `2x3` gtest entries passed on
  the Quasar emulator in the release run.
- **One failure is open.** On the Horizon emulator, `test_resnet50_e2e` gives
  a different top-1 class in each run (549, then 783; golden 610) with a
  final-logits PCC above 0.99. The first op with a different output is op011,
  `layer1_module2.conv2`. It is under investigation by the owners of
  `models/demos/vision/classification/resnet50/quasar/`. The same test passes
  on the Quasar emulator.
- **AIIPSW-15 is verified in the tt-umd-horizon repository.** Its "Release
  Horizon" workflow passed all six suites on 2026-09-21. tt-umd-horizon is a
  separate repository released on its own cadence.
- **LLK coverage in the release pipeline grows from the next release.** Twelve
  `unit_tests_llk` gtest entries (SFPU unary, binary and ternary, typecast,
  tilize and untilize, transpose, reduce, binary compute, matmul, copy block,
  cumsum and the `LLKQuasarMeshDeviceSingleCardFixture` suite) were added to
  `quasar_regression_tests.yaml` by PR 59926, merged 2026-10-14.
- **Debug-tool deliverables ship outside tt-metal.** TT-Exalens,
  TTNN-Visualizer and TT-NPE are released as separate packages. The Sep-15
  rows reference the versions they were validated against, tt-exalens 0.3.31
  and ttnn-visualizer 0.101.0, rather than bundling them.
