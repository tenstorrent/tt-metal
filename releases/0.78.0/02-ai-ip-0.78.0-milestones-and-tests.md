<!--
AI-IP SW milestones and tests, cumulative through the Sep-15 release (v0.78.0).

Provenance differs by milestone, and that matters when checking this document:

  Jul-15 and Aug-15 rows
    Carried forward unchanged from the published 0.76.0 document, which was a
    verbatim copy of Confluence space AIIP, page 2664202491 ("AI IP SW
    Milestones and Tests"), last modified 2026-08-06.

  Sep-15 rows
    Compiled from the Jira feature tickets under epic AIIPSW-10 (Sept-15
    Release). The Confluence page was NOT updated for this release -- it still
    holds only the Jul-15 and Aug-15 rows. On 2026-09-01 the collection of PR
    and test evidence was moved into the individual Jira tickets, so those
    tickets are the source for Sep-15. Confluence should be reconciled to this
    document, or retired as the source of truth.

Known gaps, carried from the sources rather than omitted:
  - AIIPSW-9  (LLK: ResNet on Horizon)    - still "to be filled out"
  - AIIPSW-53 (Qwen3-VL-2B LLK features)  - no test evidence supplied

Out of scope for Sep-15, so not listed: AIIPSW-26 (Trinity port of ResNet
Kernel Ops) and AIIPSW-55 (Trinity runtime basic functionality). Neither
landed in v0.78.0.

This document is cumulative by design: it lists the tests for every AI-IP
milestone to date, not only the ones new in this release.
-->

# AI IP SW Milestones and Tests

Cumulative test inventory for the AI-IP software deliverables, covering the
Jul-15, Aug-15 and Sep-15 milestones.

## Coverage summary

| Milestone | Release | Requirements | With tests recorded | Outstanding |
| --- | --- | --- | --- | --- |
| Jul-15 | — | 3 | 2 | 1 (AIIPSW-9) |
| Aug-15 | v0.76.0 | 5 | 5 | 0 |
| Sep-15 | v0.78.0 | 4 | 3 | 1 (AIIPSW-53) |
| **Total** | | **12** | **10** | **2** |

Of the three Sep-15 requirements with tests recorded, **AIIPSW-54 has no Quasar
test** - the automated test supplied for it runs on Wormhole or Blackhole only.
Counting only Quasar coverage, Sep-15 is 2 of 4.

## Milestones

| **Milestone** | **Requirement (ticket)** | **Existing Quasar tests** | **Team** | **Manager (escalation)** |
| --- | --- | --- | --- | --- |
| Jul-15 | **Runtime: FD support** (AIIPSW-2) | `DmLoopback`, `QuasarComputeKernelMultipleThread`, `MultiDmAddTwoInts` (fast-dispatch) | Trinity | [@Kevin Stevens](https://tenstorrent.enterprise.slack.com/team/U093BKF1FF1) |
| Jul-15 | **LLK: Yolo LLK API** (AIIPSW-3) | Consolidated LLK tests under `tt_metal/tt-llk/tests/` (one file per op family, extended per kernel): `test_eltwise_unary_sfpu_quasar.py`, `test_eltwise_binary_sfpu_quasar.py`, `eltwise_unary_sfpu_quasar_test.cpp`, `eltwise_binary_sfpu_quasar_test.cpp`, `sfpu_topk_quasar_test.cpp`, `unpack_tilize_quasar_test.cpp`, `reduce_quasar_test.cpp`. YOLO8-op coverage added in PR 51478. | Trinity | [@Filip Vranic](https://tenstorrent.enterprise.slack.com/team/U08AU2A435Z) |
| Jul-15 | **LLK: ResNet on Horizon** (AIIPSW-9) | _to be filled out_ (Horizon env, not Quasar) | Horizon | [@Filip Vranic](https://tenstorrent.enterprise.slack.com/team/U08AU2A435Z) |
| Aug-15 | **TTNN/Kernel Ops: ResNet** (AIIPSW-4) | `models/demos/vision/classification/resnet50/quasar/tests/ops/` (45 op tests). Passing: `test_untilize_with_unpadding`, `test_to_memory_config`, `test_to_layout`, `test_tilize`, `test_reshape`, `test_reshape_tiled`, `test_slice_write`, `test_sharded_to_interleaved`, `test_reallocate`, `test_padded_slice`. Not yet supported: `test_conv2d*`, `test_max_pool2d*`/`test_avg_pool2d`, `test_linear`. Run: slow dispatch, watcher on (`TT_METAL_WATCHER=10`), NoC-sanitize disabled, serialized ops. | Trinity | [@Borys Bradel](https://tenstorrent.enterprise.slack.com/team/U084B1CES7M) |
| Aug-15 | **Runtime: FD for dispatch engine** (AIIPSW-6) | `TensixSingleCoreDirectDramReaderDatacopyWriter`, `2x3_DISPATCH` config, `QuasarCRTA*` | Trinity | [@Kevin Stevens](https://tenstorrent.enterprise.slack.com/team/U093BKF1FF1) |
| Aug-15 | **Quant/dequant · SFPU LLK** (AIIPSW-8 INT8) | - SFPU (unary/binary), consolidated: `test_eltwise_unary_sfpu_quasar.py`, `test_eltwise_binary_sfpu_quasar.py` (+ `eltwise_unary_sfpu_quasar_test.cpp`, `eltwise_binary_sfpu_quasar_test.cpp`) — extended for INT8 add_int/mul_int and quant/requant/dequant (PR 48208). INT8→INT32 reduce: `reduce_quasar_test.cpp` (PR 49390). Shared helpers extended: `helpers/golden_generators.py`, `helpers/include/sfpu_operations_quasar.h`, `helpers/sfpu_domains.py`. | Trinity | [@Filip Vranic](https://tenstorrent.enterprise.slack.com/team/U08AU2A435Z) |
| Aug-15 | **FPU / tilize — 32-bit dest mode** | - Consolidated FPU tests, extended from 16-bit to 32-bit dest: `unpack_tilize_quasar_test.cpp`, `pack_untilize_quasar_test.cpp`, `reduce_quasar_test.cpp`, `transpose_dest_quasar_test.cpp`, `matmul_quasar_test.cpp`, `pack_l1_acc_quasar_test.cpp` (e.g. 32-bit dest TilizeA_B, PR 49579). | Trinity | [@Filip Vranic](https://tenstorrent.enterprise.slack.com/team/U08AU2A435Z) |
| Aug-15 | **Profiler** (AIIPSW-13) | Quasar LLK performance-test suite: eltwise binary (+broadcast, reuse_dest), unary broadcast/datacopy, pack / pack_l1_acc / pack_untilize, unpack_tilize / unpack_unary_operand, reduce, and transpose_dest (PRs 50584–50596); unary and binary SFPU tests (PR 51325); and the Quasar cache-write benchmark (PR 50898). | Trinity | [@Filip Vranic](https://tenstorrent.enterprise.slack.com/team/U08AU2A435Z) |
| Sep-15 | **TTNN/Kernel Ops: Quasar ResNet with conv2D, pool and linear** (AIIPSW-16) | End-to-end model test on a two-compute-node (2x3) grid, plus the individual op tests used for debugging. `models/demos/vision/classification/resnet50/quasar/tests/ops/` now holds **50** op tests, up from 45 at v0.76.0. Commands and environment are given in full below. Verified on `main` at commit `5c73430ac16`. | Trinity | [@Borys Bradel](https://tenstorrent.enterprise.slack.com/team/U084B1CES7M) |
| Sep-15 | **Debug tools: Exalens advanced debugging** (AIIPSW-52) | `rocket_step_test.py` (step-by-step debugging) and `rocket_callstack_test/rocket_callstack_test.py` (call-stack retrieval), both in `tenstorrent/tt-exalens` on branch `adjordjevic/release_testing`. See the note on test location below. | Debug Tools | [@Aleksandar Đorđević](https://tenstorrent.enterprise.slack.com/team/U082G4QEVGV) |
| Sep-15 | **LLK: Qwen3-VL-2B related LLK features** (AIIPSW-53) | _to be filled out._ No PRs and no test evidence supplied at the v0.78.0 cut-off. Qwen3-VL op tests for Quasar do exist (PRs 54588, 54625) but were not offered as evidence for this requirement. | Trinity | [@Filip Vranic](https://tenstorrent.enterprise.slack.com/team/U08AU2A435Z) |
| Sep-15 | **Debug tools: dynamic visualizer / NPE support** (AIIPSW-54) | **Not Quasar coverage - see the note below.** SoC-descriptor dump: `pytest tests/tt_metal/tools/profiler/test_device_profiler.py::test_noc_event_profiler`, which runs on Wormhole or Blackhole only. Visualizer compatibility with the new SoC-descriptor information in the timeline file was verified manually, not by an automated test. | Debug Tools | [@Sohaib Nadeem](https://tenstorrent.enterprise.slack.com/team/U08M5PK2492) · [@Denis Kartashevsky](https://tenstorrent.enterprise.slack.com/team/U05BL8X4BED) |

## Sep-15 test detail

### AIIPSW-16 — Quasar ResNet Kernel Ops with conv2D, pool and linear

End-to-end model test on a two-compute-node grid (2x3). Takes at least ~25 minutes.

**Use the recipe documented in the test itself.** `test_resnet50_e2e.py` carries a
`REQUIRES` note: until the K-spill `0x10000` hazard has its full fix, the run needs
the DPRINT mask on, so that the K-spill matmul's `mm_partials` wait-then-pop hazard
stays masked. The `copy_tile` interpose is only a partial fix. Its own run recipe is:

```
unset TT_METAL_LLK_ASSERTS
TT_METAL_DPRINT_CORES=all TT_METAL_QSR_CONV_SPLIT_PROGRAM=1 TT_METAL_FORCE_JIT_COMPILE=1 \
TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false, "enable_logging": true}' \
pytest -q models/demos/vision/classification/resnet50/quasar/tests/test_resnet50_e2e.py
```

The command supplied as release evidence was the shorter form below. It omits the
DPRINT mask, the Quasar split path, forced JIT and the fast-runtime-mode override,
so it does not reproduce the validated setup and can hit the K-spill hazard. It is
recorded for traceability; prefer the recipe above:

```
RESNET_PCC_LOG=1 TT_METAL_SLOW_DISPATCH_MODE=1 pytest -q models/demos/vision/classification/resnet50/quasar/tests/test_resnet50_e2e.py::test_resnet50_e2e[pretrained-device_params0]
```

Individual op tests used for debugging, additional to those already listed for
AIIPSW-4 in the Aug-15 milestone:

```
TT_METAL_SLOW_DISPATCH_MODE=1 pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_linear.py

TT_METAL_SLOW_DISPATCH_MODE=1 pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_avg_pool2d.py
TT_METAL_SLOW_DISPATCH_MODE=1 pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_global_avgpool.py

TT_METAL_QSR_CONV_SPLIT_PROGRAM=1 TT_METAL_SLOW_DISPATCH_MODE=1 pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_conv2d_stem.py
TT_METAL_SLOW_DISPATCH_MODE=1 pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_conv2d_layer3_downsample_split.py
TT_METAL_SLOW_DISPATCH_MODE=1 pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_conv2d_layer_conv2_modelcfg.py

TT_METAL_SLOW_DISPATCH_MODE=1 pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_max_pool2d.py --timeout=1200
TT_METAL_SLOW_DISPATCH_MODE=1 pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_max_pool2d_1x3.py
TT_METAL_SLOW_DISPATCH_MODE=1 pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_max_pool2d_dprint_debug.py --timeout=1200
TT_METAL_SLOW_DISPATCH_MODE=1 pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_max_pool2d_strided_reduce.py
```

Environment used for the runs above. Several watcher variables have no effect
while the watcher is disabled; they are recorded as supplied:

```
TT_METAL_WATCHER_DUMP_ALL=1
TT_METAL_WATCHER_NOINLINE=1
TT_METAL_DPRINT_ONE_FILE_PER_RISC=1
TT_METAL_ENV=dev
TT_METAL_WATCHER_DISABLE_RING_BUFFER=1
TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1
TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1
TT_METAL_SLOW_DISPATCH_MODE=1
TT_METAL_HOME=<path to your tt-metal checkout>
TT_METAL_LOGGER_LEVEL=DEBUG
TT_METAL_SIMULATOR=<path to your tt-umd-simulators build>/emu-quasar-2x3
TT_METAL_WATCHER_DISABLE_PAUSE=1
```

### AIIPSW-52 — Exalens advanced debugging

The two tests are held on the branch `adjordjevic/release_testing` in
`tenstorrent/tt-exalens`, not on that repository's default branch:

- [`rocket_step_test.py`](https://github.com/tenstorrent/tt-exalens/blob/adjordjevic/release_testing/rocket_step_test.py) — step-by-step debugging
- [`rocket_callstack_test/rocket_callstack_test.py`](https://github.com/tenstorrent/tt-exalens/blob/adjordjevic/release_testing/rocket_callstack_test/rocket_callstack_test.py) — call-stack retrieval

TT-Exalens is versioned and released as its own Python package. This release was
validated against **tt-exalens 0.3.31**, available from
[PyPI](https://pypi.org/project/tt-exalens/); install a specific version with
`pip install tt-exalens==0.3.31`.

## Notes on this release

- **Debug-tool deliverables ship outside tt-metal.** TT-Exalens, TTNN-Visualizer
  and TT-NPE are released as separate packages on their own cadence. This release
  references the versions it was validated against — tt-exalens 0.3.31 and
  ttnn-visualizer 0.101.0 — rather than bundling them.
- **One test in this release is manual.** Visualizer compatibility with the new
  SoC-descriptor timeline information (AIIPSW-54) was checked by hand. It has no
  automated coverage yet.
- **AIIPSW-54 has no Quasar test.** The automated test supplied for the
  SoC-descriptor dump, `test_noc_event_profiler`, asserts that `ARCH_NAME` is one
  of `grayskull`, `wormhole_b0` or `blackhole`, so it cannot run on Quasar. The
  feature itself is architecture-neutral and the test does validate the descriptor
  it emits, but Quasar coverage for this requirement is still outstanding.
