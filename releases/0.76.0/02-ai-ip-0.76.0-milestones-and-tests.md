<!--
Verbatim copy. Source: Confluence, space AIIP (AI IP), page ID 2664202491
URL:      https://tenstorrent.atlassian.net/wiki/spaces/AIIP/pages/2664202491/AI+IP+SW+Milestones+and+Tests
Author:   Dimitri Gnidash
Modified: 2026-08-06
Retrieved: 2026-08-13
Reproduced as authored; no content changes.
Known gap, carried from the source: AIIPSW-9 (LLK: ResNet on Horizon) reads
"to be filled out". No test data had been supplied for that milestone as of the
2026-08-06 snapshot -- it is an open item with the Horizon team, not an omission
in transcription.
-->

# AI IP SW Milestones and Tests

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
