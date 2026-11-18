# gpt-oss on a 2-core Quasar grid (craq-sim): results

This report covers the gpt-oss op test suite, `models/experimental/ops/quasar/tests/gpt_oss_ops/` (119 cases in 38 files), on a Quasar device limited to 2 worker cores, the `_small_grid.py` companion tests written for the cases that fail there, and a run of the gpt-oss demo on 2 cores. The report on the full 8x4 device is [gpt_oss_ops_analysis.md](gpt_oss_ops_analysis.md); it defines the problem letters used here (G, L).

## How the grid was limited

`TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="3,2"` sets the last worker core of the compute grid to logical (3, 2) ([core_descriptor.cpp:202-234](../../../../../tt_metal/llrt/core_descriptor.cpp#L202-L234)). The first worker core stays where the core descriptor puts it: (2, 2) on the 8x4 craq-sim device ([quasar_simulation_8x4_arch.yaml:11-13](../../../../../tt_metal/core_descriptors/quasar_simulation_8x4_arch.yaml#L11-L13)). The grid is therefore (3 − 2 + 1) x (2 − 2 + 1) = 2x1 cores. On a simulator the override also applies in slow dispatch ([core_descriptor.cpp:241-263](../../../../../tt_metal/llrt/core_descriptor.cpp#L241-L263)). Every log of the suite run (119 of 119) and of the 2x1 companion runs contains "Overrode compute_with_storage_end".

A 2x1 grid is the grid of the 2-core Quasar emulator ([quasar_simulation_2x3_arch.yaml:11-13](../../../../../tt_metal/core_descriptors/quasar_simulation_2x3_arch.yaml#L11-L13)). The Quasar llama tests use the same override, for example [test_quasar_matmul_2d_single_row.py:15-26](../../../llama32_1b_quasar/tests/debug_ops/test_quasar_matmul_2d_single_row.py#L15-L26).

## Setup

The suite and the companion tests (2026-11-12):

- Branch `fplavec/gpt_oss_quasar_bringup_2` on `main` `737452e6d8f`, with the four PRs of run 10 of the main report applied as uncommitted changes (#64497, the own commit of #63905, #64567, and #59572 with a fix to 3 of its lines), and the model workarounds for G. This is the tree on which the gpt-oss demo passed with every op on the 8x4 device.
- craq-sim built on 2026-10-30 from its `quasar` branch at `450de768`, with 3 GiB of DRAM per channel.
- bfloat16 for the block float cases (`TTNN_GRAPH_OPS_BF8_AS_BF16=1`), one pytest process per case, 15 min per case.
- Run of the suite (run 11, 2026-11-12, 98 min): `/localdev/fplavec/temp/gpt_oss_ops_runs/20261112_run11_grid3x2/suite/` (`summary.tsv` and one log per case). Logs of the companion tests: `small_grid/<op>/` in the folder above it.

The work on L and the demo run (2026-11-17 and 2026-11-18):

- The same branch on `main` `afff2ea28a6`, which has #64497, #64567 and #59572. On top of it, the runs had these local changes: the commit of #63905 itself (without the commits it merges from other branches), the sparse_matmul fix for L, the decode down projection change, and `import functools` in `models/common/moe/compute_helpers.py` (next sections).
- craq-sim rebuilt on 2026-11-17, with 3 GiB of DRAM per channel. On this `main`, every test stops at device open with "UnsupportedFunctionality: write_csr: qsr chicken_bits: data=0x8", because craq-sim rejects the write that #64588 added to turn off the TRISC data cache (bit 3 of CSR 0x7C0) (craq-sim#549). All runs on this tree set `TT_METAL_ENABLE_L1_DATA_CACHE_CORES=all`, the workaround that craq-sim#549 gives, or skipped the write with a temporary code change.
- Logs of the L work: `/localdev/fplavec/temp/gpt_oss_ops_runs/20261112_run11_grid3x2/small_grid/sparse_trace/`. Log of the demo: `/localdev/fplavec/temp/gpt_oss_ops_runs/20261117_demo_grid2x1/demo3.log`.

## Results of the suite on 2 cores

| Result | Run 10 (8x4) | Run 11 (2x1) |
|---|---|---|
| Passed | 114 | 106 |
| Failed | 3 | 9 |
| Timeout | 0 | 2 |
| Skipped | 2 | 2 |
| **Total** | **119** | **119** |

The 11 cases that do not pass on 2 cores are in 5 test files:

| Test cases | Error on 2 cores | Cause |
|---|---|---|
| `test_matmul[03_1x2880_bf8_int-l1]` and `[04_32x2880_bf8_int-dram]` | timeout after 15 min | Slow, not hung. These are the 262144-wide `lm_head` matmuls. Run alone on 2 cores, they pass in 39 min and 31 min (75 to 114 million simulated cycles, against 3 to 7 million on 8x4). |
| `test_nlp_concat_heads_decode[00_64x64_bf16_hs-l1]` | "Target number of cores 64 is greater than total number of available cores 2" | The op places one head on each core (G). It fails on 8x4 too. |
| `test_paged_scaled_dot_product_attention_decode[00_...]` and `[01_...]` | "Cores available (32) exceeds grid size (2)" | The captured program config asks for an 8x4 grid. |
| `test_scaled_dot_product_attention[00_...]` and `[01_...]` | "Provided grid must not contain more cores than the device. Got 32 cores, expected at most 2 cores." | The captured program config asks for an 8x4 grid. |
| `test_sparse_matmul[00_...]`, `[01_...]`, `[02_...]` and `[03_...]` | "Multicast kernel is placed outside the worker grid" (00, 02); "Out of Memory ... 5898240 B L1 buffer across 2 banks" (01); "No core coordinate found" (03) | The captured program configs ask for (3, 4) and (5, 6) grids; case 01 also keeps a 5.9 MB output in SRAM. |

The same 2 cases are skipped as on 8x4. Their captured SRAM shard grids are larger than the device.

## The `_small_grid.py` companion tests

Each companion is a manual file next to the generated test file, in the style of the companions of the Quasar llama suite (`models/experimental/llama32_1b_quasar/tests/graph_ops/`) and of the yolo suite (`models/experimental/ops/quasar/tests/yolo_ops/`). Its header says why the captured case fails on a small grid and what the companion changes. Each companion reuses the `CASES` of the generated file and changes only what the small grid forces. Where the gpt-oss model has a rule for small grids, the companion uses that rule, so that it tests the call that the model makes. A variant whose grid does not fit the device skips, so each file is safe on any device. Each file was run with the override (2x1) and without it (8x4).

| File | What it changes | 2x1 | 8x4 |
|---|---|---|---|
| [test_matmul_small_grid.py](../tests/gpt_oss_ops/test_matmul_small_grid.py#L4-L35) | Cases 03 and 04 at a weight width of 2048 instead of 262144; everything else as captured. On 2 cores, the narrow form of case 03 still takes the plain matmul (each core reads its own inputs), as the captured call does. On 8x4 it takes the 1D multicast matmul (one core reads the activations and sends the same data to the other cores), and the generated file covers the plain matmul at the captured width. | 2 passed, 43 s | 2 passed, 20 s |
| [test_nlp_concat_heads_decode_small_grid.py](../tests/gpt_oss_ops/test_nlp_concat_heads_decode_small_grid.py#L4-L34) | Two tests. The op itself with 2 heads, which fit 2 cores; the values are compared with the torch reference. And the captured 64-head case through `merge_heads_on_device`, the head merge that the model runs on a device with fewer cores than heads, with the expected output placement changed to interleaved DRAM. | 2 passed, 20 s | 2 passed, 18 s |
| [test_paged_scaled_dot_product_attention_decode_small_grid.py](../tests/gpt_oss_ops/test_paged_scaled_dot_product_attention_decode_small_grid.py#L4-L31) | Both cases with a 2x1 grid in the program config, the grid that the model passes on a 2x1 device ([attention/config.py:101-108](tt/attention/config.py#L101-L108)). | 2 passed, 68 s | 2 passed, 53 s |
| [test_scaled_dot_product_attention_small_grid.py](../tests/gpt_oss_ops/test_scaled_dot_product_attention_small_grid.py#L4-L26) | Both cases with a 2x1 grid, as the model passes for prefill ([attention/config.py:110-120](tt/attention/config.py#L110-L120)). | 2 passed, 191 s | 2 passed, 185 s |
| [test_sparse_matmul_small_grid.py](../tests/gpt_oss_ops/test_sparse_matmul_small_grid.py#L4-L51) | All 4 cases with the grid that the model's `fit_core_grid` picks for the device, and `per_core_N` (output width per core, in tiles) to match; case 01 with its output in DRAM. On a 1- or 2-core grid, every variant is marked `xfail` and does not run, because of L (next sections). | 4 xfailed, 6 s | 2 passed (01 and 03, on the model's (5, 3) grid), 2 skipped (00 and 02 need no change on 8x4), 10 min |

The scaled dot product attention (SDPA) companion and the paged SDPA decode companion check only the output shape, type, placement and that the values are finite: the suite has no torch reference for these ops, as for the generated files. The `sparse_matmul` companion has the same limit.

The five companion files were written on 2026-11-12 and, as of that date, had not run in CI.

## L on small grids: cause and fix

All 4 `sparse_matmul` cases stop craq-sim with the tile counter error of problem L on the only grids a 2-core device has (logs in `small_grid/sparse/`):

| Case | Grid | Error: occupancy / capacity / acked (posted = 0 in all) | Log |
|---|---|---|---|
| 00 | (2, 1) | 1502 / 60 / 64034 | `ovr_00.log` |
| 01 | (2, 1) | 1482 / 20 / 64054 | `ovr_01.log` |
| 02 | (2, 1) | 5918 / 60 / 59618 | `ovr_02.log` |
| 03 | (2, 1) | 5898 / 20 / 59638 | `ovr_03.log` |
| 00 | (1, 1) | 2974 / 60 / 62562 | `ovr_00_grid1x1.log` |
| 01 | (1, 1) | 2954 / 20 / 62582 | `ovr_01_grid1x1.log` |

The error message reads "UndefinedBehavior: qsr_tile_counter_check_error: tile counter occupancy=X exceeds capacity=Y (posted=0 acked=Z)". A DFB (dataflow buffer) is the circular buffer in SRAM that passes tiles from one kernel to the next. A tile counter is the hardware counter of a DFB: "posted" counts the tiles that the producer added, and "acked" the tiles that the consumer removed. Both are 16-bit and wrap at 65536. The failure depends on the grid, not on the override: on the full 8x4 device, case 01 forced onto (2, 1) fails with the same counts, and on (3, 1) it passes. On 8x4, the (6, 3) grid with the 128-row case 03 fails in the same way (the main report, L).

### Cause

The other matmul program factories (multicore, reuse-optimized, 1D and 2D multicast, DRAM-sharded) set `disable_dfb_implicit_sync_for_all = true` for their data movement kernels on Quasar, for example [matmul_multicore_reuse_mcast_1d_program_factory.cpp:3846-3869](../../../../../ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp#L3846-L3869). On `main` `afff2ea28a6`, the sparse_matmul factory sets nothing for Quasar on its in0 and in1 kernels (`in0_hw_config` and `in1_hw_config` have only `config_1xx`), so every DFB that these kernels use is in implicit sync, the default ([helpers.hpp:169-184](../../../../../tt_metal/impl/metal2_host_api/helpers.hpp#L169-L184)). In implicit sync, a kernel records its progress (`DFBImplicitProgress`) as it reads and writes, and at kernel exit the firmware tops up each counter to the count that this progress implies ([dataflow_buffer_finish.h:192-221](../../../../../tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer/dataflow_buffer_finish.h#L192-L221)). The sparse_matmul kernels fill their DFBs with explicit `reserve_back` and `push_back`, so their progress stays 0. The firmware then computes `behind = expected - posted` on 16 bits with `expected` = 0, and adds `behind` to "posted" when it is positive as a signed 16-bit number. When the producer has posted all its tiles, `behind` = 65536 − posted. This number is positive as a signed 16-bit number when "posted" (mod 65536) is more than 32768. The addition then wraps "posted" to exactly 0.

Evidence (case 01 on (2, 1), with a temporary DPRINT in `dfb_dm0_flush_side` and in the kernel exit code, logs `dp*_c01_grid2x1.*` in `small_grid/sparse_trace/`):

- At kernel exit, the progress of every DFB is 0 (`loop=0 pos=0 share=0`).
- For the in0 counter (capacity 20), the firmware sees `expected` = 0 and "posted" = 64064, and computes `behind` = 1472. 64064 is 129600 mod 65536, and 129600 is the number of tiles that craq-sim's counter trace (`TTSIM_QSR_DFB_COUNTER_TRACE=1`, `ctr2_grid2x1.log`) shows posted to this counter. 64064 + 1472 = 65536, so "posted" wraps to 0, and occupancy = 0 − 64054 mod 65536 = 1482, the error above.
- With this write skipped, case 01 on (2, 1) passes.
- On (3, 1), each counter gets 86400 tiles, 20864 mod 65536, so `behind` is negative and the firmware writes nothing. On (1, 1), each counter should get twice the tiles of (2, 1), 259200, which is 62592 mod 65536, and the case fails (this count is derived, not traced).

### Fix

With `.config_2xx = {.disable_dfb_implicit_sync_for_all = true}` on the in0 and in1 kernel configs of the sparse_matmul factory, as in the other matmul factories, these cases pass (logs `fix_*.log`):

| Case | Grid | Without the fix | With the fix |
|---|---|---|---|
| 01 | (2, 1) | tile counter error | passed, 4 min |
| 01 | (1, 1) | tile counter error | passed, 7 min |
| 03 | (6, 3) | tile counter error | passed, 8 min |

Cases 00, 02 and 03 on (2, 1) and case 00 on (1, 1) were not run with the fix. The fix runs forced the grid on the full 8x4 device, without the override. Every run with the fix also had #63905 applied. Without #63905 and without the fix, on `main` `afff2ea28a6`, the repro test of #66114 fails in the same way (`repro_main_*.log`), so the problem does not depend on #63905.

tt-metal issue #66114 tracks the problem, with a repro test that needs only files on `main`. The issue gives the repro test, to be saved as `models/experimental/ops/quasar/tests/gpt_oss_ops/test_sparse_matmul_repro.py`.

Until the fix is on `main`, the model excludes the (6, 3) grid for its expert matmuls (`excluded_grids` in `tt/mlp.py`), and the `sparse_matmul` companion marks 1- and 2-core grids `xfail`. Both can be removed once the fix is on `main`.

## The decode down projection on 2 cores

The decode down projection keeps its input in SRAM: its input comes from the gate and up projections, whose outputs are placed in SRAM ([experts/decode.py:102](tt/experts/decode.py#L102-L102) and [experts/decode.py:128](tt/experts/decode.py#L128-L128)). Input and output are each [1, 32, 1, 2880]. Tile layout pads the dimension of size 1 to 32 rows, so each tensor is 5.9 MB in bfloat16, which is 2.95 MB per core on 2 cores. Together they need 5.9 MB per core, and the SRAM bank of each core holds 3.86 MB. In `bfloat8_b`, which the model asks for, the two would just fit, but Quasar has no block float formats, and the model uses bfloat16 instead.

The model puts this output in DRAM when input and output together need more than half of each core's SRAM bank, and keeps it in SRAM otherwise ([experts/decode.py:14-28](tt/experts/decode.py#L14-L28)). On 8x4, the two need about 370 KB per bank, so the output stays in SRAM. On 2 cores, the output goes to DRAM, as the companion test does for case 01.

## The gpt-oss demo on 2 cores

With the sparse_matmul fix and the decode down projection change, the gpt-oss demo passes on the 2x1 grid with every op on the device:

```
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="3,2" TT_METAL_ENABLE_L1_DATA_CACHE_CORES=all QSR_N_LAYERS=2 pytest "models/experimental/ops/quasar/gpt_oss/demo/text_demo.py::test_gpt_oss_demo[quasar-1x1-prefill_128]"
```

The run used the first 2 decoder layers, as the 8x4 run of the main report, so the generated text has no meaning. It decoded 3 tokens instead of 200: a pytest plugin used for the run, not in the repository, sets `max_generated_tokens`. The run took 3 h 34 min (log `demo3.log`).

| Step | 2x1 | 8x4 (main report) |
|---|---|---|
| Setup and weight loading | 3 min | 4 min |
| Warmup prefill (128 tokens) | 92 min | 51 min |
| Prefill (128 tokens) | 92 min | 52 min |
| Decode | 491 s per token (last 2 of 3 tokens; the first took 573 s, including kernel compiles) | 349 s per token (200 tokens) |

The demo also needs `import functools` in `models/common/moe/compute_helpers.py`. Without it, the demo stops with "NameError: name 'functools' is not defined" before it runs any op, on every architecture. `main` has had this bug since #64223 (merged 2026-11-13), and PR #65742 (a draft since 2026-11-16) adds the import.

## Open problems

### The `lm_head` matmuls are slow on 2 cores

Each of the two captured `lm_head` matmul calls takes 31 to 39 min on a 2-core grid on craq-sim, against 6.5 to 7 min on 8x4. These times include test setup and kernel compile. The generated cases 03 and 04 need a timeout of at least 45 min per case there, or the suite should skip them on small grids and run the companion instead.

## Summary

On a 2-core grid, 106 of the 119 cases pass. With the 5 companion files, every failing case has a small-grid form that passes, except `sparse_matmul`, which stops the simulator on 1- and 2-core grids because of L. L comes from the sparse_matmul factory, which leaves implicit sync on for kernels that use explicit sync; at kernel exit, the firmware then adds tiles to a counter that already holds all of them. With implicit sync turned off, as in the other matmul factories, the 3 tested cases pass (case 01 on (2, 1) and (1, 1), case 03 on (6, 3); #66114). With that fix and the DRAM output for the decode down projection, the gpt-oss demo (2 layers) passes on 2 cores.

Next steps:
1. Fix L on `main` (#66114). Then remove the (6, 3) exclusion from `tt/mlp.py` and the `xfail` from the `sparse_matmul` companion, and run cases 00, 02 and 03 on (2, 1) and case 00 on (1, 1).
2. Get the DRAM output for the decode down projection reviewed and merged.
3. Skip the generated `lm_head` cases on small grids, or give them a longer timeout.
