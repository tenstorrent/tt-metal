# gpt-oss op tests on a 2-core Quasar grid (craq-sim): results

This report covers the gpt-oss op test suite, `models/experimental/ops/quasar/tests/gpt_oss_ops/` (119 cases in 38 files), on a Quasar device limited to 2 worker cores, and the `_small_grid.py` companion tests written for the cases that fail there. The report on the full 8x4 device is [gpt_oss_ops_analysis.md](gpt_oss_ops_analysis.md); it defines the problem letters used here (G, L).

## How the grid was limited

`TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="3,2"` sets the last worker core of the compute grid to logical (3, 2) ([core_descriptor.cpp:202-234](../../../../../tt_metal/llrt/core_descriptor.cpp#L202-L234)). The first worker core stays where the core descriptor puts it: (2, 2) on the 8x4 craq-sim device ([quasar_simulation_8x4_arch.yaml:11-13](../../../../../tt_metal/core_descriptors/quasar_simulation_8x4_arch.yaml#L11-L13)). The grid is therefore (3 − 2 + 1) x (2 − 2 + 1) = 2x1 cores. On a simulator the override also applies in slow dispatch ([core_descriptor.cpp:241-263](../../../../../tt_metal/llrt/core_descriptor.cpp#L241-L263)). Every log of the suite run (119 of 119) and of the 2x1 companion runs contains "Overrode compute_with_storage_end".

A 2x1 grid is the grid of the 2-core Quasar emulator ([quasar_simulation_2x3_arch.yaml:11-13](../../../../../tt_metal/core_descriptors/quasar_simulation_2x3_arch.yaml#L11-L13)). The Quasar llama tests use the same override, for example [test_quasar_matmul_2d_single_row.py:15-26](../../../llama32_1b_quasar/tests/debug_ops/test_quasar_matmul_2d_single_row.py#L15-L26).

## Setup

- Branch `fplavec/gpt_oss_quasar_bringup_2` on `main` `737452e6d8f`, with the four PRs of run 10 of the main report applied as uncommitted changes (#64497, the own commit of #63905, #64567, and #59572 with a fix to 3 of its lines), and the model workarounds for G. This is the tree on which the gpt-oss demo passed with every op on the 8x4 device.
- craq-sim built on 2026-10-30 from its `quasar` branch at `450de768`, with 3 GiB of DRAM per channel.
- bfloat16 for the block float cases (`TTNN_GRAPH_OPS_BF8_AS_BF16=1`), one pytest process per case, 15 min per case.
- Run of the suite (run 11, 2026-11-12, 98 min): `/localdev/fplavec/temp/gpt_oss_ops_runs/20261112_run11_grid3x2/suite/` (`summary.tsv` and one log per case). Logs of the companion tests: `small_grid/<op>/` in the folder above it.

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
| [test_sparse_matmul_small_grid.py](../tests/gpt_oss_ops/test_sparse_matmul_small_grid.py#L4-L51) | All 4 cases with the grid that the model's `fit_core_grid` picks for the device, and `per_core_N` (output width per core, in tiles) to match; case 01 with its output in DRAM. On a 1- or 2-core grid, every variant is marked `xfail` and does not run, because of L (next section). | 4 xfailed, 6 s | 2 passed (01 and 03, on the model's (5, 3) grid), 2 skipped (00 and 02 need no change on 8x4), 10 min |

The scaled dot product attention (SDPA) companion and the paged SDPA decode companion check only the output shape, type, placement and that the values are finite: the suite has no torch reference for these ops, as for the generated files. The `sparse_matmul` companion has the same limit.

The five companion files were written on 2026-11-12 and, as of that date, had not run in CI.

## Open problems

### L on 2 cores: the expert `sparse_matmul` cannot run on a 1- or 2-core grid

All 4 `sparse_matmul` cases stop craq-sim with the tile counter error of problem L, on the only grids a 2-core device has:

| Case | Grid | Error: occupancy / capacity / acked (posted = 0 in all) | Log in `small_grid/sparse/` |
|---|---|---|---|
| 00 | (2, 1) | 1502 / 60 / 64034 | `ovr_00.log` |
| 01 | (2, 1) | 1482 / 20 / 64054 | `ovr_01.log` |
| 02 | (2, 1) | 5918 / 60 / 59618 | `ovr_02.log` |
| 03 | (2, 1) | 5898 / 20 / 59638 | `ovr_03.log` |
| 00 | (1, 1) | 2974 / 60 / 62562 | `ovr_00_grid1x1.log` |
| 01 | (1, 1) | 2954 / 20 / 62582 | `ovr_01_grid1x1.log` |

- The error message reads "UndefinedBehavior: qsr_tile_counter_check_error: tile counter occupancy=X exceeds capacity=Y (posted=0 acked=Z)". A DFB (dataflow buffer) is the circular buffer in SRAM that passes tiles from one kernel to the next. A tile counter is the hardware counter of a DFB: "posted" counts the tiles that the producer added, and "acked" the tiles that the consumer removed.
- The failure depends on the grid, not on the override: on the full 8x4 device, case 01 forced onto (2, 1) fails with the same counts (`full_01_grid2x1.log`), and on (3, 1) it passes in 4 min (`full_01_grid3x1.log`). On 8x4, the (6, 3) grid with the 128-row case 03 fails in the same way (the main report, L).
- The capacity in each error is the size of the in0 DFB of the matmul, which is the DFB for the first input (the activations). The in0 DFB holds two blocks of `in0_block_w` tiles, which is 60 tiles for the gate and up projections and 20 tiles for the down projection. With a wider output block, the error moves to the in1 DFB, which is the DFB for the second input (the weights). The in1 DFB has a capacity of 100 tiles, and the counts follow the same pattern (`ovr_01_blockw5.log`).
- "posted" is 0 in all failures, so the reported occupancy is just 65536 − acked. The tile counters are 16-bit and are meant to wrap at 65536; the tt-metal DFB code compares them modulo 65536 ([dataflow_buffer.inl:175-187](../../../../../tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer.inl#L175-L187)). A producer waits for free space before it posts, so the consumer can never be more than one capacity behind. So a plain wrap of the counter does not explain a gap of thousands of tiles.
- No issue or PR reports this case (searched on 2026-11-12). tt-metal issue #57771 has the same error text, with a different cause: a kernel that removed tiles it never added. tt-metal #61931 (closed) confirms that the counters wrap at 65536.
- Effect on the model: my guess is that every expert `sparse_matmul` call of the model would stop the simulator on a 2-core device, because the model's calls use the same grids as the 4 tested companion cases. The model was not run on 2 cores.

#### What the counter trace shows

Case 01 was run on (2, 1) and on (3, 1) on the full 8x4 device, with craq-sim's trace options `TTSIM_QSR_DFB_COUNTER_TRACE=1` (every write to a tile counter, as seen by the overlay, the block that connects the data movement processors to the tile counters, with the old and new values of "posted", "acked" and the capacity) and `TTSIM_QSR_DFB_TRACE=1` (also the posts and acks of the compute processors). The logs are in `small_grid/sparse_trace/` (`ctr2_grid2x1.log`, `ctr2_grid3x1.log`, `dfb_grid2x1.log`). The results:

- The in0 DFB uses tile counters 0 and 1 of Tensix 0 and Tensix 1 (the trace's T0 and T1; a Tensix ID selects the set of tile counters that a register address refers to). Each counter is set to capacity 20 at the start of the op. On (2, 1), each counter gets 12960 posts of 10 tiles, 129600 tiles in total. On (3, 1), each counter gets 86400 tiles, and the run passes. 129600 is 1.5 times 86400, the same ratio as 3 cores to 2 cores, so the run on (2, 1) had made every post it should make when it stopped.
- On both grids, "posted" wraps past 65535 in the middle of the op with no error (`posted=65530->4`). The trace shows no counter reset after the start of the op.
- On (2, 1), the last traced write takes "posted" from 64054 to 64064 (129600 mod 65536) while "acked" is 64054: the last 10 tiles wait for the compute kernel. Then a write that the trace does not show sets "posted" to 0, and the simulator stops. The trace has no clock and no processor number, so it does not name the writer.

The counts fit a write at the end of the op that adds 65536 − 64064 = 1472 tiles to "posted", which makes it exactly 0. The trace does not show this write, so it does not tell whether tt-metal code or the simulator makes it. No reset appears in the trace.

My guess is that the writer is the firmware code on DM0, the first data movement processor of the core, that finishes each DFB when the kernels exit ([dataflow_buffer_finish.h:192-221](../../../../../tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer/dataflow_buffer_finish.h#L192-L221)). For each counter, the code computes `expected`, the count that the exiting kernel recorded, and `behind = expected − actual` on 16 bits. When `behind` is positive as a signed 16-bit number, the code adds `behind` to "posted". If `expected` is wrongly 0 mod 65536, `behind` is 65536 minus the real count mod 65536. This addition happens only when the real count mod 65536 is more than 32768, and that fits the grids that fail and the grid that passes:

| Grid (case 01) | Tiles per counter | Mod 65536 | `behind` as signed 16-bit | Result |
|---|---|---|---|---|
| (2, 1) | 129600 | 64064 | +1472, added to "posted" | fails; acked = 64054 |
| (1, 1) | 259200 | 62592 | +2944, added to "posted" | fails; acked = 62582 |
| (3, 1) | 86400 | 20864 | −20864, nothing added | passes |

The trace was run only on (2, 1) and (3, 1); the (1, 1) row uses the tile count that the (2, 1) trace gives, doubled, and the counts in `ovr_01_grid1x1.log`. In both failures, "acked" is 10 tiles below the real count, as in the trace on (2, 1). This firmware code has been on `main` since #62204. The test tree also had the commit of #63905, which changes how the exit record stores its counts, so the cause may be in either version of the code. Why `expected` would be 0 mod 65536 was not tested. A DPRINT (a print from kernel or firmware code to the host) in `dfb_dm0_flush_side` would show it, if it printed for this DFB: the exit record (the count of batches and entries that the exiting kernel moved), `per_tc` (the tiles per batch on each counter), `expected` and `actual`.

### The decode down projection does not fit SRAM on 2 cores in bfloat16

The decode down projection keeps both its input and its output in SRAM. Its input comes from the gate and up projections, whose outputs are placed in SRAM ([experts/decode.py:85](tt/experts/decode.py#L85-L85) and [experts/decode.py:111](tt/experts/decode.py#L111-L111)), and its own output is placed in SRAM ([experts/decode.py:131-143](tt/experts/decode.py#L131-L143)). Each is [1, 32, 1, 2880], 2.95 MB per core on 2 cores in bfloat16; together they need 5.9 MB per core, and the SRAM bank of each core holds 3.86 MB. In `bfloat8_b`, which the model asks for, the two would just fit, but Quasar has no block float formats, and the model uses bfloat16 instead. On a 2-core device the model would have to put this output in DRAM, as the companion does for case 01. This was not tested in the model, because the model was not run on 2 cores.

### The `lm_head` matmuls are slow on 2 cores

Each of the two captured `lm_head` matmul calls takes 31 to 39 min on a 2-core grid on craq-sim, against 6.5 to 7 min on 8x4. These times include test setup and kernel compile. The generated cases 03 and 04 need a timeout of at least 45 min per case there, or the suite should skip them on small grids and run the companion instead.

## Summary

On a 2-core grid, 106 of the 119 cases pass. With the 5 companion files, every failing case has a small-grid form that passes, except `sparse_matmul`, which cannot run on a 1- or 2-core grid because of L. A counter trace shows that the counts fit a write at the end of the op that adds tiles to a counter that already has every tile. The trace does not show that write. My guess is that it comes from the tt-metal firmware code that finishes a DFB at kernel exit. The matmul, attention and head-merge paths that the gpt-oss model takes on a small grid all run on 2 cores. The gpt-oss model itself was not run on 2 cores. My guess is that L would stop it in the first expert call.

Next steps:
1. Add a DPRINT in `dfb_dm0_flush_side` to confirm that the DM0 finish code makes the write that sets "posted" to 0, and find why its expected count is 0 mod 65536. Then file a tt-metal issue.
2. Put the output of the decode down projection in DRAM when it does not fit SRAM, in the model.
3. Skip the generated `lm_head` cases on small grids, or give them a longer timeout.
