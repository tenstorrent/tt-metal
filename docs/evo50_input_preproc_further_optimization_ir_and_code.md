# EVO50 input pre-processing: further optimization of the channels-last stem (s2d_nhwc → s2d_nhwc_opt)

**Date:** 20 Nov 2026
**Branch:** `pchandrasekaran/evo50_model_perf` @ `47d0cdb2424` plus the uncommitted changes listed in section 9
**Hardware:** Wormhole n150, 8x8 worker grid (64 Tensix cores, 5 RISC-V cores each: BRISC, NCRISC, TRISC0 UNPACK, TRISC1 MATH, TRISC2 PACK)
**Test file:** `/proj_sw/user_dev/pchandrasekaran/tt-metal/tests/ttnn/unit_tests/operations/conv/test_evo50_input_preproc.py`
**Before:** `test_evo50_input_preproc_s2d_nhwc`. **After:** `test_evo50_input_preproc_s2d_nhwc_opt`.
**Previous spec:** `/proj_sw/user_dev/pchandrasekaran/tt-metal/docs/evo50_input_preproc_optimization_for_ttnn_ir.md` (section 4 is the s2d_nhwc IR this round starts from)

Rules kept for every change:
- same math: HiFi2, fp32 accumulation (`fp32_dest_acc_en = true`), bf16 weights, RELU fused into the packer;
- no new TTNN ops;
- every kernel change is generic and guarded, so the cases it does not target behave exactly as before;
- PCC ≥ 0.99 (measured 0.999981).

---

## 1. Result

| Metric (between the inference signposts) | baseline (IR) | s2d | **s2d_nhwc (before)** | s2d_nhwc, new kernels only | **s2d_nhwc_opt (after)** |
|---|---:|---:|---:|---:|---:|
| Device ops | 21 | 14 | 9 | 9 | **8** |
| **Total device kernel duration (µs)** | 2,604.5 | 1,249.9 | **669.2** | 595.8 | **416.4** |
| Total device FW duration (µs) | 3,067.4 | 1,342.3 | 713.4 | 632.3 | **427.2** |
| FW − kernel overhead (µs) | 462.9 | 92.4 | 44.1 | 36.5 | **10.8** |
| Total host duration (µs) | 45.7 | 36.6 | 27.2 | 30.9 | **26.3** |
| PCC vs torch golden | 0.999978 | 0.999980 | 0.999980 | 0.999980 | **0.999981** |

- **s2d_nhwc → s2d_nhwc_opt: 669.2 → 416.4 µs, −252.8 µs (−37.8 %).**
- Against the compiled IR baseline: 2,604.5 → 416.4 µs, **6.3x**.
- The "new kernels only" column runs the *unchanged* s2d_nhwc test on the rebuilt branch. It isolates the kernel-side
  gains (pixel_unshuffle, halo; −73.4 µs) from the graph and config changes made in the opt test (−179.4 µs).

Where the 252.8 µs came from:

| Change | Kind | Saved (µs) | Section |
|---|---|---:|---|
| Output tail rebuilt (row-wise transpose → sharded untilize → stick transpose) | graph (existing ops) | 173.8 | 5.4 |
| pixel_unshuffle gather spread over all 5 RISCs | kernel | 60.7 | 5.1 |
| conv2 `full_inner_dim` (no fp32 partial spills) | kernel + config | 13.7 | 5.2 |
| conv2 halo: NOC direction + 64-row untilize blocks | kernel | 4.4 | 5.3 |
| other (WH transpose, conv1 halo, run-to-run) | – | 0.2 | |
| **Total** | | **252.8** | |

Op-to-op latency is not compared. The gaps between ops (250-450 µs before the halos and the tail) are host-side
dispatch time of conv2d / transpose, not device time. The model function is `@trace_0_forward`, which runs under metal
trace, where those gaps disappear.

---

## 2. Ops perf report: before and after

All numbers come from tracy (`python -m tracy -r -v --op-support-count=20000`), scoped to each test's
`*_inference-start` / `*_inference-end` signposts after a warm-up pass has filled the program cache.
`ttnn.reshape` calls here are views and produce no device op.

- "kernel ns" is the op's device kernel duration (first core start to last core end).
- "FW ns" adds firmware launch/teardown.
- BRISC / NCRISC / TRISC1 are the longest per-core duration of that RISC's kernel. BRISC and NCRISC are the two data
  movement RISCs; TRISC1 (MATH) is where compute-bound ops show up.

### 2.1 Before: `test_evo50_input_preproc_s2d_nhwc` (start of the round, unmodified branch)

Report: `/proj_sw/user_dev/pchandrasekaran/tt-metal/generated/profiler/reports/evo50_cmp/test_evo50_input_preproc_s2d_nhwc/reports/test_evo50_input_preproc_s2d_nhwc/2026_11_19_08_48_07/ops_perf_results_test_evo50_input_preproc_s2d_nhwc_2026_11_19_08_48_07.csv`

IR op → device op(s):

| IR op | device op(s) | rows |
| --- | --- | --- |
| `%1 pixel_unshuffle` (channels_last) | `PixelUnshuffle` | 0 |
| `%3 conv2d` conv1 4x4 | `Halo`, `Conv2d` | 1-2 |
| `%4 conv2d` conv2 grouped 3x3 | `Halo`, `Conv2d` | 3-4 |
| `%6 permute [0,3,1,2]` | `Transpose` (WH), `Transpose` (HC) | 5-6 |
| `%7 to_memory_config` → DRAM | `Copy` | 7 |
| `%8 to_layout` → ROW_MAJOR | `Untilize` | 8 |

| # | device op | in -> out | in layout | cores | kernel ns | FW ns | BRISC ns | NCRISC ns | TRISC1 ns | share |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `PixelUnshuffle` | DRAM_INTERLEAVED -> L1_HEIGHT_SHARDED | ROW_MAJOR | 64 | 136,900 | 137,895 | 136,864 | 128,547 | - | 20.5 % |
| 1 | `Halo` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | ROW_MAJOR | 64 | 15,464 | 16,419 | 15,435 | 12,703 | - | 2.3 % |
| 2 | `Conv2d` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | ROW_MAJOR | 64 | 131,203 | 140,077 | 119,098 | 112,359 | 130,868 | 19.6 % |
| 3 | `Halo` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | TILE | 64 | 24,405 | 25,103 | 23,665 | 24,048 | 22,984 | 3.6 % |
| 4 | `Conv2d` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | ROW_MAJOR | 64 | 123,526 | 131,991 | 100,943 | 98,223 | 123,185 | 18.5 % |
| 5 | `Transpose` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | TILE | 64 | 8,198 | 9,759 | 8,188 | 2,034 | 7,796 | 1.2 % |
| 6 | `Transpose` | L1_HEIGHT_SHARDED -> L1_INTERLEAVED | TILE | 64 | 129,940 | 130,847 | 129,940 | 126,665 | - | 19.4 % |
| 7 | `Copy` | L1_INTERLEAVED -> DRAM_INTERLEAVED | TILE | 64 | 39,529 | 40,377 | 39,511 | 39,060 | - | 5.9 % |
| 8 | `Untilize` | DRAM_INTERLEAVED -> DRAM_INTERLEAVED | TILE | 48 | 60,070 | 80,886 | 60,070 | 43,359 | 41,597 | 9.0 % |
| | **total** | | | | **669,235** | **713,354** | | | | 100 % |

Where the time went:

| Group | ns | share | Bound by |
|---|---:|---:|---|
| output tail (rows 5-8) | 237,737 | 35.5 % | data movement only (TRISC idle in rows 6-7) |
| pixel_unshuffle (row 0) | 136,900 | 20.5 % | BRISC/NCRISC L1 gather; compute RISCs idle |
| conv1 (rows 1-2) | 146,667 | 21.9 % | TRISC1 = kernel time (compute) |
| conv2 (rows 3-4) | 147,931 | 22.1 % | TRISC1 = kernel time (compute) |

### 2.2 Intermediate: the same test after the kernel changes only

Report: `/proj_sw/user_dev/pchandrasekaran/tt-metal/generated/profiler/reports/evo50_final/test_evo50_input_preproc_s2d_nhwc/reports/test_evo50_input_preproc_s2d_nhwc/2026_11_19_10_21_48/ops_perf_results_test_evo50_input_preproc_s2d_nhwc_2026_11_19_10_21_48.csv`

| # | device op | in -> out | in layout | cores | kernel ns | FW ns | BRISC ns | NCRISC ns | TRISC1 ns | share |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `PixelUnshuffle` | DRAM_INTERLEAVED -> L1_HEIGHT_SHARDED | ROW_MAJOR | 64 | 76,224 | 76,905 | 63,177 | 53,771 | 75,176 | 12.8 % |
| 1 | `Halo` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | ROW_MAJOR | 64 | 15,651 | 16,619 | 15,617 | 13,580 | - | 2.6 % |
| 2 | `Conv2d` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | ROW_MAJOR | 64 | 131,149 | 140,165 | 119,006 | 112,149 | 130,815 | 22.0 % |
| 3 | `Halo` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | TILE | 64 | 19,811 | 20,496 | 19,397 | 15,560 | 18,127 | 3.3 % |
| 4 | `Conv2d` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | ROW_MAJOR | 64 | 122,204 | 123,762 | 99,661 | 96,969 | 121,875 | 20.5 % |
| 5 | `Transpose` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | TILE | 64 | 6,712 | 7,366 | 6,393 | 243 | 6,308 | 1.1 % |
| 6 | `Transpose` | L1_HEIGHT_SHARDED -> L1_INTERLEAVED | TILE | 64 | 125,632 | 126,528 | 125,632 | 122,467 | - | 21.1 % |
| 7 | `Copy` | L1_INTERLEAVED -> DRAM_INTERLEAVED | TILE | 64 | 39,875 | 40,749 | 39,848 | 39,402 | - | 6.7 % |
| 8 | `Untilize` | DRAM_INTERLEAVED -> DRAM_INTERLEAVED | TILE | 48 | 58,520 | 79,728 | 58,520 | 43,031 | 40,878 | 9.8 % |
| | **total** | | | | **595,778** | **632,318** | | | | 100 % |

Note row 0: the compute RISCs now take part in pixel_unshuffle (TRISC1 75.2 µs, previously none). In row 3, NCRISC drops
from 24.0 to 15.6 µs: the blocks whose halo rows go to the previous core now use NOC 1.

### 2.3 After: `test_evo50_input_preproc_s2d_nhwc_opt`

Report: `/proj_sw/user_dev/pchandrasekaran/tt-metal/generated/profiler/reports/evo50_final/test_evo50_input_preproc_s2d_nhwc_opt/reports/test_evo50_input_preproc_s2d_nhwc_opt/2026_11_19_10_22_13/ops_perf_results_test_evo50_input_preproc_s2d_nhwc_opt_2026_11_19_10_22_13.csv`

IR op → device op(s):

| IR op | device op(s) | rows |
| --- | --- | --- |
| `%1 pixel_unshuffle` (channels_last) | `PixelUnshuffle` | 0 |
| `%3 conv2d` conv1 4x4 | `Halo`, `Conv2d` | 1-2 |
| `%4 conv2d` conv2 grouped 3x3, `full_inner_dim` | `Halo`, `Conv2d` | 3-4 |
| `%6 transpose(2,3)` | `Transpose` (WH) | 5 |
| `%7 to_layout` → ROW_MAJOR, sharded | `UntilizeWithUnpadding` | 6 |
| `%8 transpose(1,2)` → DRAM | `Transpose` (HC, ROW_MAJOR) | 7 |

| # | device op | in -> out | in layout | cores | kernel ns | FW ns | BRISC ns | NCRISC ns | TRISC1 ns | share |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `PixelUnshuffle` | DRAM_INTERLEAVED -> L1_HEIGHT_SHARDED | ROW_MAJOR | 64 | 76,215 | 76,893 | 63,050 | 53,436 | 75,008 | 18.3 % |
| 1 | `Halo` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | ROW_MAJOR | 64 | 15,184 | 16,157 | 15,154 | 12,689 | - | 3.6 % |
| 2 | `Conv2d` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | ROW_MAJOR | 64 | 131,136 | 134,249 | 119,030 | 112,295 | 130,819 | 31.5 % |
| 3 | `Halo` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | TILE | 64 | 19,985 | 20,669 | 19,570 | 15,488 | 18,304 | 4.8 % |
| 4 | `Conv2d` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | ROW_MAJOR | 64 | 109,871 | 112,999 | 101,823 | 95,710 | 109,508 | 26.4 % |
| 5 | `Transpose` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | TILE | 64 | 6,648 | 7,312 | 6,351 | 222 | 6,237 | 1.6 % |
| 6 | `UntilizeWithUnpadding` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | TILE | 64 | 14,941 | 15,566 | 14,571 | 295 | 12,846 | 3.6 % |
| 7 | `Transpose` | L1_HEIGHT_SHARDED -> DRAM_INTERLEAVED | ROW_MAJOR | 64 | 42,443 | 43,353 | 42,443 | 30,753 | - | 10.2 % |
| | **total** | | | | **416,423** | **427,198** | | | | 100 % |

### 2.4 Before vs after, per op

| Op | Before kernel ns | After kernel ns | Δ ns | Before BRISC / NCRISC / TRISC1 | After BRISC / NCRISC / TRISC1 |
|---|---:|---:|---:|---|---|
| `PixelUnshuffle` | 136,900 | 76,215 | −60,685 | 136,864 / 128,547 / – | 63,050 / 53,436 / 75,008 |
| `Halo` (conv1) | 15,464 | 15,184 | −280 | 15,435 / 12,703 / – | 15,154 / 12,689 / – |
| `Conv2d` conv1 | 131,203 | 131,136 | −67 | 119,098 / 112,359 / 130,868 | 119,030 / 112,295 / 130,819 |
| `Halo` (conv2) | 24,405 | 19,985 | −4,420 | 23,665 / 24,048 / 22,984 | 19,570 / 15,488 / 18,304 |
| `Conv2d` conv2 | 123,526 | 109,871 | −13,655 | 100,943 / 98,223 / 123,185 | 101,823 / 95,710 / 109,508 |
| `Transpose` WH | 8,198 | 6,648 | −1,550 | 8,188 / 2,034 / 7,796 | 6,351 / 222 / 6,237 |
| `Transpose` HC (TILE) | 129,940 | – | −129,940 | 129,940 / 126,665 / – | removed |
| `Copy` L1 → DRAM | 39,529 | – | −39,529 | 39,511 / 39,060 / – | removed |
| `Untilize` DRAM → DRAM | 60,070 | – | −60,070 | 60,070 / 43,359 / 41,597 | replaced |
| `UntilizeWithUnpadding` sharded → sharded | – | 14,941 | +14,941 | – | 14,571 / 295 / 12,846 |
| `Transpose` HC (ROW_MAJOR) → DRAM | – | 42,443 | +42,443 | – | 42,443 / 30,753 / – |
| **Total** | **669,235** | **416,423** | **−252,812** | | |

In the after column the two convs are 57.9 % of the block (they were 38.1 %). The data-movement plumbing
(pixel_unshuffle + halos + tail) dropped from 61.9 % to 42.1 %.

---

## 3. TTNN IR: before → after

### 3.1 Layouts

Layouts already defined in section 1 of the previous spec are reused: `#ttnn_layout1` input, `#ttnn_layout10` TILE
height-sharded `<64x1>`, `#ttnn_layout12`, `#ttnn_layout4`. The tail adds these:

```mlir
// stem output, as before: NHWC ROW_MAJOR height-sharded, 2496 pixels x 16 channels per core
#ttnn_layout_pu      = #ttnn.ttnn_layout<(d0,d1,d2,d3) -> (d0*159744 + d1*832 + d2, d3), <64x1>, memref<2496x16xbf16, #l1>, <height_sharded>,
                                         core_ranges = <[#ttnn.core_range<(0,0), (7,7)>]>>
#ttnn_layout_pu_flat = #ttnn.ttnn_layout<(d0,d1,d2,d3) -> (d0*159744 + d1*159744 + d2, d3), <64x1>, memref<2496x16xbf16, #l1>, <height_sharded>,
                                         core_ranges = <[#ttnn.core_range<(0,0), (7,7)>]>>

// NEW - conv2 output viewed as [1,192,832,16]: same buffer as #ttnn_layout10 (832 is tile-aligned, a view)
#ttnn_layout_nhwc_hs = #ttnn.ttnn_layout<(d0,d1,d2,d3) -> (d0*159744 + d1*832 + d2, d3), <64x1>, memref<78x1x!ttcore.tile<32x32,bf16>, #l1>, <height_sharded>,
                                         core_ranges = <[#ttnn.core_range<(0,0), (7,7)>]>>
// NEW - after the per-row WH transpose [1,192,16,832]: 16 rows per image row, padded to one tile row; shard [96, 832]
#ttnn_layout_hcw_hs  = #ttnn.ttnn_layout<(d0,d1,d2,d3) -> (d0*3072 + d1*16 + d2, d3), <64x1>, memref<3x26x!ttcore.tile<32x32,bf16>, #l1>, <height_sharded>,
                                         core_ranges = <[#ttnn.core_range<(0,0), (7,7)>]>>
// NEW - the same untilized into ROW_MAJOR shards: one 1664 B stick per (h, c)
#ttnn_layout_hcw_rm  = #ttnn.ttnn_layout<(d0,d1,d2,d3) -> (d0*3072 + d1*16 + d2, d3), <64x1>, memref<96x832xbf16, #l1>, <height_sharded>,
                                         core_ranges = <[#ttnn.core_range<(0,0), (7,7)>]>>
// NEW - model output, NCHW ROW_MAJOR in DRAM (what the runtime hands back to the host)
#ttnn_layout_out_rm  = #ttnn.ttnn_layout<(d0,d1,d2,d3) -> (d0*3072 + d1*192 + d2, d3), <1x1>, memref<3072x832xbf16, #dram>, <interleaved>>
```

### 3.2 Before: `test_evo50_input_preproc_s2d_nhwc` (9 device ops, 669.2 µs)

```mlir
// (1) space-to-depth straight to the conv's NHWC height-sharded input                        PixelUnshuffle 136.9 us
%1 = "ttnn.pixel_unshuffle"(%arg0) <{downscale_factor = 2 : i32, channel_order = channel_major,
                                      channels_last = true, padded_channels = 16 : i32,
                                      memory_config = #ttnn.memory_config<#l1, <height_sharded>, #ttnn.shard_spec<<[(0,0)-(7,7)]>, <2496x16>, <row_major>>>}>
      : (tensor<1x3x384x1664xbf16, #ttnn_layout1>) -> tensor<1x192x832x16xbf16, #ttnn_layout_pu>
%2 = "ttnn.reshape"(%1) <{shape = [1, 1, 159744, 16]}> : (...) -> tensor<1x1x159744x16xbf16, #ttnn_layout_pu_flat>     // view

// (2) conv1: 4x4/s1 on the re-packed weights                                                Halo 15.5 + Conv2d 131.2 us
%3 = "ttnn.conv2d"(%2, %w_packed16, %b1, %dev) <{
        batch_size = 1, in_channels = 16, out_channels = 16, input_height = 192, input_width = 832,
        kernel_size = array<i32: 4, 4>, stride = array<i32: 1, 1>, padding = array<i32: 1, 2, 1, 2>, dilation = array<i32: 1, 1>, groups = 1,
        compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi2, fp32_dest_acc_en = true>,
        conv2d_config = #ttnn.conv2d_config<weights_dtype = bf16, activation = <op_type = relu>, deallocate_activation = true,
                                            reallocate_halo_output = false, act_block_h_override = 832, enable_act_double_buffer = true,
                                            shard_layout = height_sharded, enable_kernel_stride_folding = false, config_tensors_in_dram = true>,
        conv2d_slice_config = #ttnn.conv2d_slice_config<l1_full, 0>}>
      : (...) -> tensor<1x1x159744x16xbf16, #ttnn_layout10>

// (3) conv2: grouped 3x3, inner dim sliced per kernel row                                  Halo 24.4 + Conv2d 123.5 us
%4 = "ttnn.conv2d"(%3, %arg2, %arg4, %dev) <{
        batch_size = 1, in_channels = 16, out_channels = 16, input_height = 192, input_width = 832,
        kernel_size = array<i32: 3, 3>, stride = array<i32: 1, 1>, padding = array<i32: 1, 1, 1, 1>, dilation = array<i32: 1, 1>, groups = 2,
        compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi2, fp32_dest_acc_en = true>,
        conv2d_config = #ttnn.conv2d_config<weights_dtype = bf16, activation = <op_type = relu>, deallocate_activation = true,
                                            reallocate_halo_output = false, act_block_h_override = 832, enable_act_double_buffer = true,
                                            shard_layout = height_sharded, enable_kernel_stride_folding = false, config_tensors_in_dram = true>,
        conv2d_slice_config = #ttnn.conv2d_slice_config<l1_full, 0>}>
      : (...) -> tensor<1x1x159744x16xbf16, #ttnn_layout10>

// (4) IR tail: NHWC -> NCHW on TILE data, 16 channels padded to 32 in every tile
%5 = "ttnn.reshape"(%4) <{shape = [1, 192, 832, 16]}> : (...) -> tensor<1x192x832x16xbf16, #ttnn_layout_nhwc_hs>          // view
%6 = "ttnn.permute"(%5) <{permutation = array<i64: 0, 3, 1, 2>}>                                                          // Transpose WH 8.2 + Transpose HC 129.9 us
      : (...) -> tensor<1x16x192x832xbf16, #ttnn_layout12>                                                               // TILE L1 interleaved
%7 = "ttnn.to_memory_config"(%6) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}>                             // Copy 39.5 us
      : (...) -> tensor<1x16x192x832xbf16, #ttnn_layout4>                                                                // TILE DRAM
%8 = "ttnn.to_layout"(%7) <{layout = #ttnn.layout<row_major>}>                                                            // Untilize 60.1 us (runtime output untilize)
      : (...) -> tensor<1x16x192x832xbf16, #ttnn_layout_out_rm>
```

### 3.3 After: `test_evo50_input_preproc_s2d_nhwc_opt` (8 device ops, 416.4 µs)

Changes are marked `// CHANGED`. `%1`-`%3` have the same IR as before; `%1` and the halos inside `%3`/`%4` run faster
because of the kernel changes in section 5.

```mlir
// (1) unchanged IR; kernel gathers on all 5 RISCs (section 5.1)                              PixelUnshuffle 76.2 us
%1 = "ttnn.pixel_unshuffle"(%arg0) <{downscale_factor = 2 : i32, channel_order = channel_major,
                                      channels_last = true, padded_channels = 16 : i32,
                                      memory_config = #ttnn.memory_config<#l1, <height_sharded>, #ttnn.shard_spec<<[(0,0)-(7,7)]>, <2496x16>, <row_major>>>}>
      : (tensor<1x3x384x1664xbf16, #ttnn_layout1>) -> tensor<1x192x832x16xbf16, #ttnn_layout_pu>
%2 = "ttnn.reshape"(%1) <{shape = [1, 1, 159744, 16]}> : (...) -> tensor<1x1x159744x16xbf16, #ttnn_layout_pu_flat>

// (2) conv1: unchanged (full_inner_dim measured slower here, section 5.2)                    Halo 15.2 + Conv2d 131.1 us
%3 = "ttnn.conv2d"(%2, %w_packed16, %b1, %dev) <{ ... same attributes as before ... }>
      : (...) -> tensor<1x1x159744x16xbf16, #ttnn_layout10>

// (3) conv2: whole 3x3 window as one inner-dim block                                         Halo 20.0 + Conv2d 109.9 us
%4 = "ttnn.conv2d"(%3, %arg2, %arg4, %dev) <{
        batch_size = 1, in_channels = 16, out_channels = 16, input_height = 192, input_width = 832,
        kernel_size = array<i32: 3, 3>, stride = array<i32: 1, 1>, padding = array<i32: 1, 1, 1, 1>, dilation = array<i32: 1, 1>, groups = 2,
        compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi2, fp32_dest_acc_en = true>,
        conv2d_config = #ttnn.conv2d_config<weights_dtype = bf16, activation = <op_type = relu>, deallocate_activation = true,
                                            reallocate_halo_output = false,
                                            act_block_h_override = 192,          // CHANGED (832 overflows L1 with the 3x wider block)
                                            enable_act_double_buffer = true,
                                            full_inner_dim = true,               // CHANGED
                                            shard_layout = height_sharded, enable_kernel_stride_folding = false, config_tensors_in_dram = true>,
        conv2d_slice_config = #ttnn.conv2d_slice_config<l1_full, 0>}>
      : (...) -> tensor<1x1x159744x16xbf16, #ttnn_layout10>

// (4) CHANGED tail: transpose each image row on its own core, untilize into RM shards, swap H and C on whole sticks
%5 = "ttnn.reshape"(%4) <{shape = [1, 192, 832, 16]}> : (...) -> tensor<1x192x832x16xbf16, #ttnn_layout_nhwc_hs>          // view
%6 = "ttnn.transpose"(%5) <{dim0 = 2 : si32, dim1 = 3 : si32}>                                                            // Transpose WH 6.6 us
      : (...) -> tensor<1x192x16x832xbf16, #ttnn_layout_hcw_hs>                                                          // TILE, still height-sharded
%7 = "ttnn.to_layout"(%6) <{layout = #ttnn.layout<row_major>,                                                             // UntilizeWithUnpadding 14.9 us
                            memory_config = #ttnn.memory_config<#l1, <height_sharded>, #ttnn.shard_spec<<[(0,0)-(7,7)]>, <96x832>, <row_major>>>}>
      : (...) -> tensor<1x192x16x832xbf16, #ttnn_layout_hcw_rm>
%8 = "ttnn.transpose"(%7) <{dim0 = 1 : si32, dim1 = 2 : si32,                                                             // Transpose HC (ROW_MAJOR) 42.4 us
                            memory_config = #ttnn.memory_config<#dram, <interleaved>>}>
      : (...) -> tensor<1x16x192x832xbf16, #ttnn_layout_out_rm>                                                          // model output
```

---

## 4. TTNN Python: before → after (test)

`/proj_sw/user_dev/pchandrasekaran/tt-metal/tests/ttnn/unit_tests/operations/conv/test_evo50_input_preproc.py`
(+114 lines: `_tail_opt` and `test_evo50_input_preproc_s2d_nhwc_opt`; the earlier three tests are untouched)

### 4.1 conv2 config

```diff
     tuned = dict(
         weights_dtype=ttnn.bfloat16,
         activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
         deallocate_activation=True,
         reallocate_halo_output=False,
         act_block_h_override=832,
         enable_act_double_buffer=True,
         shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
         enable_kernel_stride_folding=False,
         config_tensors_in_dram=True,
     )
     conv1_cfg = ttnn.Conv2dConfig(**tuned)
-    conv2_cfg = ttnn.Conv2dConfig(**tuned)
+    # conv2 (grouped 3x3): the whole window as one inner-dim block, 123 -> 110 us. abh 832/416 overflow L1 with the
+    # wider activation block, 96 is slower (127 us).
+    conv2_cfg = ttnn.Conv2dConfig(**{**tuned, "full_inner_dim": True, "act_block_h_override": 192})
```

### 4.2 Output tail

```diff
-    # %8 = ttnn.reshape(%7) {shape = [1, 192, 832, 16]}
-    t8 = ttnn.reshape(t7, (BATCH, MID_H, MID_W, C2_OUT))
-    _free_unless_alias(t7, t8)
-    # %9 = ttnn.permute(%8) {permutation = [0, 3, 1, 2]}  -> TILE L1 interleaved
-    t9 = ttnn.permute(t8, (0, 3, 1, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
-    ttnn.deallocate(t8)
-    # %10 = ttnn.to_memory_config(%9)  -> TILE DRAM
-    t10 = ttnn.to_memory_config(t9, ttnn.DRAM_MEMORY_CONFIG)
-    ttnn.deallocate(t9)
-    # runtime output untilize
-    out_rm = ttnn.to_layout(t10, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
-    ttnn.deallocate(t10)
-    return out_rm
+def _tail_opt(t7):
+    """conv2's output is [1,1,159744,16] TILE height-sharded, 2496 pixels (= 3 whole image rows) per core."""
+    t8 = ttnn.reshape(t7, (BATCH, MID_H, MID_W, C2_OUT))          # view: 832 is tile-aligned
+    _free_unless_alias(t7, t8)
+    t9 = ttnn.transpose(t8, 2, 3)                                 # WH tile transpose on the shards -> [1,192,16,832]
+    ttnn.deallocate(t8)
+    t10 = ttnn.to_layout(t9, ttnn.ROW_MAJOR_LAYOUT,
+                         memory_config=t9.memory_config())        # sharded -> sharded untilize
+    ttnn.deallocate(t9)
+    out = ttnn.transpose(t10, 1, 2,
+                         memory_config=ttnn.DRAM_MEMORY_CONFIG)   # H<->C on 1664 B sticks -> [1,16,192,832] RM DRAM
+    ttnn.deallocate(t10)
+    return out
```

---

## 5. The optimizations in detail

Sizes used below: input `[1,3,384,1664]` bf16 = 3.83 MB. The stem output and every later activation hold 159,744 pixels
(192 × 832) × 16 channels. Each of the 64 cores owns 2,496 pixels, which is exactly 3 image rows of 832 pixels.

### 5.1 pixel_unshuffle: spread the L1 gather over all five RISCs (136.9 → 76.2 µs, −60.7)

#### What the op does

`pixel_unshuffle(r=2, channels_last=True, padded_channels=16)` turns NCHW `[1,3,384,1664]` (DRAM) into NHWC
`[1,192,832,16]` (height-sharded in L1, ROW_MAJOR). Output pixel (ho, wo) takes, for each of the 3 channels and 2
sub-rows dy, the two adjacent input elements at columns 2·wo and 2·wo+1. For bf16 those two elements are one 32-bit word.
So one output pixel = **6 word loads** (one from each of the r·C = 6 input rows) + **8 word stores** (6 data words + 2
zero words for channels 12..15). That is 14 L1 memory operations per pixel and **34,944 per core** (2,496 pixels).

Per core and per output image row, the kernel reads the 6 input rows it needs from DRAM (6 × 3,328 B = 20 KB, split
between the two readers) into a staging buffer, then gathers from it into its own output shard with plain RISC-V
loads/stores. No NOC writes happen at all.

#### Why it was slow

On the branch, the two data-movement RISCs did both the DRAM reads **and** the whole gather. The three compute RISCs
had no kernel (TRISC column "-" in 2.1). Each reader ran ~17.5 k scalar L1 operations plus its reads: BRISC 136.9 /
NCRISC 128.5 µs. DRAM is not the limit: 3.83 MB in 137 µs is 28 GB/s, an order of magnitude below the read bandwidth.

#### Fix, part 1: port the bev_model_perf kernels

`origin/pchandrasekaran/bev_model_perf` has three later pixel_unshuffle commits on top of the same base (`6eed7b6`
there is identical to `60ca345` here). They touch only pixel_unshuffle, its unit test and `sources.cmake`, and were
applied with `git cherry-pick -n a527966f487 fcf4827c393 4e1f1b33715`:

| Commit | Change | Effect here |
|---|---|---|
| `a527966` | large pixels (r=4, C=3: 24 words) gathered one pixel at a time in 12-word chunks so loads do not spill registers | none (r=2 uses the paired-pixel path) |
| `fcf4827` | gather moved into a shared header `pixel_unshuffle_nhwc_common.hpp`; the readers stage each image row in a 2-page CB; a new compute kernel lets UNPACK, MATH and PACK gather a compile-time share of every staged row | **the speed-up** |
| `4e1f1b3` | new `COLUMN_MAJOR_PADDED` channel order produced on the compute engine (tilize → transpose → untilize), with its own program factory | not used by this graph |

The evo50 fix "drop NOC transaction ids" (`657c92f`) stays in. It is in the NCHW kernel, which these commits do not
touch, and the ported channels_last kernels use plain barriers.

How the five RISCs work together after the port:
1. Each reader (BRISC, NCRISC) walks its half of every image row in the core's pixel range ("items"). It issues the
   DRAM reads for item n+1 into CB page (n+1) mod 2, waits for item n's reads, pushes the page, then gathers its own
   share of item n.
2. UNPACK, MATH and PACK walk the same item sequence, recomputed from the same compile-time geometry and runtime pixel
   range. That way the readers never have to describe the work.
3. Synchronisation uses only the CB stream registers (received/acked counters), read and written directly, with no
   LLK push/pop. The helpers poll `received`. PACK and MATH report "my share is done" through two small counter CBs;
   UNPACK waits for both, then writes `acked`, which frees the page for the reader.

Reader loop, before (`dataflow/pixel_unshuffle_nhwc_sharded.cpp` @ HEAD). It reads one row ahead into a private
slot and gathers the whole span itself:

```cpp
    uint32_t slot = 0;
    issue_reads(cur, slot);
    while (true) {
        const bool has_next = advance(cursor, nxt);
        noc_async_read_barrier();  // only cur's reads are outstanding
        if (has_next) {
            issue_reads(nxt, slot ^ 1);  // overlaps the gather below
        }
        gather(cur, slot);               // the reader gathers ALL of its span
        if (!has_next) break;
        cur = nxt;
        slot ^= 1;
    }
```

Reader loop, after. Pages go through a real CB that the compute RISCs can see, and the reader gathers only the first
`24 - helper24` 24ths of its span:

```cpp
    cb_reserve_back(cb_in, 1);
    uint32_t page = get_write_ptr(cb_in);
    issue_reads(cur, page);
    while (true) {
        const bool has_next = G::advance(risc, pix0, end, cursor, nxt);
        noc_async_read_barrier();  // only cur's reads are outstanding
        cb_push_back(cb_in, 1);    // the helpers may start on cur
        uint32_t next_page = 0;
        if (has_next) {
            cb_reserve_back(cb_in, 1);    // waits until the helpers are done with the row before last
            next_page = get_write_ptr(cb_in);
            issue_reads(nxt, next_page);  // overlaps the gather below
        }
        uint32_t a, b;
        G::part(cur, 0, own_hi, a, b);    // own_hi = 24 - helper24
        G::gather(cur, a, b, page, out_base);
        if (!has_next) break;
        cur = nxt;
        page = next_page;
    }
```

New compute kernel `compute/pixel_unshuffle_nhwc_gather.cpp` (no math, only L1 copies):

```cpp
            UNPACK({
                wait_count(in_received[s], n[s] + 1);                        // reader s pushed the row
                G::part(it[s], 24 - helper24, 24 - helper24 + up24, a, b);
                G::gather(it[s], a, b, page, out_base);
                wait_count(done_math[s], n[s] + 1);
                wait_count(done_pack[s], n[s] + 1);
                *in_acked[s] = (uint16_t)(n[s] + 1);                         // page free for the reader
            });
            MATH({
                wait_count(in_received[s], n[s] + 1);
                G::part(it[s], 24 - helper24 + up24, 24 - pk24, a, b);
                G::gather(it[s], a, b, page, out_base);
                *done_math[s] = (uint16_t)(n[s] + 1);
            });
            PACK({
                wait_count(in_received[s], n[s] + 1);
                G::part(it[s], 24 - pk24, 24, a, b);
                G::gather(it[s], a, b, page, out_base);
                *done_pack[s] = (uint16_t)(n[s] + 1);
            });
```

The paired-pixel gather in `pixel_unshuffle_nhwc_common.hpp` (used by every RISC) issues all loads for two pixels
before any store, so up to 12 loads are in flight:

```cpp
            if constexpr (2 * wpp <= 16) {
                // Small pixels (r=2, C<=4): two pixels per iteration, all 2*wpp loads issued before any store.
                for (; v + 1 < vb; v += 2) {
                    uint32_t w0[wpp], w1[wpp];
                    for (uint32_t k = 0; k < rows_per_item; k++)
                        for (uint32_t t = 0; t < wpr; t++) {
                            w0[k * wpr + t] = rows[k][sw + t];
                            w1[k * wpr + t] = rows[k][sw + wpr + t];
                        }
                    for (uint32_t q = 0; q < wpp; q++)    { d0[q] = w0[q]; d1[q] = w1[q]; }
                    for (uint32_t q = wpp; q < wpad; q++) { d0[q] = 0;     d1[q] = 0;     }   // zero pad channels
                    dst += 2 * out_row_nbytes;
                    sw += 2 * wpr;
                }
            }
```

#### Fix, part 2: retune the split for small pixels

The ported split was tuned on BEV's r=4 stems (24 data words per pixel): all of the gather on UNPACK + PACK, none on
MATH or the readers. That gave 82.6 µs here, with UNPACK/PACK at ~80 µs while the readers finished at BRISC 45 /
NCRISC 28 µs and then idled. Sweep on
`/proj_sw/user_dev/pchandrasekaran/tt-metal/tests/ttnn/unit_tests/operations/conv/test_evo50_pixel_unshuffle_micro.py`
(the op alone, 3 timed runs each, output bit-exact vs torch). Each column is helpers / 24, of which MATH / 24:

| Split | 24/0 (ported) | 22/4 | 20/4 | 20/6 | 18/4 | **18/6** | 16/6 | 16/0 | 12/0 | 8/0 | 0/0 (readers only) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| µs | 82.6 | 82.3 | 78.1 | 79.6 | 80.9 | **76.0** | 78.9 | 95.3 | 101.7 | 111.5 | 131.9 |

With 18/6, every RISC takes 6/24 of each reader's span. The readers do one span each (plus the DRAM reads); UNPACK,
MATH and PACK each do both spans. Per core that is 1/8 of the pixels per reader and 1/4 per compute RISC.

`device/pixel_unshuffle_channels_last_program_factory.cpp`:

```diff
-    constexpr uint32_t helper24 = 24;
-    constexpr uint32_t math24 = 0;  // MATH measured slower than UNPACK/PACK at the L1 copies (... on the block C stem)
+    // Gather split per reader span, in 24ths: the reader keeps 24 - helper24; of the helpers' part MATH takes math24
+    // and UNPACK / PACK split the rest.
+    //  * Large pixels (e.g. r=4, C=3: 24 data words; the BEV stems): the readers gather slowest (they also issue the
+    //    NOC reads) and MATH is slower than UNPACK/PACK at the L1 copies (math 8/24: 382 us, 4/24: 353 us, 0: 335 us on
+    //    the block C stem), so UNPACK and PACK take everything.
+    //  * Small pixels on the fast path (2-byte data, r even, CHANNEL_MAJOR, C*r*r <= 16: the paired-pixel gather; e.g.
+    //    the EVO50 stem r=2, C=3): the gather is cheap enough that UNPACK/PACK alone become the limit while the readers
+    //    idle, so all five RISCs share it. EVO50 [1,3,384,1664] -> 64 cores: 24/0 82.6 us, 20/4 78.1, 18/6 76.0,
+    //    16/6 78.9, 0/0 131.9.
+    const bool small_fast_pixels = datum_sz == 2 && (r % 2) == 0 &&
+                                   op_attr.channel_order != ttnn::PixelUnshuffleChannelOrder::SPATIAL_MAJOR &&
+                                   C * r * r <= 16;
+    const uint32_t helper24 = small_fast_pixels ? 18 : 24;
+    const uint32_t math24 = small_fast_pixels ? 6 : 0;
```

Scope: the condition is exactly the kernel's paired-pixel fast path (`fast && 2 * wpp <= 16`). Large pixels (the BEV
r=4 stems), SPATIAL_MAJOR order and 4-byte elements keep the BEV setting unchanged.

#### Result

136.9 → **76.2 µs** (BRISC 63.1 / NCRISC 53.4 / TRISC 75.0: all five RISCs now busy).
`test_pixel_unshuffle.py`: 161 passed, 28 skipped (bit-exact golden suite).

### 5.2 conv2 (grouped 3x3): `full_inner_dim` on a height-sharded conv (123.5 → 109.9 µs, −13.7)

#### How a height-sharded conv2d computes

Each core turns its pixels into an im2col matrix `[pixels, K]` and multiplies it by the weight matrix `[K, out_ch]`.
- The activation is processed in blocks of `act_block_h` rows (output pixels) by `act_block_w` columns (part of K).
- For every block, the reader copies the window sticks into a CB, the compute kernel **tilizes** the block, runs the
  matmul, and accumulates in DEST.
- With `fp32_dest_acc_en`, when K is split into several inner blocks, every non-final inner block **packs the fp32
  partial sums out to L1** (the matmul-partials CB) and the next one **reloads** them into DEST.
- A height-sharded conv **always split K per kernel row** before this change (`slice_inner_dim`). `full_inner_dim`
  existed only for block-sharded convs.

#### conv2 before and after (from the ops perf report's `block_config` attribute)

The halo output is TILE, so conv2's 16 input channels are padded to 32. One kernel row = 3 taps × 32 = 96 = 3 tiles.

| | before (sliced) | after (`full_inner_dim`, abh 192) |
|---|---|---|
| `act_block_h_ntiles` × `act_block_w_ntiles` | 26 × 3 | 6 × 9 |
| inner-dim blocks per output block | 3 (one per kernel row) | 1 (whole 3x3 window) |
| output blocks per core (78 tile rows) | 3 | 13 |
| matmul tiles per core (78 × 9 × 1) | 702 | 702 (same) |
| fp32 partial spill + reload per output block | 2 packs + 2 unpacks of 26 tiles | none |
| fp32 partial tile moves per core | 3 × 4 × 26 = **312** | **0** |
| out subblock (h × w) | 2 × 1 | 3 × 1 |
| activation CBs (double-buffered, bf16) | 2 × 26 × 3 × 2 KB = 312 KB | 2 × 6 × 9 × 2 KB = 216 KB |
| TRISC1 (MATH) | 123.2 µs | 109.5 µs |

The arithmetic is identical: same K, same tiles, same HiFi2 + fp32 accumulation. The saving is the 312 fp32 partial
tile round trips per core, plus the per-inner-block setup. abh had to drop to 192 because the whole-window activation
block is 3x wider: at abh 832 the CBs would need 2 × 26 × 9 × 2 KB = 936 KB, and at 416 they clash with the resident
L1 buffers. 192 (6 tiles) divides the 78 tile rows exactly.

#### Fix: implement `full_inner_dim` for height-sharded convs

This ports BEV round 5. It had only been in the bev working tree (now `stash@{0}`) and was never committed. The
RELU6-fusion hunks in the same files were left out.

1. **Block config** (`conv2d_utils.cpp`, `determine_per_core_conv_block_config`): the activation block width covers the
   whole window.

```diff
-            act_block_w = enable_activation_reuse
+            // Activation reuse and full_inner_dim both put the whole kh x kw window of a pixel in one activation
+            // block row (one inner-dim block, no partial-sum spill/reload between kernel rows).
+            act_block_w = (enable_activation_reuse || full_inner_dim)
                               ? tt::round_up(padded_in_channels * window_h * window_w, tt::constants::TILE_WIDTH)
                               : tt::round_up(padded_in_channels * window_w, tt::constants::TILE_WIDTH);
```

2. **Program factory** (`device/conv2d_op_sharded_program_factory.cpp`): stop slicing, and tell every activation reader
   (reader, writer and the separate mcast-sender define map) to read `filter_h` kernel rows per pixel.

```diff
-    const bool slice_inner_dim = (height_sharded && !enable_activation_reuse) || (block_sharded && !full_inner_dim);
+    // Height sharded: the inner dim is sliced per kernel row unless activation reuse or full_inner_dim is on (both
+    // read the whole window per pixel). Block sharded: sliced unless full_inner_dim.
+    const bool slice_inner_dim =
+        (height_sharded && !enable_activation_reuse && !full_inner_dim) || (block_sharded && !full_inner_dim);
 ...
+    if (height_sharded && !slice_inner_dim && !enable_activation_reuse && !is_conv_1d_depthwise_conv) {
+        // full_inner_dim on a height sharded conv: every activation reader reads filter_h kernel rows per pixel
+        // (one window_outer_offset apart) into one activation block row instead of one kernel row per block.
+        reader_defines["ACT_WINDOW_H_READS"] = std::to_string(filter_h);
+        writer_defines["ACT_WINDOW_H_READS"] = std::to_string(filter_h);
+        writer_mcast_sender_defines["ACT_WINDOW_H_READS"] = std::to_string(filter_h);
+    }
```

3. **Activation reader** (`device/kernels/conv_reader_common.hpp`): `read_sticks` issues `window_h_reads` reads per
   pixel, one input row (`window_outer_offset` bytes) apart. The default of 1 compiles to exactly the old code.

```diff
-    uint32_t stride_w>
+    uint32_t stride_w,
+    uint32_t window_h_reads = 1,
+    uint32_t window_h_stride_bytes = 0>
 FORCE_INLINE void read_sticks(...) {
         if constexpr (dilation_w == 1) {
             for (uint16_t ind = start_ind; ind <= end_ind; ind += stride_w) {
                 uint32_t src_addr = reader_offset + (ind * conv_act_c_read_bytes);
-                experimental::read_with_state(noc, l1_write_addr_act, src_addr);
-                l1_write_addr_act += (coalesced_read_bytes + act_block_w_extra_align_bytes);
+#pragma GCC unroll(window_h_reads)
+                for (uint32_t kh = 0; kh < window_h_reads; kh++) {
+                    experimental::read_with_state(noc, l1_write_addr_act, src_addr);
+                    l1_write_addr_act += coalesced_read_bytes;
+                    src_addr += window_h_stride_bytes;
+                }
+                l1_write_addr_act += act_block_w_extra_align_bytes;
             }
         } else { ... the same kh loop around the dilated kw reads ... }
 ...
+#ifdef ACT_WINDOW_H_READS
+constexpr uint32_t act_window_h_reads = ACT_WINDOW_H_READS;
+#else
+constexpr uint32_t act_window_h_reads = 1;
+#endif
```

   The three activation readers (`reader_conv_activations_padded_with_halo_3x3_weights_v2.cpp`,
   `reader_writer_tiled_out_1d_mcast_{sender,receiver}_conv_weights_tiled_col_to_rm_blocks.cpp`) pass the new template
   arguments:

```diff
                     weight_size_w,
-                    stride_w>(noc, packed_reader_indices_ptr, reader_offset, l1_write_addr_act, reader_idx);
+                    stride_w,
+                    act_window_h_reads,
+                    window_outer_offset>(noc, packed_reader_indices_ptr, reader_offset, l1_write_addr_act, reader_idx);
```

4. **Weights** (`prepare_conv2d_weights.cpp`): the weight rows must be (kh, kw, c)-contiguous, matching the new
   activation row. Without this the rows for kh ≥ 1 are zero.

```diff
                 weight_tensor_ = convert_conv_weight_tensor_to_special_padding_tiled_layout(
                     weight_tensor_,
                     params.weight_block_h_ntiles,
                     params.weight_block_w_ntiles,
-                    params.enable_activation_reuse,
+                    params.enable_activation_reuse || params.full_inner_dim,
                     weight_tensor_.dtype());
```

5. **Weight CB** (`conv2d_op_program_factory_common.cpp`): a fully-buffered weight CB must hold exactly the
   `num_blocks_act_w` blocks the writer pushes per output block. That is 1 with `full_inner_dim`, not `kernel_size[0]`.

```diff
-                    weight_block_num_tiles *= is_1d_depthwise_conv ? num_blocks_act_w : kernel_size[0];
+                    weight_block_num_tiles *= num_blocks_act_w;
```

#### Measurements (`test_evo50_input_preproc_opt.py`, `conv=` parameter)

| Conv | Config | Conv2d µs |
|---|---|---:|
| conv2 | sliced, abh 832 (before) | 123.3 |
| conv2 | `full_inner_dim`, abh 832 / 416 | L1 overflow / CB clash |
| conv2 | **`full_inner_dim`, abh 192 (adopted)** | **109.8** |
| conv2 | `full_inner_dim`, abh 96 | 126.6 |
| conv1 | sliced, abh 832 (kept) | 131.2 |
| conv1 | `full_inner_dim`, abh 416 / 192 / 832 without act double buffer | 148.2 / 135.9 / 168.0 |

Why conv1 does not benefit: conv1's input is ROW_MAJOR with 16 unpadded channels. One kernel row is 4 taps × 16 = 64 =
2 tiles, so it has 4 small inner blocks. With the full window, the readers issue 4 kernel-row reads per pixel into
8-tile rows and become the limit (abh 192: BRISC 125.7 / NCRISC 129.3 µs, against 119.1 / 112.3 sliced). The smaller
block heights that L1 forces add per-block overhead on top. The same pattern was seen on BEV's small height-sharded
convs, so conv1 stays sliced.

#### Result

conv2 123.5 → **109.9 µs**.
`tests/ttnn/unit_tests/operations/conv/test_conv2d_full_inner_dim_height_sharded.py` (8 shapes: grouped, stride 2,
5x5, dilation, split reader on/off, bfp8 output; compared against the sliced conv): 8 passed. `test_conv2d.py`: 161
passed.

### 5.3 conv2 halo: NOC direction per block + 64-row untilize blocks (24.4 → 20.0 µs, −4.4)

#### What the halo does

Before a 3x3 conv, each core needs the image row above and below its own pixels (plus the padding), which live on its
neighbours. For a TILE input, the halo op:
1. untilizes the shard on the compute RISCs, in blocks of 32 rows;
2. lets the two data-movement RISCs (RISC 0 on NOC 0, RISC 1 on NOC 1) gather the sticks into the padded output of
   this core and its neighbours.

Blocks were handed to the two gather RISCs round-robin (even blocks → RISC 0, odd → RISC 1).

#### Why it was slow

Each Wormhole NOC is a one-way torus: NOC 0 routes +x then +y, NOC 1 routes −x then −y. The halo rows a core sends to
the *previous* core (smaller x, or same x and smaller y) take one hop on NOC 1 but nearly a full ring on NOC 0. With
round-robin, half of those blocks went the long way, and every core sends at the same time, so the NOC-0 ring
congests. The untilize/gather handshake also ran once per 32 rows.

#### Fix

Ported BEV rounds 6/7 unchanged (also stash-only before).

1. **Host** (`sliding_window.cpp`): compute each transfer's direction by hop count on the actual NOC grid. The blocks
   that write behind form a prefix `[0, left_blocks)` and go to RISC 1; the blocks that write ahead form a suffix
   `[right_block_start, n)` and go to RISC 0; the middle alternates. Each side is capped at half the blocks to keep the
   two RISCs balanced.

```diff
-// Round-robin blocks between two cores
-static std::tuple<GatherConfig, GatherConfig, uint32_t> divide_blocks_between_cores(
-    const GatherConfig& input, uint32_t block_size) {
+static int transfer_direction(
+    const DestinationTransferPair& t, uint16_t my_noc_x, uint16_t my_noc_y, uint32_t noc_w, uint32_t noc_h) {
+    const int dx = static_cast<int>(t.noc_x) - static_cast<int>(my_noc_x);
+    const int dy = static_cast<int>(t.noc_y) - static_cast<int>(my_noc_y);
+    if (dx == 0 && dy == 0) {
+        return 0;
+    }
+    const int w = static_cast<int>(noc_w), h = static_cast<int>(noc_h);
+    const int hops0 = (dx >= 0 ? dx : w + dx) + (dy >= 0 ? dy : h + dy);
+    const int hops1 = (dx <= 0 ? -dx : w - dx) + (dy <= 0 ? -dy : h - dy);
+    return hops1 < hops0 ? -1 : (hops0 < hops1 ? 1 : 0);
+}
+
+static BlockDivision divide_blocks_between_cores(
+    const GatherConfig& input, uint32_t block_size, uint16_t my_noc_x, uint16_t my_noc_y, uint32_t noc_w, uint32_t noc_h) {
     ...
-        (block_id % NUM_RISCV_DATA_MOVEMENT_CORES == 0 ? first : second).push_back(transfer);
+    const uint32_t half = (number_of_blocks + 1) / 2;
+    left_blocks = std::min(left_blocks, half);
+    right_block_start = std::max({right_block_start, number_of_blocks - half, left_blocks});
+    for (const auto& transfer : all) {
+        const auto block_id = transfer.src_id / block_size;
+        (block_owner(block_id, left_blocks, right_block_start) == 0 ? first : second).push_back(transfer);
+    }
```

2. **Gather kernel and untilize kernel** (`halo/device/kernels/dataflow/halo_gather.cpp`,
   `halo/device/kernels/compute/pack_untilize.cpp`): both apply the same owner rule, read from runtime args 1-3 instead
   of a fixed start offset and stride of 2. The compute kernel pushes each untilized block into the CB of the RISC that
   owns it.

```diff
+FORCE_INLINE uint32_t halo_block_owner(uint32_t block_id, uint32_t left_blocks, uint32_t right_block_start) {
+    if (block_id < left_blocks) {
+        return 1;
+    }
+    if (block_id >= right_block_start) {
+        return 0;
+    }
+    return (block_id - left_blocks) & 1;
+}
 ...
-                    block_boundary_offset += BlockSizeHeight * BlockStride;
-                    block_id += BlockStride;
+                    block_id++;
+                    while (block_id < total_blocks &&
+                           halo_block_owner(block_id, left_blocks, right_block_start) != MyRisc) {
+                        block_id++;
+                    }
+                    block_boundary_offset = BlockSizeHeight * (block_id + 1);
```

```diff
-        if (block_idx % 2 == 0) {
+        const uint32_t owner =
+            block_idx < left_blocks ? 1 : (block_idx >= right_block_start ? 0 : ((block_idx - left_blocks) & 1));
+        if (owner == 0) {
```

3. **Block size** (`halo/device/untilize_with_halo_program_factory.cpp`): use 64-row untilize blocks when the per-core
   tile rows are even and the row is at most 8 tiles. Here that is 78 tile rows and 1 tile per row, so 39 blocks
   instead of 78. Block-direction runtime args are passed to both gather RISCs and the compute kernel.

```diff
+int select_untilize_block_size(const Tensor& input_tensor, uint32_t ncores_nhw) {
+    if (input_tensor.layout() != Layout::TILE) {
+        return UNTILIZE_BLOCK_SIZE;
+    }
+    ...
+    if (tile_rows_per_core >= 2 * large_tile_rows && tile_rows_per_core % large_tile_rows == 0 &&
+        ntiles_per_row <= UNTILIZE_LARGE_BLOCK_MAX_TILES_PER_ROW) {
+        return UNTILIZE_BLOCK_SIZE_LARGE;   // 64
+    }
+    return UNTILIZE_BLOCK_SIZE;             // 32
+}
```

The row-major halo path (conv1's halo, which has no untilize) keeps the round-robin split; splitting it by direction
was measured slower on BEV.

#### Result

conv2 halo 24.4 → **20.0 µs** (BRISC 23.7 → 19.6, NCRISC 24.0 → 15.5, TRISC1 23.0 → 18.3). The conv1 halo is
unchanged at 15.2 µs. `test_maxpool2d.py`: 104 passed. `test_avgpool2d.py`: 276 passed. `test_conv2d.py`: 161 passed.

### 5.4 Output tail: row-wise transpose instead of the TILE permute (237.7 → 64.0 µs, −173.7)

#### What the tail must do

conv2 produces `[1,1,159744,16]` (NHWC, TILE, height-sharded, 3 image rows per core). The model output is
`[1,16,192,832]` NCHW ROW_MAJOR in DRAM: 16 × 192 × 832 × 2 B = **5.11 MB**.

#### Why the IR tail was slow (237.7 µs, 35.5 % of the block)

`permute(0,3,1,2)` on TILE data runs as two transposes:
1. **WH transpose** (8.2 µs, compute, core-local).
2. **HC transpose on TILE** (129.9 µs, data movement only; TRISC idle). In TILE layout the 16 channels are padded to
   32 rows, so the HC transpose moves twice the useful bytes and rebuilds every tile. It writes an interleaved L1
   tensor, spread over all cores through the NOC.
3. **Copy** L1 → DRAM (39.5 µs): another pass over 5.1 MB.
4. **Untilize** DRAM → DRAM (60.1 µs, 48 cores): reads 5.1 MB back from DRAM and writes it again.

The data crosses the NOC or DRAM four times, and half of the bytes in step 2 are padding.

#### The new tail

`permute(0,3,1,2)` = `transpose(2,3)` then `transpose(1,2)`. Doing the steps in that order, with the right layouts at
each step, gives:

| Step | Op | What happens | µs |
|---|---|---|---:|
| 1 | `reshape [1,192,832,16]` | view (832 is tile-aligned), no device op | 0 |
| 2 | `transpose(2,3)` | WH tile transpose **inside each image row**. A core holds whole image rows, so this is core-local compute on the shard → `[1,192,16,832]` TILE, height-sharded | 6.6 |
| 3 | `to_layout(ROW_MAJOR, memory_config=<same shard spec>)` | untilize into ROW_MAJOR shards. No DRAM, no interleaving; the reader is idle (NCRISC 0.3 µs) because the data is already in place | 14.9 |
| 4 | `transpose(1,2, memory_config=DRAM)` | each (h, c) row of 832 values is now one contiguous 1,664 B stick; swapping H and C is a pure stick copy written straight into the ROW_MAJOR DRAM output | 42.4 |

The final output is produced in its final layout, so the `Copy` and the DRAM → DRAM `Untilize` disappear. No step
moves tile padding over the NOC or to DRAM.

#### Tail variants measured (`test_evo50_input_preproc_opt.py`, `tail=` parameter; all PCC 0.999980-0.999981)

| Tail | Device ops (µs) | Tail µs |
|---|---|---:|
| `ir`: reshape → permute → to_memory_config → to_layout | transpose 6.7 + transpose 125.7 + copy 39.9 + untilize 59.4 | 231.7 |
| `tw`: transpose(2,3) on the flat `[1,1,159744,16]` → untilize to DRAM → reshape `[1,16,192,832]` | 7.7 + 56.7 + **174.7** (ROW_MAJOR reshape copy) | 239.1 |
| `rowt`: row-wise, untilize to L1 interleaved | 6.6 + 30.6 + 57.2 | 94.4 |
| `rowtd`: row-wise, untilize to DRAM | 6.6 + 42.2 + 91.8 | 140.6 |
| **`rowts`: row-wise, sharded untilize (adopted)** | 6.6 + 14.9 + 42.4 | **63.9** |
| row-wise, untilize into 48-row ROW_MAJOR shards on all 64 cores | – | rejected by the tensor-layout shard-alignment check |

`tw` shows why the order matters. Its transpose is just as cheap, but each core's channel rows end up 2,496 elements
long, which is not the output's row length (832). Fixing that needs a ROW_MAJOR reshape that copies everything again.

#### Result

Tail 237.7 → **64.0 µs**, one op fewer, PCC 0.999981. This is a graph change only: no kernel was modified, and the
same pattern applies wherever a height-sharded NHWC conv output with whole image rows per core has to become NCHW.

---

## 6. Tried and not adopted

| Attempt | Result |
|---|---|
| conv1 `full_inner_dim` (abh 416 / 192 / 832 without act double buffer) | 148.2 / 135.9 / 168.0 µs vs 131.2 (reader-bound, 5.2) |
| conv2 `full_inner_dim` at abh 832 / 416 | static CBs overflow L1 / clash with resident buffers |
| conv2 `full_inner_dim` at abh 96 | 126.6 µs |
| tail via flat transpose + ROW_MAJOR reshape (`tw`) | reshape alone 174.7 µs |
| tail untilize to DRAM / L1 interleaved instead of sharded | 140.6 / 94.4 µs vs 63.9 |
| 48-row ROW_MAJOR shards for the untilize output | rejected by the shard-alignment check |
| pixel_unshuffle split with more on MATH (24/8, 20/8) | 92.9 / 84.8 µs |

## 7. Remaining

- **Final HC transpose (42.4 µs, ~120 GB/s).** It uses the generic interleaved row-major kernels
  (`reader_unary_transpose_hc_interleaved_partitioned_rm.cpp` / `writer_unary_transpose_hc_interleaved_start_id_rm.cpp`):
  each core reads its output sticks from other cores' shards over the NOC, then writes them to DRAM. The untilize
  output keeps the tile-padded 96-row shard, so only 32 cores hold data. A sharded-input path that walks the core's
  own shard and only scatters the writes would be a generic `transpose_hc_rm` change; estimated 20-25 µs.
- **conv1 (131 µs) and conv2 (110 µs)** are compute-bound (TRISC1 ≈ kernel time) at HiFi2 + fp32 accumulation, which
  stay unchanged.

## 8. Validation

| Suite | Result |
|---|---|
| `tests/ttnn/unit_tests/operations/data_movement/test_pixel_unshuffle.py` | 161 passed, 28 skipped |
| `tests/ttnn/unit_tests/operations/conv/test_conv2d_full_inner_dim_height_sharded.py` | 8 passed |
| `tests/ttnn/unit_tests/operations/conv/test_conv2d.py` | 161 passed, 48 skipped |
| `tests/ttnn/unit_tests/operations/pool/test_maxpool2d.py` | 104 passed, 13 skipped |
| `tests/ttnn/unit_tests/operations/pool/test_avgpool2d.py` | 276 passed, 144 skipped |
| `tests/ttnn/unit_tests/operations/conv/test_evo50_input_preproc.py` (all 4 tests) | 4 passed, PCC 0.999978-0.999981 |

---

## 9. Files modified

All paths are under `/proj_sw/user_dev/pchandrasekaran/tt-metal/`. Committed on `pchandrasekaran/evo50_model_perf`
except the rows marked "local, not committed" (helper tools and side notes; the `tools/evo50/*.py` commands in
section 10 need a local copy of them).

### 9.1 pixel_unshuffle (section 5.1; from bev `a527966`, `fcf4827`, `4e1f1b3` + the split heuristic)

| File | Status | +/− | Change |
|---|---|---:|---|
| `ttnn/cpp/ttnn/operations/data_movement/pixel_unshuffle/device/pixel_unshuffle_channels_last_program_factory.cpp` | modified | +69 / −7 | staging CB pages, two "done" counter CBs, compute gather kernel, small-pixel split 18/6, dispatch to the tiled factory for COLUMN_MAJOR_PADDED |
| `ttnn/cpp/ttnn/operations/data_movement/pixel_unshuffle/device/kernels/dataflow/pixel_unshuffle_nhwc_sharded.cpp` | modified | +29 / −203 | reader: CB pages instead of private slots, gathers only its share; gather code moved to the header |
| `ttnn/cpp/ttnn/operations/data_movement/pixel_unshuffle/device/kernels/pixel_unshuffle_nhwc_common.hpp` | **new** | +253 | shared work walk (`advance`, `part`) and the L1 gather (paired-pixel, 12-word chunk and element paths) |
| `ttnn/cpp/ttnn/operations/data_movement/pixel_unshuffle/device/kernels/compute/pixel_unshuffle_nhwc_gather.cpp` | **new** | +135 | UNPACK / MATH / PACK gather helpers synchronised through CB stream registers |
| `ttnn/cpp/ttnn/operations/data_movement/pixel_unshuffle/device/pixel_unshuffle_channels_last_tiled_program_factory.cpp` | **new** | +140 | COLUMN_MAJOR_PADDED factory (`4e1f1b3`; not used here) |
| `ttnn/cpp/ttnn/operations/data_movement/pixel_unshuffle/device/kernels/compute/pixel_unshuffle_nhwc_tiled.cpp` | **new** | +95 | COLUMN_MAJOR_PADDED compute kernel (not used here) |
| `ttnn/cpp/ttnn/operations/data_movement/pixel_unshuffle/device/kernels/dataflow/pixel_unshuffle_nhwc_tiled_reader.cpp` | **new** | +78 | COLUMN_MAJOR_PADDED reader (not used here) |
| `ttnn/cpp/ttnn/operations/data_movement/pixel_unshuffle/device/pixel_unshuffle_device_op.cpp` | modified | +24 / −0 | validation / output spec for COLUMN_MAJOR_PADDED |
| `ttnn/cpp/ttnn/operations/data_movement/pixel_unshuffle/device/pixel_unshuffle_device_op.hpp` | modified | +6 / −0 | `create_descriptor_tiled` declaration |
| `ttnn/cpp/ttnn/operations/data_movement/pixel_unshuffle/pixel_unshuffle.cpp` | modified | +7 / −1 | COLUMN_MAJOR_PADDED routing |
| `ttnn/cpp/ttnn/operations/data_movement/pixel_unshuffle/pixel_unshuffle.hpp` | modified | +9 / −0 | `PixelUnshuffleChannelOrder::COLUMN_MAJOR_PADDED` |
| `ttnn/cpp/ttnn/operations/data_movement/pixel_unshuffle/pixel_unshuffle_nanobind.cpp` | modified | +11 / −1 | Python binding of the new channel order |
| `ttnn/cpp/ttnn/operations/data_movement/sources.cmake` | modified | +1 / −0 | tiled program factory source |
| `tests/ttnn/unit_tests/operations/data_movement/test_pixel_unshuffle.py` | **new** | +603 | pixel_unshuffle golden suite (from bev `4e1f1b3`) |

### 9.2 conv2d `full_inner_dim` for height-sharded convs (section 5.2)

| File | Status | +/− | Change |
|---|---|---:|---|
| `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp` | modified | +3 / −1 | `act_block_w` = whole window when `full_inner_dim` |
| `ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op_sharded_program_factory.cpp` | modified | +11 / −1 | no inner-dim slicing for HS + `full_inner_dim`; `ACT_WINDOW_H_READS` define on reader, writer and mcast-sender |
| `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_reader_common.hpp` | modified | +28 / −8 | `read_sticks` template params `window_h_reads`, `window_h_stride_bytes`; `act_window_h_reads` constant |
| `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_conv_activations_padded_with_halo_3x3_weights_v2.cpp` | modified | +3 / −1 | pass the new template args |
| `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp` | modified | +3 / −1 | pass the new template args |
| `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp` | modified | +3 / −1 | pass the new template args |
| `ttnn/cpp/ttnn/operations/conv/conv2d/prepare_conv2d_weights.cpp` | modified | +3 / −1 | (kh, kw, c)-contiguous weight rows for `full_inner_dim` |
| `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_op_program_factory_common.cpp` | modified | +4 / −2 | fully-buffered weight CB = `num_blocks_act_w` blocks |
| `tests/ttnn/unit_tests/operations/conv/test_conv2d_full_inner_dim_height_sharded.py` | **new** (restored from `stash@{0}`) | +52 | `full_inner_dim` vs sliced, 8 shapes |

### 9.3 Halo (section 5.3)

| File | Status | +/− | Change |
|---|---|---:|---|
| `ttnn/cpp/ttnn/operations/sliding_window/sliding_window.cpp` | modified | +110 / −16 | `transfer_direction` (hop count), `divide_blocks_between_cores` by direction with half caps, per-core `left_blocks` / `right_block_start` |
| `ttnn/cpp/ttnn/operations/sliding_window/sliding_window.hpp` | modified | +5 / −0 | `HaloGatherKernelConfig::left_blocks_per_core`, `right_block_start_per_core` |
| `ttnn/cpp/ttnn/operations/sliding_window/halo/device/untilize_with_halo_program_factory.cpp` | modified | +63 / −18 | `select_untilize_block_size` (64-row blocks), block-direction runtime args to the readers and the compute kernel |
| `ttnn/cpp/ttnn/operations/sliding_window/halo/device/kernels/dataflow/halo_gather.cpp` | modified | +37 / −10 | `halo_block_owner`; walk only the owned blocks |
| `ttnn/cpp/ttnn/operations/sliding_window/halo/device/kernels/compute/pack_untilize.cpp` | modified | +7 / −1 | push each block to its owner's CB |

### 9.4 Tests, tools and docs

| File | Status | +/− | Purpose |
|---|---|---:|---|
| `tests/ttnn/unit_tests/operations/conv/test_evo50_input_preproc.py` | modified | +114 / −0 | docstring + `_tail_opt` + `test_evo50_input_preproc_s2d_nhwc_opt` |
| `tests/ttnn/unit_tests/operations/conv/test_evo50_input_preproc_opt.py` | **new** | +201 | sweep: `tail` (ir, tw, rowt, rowtd, rowts) × `conv` (base, fid*, a416/a192/a96, nd) |
| `tests/ttnn/unit_tests/operations/conv/test_evo50_pixel_unshuffle_micro.py` | **new** | +65 | pixel_unshuffle alone on the EVO50 shape, bit-exact check, 3 timed runs |
| `tools/evo50/ops_summary.py` | local, not committed | +79 | signpost-scoped totals table for several ops_perf CSVs (`--ops` per-op lines) |
| `tools/evo50/ops_table.py` | local, not committed | +37 | the per-op tables in section 2 |
| `docs/evo50_input_preproc_further_optimization_ir_and_code.md` | **new** | – | this document |
| `docs/evo50_input_preproc_s2d_nhwc_opt.md` | local, not committed | – | short round summary |
| `docs/evo50_input_preproc_three_variant_perf.md` | local, not committed | – | the morning comparison of the three original tests |

Totals for the source changes (`git diff 47d0cdb2424 --stat -- ttnn`): 26 files, +1,137 / −273.

---

## 10. Reproduce

```bash
cd /proj_sw/user_dev/pchandrasekaran/tt-metal
ninja -C build && ninja -C build install
T=tests/ttnn/unit_tests/operations/conv/test_evo50_input_preproc.py
R=generated/profiler/reports/evo50_final
for t in test_evo50_input_preproc_s2d_nhwc test_evo50_input_preproc_s2d_nhwc_opt; do
  python -m tracy -r -v --op-support-count=20000 -o $R/$t -n $t -m "pytest $T::$t"
done
# totals side by side
python3 tools/evo50/ops_summary.py --ops \
  before=$(find $R/test_evo50_input_preproc_s2d_nhwc -name 'ops_perf*.csv') \
  after=$(find $R/test_evo50_input_preproc_s2d_nhwc_opt -name 'ops_perf*.csv')
# per-op table (section 2 format)
python3 tools/evo50/ops_table.py $(find $R/test_evo50_input_preproc_s2d_nhwc_opt -name 'ops_perf*.csv')
```
