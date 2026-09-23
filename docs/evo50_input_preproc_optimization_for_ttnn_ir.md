# EVO50 input pre-processing: baseline, space-to-depth stem, and the channels-last stem — a specification for the TTNN IR

Status 2026-09-23. tt-metal branch `pchnadrasekaran/evo50_model_perf` (base `f1f4ff75579`, plus the
`pixel_unshuffle` op). Every number below is a `DEVICE KERNEL DURATION` from one Tracy run of the three
tests in `tests/ttnn/unit_tests/operations/conv/test_evo50_input_preproc.py`, Wormhole B0, 8x8 grid,
report `ops_perf_results_evo50_final_2026_09_23_12_56_59.csv`. The tests are written op-for-op against the TTNN IR, so each
section gives (a) the TTNN IR pattern, (b) the TTNN Python that executes it, (c) the device ops report,
and (d) what to change in the compiler so it emits the pattern. Section 5 is the side-by-side comparison;
section 6 is the implementation checklist for tt-mlir.

Contents

1. The block and the reference IR
2. Test 1 — baseline (`@trace_0_forward` as compiled): IR, code, ops report, the bottleneck
3. Test 2 — space-to-depth stem: IR before/after, code, ops report
4. Test 3 — channels-last `pixel_unshuffle` feeding conv2d directly + conv blocking: IR proposed, code, ops report
5. Comparison 1 vs 2 vs 3
6. What the compiler has to emit (checklist, matching conditions, pitfalls)

---

## 1. The block

EVO50's input pre-processing, `@trace_0_forward` of
`FULL_EVO50_MODEL/INPUT_PREPROC/WITH_MLIR_TRACE/IR/ttnn_evo50_input_preproc.mlir`:

```
input_luv [1,3,384,1664] bf16 NCHW
  -> Conv2d 7x7, stride 2, pads (top 2, left 2, bottom 3, right 3), 3 -> 16 ch, ReLU     -> [1,16,192,832]
  -> Conv2d 3x3, stride 1, pad 1, groups 2, 16 -> 16 ch, ReLU                            -> [1,16,192,832]
```

Arguments and layouts as the IR declares them:

```mlir
#dram = #ttnn.buffer_type<dram>
#l1   = #ttnn.buffer_type<l1>
#ttnn_layout1  = #ttnn.ttnn_layout<(d0,d1,d2,d3) -> (d0*1152 + d1*384 + d2, d3), <1x1>, memref<1152x1664xbf16, #dram>, <interleaved>>                 // input, ROW_MAJOR
#ttnn_layout5  = #ttnn.ttnn_layout<(d0,d1,d2,d3) -> (d0*1152 + d1*384 + d2, d3), <1x1>, memref<36x52x!ttcore.tile<32x32,bf16>, #dram>, <interleaved>> // tilized input
#ttnn_layout6  = #ttnn.ttnn_layout<(d0,d1,d2,d3) -> (d0*638976 + d1*1664 + d2, d3), <8x8>, memref<1x312x!ttcore.tile<32x32,bf16>, #l1>, <interleaved>>  // NHWC, C=3 padded to 32
#ttnn_layout8  = #ttnn.ttnn_layout<(d0,d1,d2,d3) -> (d0*638976 + d1*638976 + d2, d3), <1x1>, memref<19968x1x!ttcore.tile<32x32,bf16>, #dram>, <interleaved>>
#ttnn_layout9  = #ttnn.ttnn_layout<(d0,d1,d2,d3) -> (d0*159744 + d1*159744 + d2, d3), <1x1>, memref<4992x1x!ttcore.tile<32x32,bf16>, #dram>, <interleaved>>
#ttnn_layout10 = #ttnn.ttnn_layout<(d0,d1,d2,d3) -> (d0*159744 + d1*159744 + d2, d3), <64x1>, memref<78x1x!ttcore.tile<32x32,bf16>, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,7)>]>>
#ttnn_layout12 = #ttnn.ttnn_layout<(d0,d1,d2,d3) -> (d0*3072 + d1*192 + d2, d3), <8x8>, memref<1x39x!ttcore.tile<32x32,bf16>, #l1>, <interleaved>>
#ttnn_layout4  = #ttnn.ttnn_layout<(d0,d1,d2,d3) -> (d0*3072 + d1*192 + d2, d3), <1x1>, memref<96x26x!ttcore.tile<32x32,bf16>, #dram>, <interleaved>>   // output

func.func private @trace_0_forward(
    %arg0: tensor<1x3x384x1664xbf16, #ttnn_layout1>  {ttir.name = "input_luv"},
    %arg1: tensor<16x3x7x7xbf16,     #ttnn_layout2>  {ttir.name = "fused_7x7_weights"},   // host
    %arg2: tensor<16x8x3x3xbf16,     #ttnn_layout3>  {ttir.name = "grouped_conv_weights"}, // host
    %arg3: tensor<1x1x1x16xbf16,     #ttnn_layout>,                                          // conv1 bias, host
    %arg4: tensor<1x1x1x16xbf16,     #ttnn_layout>)                                          // conv2 bias, host
    -> tensor<1x16x192x832xbf16, #ttnn_layout4>
```

Layout convention used throughout: a `memref<...xbf16>` element type is ROW_MAJOR, a
`memref<...x!ttcore.tile<32x32,bf16>>` element type is TILE. Padding order differs per API and is the
classic mistake: ONNX/TTIR `(top, left, bottom, right)`; TTNN IR and `ttnn.conv2d` `(top, bottom, left, right)`;
`torch.nn.functional.pad` `(left, right, top, bottom)`.

The torch golden all three tests are checked against (fp32):

```python
def torch_golden(x, w1, b1, w2, b2):
    """The two relu-fused convolutions, in NCHW."""
    # F.pad order is (left, right, top, bottom); C1_PAD is (top, bottom, left, right).
    xp = F.pad(x, (C1_PAD[2], C1_PAD[3], C1_PAD[0], C1_PAD[1]))
    y1 = F.relu(F.conv2d(xp, w1, b1, stride=C1_STRIDE, padding=0, groups=C1_GROUPS))
    y2 = F.relu(F.conv2d(y1, w2, b2, stride=C2_STRIDE, padding=C2_PAD[0], groups=C2_GROUPS))
    return y1, y2
```

Shared tail (`%7`..`%10` + the runtime's output untilize), identical in all three tests:

```python
def _tail(device, t6, tt_w2, tt_b2, conv2_cfg, compute_cfg, slice2):
    """@trace_0_forward %7..%10 plus the runtime's output untilize. Identical for both stems."""
    # %7 = ttnn.conv2d(%6, %arg2, %arg4, %0)
    t7 = ttnn.conv2d(
        input_tensor=t6,
        weight_tensor=tt_w2,
        bias_tensor=tt_b2,
        device=device,
        in_channels=C2_OUT,
        out_channels=C2_OUT,
        batch_size=BATCH,
        input_height=MID_H,
        input_width=MID_W,
        kernel_size=C2_K,
        stride=C2_STRIDE,
        padding=C2_PAD,
        dilation=(1, 1),
        groups=C2_GROUPS,
        conv_config=conv2_cfg,
        compute_config=compute_cfg,
        slice_config=slice2,
    )

    # %8 = ttnn.reshape(%7) {shape = [1, 192, 832, 16]}
    t8 = ttnn.reshape(t7, (BATCH, MID_H, MID_W, C2_OUT))
    _free_unless_alias(t7, t8)

    # %9 = ttnn.permute(%8) {permutation = [0, 3, 1, 2]}  -> TILE L1 interleaved
    t9 = ttnn.permute(t8, (0, 3, 1, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(t8)

    # %10 = ttnn.to_memory_config(%9)  -> TILE DRAM
    t10 = ttnn.to_memory_config(t9, ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t9)

    # The runtime untilizes the TILE DRAM result (#ttnn_layout4) on device before handing it
    # back to the host; this is the trailing UntilizeDeviceOperation of the reference report.
    out_rm = ttnn.to_layout(t10, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t10)
    return out_rm
```

---

## 2. Test 1 — baseline: `@trace_0_forward` as compiled

### 2.1 TTNN IR

```mlir
%1  = "ttnn.to_layout"(%arg0) : (tensor<1x3x384x1664xbf16, #ttnn_layout1>) -> tensor<1x3x384x1664xbf16, #ttnn_layout5>       // tilize the image
%2  = "ttnn.permute"(%1) <{permutation = array<i64: 0, 2, 3, 1>}> : (...) -> tensor<1x384x1664x3xbf16, #ttnn_layout6>          // NCHW -> NHWC, C=3 -> 32-wide tile
"ttnn.deallocate"(%1)
%3  = "ttnn.reshape"(%2) <{shape = [1, 1, 638976, 3]}> : (...) -> tensor<1x1x638976x3xbf16, #ttnn_layout7>
"ttnn.deallocate"(%2)
%4  = "ttnn.to_memory_config"(%3) : (...) -> tensor<1x1x638976x3xbf16, #ttnn_layout8>                                           // L1 -> DRAM spill
"ttnn.deallocate"(%3)
%5  = "ttnn.conv2d"(%4, %arg1, %arg3, %0) <{
        batch_size = 1, in_channels = 3, out_channels = 16, input_height = 384, input_width = 1664,
        kernel_size = array<i32: 7, 7>, stride = array<i32: 2, 2>, padding = array<i32: 2, 3, 2, 3>, dilation = array<i32: 1, 1>, groups = 1,
        compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi2, fp32_dest_acc_en = true>,
        conv2d_config = #ttnn.conv2d_config<weights_dtype = bf16, activation = <op_type = relu>, act_block_h_override = 1024,
                                            enable_kernel_stride_folding = false, config_tensors_in_dram = true>,
        conv2d_slice_config = #ttnn.conv2d_slice_config<dram_width, 0>}>
      : (...) -> tensor<1x1x159744x16xbf16, #ttnn_layout9>                                                                        // DRAM-sliced conv
"ttnn.deallocate"(%4)
%6  = "ttnn.to_memory_config"(%5) : (...) -> tensor<1x1x159744x16xbf16, #ttnn_layout10>                                          // DRAM -> L1 height-sharded
"ttnn.deallocate"(%5)
%7  = "ttnn.conv2d"(%6, %arg2, %arg4, %0) <{
        batch_size = 1, in_channels = 16, out_channels = 16, input_height = 192, input_width = 832,
        kernel_size = array<i32: 3, 3>, stride = array<i32: 1, 1>, padding = array<i32: 1, 1, 1, 1>, dilation = array<i32: 1, 1>, groups = 2,
        compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi2, fp32_dest_acc_en = true>,
        conv2d_config = #ttnn.conv2d_config<weights_dtype = bf16, activation = <op_type = relu>, deallocate_activation = true,
                                            act_block_h_override = 64, shard_layout = height_sharded,
                                            enable_kernel_stride_folding = false, config_tensors_in_dram = true>,
        conv2d_slice_config = #ttnn.conv2d_slice_config<l1_full, 0>}>
      : (...) -> tensor<1x1x159744x16xbf16, #ttnn_layout10>
%8  = "ttnn.reshape"(%7) <{shape = [1, 192, 832, 16]}> : (...) -> tensor<1x192x832x16xbf16, #ttnn_layout11>
"ttnn.deallocate"(%7)
%9  = "ttnn.permute"(%8) <{permutation = array<i64: 0, 3, 1, 2>}> : (...) -> tensor<1x16x192x832xbf16, #ttnn_layout12>          // NHWC -> NCHW
"ttnn.deallocate"(%8)
%10 = "ttnn.to_memory_config"(%9) : (...) -> tensor<1x16x192x832xbf16, #ttnn_layout4>                                            // L1 -> DRAM
"ttnn.deallocate"(%9)
return %10
```

### 2.2 TTNN Python (the test)

```python
def _stem_baseline(device, tt_in, tt_w1, tt_b1, hs_mem_cfg, conv1_cfg, compute_cfg, slice1):
    """@trace_0_forward %1..%6: tilize, permute, reshape, spill to DRAM, 7x7/s2 DRAM-sliced conv,
    re-shard into L1. Returns the <64x1> height-sharded L1 tensor the grouped conv consumes."""
    # %1 = ttnn.to_layout(%arg0)  -> TILE DRAM
    t1 = ttnn.to_layout(tt_in, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # %2 = ttnn.permute(%1) {permutation = [0, 2, 3, 1]}  -> TILE L1 interleaved
    t2 = ttnn.permute(t1, (0, 2, 3, 1), memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(t1)

    # %3 = ttnn.reshape(%2) {shape = [1, 1, 638976, 3]}
    t3 = ttnn.reshape(t2, (1, 1, IN_H * IN_W, IN_C))
    _free_unless_alias(t2, t3)

    # %4 = ttnn.to_memory_config(%3)  -> TILE DRAM
    t4 = ttnn.to_memory_config(t3, ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t3)  # frees the t2/t3 buffer

    # %5 = ttnn.conv2d(%4, %arg1, %arg3, %0)
    t5 = ttnn.conv2d(
        input_tensor=t4,
        weight_tensor=tt_w1,
        bias_tensor=tt_b1,
        device=device,
        in_channels=IN_C,
        out_channels=C1_OUT,
        batch_size=BATCH,
        input_height=IN_H,
        input_width=IN_W,
        kernel_size=C1_K,
        stride=C1_STRIDE,
        padding=C1_PAD,
        dilation=(1, 1),
        groups=C1_GROUPS,
        conv_config=conv1_cfg,
        compute_config=compute_cfg,
        slice_config=slice1,
    )
    ttnn.deallocate(t4)

    # %6 = ttnn.to_memory_config(%5)  -> TILE L1 height_sharded <64x1>
    t6 = ttnn.to_memory_config(t5, hs_mem_cfg)
    ttnn.deallocate(t5)
    return t6


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_evo50_input_preproc(device):
    """@trace_0_forward as compiled: the 2.55 ms baseline."""
    x, w1, b1, w2, b2 = _make_inputs()
    _, golden = torch_golden(x.float(), w1.float(), b1.float(), w2.float(), b2.float())

    # %arg0: [1,3,384,1664] bf16 ROW_MAJOR DRAM interleaved (#ttnn_layout1)
    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    # %arg1/%arg2 weights, %arg3/%arg4 biases: host (#system_memory)
    tt_w1 = ttnn.from_torch(w1, dtype=ttnn.bfloat16)
    tt_b1 = ttnn.from_torch(b1.reshape(1, 1, 1, C1_OUT), dtype=ttnn.bfloat16)
    tt_w2 = ttnn.from_torch(w2, dtype=ttnn.bfloat16)
    tt_b2 = ttnn.from_torch(b2.reshape(1, 1, 1, C2_OUT), dtype=ttnn.bfloat16)

    compute_cfg, conv2_cfg, slice_l1_full = _common_configs(device)

    # #ttnn.conv2d_config<weights_dtype = bf16, activation = <op_type = relu>,
    #                     act_block_h_override = 1024, enable_kernel_stride_folding = false,
    #                     config_tensors_in_dram = true>
    conv1_cfg = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),  # activation = <op_type = relu>
        act_block_h_override=1024,
        enable_kernel_stride_folding=False,
        config_tensors_in_dram=True,
    )
    # #ttnn.conv2d_slice_config<dram_width, 0>
    slice1 = ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0)

    # #ttnn_layout10: <64x1>, memref<78x1 x tile<32x32,bf16>, #l1>, height_sharded,
    #                 core_ranges = <[(0,0) - (7,7)]>
    hs_mem_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(HS_CORE_GRID[0] - 1, HS_CORE_GRID[1] - 1))}
            ),
            HS_SHARD_SHAPE,
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    def run():
        t6 = _stem_baseline(device, tt_in, tt_w1, tt_b1, hs_mem_cfg, conv1_cfg, compute_cfg, slice1)
        return _tail(device, t6, tt_w2, tt_b2, conv2_cfg, compute_cfg, slice_l1_full)

    # warm-up fills the program cache
    ttnn.deallocate(run())
    ttnn.synchronize_device(device)

    signpost(header="evo50_inference-start")
    out = run()
    ttnn.synchronize_device(device)
    result = ttnn.to_torch(out)
    signpost(header="evo50_inference-end")

    _check(result, golden)
```

### 2.3 Ops perf report (scoped to the signposts)

IR op -> device op(s). `ttnn.reshape` is a view and produces no device op.

| IR op | device op(s) | rows |
| --- | --- | --- |
| `%1 to_layout` | `Tilize` | 0 |
| `%2 permute` | `Permute` | 1 |
| `%4 to_memory_config` | `Copy` | 2 |
| `%5 conv2d` (dram_width, auto -> 2 slices) | 2 x (`PaddedSlice`, `Halo`, `Move`, `Conv2d`, `SliceWrite`) | 3-12 |
| `%6 to_memory_config` | `InterleavedToSharded` | 13 |
| `%7 conv2d` (l1_full) | `Halo`, `Move`, `Conv2d` | 14-16 |
| `%9 permute` | `Transpose`, `Transpose` | 17-18 |
| `%10 to_memory_config` | `Copy` | 19 |
| runtime output conversion | `Untilize` | 20 |

| # | device op | in -> out | cores | kernel ns | share |
| ---: | --- | --- | ---: | ---: | ---: |
| 0 | `Tilize` | DRAM_INTERLEAVED -> DRAM_INTERLEAVED | 39 | 171,010 | 6.6 % |
| 1 | `Permute` | DRAM_INTERLEAVED -> L1_INTERLEAVED | 64 | 905,853 | 34.7 % |
| 2 | `Copy` | L1_INTERLEAVED -> DRAM_INTERLEAVED | 64 | 306,925 | 11.8 % |
| 3 | `PaddedSlice` | DRAM_INTERLEAVED -> L1_HEIGHT_SHARDED | 64 | 142,933 | 5.5 % |
| 4 | `Halo` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 11,924 | 0.5 % |
| 5 | `Move` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 7,339 | 0.3 % |
| 6 | `Conv2d` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 130,113 | 5.0 % |
| 7 | `SliceWrite` | L1_HEIGHT_SHARDED -> DRAM_INTERLEAVED | 64 | 44,794 | 1.7 % |
| 8 | `PaddedSlice` | DRAM_INTERLEAVED -> L1_HEIGHT_SHARDED | 64 | 143,875 | 5.5 % |
| 9 | `Halo` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 11,732 | 0.4 % |
| 10 | `Move` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 7,381 | 0.3 % |
| 11 | `Conv2d` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 130,183 | 5.0 % |
| 12 | `SliceWrite` | L1_HEIGHT_SHARDED -> DRAM_INTERLEAVED | 64 | 43,280 | 1.7 % |
| 13 | `InterleavedToSharded` | DRAM_INTERLEAVED -> L1_HEIGHT_SHARDED | 64 | 51,905 | 2.0 % |
| 14 | `Halo` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 24,936 | 1.0 % |
| 15 | `Move` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 11,541 | 0.4 % |
| 16 | `Conv2d` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 228,963 | 8.8 % |
| 17 | `Transpose` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 8,184 | 0.3 % |
| 18 | `Transpose` | L1_HEIGHT_SHARDED -> L1_INTERLEAVED | 64 | 126,808 | 4.9 % |
| 19 | `Copy` | L1_INTERLEAVED -> DRAM_INTERLEAVED | 64 | 39,463 | 1.5 % |
| 20 | `Untilize` | DRAM_INTERLEAVED -> DRAM_INTERLEAVED | 48 | 59,673 | 2.3 % |
| | **total** | | | **2,608,815** | 100 % |

This reproduces the model run: the reference report's scoped region (rows 62-82 of the Forge Tracy CSV)
totals 2,552,407 ns over the same 20 timed ops; this test measures 2,549,142 ns for them (-0.13 %),
with the same op sequence, core counts and memory transitions.

### 2.4 The bottleneck

| | ns | share of block |
| --- | ---: | ---: |
| stem (`%1`..`%6`, 14 device ops) | 2,109,247 | 80.9 % |
| `Permute` `%2` alone | 905,853 | 34.7 % |
| `Permute` + `Copy` + 2 x `PaddedSlice` (the padded tensor's round trip) | 1,499,586 | 57.5 % |
| `PaddedSlice` + `SliceWrite` (pure data movement created by DRAM slicing) | 374,882 | 14.4 % |

Three causes, all layout, none arithmetic:

1. **Padding inflation.** `%2` permutes a TILE tensor so the 3 input channels become the tile's last dim
   and are padded to 32. The activation is `638976 x 3 x 2 B = 3.83 MB` of data but `638976 x 32 x 2 B = 40.9 MB`
   on device - a **10.67x write amplification**. `Permute` (906 us) writes it, `Copy` (316 us) spills it to DRAM,
   both `PaddedSlice`s (287 us) read it back. Every byte of the padding is moved three times.
2. **DRAM slicing of conv1.** A 7x7/s2 conv on a `384x1664x32`-padded input does not fit L1, so it runs as
   `dram_width` slices: each slice is copied DRAM -> L1 (`PaddedSlice`), convolved, written back (`SliceWrite`),
   then `%6` re-shards the DRAM result into L1 for conv2 (`InterleavedToSharded`). 418 us of the stem is this
   plumbing; the two `Conv2d`s themselves are 260 us.
3. **Tilize of the whole image** (`%1`, 170 us) exists only so that `%2` can permute in TILE layout.

---

## 3. Test 2 — space-to-depth stem (first solution)

### 3.1 The idea

A stride-2 `k x k` conv is exactly `pixel_unshuffle(2)` (space-to-depth: 4x channels, half resolution) followed
by a **stride-1** `ceil(k/2) x ceil(k/2)` conv whose weights are the original weights re-packed. Same multiply-adds,
regrouped. For EVO50: `7x7/s2, 3 ch` -> `pixel_unshuffle(2)` + `4x4/s1, 12 ch`.

```
S[c*4 + dy*2 + dx, u, v]        = X[c, 2u + dy, 2v + dx]                  (pixel_unshuffle, CHANNEL_MAJOR / torch order)
W'[oc, c*4 + dy*2 + dx, i, j]   = Wpad[oc, c, 2i + dy, 2j + dx]           (Wpad = W zero-padded 7x7 -> 8x8)
pT' = pT/2 = 1,  pL' = pL/2 = 1,  pB' = outH + kHp - 1 - H/2 - pT' = 192 + 4 - 1 - 192 - 1 = 2,  pR' = 2
   -> padding (top 1, bottom 2, left 1, right 2)  ->  output 192 x 832, unchanged
```

Weight re-pack and torch reference (this is what the compiler const-evals; `pack_weight_s2d` is the rank-6
form of the rank-4 pad/reshape/permute chain listed in section 6.2):

```python
def pack_weight_s2d(w, r=S2D_R):
    """[OC, C, kH, kW] -> [OC, C*r*r, ceil(kH/r), ceil(kW/r)], CHANNEL_MAJOR order
    (c' = c*r*r + dy*r + dx), i.e. W'[oc, c', i, j] = Wpad[oc, c, r*i + dy, r*j + dx].
    docs/evo50_stem_space_to_depth.md 2A.4."""
    OC, C, kH, kW = w.shape
    kHp, kWp = -(-kH // r), -(-kW // r)
    wp = F.pad(w, (0, kWp * r - kW, 0, kHp * r - kH))  # zero-pad kernel to (kHp*r, kWp*r): 7x7 -> 8x8
    wp = wp.reshape(OC, C, kHp, r, kWp, r)  # [oc, c, i, dy, j, dx]
    wp = wp.permute(0, 1, 3, 5, 2, 4)  # [oc, c, dy, dx, i, j]
    return wp.reshape(OC, C * r * r, kHp, kWp).contiguous()


def torch_stem_s2d(x, w1, b1):
    """conv1 in its space-to-depth form; must equal torch_golden()[0]."""
    s = F.pixel_unshuffle(x, S2D_R)  # [N, 12, 192, 832], channel_major
    s = F.pad(s, (S2D_PAD[2], S2D_PAD[3], S2D_PAD[0], S2D_PAD[1]))
    return F.relu(F.conv2d(s, pack_weight_s2d(w1), b1, stride=1))
```

Checked in torch before touching the device: `max |torch_stem_s2d - conv7x7| = 1.5e-5` (fp32).

### 3.2 TTNN IR, before -> after

Before (test 1, `%1`..`%6`):

```mlir
%1 = "ttnn.to_layout"(%arg0)                          -> tensor<1x3x384x1664xbf16, #tile_dram>
%2 = "ttnn.permute"(%1) {permutation = [0,2,3,1]}     -> tensor<1x384x1664x3xbf16, #tile_l1_8x8>        // 40.9 MB
%3 = "ttnn.reshape"(%2) {shape = [1,1,638976,3]}
%4 = "ttnn.to_memory_config"(%3)                      -> tensor<1x1x638976x3xbf16, #tile_dram>
%5 = "ttnn.conv2d"(%4, %w7x7, %b, %dev) {k=7x7, s=2, pad=[2,3,2,3], in=3, H=384, W=1664, dram_width}
                                                       -> tensor<1x1x159744x16xbf16, #tile_dram>
%6 = "ttnn.to_memory_config"(%5)                      -> tensor<1x1x159744x16xbf16, #tile_l1_hs_64x1>
```

After (what tt-mlir's `--ttir-strided-conv-space-to-depth-opt` emits; Forge4 `ttnn_evo50_onnx_deploy.mlir`, pass on):

```mlir
// input stays ROW_MAJOR in DRAM: nothing tilizes the image any more
%1 = "ttnn.pixel_unshuffle"(%arg0) <{downscale_factor = 2 : i32, channel_order = channel_major,
                                      memory_config = #ttnn.memory_config<#l1, <interleaved>>}>
      : (tensor<1x3x384x1664xbf16, #rm_dram>) -> tensor<1x12x192x832xbf16, #tile_l1_8x8>          // op emits ROW_MAJOR, runtime tilizes
%2 = "ttnn.permute"(%1) <{permutation = array<i64: 0, 2, 3, 1>}>
      : (...) -> tensor<1x192x832x12xbf16, #tile_l1_8x8>                                           // 10.2 MB, 4x less than before
%3 = "ttnn.reshape"(%2) <{shape = [1, 1, 159744, 12]}> : (...) -> tensor<1x1x159744x12xbf16, #tile_l1_8x8>
%4 = "ttnn.conv2d"(%3, %w_packed /* const-eval: 16x12x4x4 -> prepared 1x1x192x16 */, %b, %dev) <{
        batch_size = 1, in_channels = 12, out_channels = 16, input_height = 192, input_width = 832,
        kernel_size = array<i32: 4, 4>, stride = array<i32: 1, 1>, padding = array<i32: 1, 2, 1, 2>, dilation = array<i32: 1, 1>, groups = 1,
        compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi2, fp32_dest_acc_en = true>,
        conv2d_config = #ttnn.conv2d_config<weights_dtype = bf16, activation = <op_type = relu>, deallocate_activation = true,
                                            act_block_h_override = 64, shard_layout = height_sharded,
                                            enable_kernel_stride_folding = false, config_tensors_in_dram = true>,
        conv2d_slice_config = #ttnn.conv2d_slice_config<l1_full, 0>}>
      : (...) -> tensor<1x1x159744x16xbf16, #tile_l1_hs_64x1>                                      // fits L1: no slicing
// the to_memory_config (%6) is gone: %7 conv2d consumes %4 directly
```

### 3.3 TTNN Python (the test)

```python
def _stem_s2d(device, tt_in, tt_w1_packed, tt_b1, conv1_s2d_cfg, compute_cfg, slice_l1_full):
    """The stem after --ttir-strided-conv-space-to-depth-opt (doc section 2 / 2A.6 'After').
    Returns the same <64x1> height-sharded L1 tensor as _stem_baseline."""
    # (1') ttnn.pixel_unshuffle(%arg0) {downscale_factor = 2, channel_order = channel_major,
    #      memory_config = <#l1, interleaved>}  -> [1,12,192,832] TILE L1
    #      input stays ROW_MAJOR (no tilize of the image); kernel emits ROW_MAJOR, tilizes on request
    s1 = ttnn.pixel_unshuffle(
        tt_in,
        S2D_R,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_layout=ttnn.TILE_LAYOUT,
        channel_order=ttnn.PixelUnshuffleChannelOrder.CHANNEL_MAJOR,
    )

    # (2') ttnn.permute {permutation = [0, 2, 3, 1]}  -> [1,192,832,12] TILE L1  (4x fewer bytes than %2)
    s2 = ttnn.permute(s1, (0, 2, 3, 1), memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(s1)

    # (3') ttnn.reshape {shape = [1, 1, 159744, 12]}
    s3 = ttnn.reshape(s2, (1, 1, S2D_H * S2D_W, S2D_C))
    _free_unless_alias(s2, s3)  # s3 is a view of s2; conv2d(deallocate_activation=True) frees it below

    # (4') ttnn.conv2d 4x4 / s1 / pad [1,2,1,2] on the re-packed weights, l1_full, height sharded.
    #      deallocate_activation=true frees s3.  Output: [1,1,159744,16] TILE L1 height_sharded <64x1>
    y1 = ttnn.conv2d(
        input_tensor=s3,
        weight_tensor=tt_w1_packed,
        bias_tensor=tt_b1,
        device=device,
        in_channels=S2D_C,
        out_channels=C1_OUT,
        batch_size=BATCH,
        input_height=S2D_H,
        input_width=S2D_W,
        kernel_size=S2D_K,
        stride=(1, 1),
        padding=S2D_PAD,
        dilation=(1, 1),
        groups=C1_GROUPS,
        conv_config=conv1_s2d_cfg,
        compute_config=compute_cfg,
        slice_config=slice_l1_full,
    )
    # (5') the to_memory_config is gone: the next conv consumes the sharded L1 result directly
    return y1


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_evo50_input_preproc_s2d(device):
    """Same block with the stem in space-to-depth form (docs/evo50_stem_space_to_depth.md)."""
    x, w1, b1, w2, b2 = _make_inputs()
    y1_ref, golden = torch_golden(x.float(), w1.float(), b1.float(), w2.float(), b2.float())

    # The re-packed stem is the same arithmetic regrouped: check that in torch before touching the device.
    y1_s2d = torch_stem_s2d(x.float(), w1.float(), b1.float())
    assert list(y1_s2d.shape) == [BATCH, C1_OUT, MID_H, MID_W], f"packed conv shape {y1_s2d.shape}"
    assert torch.allclose(y1_s2d, y1_ref, atol=1e-3, rtol=1e-4), (
        f"packed-weight stem differs from the 7x7/s2 conv: max |diff| {(y1_s2d - y1_ref).abs().max().item():.3e}"
    )
    assert S2D_PAD == (1, 2, 1, 2), S2D_PAD

    # %arg0 unchanged: ROW_MAJOR DRAM; the stem no longer tilizes it
    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    # conv1 weights re-packed on host to [16,12,4,4] (the compiler const-evals this chain)
    tt_w1_packed = ttnn.from_torch(pack_weight_s2d(w1.float()).to(torch.bfloat16), dtype=ttnn.bfloat16)
    tt_b1 = ttnn.from_torch(b1.reshape(1, 1, 1, C1_OUT), dtype=ttnn.bfloat16)
    tt_w2 = ttnn.from_torch(w2, dtype=ttnn.bfloat16)
    tt_b2 = ttnn.from_torch(b2.reshape(1, 1, 1, C2_OUT), dtype=ttnn.bfloat16)

    compute_cfg, conv2_cfg, slice_l1_full = _common_configs(device)

    # conv2d_config = <relu, shard_layout = height_sharded, act_block_h_override = 64,
    #                  deallocate_activation = true, config_tensors_in_dram = true>   (doc 2A.6)
    conv1_s2d_cfg = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        deallocate_activation=True,
        act_block_h_override=64,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        enable_kernel_stride_folding=False,
        config_tensors_in_dram=True,
    )

    def run():
        y1 = _stem_s2d(device, tt_in, tt_w1_packed, tt_b1, conv1_s2d_cfg, compute_cfg, slice_l1_full)
        return _tail(device, y1, tt_w2, tt_b2, conv2_cfg, compute_cfg, slice_l1_full)

    ttnn.deallocate(run())
    ttnn.synchronize_device(device)

    signpost(header="evo50_s2d_inference-start")
    out = run()
    ttnn.synchronize_device(device)
    result = ttnn.to_torch(out)
    signpost(header="evo50_s2d_inference-end")

    _check(result, golden)
```

### 3.4 Ops perf report

| IR op | device op(s) | rows |
| --- | --- | --- |
| `pixel_unshuffle` (TILE requested) | `PixelUnshuffle`, `Tilize` | 0-1 |
| `permute` | `Permute` | 2 |
| `conv2d` (l1_full, L1 interleaved input) | `InterleavedToSharded`, `Move`, `Halo`, `Move`, `Conv2d` | 3-7 |
| `%7 conv2d` (input already sharded by conv1) | `Halo`, `Conv2d` | 8-9 |
| `%9 permute`, `%10 to_memory_config`, output untilize | `Transpose`, `Transpose`, `Copy`, `Untilize` | 10-13 |

| # | device op | in -> out | cores | kernel ns | share |
| ---: | --- | --- | ---: | ---: | ---: |
| 0 | `PixelUnshuffle` | DRAM_INTERLEAVED -> L1_INTERLEAVED | 64 | 119,499 | 9.5 % |
| 1 | `Tilize` | L1_INTERLEAVED -> L1_INTERLEAVED | 36 | 50,552 | 4.0 % |
| 2 | `Permute` | L1_INTERLEAVED -> L1_INTERLEAVED | 64 | 229,259 | 18.2 % |
| 3 | `InterleavedToSharded` | L1_INTERLEAVED -> L1_HEIGHT_SHARDED | 64 | 35,356 | 2.8 % |
| 4 | `Move` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 7,526 | 0.6 % |
| 5 | `Halo` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 34,406 | 2.7 % |
| 6 | `Move` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 11,822 | 0.9 % |
| 7 | `Conv2d` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 289,919 | 23.0 % |
| 8 | `Halo` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 24,472 | 1.9 % |
| 9 | `Conv2d` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 229,435 | 18.2 % |
| 10 | `Transpose` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 6,718 | 0.5 % |
| 11 | `Transpose` | L1_HEIGHT_SHARDED -> L1_INTERLEAVED | 64 | 125,089 | 9.9 % |
| 12 | `Copy` | L1_INTERLEAVED -> DRAM_INTERLEAVED | 64 | 39,344 | 3.1 % |
| 13 | `Untilize` | DRAM_INTERLEAVED -> DRAM_INTERLEAVED | 48 | 58,841 | 4.7 % |
| | **total** | | | **1,262,238** | 100 % |

Stem: 2,109,247 -> 778,339 ns (-63.1 %), 14 -> 8 device ops. Block: 2,608,815 -> 1,262,238 ns (-51.6 %).
PCC 0.999980. This matches the Forge4 real-model measurement in `evo50_stem_space_to_depth.md` (792 us stem)
op for op within a few microseconds.

What remains in this stem is layout plumbing around a 120 us gather: `Tilize` 45-57 + `Permute` 229 (NCHW tile ->
NHWC) + `InterleavedToSharded` 35 + two `Move`s 19 + a `Halo` that must untilize 34 = ~370 us before the conv, and a
4x4 conv at 290 us that is slower than the two sliced 7x7s it replaced (260 us) because it is reader-bound at
`act_block_h_override = 64`.

---

## 4. Test 3 — channels-last `pixel_unshuffle` feeding conv2d directly + conv blocking

### 4.1 What was measured before writing any kernel

**conv2d config sweep on the test-2 stem** (only `Conv2dConfig` knobs; all PCC 0.999986):

| knob | `Conv2d` us | note |
| --- | ---: | --- |
| `act_block_h_override = 64` (IR) | 291 | reader-bound |
| 96 / 192 / 416 / 832 / 1248 | 283 / 202 / 224 / 203 / 199 | >= 192 is the floor. 2496 rows/core = 78 tiles; the block must divide it (32, 64, 96, 192, 416, 832, 1248, 2496) |
| + `enable_act_double_buffer` / `force_split_reader` / `enable_weights_double_buffer` | 202 / 202 / 290 | no further gain |
| `reallocate_halo_output = false` | - | removes the 13 us `Move` after `Halo` |
| `enable_activation_reuse = true` | fails | needs act_block_h > output row width in tiles (27+); 1248 then fails program build |

**conv2d fed a ROW_MAJOR height-sharded NHWC input directly.** Built on host: `[1,1,159744,16]` bf16 ROW_MAJOR,
`HEIGHT_SHARDED` L1, grid (0,0)-(7,7), shard `[2496, 16]`; conv2d with `in_channels = 16` (4 zero channels, 4 zero weight taps):

| act_block_h_override | device ops | `Halo` | `Conv2d` | total us |
| ---: | --- | ---: | ---: | ---: |
| 64 | `Halo`, `Conv2d` | 15.8 | 265.8 | 281.6 |
| 192 | `Halo`, `Conv2d` | 15.3 | 148.0 | 163.2 |
| 192 + act double buffer | `Halo`, `Conv2d` | 15.3 | 138.5 | 153.8 |
| **832 + act double buffer** | `Halo`, `Conv2d` | 14.9 | **130.5** | **145.4** |

No `InterleavedToSharded`, no `Move`s, and `Halo` drops 35 -> 15 us because it no longer untilizes; the conv is faster
on the row-major sharded input (130 vs 200 us on tiled). This fixed the target: make `pixel_unshuffle` emit exactly that tensor.

Why 16 channels and not 12: conv2d's row-major sharded input path aligns channels to the shard width
(`get_input_channels_alignment`: width % 32 -> 32, % 16 -> 16, % 8 -> 8, else 32 and a re-shard). A 12-wide
shard (24 B sticks) is not L1-aligned and would force a re-shard; 16 (32 B sticks) is accepted as is. The 4 pad
channels must be **zero**, not garbage: they multiply zero weights, and `NaN * 0 = NaN`.

### 4.2 The op: `ttnn.pixel_unshuffle(x, r, memory_config=<HEIGHT_SHARDED L1>, channels_last=True, padded_channels=16)`

New output mode of the tt-metal op (`ttnn/cpp/ttnn/operations/data_movement/pixel_unshuffle/`), on this branch:

| | |
| --- | --- |
| output | `[N, H/r, W/r, padded_channels]` ROW_MAJOR, `HEIGHT_SHARDED` in L1; channels `C*r^2 .. padded_channels-1` are zero |
| equals | `pixel_unshuffle(x, r).permute(0, 2, 3, 1)` zero-padded on the channel axis — the activation a height-sharded conv2d reads |
| shard spec | `[ceil(N*Ho*Wo / ncores), padded_channels]` over the conv's grid, ROW_MAJOR orientation; every core writes its own shard |
| constraints | output HEIGHT_SHARDED in L1; `shard_width == padded_channels`; `padded_channels * datum` a multiple of the L1 alignment (16 B); shards cover all pixels; `padded_channels >= C*r^2`; input ROW_MAJOR (a TILE input is untilized first); 2- or 4-byte elements |
| default `padded_channels` | `C*r^2` rounded up to the L1 alignment in elements (12 -> 16 for bf16) |
| `channel_order` | CHANNEL_MAJOR (torch, fast path) or SPATIAL_MAJOR (ONNX SpaceToDepth, element path) |
| device ops | exactly one, `PixelUnshuffle`, DRAM -> L1_HEIGHT_SHARDED, 64 cores |
| kernel | `pixel_unshuffle_nhwc_sharded.cpp`: core i owns shard i; the two dataflow RISCs split each image row at a 32 B-aligned input column, each reads only its `r*C` half-rows from DRAM, gathers into the local shard with 32-bit stores; no NOC writes |
| golden | `tests/ttnn/unit_tests/operations/data_movement/test_pixel_unshuffle_channels_last.py`, 11 cases bit-exact (both channel orders, r = 2/3/4, bf16 + fp32, batch 2, shards cutting image rows) |

### 4.3 TTNN IR, proposed

This is the pattern the compiler should emit for a matched strided conv (see section 6 for conditions).
`#rm_l1_hs` is a ROW_MAJOR height-sharded layout - note the plain `bf16` element type and the `<64x1>` grid:

```mlir
#ttnn_layout_pu = #ttnn.ttnn_layout<(d0,d1,d2,d3) -> (d0*159744 + d1*832 + d2, d3), <64x1>, memref<2496x16xbf16, #l1>, <height_sharded>,
                                    core_ranges = <[#ttnn.core_range<(0,0), (7,7)>]>>     // ROW_MAJOR, 16 = 12 packed channels + 4 zero
#ttnn_layout_pu_flat = #ttnn.ttnn_layout<(d0,d1,d2,d3) -> (d0*159744 + d1*159744 + d2, d3), <64x1>, memref<2496x16xbf16, #l1>, <height_sharded>,
                                    core_ranges = <[#ttnn.core_range<(0,0), (7,7)>]>>

// (1) space-to-depth straight from the row-major model input, NHWC, channel-padded, height-sharded, one device op
%1 = "ttnn.pixel_unshuffle"(%arg0) <{downscale_factor = 2 : i32, channel_order = channel_major,
                                      channels_last = true, padded_channels = 16 : i32,
                                      memory_config = #ttnn.memory_config<#l1, <height_sharded>, #ttnn.shard_spec<<[(0,0)-(7,7)]>, <2496x16>, <row_major>>>}>
      : (tensor<1x3x384x1664xbf16, #ttnn_layout1>) -> tensor<1x192x832x16xbf16, #ttnn_layout_pu>
// (2) flatten NHW - a view, no device op.  NO permute, NO to_layout, NO to_memory_config here.
%2 = "ttnn.reshape"(%1) <{shape = [1, 1, 159744, 16]}> : (...) -> tensor<1x1x159744x16xbf16, #ttnn_layout_pu_flat>
// (3) the packed conv, in_channels = 16, weights = pack_weight_s2d(W) zero-padded 12 -> 16 input channels
%3 = "ttnn.conv2d"(%2, %w_packed16 /* const-eval: 16x16x4x4 -> prepared */, %b, %dev) <{
        batch_size = 1, in_channels = 16, out_channels = 16, input_height = 192, input_width = 832,
        kernel_size = array<i32: 4, 4>, stride = array<i32: 1, 1>, padding = array<i32: 1, 2, 1, 2>, dilation = array<i32: 1, 1>, groups = 1,
        compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi2, fp32_dest_acc_en = true>,
        conv2d_config = #ttnn.conv2d_config<weights_dtype = bf16, activation = <op_type = relu>, deallocate_activation = true,
                                            reallocate_halo_output = false, act_block_h_override = 832, enable_act_double_buffer = true,
                                            shard_layout = height_sharded, enable_kernel_stride_folding = false, config_tensors_in_dram = true>,
        conv2d_slice_config = #ttnn.conv2d_slice_config<l1_full, 0>}>
      : (...) -> tensor<1x1x159744x16xbf16, #ttnn_layout10>                                     // TILE, height-sharded <64x1>, as before
// (4) grouped conv consumes %3 directly; same attributes as the IR except the blocking knobs:
%4 = "ttnn.conv2d"(%3, %arg2, %arg4, %dev) <{ ... groups = 2, kernel_size = [3,3], padding = [1,1,1,1], ...
        conv2d_config = #ttnn.conv2d_config<weights_dtype = bf16, activation = <op_type = relu>, deallocate_activation = true,
                                            reallocate_halo_output = false, act_block_h_override = 832, enable_act_double_buffer = true,
                                            shard_layout = height_sharded, enable_kernel_stride_folding = false, config_tensors_in_dram = true>, ...}>
// %8..%10 unchanged (reshape, permute [0,3,1,2], to_memory_config -> #ttnn_layout4)
```

### 4.4 TTNN Python (the test)

```python
def _stem_s2d_nhwc(device, tt_in, tt_w1_packed16, tt_b1, conv1_cfg, compute_cfg, slice_l1_full, hs_rm_cfg):
    """Space-to-depth stem with pixel_unshuffle emitting the conv's input directly:
    NHWC [1,192,832,16] ROW_MAJOR, height-sharded over the 64-core grid (2496 pixels per core).
    No tilize, no permute, no re-shard; conv2d's halo reads the row-major shards as they are."""
    # pixel_unshuffle(%arg0) {downscale_factor = 2, channel_order = channel_major, channels_last}
    s1 = ttnn.pixel_unshuffle(
        tt_in,
        S2D_R,
        memory_config=hs_rm_cfg,
        channel_order=ttnn.PixelUnshuffleChannelOrder.CHANNEL_MAJOR,
        channels_last=True,
        padded_channels=S2D_CP,
    )
    # reshape [1, 1, 159744, 16] (view of the sharded tensor)
    s3 = ttnn.reshape(s1, (1, 1, S2D_H * S2D_W, S2D_CP))
    _free_unless_alias(s1, s3)

    # 4x4 / s1 / pad [1,2,1,2], in_channels = 16 (4 zero taps), l1_full, height sharded on the input's grid
    y1 = ttnn.conv2d(
        input_tensor=s3,
        weight_tensor=tt_w1_packed16,
        bias_tensor=tt_b1,
        device=device,
        in_channels=S2D_CP,
        out_channels=C1_OUT,
        batch_size=BATCH,
        input_height=S2D_H,
        input_width=S2D_W,
        kernel_size=S2D_K,
        stride=(1, 1),
        padding=S2D_PAD,
        dilation=(1, 1),
        groups=C1_GROUPS,
        conv_config=conv1_cfg,
        compute_config=compute_cfg,
        slice_config=slice_l1_full,
    )
    return y1


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_evo50_input_preproc_s2d_nhwc(device):
    """S2D stem with pixel_unshuffle(channels_last=True) feeding conv2d directly, plus the
    conv2d config found by sweep for BOTH convs (act_block_h_override=832, act double buffer,
    no halo realloc): conv1 291 -> 131 us, the grouped conv 229 -> 123 us. Same arithmetic as
    the IR's config; only blocking / buffering knobs differ."""
    x, w1, b1, w2, b2 = _make_inputs()
    _, golden = torch_golden(x.float(), w1.float(), b1.float(), w2.float(), b2.float())

    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    # packed [16,12,4,4] -> [16,16,4,4]: input channels 12..15 are the zero pad channels of the activation
    w1_packed16 = F.pad(pack_weight_s2d(w1.float()), (0, 0, 0, 0, 0, S2D_CP - S2D_C))
    tt_w1_packed16 = ttnn.from_torch(w1_packed16.to(torch.bfloat16), dtype=ttnn.bfloat16)
    tt_b1 = ttnn.from_torch(b1.reshape(1, 1, 1, C1_OUT), dtype=ttnn.bfloat16)
    tt_w2 = ttnn.from_torch(w2, dtype=ttnn.bfloat16)
    tt_b2 = ttnn.from_torch(b2.reshape(1, 1, 1, C2_OUT), dtype=ttnn.bfloat16)

    compute_cfg, _, slice_l1_full = _common_configs(device)

    # sweep result (docs/evo50_input_preproc_ttnn_repro.md section 7): the two convs are
    # reader-bound at the IR's act_block_h_override=64; 832 (= one 26-tile output row block,
    # 2496/832 = 3 blocks per core) is the floor, double-buffered activations add a little,
    # and reallocate_halo_output=False removes a Move.
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
    conv2_cfg = ttnn.Conv2dConfig(**tuned)

    # pixel_unshuffle output: [1,192,832,16] ROW_MAJOR, <64x1> height-sharded, shard [2496, 16]
    hs_rm_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(HS_CORE_GRID[0] - 1, HS_CORE_GRID[1] - 1))}
            ),
            (S2D_H * S2D_W // (HS_CORE_GRID[0] * HS_CORE_GRID[1]), S2D_CP),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    def run():
        y1 = _stem_s2d_nhwc(device, tt_in, tt_w1_packed16, tt_b1, conv1_cfg, compute_cfg, slice_l1_full, hs_rm_cfg)
        return _tail(device, y1, tt_w2, tt_b2, conv2_cfg, compute_cfg, slice_l1_full)

    ttnn.deallocate(run())
    ttnn.synchronize_device(device)

    signpost(header="evo50_s2d_nhwc_inference-start")
    out = run()
    ttnn.synchronize_device(device)
    result = ttnn.to_torch(out)
    signpost(header="evo50_s2d_nhwc_inference-end")

    _check(result, golden)
```

### 4.5 Ops perf report

| IR op | device op(s) | rows |
| --- | --- | --- |
| `pixel_unshuffle` (channels_last) | `PixelUnshuffle` | 0 |
| `reshape` | none | |
| `conv2d` 4x4 (row-major sharded input) | `Halo`, `Conv2d` | 1-2 |
| grouped `conv2d` | `Halo`, `Conv2d` | 3-4 |
| `permute`, `to_memory_config`, output untilize | `Transpose`, `Transpose`, `Copy`, `Untilize` | 5-8 |

| # | device op | in -> out | cores | kernel ns | share |
| ---: | --- | --- | ---: | ---: | ---: |
| 0 | `PixelUnshuffle` | DRAM_INTERLEAVED -> L1_HEIGHT_SHARDED | 64 | 137,286 | 20.6 % |
| 1 | `Halo` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 15,913 | 2.4 % |
| 2 | `Conv2d` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 131,162 | 19.7 % |
| 3 | `Halo` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 24,246 | 3.6 % |
| 4 | `Conv2d` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 123,271 | 18.5 % |
| 5 | `Transpose` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 6,667 | 1.0 % |
| 6 | `Transpose` | L1_HEIGHT_SHARDED -> L1_INTERLEAVED | 64 | 130,073 | 19.5 % |
| 7 | `Copy` | L1_INTERLEAVED -> DRAM_INTERLEAVED | 64 | 39,734 | 6.0 % |
| 8 | `Untilize` | DRAM_INTERLEAVED -> DRAM_INTERLEAVED | 48 | 58,908 | 8.8 % |
| | **total** | | | **667,260** | 100 % |

Stem: 778,339 -> 284,361 ns (-63.5 % vs test 2, -86.5 % vs test 1), 3 device ops.
Tail: 483,899 -> 382,899 ns from the grouped conv's blocking (`Conv2d` 229 -> 123 us, `Move` gone).
Block: 2,608,815 -> 667,260 ns (-74.4 %). PCC 0.999980.

The new `PixelUnshuffle` is 137 us against 119 us for the NCHW kernel of test 2: it does 14 memory ops per pixel
(12 data words + 2 zero-pad words) against 12, and both kernels sit at the per-core L1 scalar bandwidth ceiling
(a 4-pixel-per-iteration gather was tried: 139 us, no gain). It still removes 45 + 229 + 35 + 19 us of ops after it.

---

## 5. Comparison: test 1 vs test 2 vs test 3

### 5.1 Totals

| | test 1 baseline | test 2 S2D | test 3 S2D + channels_last + blocking |
| --- | ---: | ---: | ---: |
| stem device ops | 14 | 8 | **3** |
| stem kernel time | 2,109,247 ns | 778,339 ns | **284,361 ns** |
| stem vs baseline | - | -63.1 % | **-86.5 %** |
| tail (grouped conv -> output) | 499,568 ns | 483,899 ns | **382,899 ns** |
| block device ops | 21 | 14 | **9** |
| block kernel time | 2,608,815 ns | 1,262,238 ns | **667,260 ns** |
| block vs baseline | - | -51.6 % | **-74.4 %** |
| speed-up | 1.00x | 2.07x | **3.91x** |
| PCC vs torch golden | 0.999978 | 0.999980 | 0.999980 |
| bytes the permute moves | 40.9 MB | 10.2 MB | 0 (no permute) |
| conv1 input | `[1,1,638976,3]` TILE DRAM, 2 DRAM slices | `[1,1,159744,12]` TILE L1, one program | `[1,1,159744,16]` ROW_MAJOR L1 height-sharded, one program |

### 5.2 Per device-op type

| device op | test 1 ns | test 2 ns | test 3 ns | 1 -> 3 |
| --- | ---: | ---: | ---: | ---: |
| `Permute` | 905,853 | 229,259 | 0 | -905,853 |
| `Conv2d` | 489,259 | 519,354 | 254,433 | -234,826 |
| `Copy` | 346,388 | 39,344 | 39,734 | -306,654 |
| `PaddedSlice` | 286,808 | 0 | 0 | -286,808 |
| `Tilize` | 171,010 | 50,552 | 0 | -171,010 |
| `Transpose` | 134,992 | 131,807 | 136,740 | +1,748 |
| `SliceWrite` | 88,074 | 0 | 0 | -88,074 |
| `Untilize` | 59,673 | 58,841 | 58,908 | -765 |
| `InterleavedToSharded` | 51,905 | 35,356 | 0 | -51,905 |
| `Halo` | 48,592 | 58,878 | 40,159 | -8,433 |
| `Move` | 26,261 | 19,348 | 0 | -26,261 |
| `PixelUnshuffle` | 0 | 119,499 | 137,286 | +137,286 |
| **total** | **2,608,815** | **1,262,238** | **667,260** | **-1,941,555** |

### 5.3 Stem, op by op, side by side

| # | test 1 | ns | test 2 | ns | test 3 | ns |
| ---: | --- | ---: | --- | ---: | --- | ---: |
| 0 | `Tilize` | 171,010 | `PixelUnshuffle` | 119,499 | `PixelUnshuffle` | 137,286 |
| 1 | `Permute` | 905,853 | `Tilize` | 50,552 | `Halo` | 15,913 |
| 2 | `Copy` | 306,925 | `Permute` | 229,259 | `Conv2d` | 131,162 |
| 3 | `PaddedSlice` | 142,933 | `InterleavedToSharded` | 35,356 |  |  |
| 4 | `Halo` | 11,924 | `Move` | 7,526 |  |  |
| 5 | `Move` | 7,339 | `Halo` | 34,406 |  |  |
| 6 | `Conv2d` | 130,113 | `Move` | 11,822 |  |  |
| 7 | `SliceWrite` | 44,794 | `Conv2d` | 289,919 |  |  |
| 8 | `PaddedSlice` | 143,875 |  |  |  |  |
| 9 | `Halo` | 11,732 |  |  |  |  |
| 10 | `Move` | 7,381 |  |  |  |  |
| 11 | `Conv2d` | 130,183 |  |  |  |  |
| 12 | `SliceWrite` | 43,280 |  |  |  |  |
| 13 | `InterleavedToSharded` | 51,905 |  |  |  |  |
| | **stem** | **2,109,247** | **stem** | **778,339** | **stem** | **284,361** |

---

## 6. What the compiler has to emit — implementation checklist for tt-mlir

### 6.1 Matching conditions (TTIR level, on a `channel_last` `ttir.conv2d` whose input is `permute{0,2,3,1}` of NCHW)

- rank-4 static shapes; `stride == (2, 2)`; `dilation == (1, 1)`; `groups == 1`
- `pT` and `pL` even; `H` and `W` even
- `C * 4 <= 32` (packed channels fit one tile / one aligned stick; EVO50: 12)
- `pB' >= 0`, `pR' >= 0`, and `pB'*2 >= pB - (Hk - kH)` (the packed padding covers what the original conv read)
- for the channels-last form additionally: the conv's parallel config is height-sharded on a grid `G`; `N*Ho*Wo` pixels are
  split as `shard_h = ceil(N*Ho*Wo / |G|)` rows per core (EVO50: 159744 / 64 = 2496); `padded_channels = round_up(C*4, 8)`
  chosen so `padded_channels % 16 == 0` or `% 32 == 0` if possible (16 for EVO50), and `in_channels` of the conv is set to it

### 6.2 Rewrite (test 2, already implemented in tt-mlir as `--ttir-strided-conv-space-to-depth-opt`)

```
before:  x(NCHW) -> permute{0,2,3,1} -> conv2d(k7 s2 pad[t2,l2,b3,r3], W[16x3x7x7])
after:   x(NCHW) -> pixel_unshuffle(2, channel_major) -> permute{0,2,3,1} -> conv2d(k4 s1 pad[t1,l1,b2,r2], W'[16x12x4x4])
```

Weight chain on the parameter (rank <= 4 ops, const-eval folds it), for `W[16,3,7,7]`:

| # | op | shape after | index meaning |
| ---: | --- | --- | --- |
| 0 | `pad` (bottom 1, right 1, value 0) | `16x3x8x8` | `Wpad[oc, c, ky, kx]` |
| 1 | `reshape` | `48x4x2x8` | `(oc,c), i, dy, kx`  (`ky = 2i + dy`) |
| 2 | `permute {0,2,1,3}` | `48x2x4x8` | `(oc,c), dy, i, kx` |
| 3 | `reshape` | `384x4x2` | `(oc,c,dy,i), j, dx`  (`kx = 2j + dx`) |
| 4 | `permute {0,2,1}` | `384x2x4` | `(oc,c,dy,i), dx, j` |
| 5 | `reshape` | `96x4x2x4` | `(oc,c,dy), i, dx, j` |
| 6 | `permute {0,2,1,3}` | `96x2x4x4` | `(oc,c,dy), dx, i, j` |
| 7 | `reshape` | `16x12x4x4` | `oc, c*4 + dy*2 + dx, i, j` = `W'` |

### 6.3 Rewrite (test 3) — what changes on top of 6.2

1. **Drop the `permute{0,2,3,1}` after `pixel_unshuffle`** and set `channels_last = true, padded_channels = Cp` on the
   `ttnn.pixel_unshuffle` op. Its result type is NHWC `[N, Ho, Wo, Cp]`, **ROW_MAJOR** (plain `bf16` memref), `height_sharded`,
   on the conv's grid with shard `[shard_h, Cp]`. This needs the op definition in the TTNN dialect to grow the two attributes,
   and the runtime to pass them through to `ttnn::pixel_unshuffle(..., channels_last, padded_channels)`.
2. **Weights**: append `8.` `pad` input channels `C*4 -> Cp` with zeros to the chain in 6.2 (`16x12x4x4 -> 16x16x4x4`), and set
   the conv's `in_channels = Cp`.
3. **Layout pass**: `TTNNLayout` must leave `pixel_unshuffle`'s **operand** row-major (already done for test 2 in the BEV branch) and
   must also leave its **result** row-major sharded - do not insert a `to_layout` (tilize) or `to_memory_config` between it and the
   conv. The `reshape` to `[1,1,N*Ho*Wo,Cp]` in between is a view on this layout.
4. **conv2d config**: `act_block_h_override = 832`, `enable_act_double_buffer = true`, `reallocate_halo_output = false` on the packed
   conv **and** on the following grouped conv (they share the geometry: 2496 rows per core). The values are geometry-specific:
   `act_block_h_override` must be a multiple of 32 that divides the per-core row count; 192 already gives most of the gain if 832
   does not fit L1 for a different shape. The op-model's L1 estimate must include the doubled activation CB.
5. **Op-model**: the conv2d query must model the row-major sharded input (no `InterleavedToSharded`, cheaper halo) and must use
   TTNN's `(top, bottom, left, right)` padding order (`convertConv2dPadding`, the fix from `evo50_stem_space_to_depth.md` 3.1).

### 6.4 Pitfalls met while building the tests (each cost real time)

- `ttnn.reshape` of a tiled tensor that only collapses tile-aligned dims is a **zero-copy view**; deallocating its source
  (mirroring the IR's `deallocate(%2)`) frees the buffer under the view and conv2d's own allocations overwrite it - PCC 0.886 with
  every stage individually bit-exact. The IR's deallocate on the reshape source is only safe if the runtime knows the aliasing.
- `Conv2dConfig.activation` is a `UnaryWithParam`, not a string.
- Padding order: `[2,3,2,3]` read as `(top, left, bottom, right)` gives a 191x832 output; assert the shape.
- `enable_activation_reuse` requires `act_block_h_override / 32 > output_width / 32`; for 832-wide rows that is 1248, which then
  fails to build a program on this commit. Leave it off.
- The channel padding of the channels-last activation must be **zero-filled** by the producer; a zero weight times an
  uninitialised NaN is NaN.
- `pixel_unshuffle` with `output_layout = TILE` costs an extra `Tilize` device op (the kernel emits ROW_MAJOR); the channels-last
  form does not accept TILE and does not need it.

### 6.5 Files

| | |
| --- | --- |
| tests | `tests/ttnn/unit_tests/operations/conv/test_evo50_input_preproc.py` (3 tests), `tests/ttnn/unit_tests/operations/data_movement/test_pixel_unshuffle_channels_last.py` |
| op | `ttnn/cpp/ttnn/operations/data_movement/pixel_unshuffle/{pixel_unshuffle.hpp,.cpp,_nanobind.cpp}`, `device/pixel_unshuffle_device_op.{hpp,cpp}`, `device/pixel_unshuffle_channels_last_program_factory.cpp`, `device/kernels/dataflow/pixel_unshuffle_nhwc_sharded.cpp` |
| detailed measurements | `docs/evo50_input_preproc_ttnn_repro.md` (sections 3-7) |
| compiler-side reference | tt-forge-onnx `docs/evo50_stem_space_to_depth.md` (the S2D pass, its fixes, real-model A/B) |

Run:

```bash
source python_env/bin/activate && export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
pytest -svv tests/ttnn/unit_tests/operations/data_movement/test_pixel_unshuffle_channels_last.py
pytest -svv tests/ttnn/unit_tests/operations/conv/test_evo50_input_preproc.py
python -m tracy -r -v -o generated/profiler/reports/evo50_final -n evo50_final \
  -m pytest tests/ttnn/unit_tests/operations/conv/test_evo50_input_preproc.py
```
