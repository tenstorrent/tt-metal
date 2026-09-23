# EVO50 input-preprocessing block — standalone TTNN reproduction

Reproduces `@trace_0_forward` of `ttnn_evo50_input_preproc.mlir` as a single TTNN pytest,
op for op, and confirms it hits the same device bottleneck as the model run.

| | |
| --- | --- |
| branch | `pchnadrasekaran/evo50_model_perf` |
| base commit | `f1f4ff75579ebd7a69c7da52d45368f273026d85` |
| test | `tests/ttnn/unit_tests/operations/conv/test_evo50_input_preproc.py` |
| source IR | `FULL_EVO50_MODEL/INPUT_PREPROC/WITH_MLIR_TRACE/IR/ttnn_evo50_input_preproc.mlir` |
| reference report | `ops_perf_results_preproc_opt2_with_trace_2026_09_23_09_06_40.csv` |
| generated report | `ops_perf_results_evo50_preproc_2026_09_23_10_12_02.csv` |
| arch | Wormhole B0, 8x8 worker grid |

---

## 1. What the test builds

Every attribute is transcribed from the IR — no defaults were relied on.

| IR value | test |
| --- | --- |
| `%arg0 input_luv` `tensor<1x3x384x1664xbf16, #ttnn_layout1>` | `ttnn.from_torch(..., bfloat16, ROW_MAJOR_LAYOUT, DRAM_MEMORY_CONFIG)` |
| `%arg1 fused_7x7_weights` `tensor<16x3x7x7xbf16, #system_memory>` | host `ttnn.from_torch(w1, bfloat16)` |
| `%arg2 grouped_conv_weights` `tensor<16x8x3x3xbf16, #system_memory>` | host `ttnn.from_torch(w2, bfloat16)` |
| `%arg3/%arg4` biases `tensor<1x1x1x16xbf16, #system_memory>` | host, reshaped `(1,1,1,16)` |
| `%1 = ttnn.to_layout` -> `#ttnn_layout5` TILE DRAM | `ttnn.to_layout(x, TILE_LAYOUT, memory_config=DRAM)` |
| `%2 = ttnn.permute {permutation = [0,2,3,1]}` -> `#ttnn_layout6` TILE L1 `<8x8>` | `ttnn.permute(t, (0,2,3,1), memory_config=L1)` |
| `%3 = ttnn.reshape [1,1,638976,3]` | `ttnn.reshape(t, (1,1,638976,3))` |
| `%4 = ttnn.to_memory_config` -> `#ttnn_layout8` TILE DRAM | `ttnn.to_memory_config(t, DRAM)` |
| `%5 = ttnn.conv2d` 7x7 s2 pad `[2,3,2,3]` g1 relu | see below |
| `%6 = ttnn.to_memory_config` -> `#ttnn_layout10` TILE L1 height_sharded `<64x1>` `memref<78x1 tile>` | explicit `MemoryConfig(HEIGHT_SHARDED, L1, ShardSpec((0,0)-(7,7), (2496,32), ROW_MAJOR))` |
| `%7 = ttnn.conv2d` 3x3 s1 pad `[1,1,1,1]` g2 relu | see below |
| `%8 = ttnn.reshape [1,192,832,16]` | `ttnn.reshape(t, (1,192,832,16))` |
| `%9 = ttnn.permute {permutation = [0,3,1,2]}` -> `#ttnn_layout12` TILE L1 `<8x8>` | `ttnn.permute(t, (0,3,1,2), memory_config=L1)` |
| `%10 = ttnn.to_memory_config` -> `#ttnn_layout4` TILE DRAM | `ttnn.to_memory_config(t, DRAM)` |
| every `ttnn.deallocate` | `ttnn.deallocate(...)` at the same point |

Conv attributes:

| attribute | `%5` (fused 7x7) | `%7` (grouped 3x3) |
| --- | --- | --- |
| `in_channels` / `out_channels` | 3 / 16 | 16 / 16 |
| `input_height` x `input_width` | 384 x 1664 | 192 x 832 |
| `kernel_size` | `[7, 7]` | `[3, 3]` |
| `stride` | `[2, 2]` | `[1, 1]` |
| `padding` | `[2, 3, 2, 3]` (top, bottom, left, right) | `[1, 1, 1, 1]` |
| `dilation` | `[1, 1]` | `[1, 1]` |
| `groups` | 1 | 2 |
| `weights_dtype` | `bfloat16` | `bfloat16` |
| `activation` | `UnaryWithParam(RELU)` | `UnaryWithParam(RELU)` |
| `act_block_h_override` | 1024 | 64 |
| `deallocate_activation` | false | **true** |
| `shard_layout` | (unset) | `HEIGHT_SHARDED` |
| `enable_kernel_stride_folding` | false | false |
| `config_tensors_in_dram` | true | true |
| `slice_config` | `Op2DSliceConfig(DRAMSliceWidth, 0)` | `Op2DSliceConfig(L1Full, 0)` |
| `compute_config` | `HiFi2`, `fp32_dest_acc_en=True` | `HiFi2`, `fp32_dest_acc_en=True` |

Output shapes follow: `(384+2+3-7)/2+1 = 192`, `(1664+2+3-7)/2+1 = 832`.

---

## 2. Golden check

```python
xp = F.pad(x, (2, 3, 2, 3))                     # (left, right, top, bottom)
y1 = F.relu(F.conv2d(xp, w1, b1, stride=2, padding=0, groups=1))
y2 = F.relu(F.conv2d(y1, w2, b2, stride=1, padding=1, groups=2))
```

| | |
| --- | --- |
| output shape | `[1, 16, 192, 832]` (matches `#ttnn_layout4`) |
| **PCC vs torch golden** | **0.999978** |

---

## 3. Device profile, scoped between the signposts

Both reports are scoped to the rows strictly between `evo50_inference-start` and
`evo50_inference-end`. The reference region is CSV rows 62-82; the generated region is
rows 21-41. The op sequence, the core counts and the memory transitions are identical.

| # | op code | in -> out | cores | reference ns | this test ns | delta | delta % |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | `Tilize` | DRAM_INTERLEAVED -> DRAM_INTERLEAVED | 39 | 171056 | 170270 | -786 | -0.46% |
| 1 | `Permute` | DRAM_INTERLEAVED -> L1_INTERLEAVED | 64 | 906228 | 906106 | -122 | -0.01% |
| 2 | `Copy` | L1_INTERLEAVED -> DRAM_INTERLEAVED | 64 | 314129 | 315262 | +1133 | +0.36% |
| 3 | `PaddedSlice` | DRAM_INTERLEAVED -> L1_HEIGHT_SHARDED | 64 | 143211 | 143448 | +237 | +0.17% |
| 4 | `Halo` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 10572 | 11467 | +895 | +8.47% |
| 5 | `Move` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 7301 | 7368 | +67 | +0.92% |
| 6 | `Conv2d` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 130082 | 130067 | -15 | -0.01% |
| 7 | `SliceWrite` | L1_HEIGHT_SHARDED -> DRAM_INTERLEAVED | 64 | 43726 | 43479 | -247 | -0.56% |
| 8 | `PaddedSlice` | DRAM_INTERLEAVED -> L1_HEIGHT_SHARDED | 64 | 142999 | 143645 | +646 | +0.45% |
| 9 | `Halo` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 11904 | 11923 | +19 | +0.16% |
| 10 | `Move` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 7370 | 7355 | -15 | -0.20% |
| 11 | `Conv2d` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 130235 | 130183 | -52 | -0.04% |
| 12 | `SliceWrite` | L1_HEIGHT_SHARDED -> DRAM_INTERLEAVED | 64 | 41761 | 42482 | +721 | +1.73% |
| 13 | `InterleavedToSharded` | DRAM_INTERLEAVED -> L1_HEIGHT_SHARDED | 64 | 50012 | 51426 | +1414 | +2.83% |
| 14 | `Halo` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 25180 | 25273 | +93 | +0.37% |
| 15 | `Move` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 11497 | 11530 | +33 | +0.29% |
| 16 | `Conv2d` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 229384 | 229031 | -353 | -0.15% |
| 17 | `Transpose` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 8158 | 8236 | +78 | +0.96% |
| 18 | `Transpose` | L1_HEIGHT_SHARDED -> L1_INTERLEAVED | 64 | 126791 | 126956 | +165 | +0.13% |
| 19 | `Copy` | L1_INTERLEAVED -> DRAM_INTERLEAVED | 64 | 40811 | 40082 | -729 | -1.79% |
| 20 | `Untilize` | DRAM_INTERLEAVED -> DRAM_INTERLEAVED | 48 | n/a | 59680 | - | - |
| | **TOTAL (20 timed ops)** | | | **2,552,407** | **2,555,589** | **+3,182** | **+0.12%** |

**2.5524 ms reference vs 2.5556 ms reproduced — +0.12%.**

The trailing `Untilize` carries no duration in the reference scoped region (it straddles
the trace boundary), but the reference's unscoped iterations record it at 59,270 ns;
this test measures 59,680 ns for the same op, +0.69%. Including it, the full 21-op region
is 2.6153 ms.

---

## 4. The bottleneck, reproduced

| op | ns | share of the 20-op region |
| --- | ---: | ---: |
| `Permute` | 906,106 | 35.5% |
| `Conv2d` | 489,281 | 19.1% |
| `Copy` | 355,344 | 13.9% |
| `PaddedSlice` | 287,093 | 11.2% |
| `Tilize` | 170,270 | 6.7% |
| `Transpose` | 135,192 | 5.3% |
| `SliceWrite` | 85,961 | 3.4% |
| `InterleavedToSharded` | 51,426 | 2.0% |
| `Halo` | 48,663 | 1.9% |
| `Move` | 26,253 | 1.0% |

### The padding-inflation chain

| step | ns | what it moves |
| --- | ---: | --- |
| `Permute` `%2` `[0,2,3,1]` | 906,106 | writes `[1,384,1664,3]` as TILE: the 3-wide channel axis is padded to a full 32-wide tile |
| `Copy` `%4` L1 -> DRAM | 315,262 | spills the same padded tensor back to DRAM for the conv |
| `PaddedSlice` x2 (conv `%5` DRAM-width slicing) | 287,093 | reads the padded tensor back into L1, two slices |
| **span total** | **1,508,461** | **59.0% of the region** |

The `[1,1,638976,3]` activation is 3.83 MB of real data. Tiled to 32 lanes it becomes
40.89 MB — a **10.67x write amplification** that every op in that span pays.
`Permute` alone is 35.5% of the block (906.1 us) and is the single largest cost.

This matches `evo50_input_preproc_bottleneck_analysis.md`, which reports the same span at
1.510 ms / 59.2% and `Permute` at 906.1 us / 35.5%. The standalone test is therefore a
faithful bench for any fix to this region.

---

## 5. How to run

```bash
source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD

# functional / golden
pytest -svv tests/ttnn/unit_tests/operations/conv/test_evo50_input_preproc.py

# profiled
python -m tracy -r -v -o generated/profiler/reports/evo50_preproc -n evo50_preproc \
  -m pytest tests/ttnn/unit_tests/operations/conv/test_evo50_input_preproc.py
```

---

## 6. The fix: space-to-depth stem (`test_evo50_input_preproc_s2d`)

Second test in the same file, the stem rewritten as `docs/evo50_stem_space_to_depth.md`
(tt-forge-onnx, pass `--ttir-strided-conv-space-to-depth-opt`) describes: the 7x7 / stride-2
conv is a `pixel_unshuffle(2)` followed by a 4x4 / stride-1 conv on re-packed weights,
`W'[oc, c*4 + dy*2 + dx, i, j] = Wpad[oc, c, 2i + dy, 2j + dx]`, padding `(1, 2, 1, 2)`.
Same MACs regrouped; the conv input becomes `[1,1,159744,12]` in L1, so nothing is tilized
up front, the permute moves 4x fewer bytes, there is no DRAM slicing and no re-shard before
the grouped conv. Everything from the grouped conv on is unchanged.

`ttnn.pixel_unshuffle` is not on `f1f4ff75579`; the three `[pixel_unshuffle]` commits were
cherry-picked from the local `pchandrasekaran/bev_model_perf` (13 files, all additions, nothing
else from that branch): `a6e29b4b1da` op + kernel, `4bc4282f2b0` NCHW kernel rewrite,
`657c92f5f04` drop NOC transaction ids.

| | |
| --- | --- |
| torch check, packed 4x4 stem vs 7x7/s2 conv | max abs diff 1.5e-5 (fp32) |
| device output `[1, 16, 192, 832]` **PCC vs torch golden** | **0.999980** (baseline test 0.999978; doc: 0.99996) |
| generated report | `ops_perf_results_evo50_s2d_2026_09_23_12_19_13.csv` |

### 6.1 Stem, device ops — this test vs the doc's real-EVO50 numbers

| # | op | in -> out | cores | this test ns | doc (EVO50, pass on) us | delta |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 0 | `PixelUnshuffle` | DRAM_INTERLEAVED -> L1_INTERLEAVED | 64 | 119,468 | 119.4 | +0.1 us |
| 1 | `Tilize` | L1_INTERLEAVED -> L1_INTERLEAVED | 36 | 56,658 | 59.3 | -2.6 us |
| 2 | `Permute` | L1_INTERLEAVED -> L1_INTERLEAVED | 64 | 229,475 | 229.2 | +0.3 us |
| 3 | `InterleavedToSharded` | L1_INTERLEAVED -> L1_HEIGHT_SHARDED | 64 | 34,548 | 36.6 | -2.1 us |
| 4 | `Move` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 7,617 | 7.5 | +0.1 us |
| 5 | `Halo` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 34,728 | 35.2 | -0.5 us |
| 6 | `Move` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 13,461 | 13.4 | +0.1 us |
| 7 | `Conv2d` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 289,964 | 291.4 | -1.4 us |
| | **stem total, 8 ops** | | | **785,919** | **792.0** | **-6.1 us (-0.8 %)** |

Op sequence, core counts and memory transitions are the ones the doc lists; every op is
within a few microseconds of the doc's measurement.

### 6.2 Stem, baseline vs S2D, same run

| | baseline stem (`%1`..`%6`) | S2D stem | change |
| --- | ---: | ---: | ---: |
| device ops | 14 | 8 | -6 |
| kernel time | 2,110,452 ns | 785,919 ns | **-62.8 %** |
| `Permute` | 906,135 | 229,475 | -74.7 % (40.9 MB -> 10.2 MB moved) |
| `Tilize` | 170,481 (whole image, DRAM) | 56,658 (pixel_unshuffle output, L1) | -66.8 % |
| `Copy` L1 -> DRAM spill | 315,631 | 0 | gone |
| `PaddedSlice` + `SliceWrite` | 368,296 | 0 | gone |
| `Conv2d` (conv1) | 260,322 (2 sliced 7x7) | 289,964 (1 4x4) | +11.4 % |
| `PixelUnshuffle` | 0 | 119,468 | new |

### 6.3 Whole block

| | baseline | S2D | change |
| --- | ---: | ---: | ---: |
| device ops (incl. trailing untilize) | 21 | 14 | -7 |
| stem | 2110.5 us | 785.9 us | -62.8 % |
| grouped conv -> output | 498.7 us | 484.0 us | -2.9 % |
| **total** | **2609.2 us** | **1270.0 us** | **-51.3 %** |

The tail differs only in that the grouped conv's `Halo` no longer needs a `Move` in front of
it (its input is conv1's own sharded L1 output rather than a fresh `InterleavedToSharded`).

### 6.4 A pitfall found on the way: `ttnn.reshape` views

First S2D run gave PCC 0.886 with pixel_unshuffle, permute and reshape each verified
bit-exact and the 4x4 conv verified correct when fed from torch. The cause was in the test:
`ttnn.reshape` of a tiled tensor that only collapses tile-aligned dims (`[1,192,832,12]` ->
`[1,1,159744,12]`) returns a **zero-copy view**, and `ttnn.deallocate(source)` right after it
(mirroring the IR's `deallocate(%2)`) freed the buffer under the view; conv2d's own L1
allocations then overwrote 50 of 192 output rows. The baseline stem has the same latent
bug at `%3` and survives only because its next op copies to DRAM before any L1 reuse. Both
stems now free the reshape source only when `buffer_address()` differs (`_free_unless_alias`).

### 6.5 Run

```bash
pytest -svv tests/ttnn/unit_tests/operations/conv/test_evo50_input_preproc.py            # both tests
python -m tracy -r -v -o generated/profiler/reports/evo50_s2d -n evo50_s2d \
  -m pytest tests/ttnn/unit_tests/operations/conv/test_evo50_input_preproc.py            # both regions in one CSV
```

---

## 7. Optimizing further: `pixel_unshuffle(channels_last=True)` + conv blocking (`test_evo50_input_preproc_s2d_nhwc`)

Starting point was the 786 us S2D stem of section 6. Its remaining cost was layout plumbing
around a 120 us gather: `Tilize` 57 + `Permute` 229 (NCHW tile -> NHWC) + `InterleavedToSharded`
35 + two `Move`s 21 + a `Halo` that untilizes 35 = 377 us before the conv even starts, and a
4x4 conv at 290 us that was slower than the two sliced 7x7s it replaced.

### 7.1 What was measured before changing code

**conv2d config sweep** (same stem, only `Conv2dConfig` knobs; all PCC 0.999986):

| knob | Conv2d us | note |
| --- | ---: | --- |
| `act_block_h_override` 64 (IR) | 291 | reader-bound |
| 96 / 192 / 416 / 832 / 1248 | 283 / **202** / 224 / 203 / 199 | 192+ is the floor; 2496 rows/core = 78 tiles, blocks must divide it |
| + `enable_act_double_buffer` / `force_split_reader` / weights double buffer | 202 / 202 / 290 | no further gain |
| `reallocate_halo_output=False` | - | removes the 13.5 us `Move` after `Halo` |
| `enable_activation_reuse` | fails | needs act_block_h > output row (27+ tiles); 1248 then fails program build |

**conv2d fed a ROW_MAJOR height-sharded NHWC input directly** (`[1,1,159744,16]`, shard
`[2496,16]`, built on host to test the idea before writing the kernel):

| act_block_h_override | ops | Halo | Conv2d | total |
| ---: | --- | ---: | ---: | ---: |
| 64 | Halo, Conv2d | 15.8 | 265.8 | 281.6 |
| 192 | Halo, Conv2d | 15.3 | 148.0 | 163.2 |
| 192 + act dbuf | Halo, Conv2d | 15.3 | 138.5 | 153.8 |
| **832 + act dbuf** | Halo, Conv2d | 14.9 | **130.5** | **145.4** |

No `InterleavedToSharded`, no `Move`s, and `Halo` drops 35 -> 15 us because it no longer untilizes.
The conv itself is faster on the row-major sharded input than on the tiled one (130 vs 200).
This fixed the target: make `pixel_unshuffle` emit exactly that tensor.

### 7.2 The kernel: `pixel_unshuffle(..., channels_last=True, padded_channels=16)`

New output mode of the op (`ttnn/cpp/ttnn/operations/data_movement/pixel_unshuffle/`):

| | |
| --- | --- |
| output | `[N, H/r, W/r, padded_channels]` ROW_MAJOR, HEIGHT_SHARDED in L1; channels `C*r^2..padded_channels` are zero |
| = | `pixel_unshuffle(x, r).permute(0,2,3,1)` zero-padded on the channel axis - what a height-sharded conv2d reads |
| factory | `MultiCoreChannelsLast` (`pixel_unshuffle_channels_last_program_factory.cpp`): output buffer bound to a CB (the interleaved_to_sharded pattern), cores in shard order, `emplace_runtime_args(core, {src_buffer, pix0, npix})` |
| kernel | `pixel_unshuffle_nhwc_sharded.cpp`, one source on both dataflow RISCs |
| work split | core i owns shard i = pixels `[i*shard_h, (i+1)*shard_h)`; each image row in that range is split between the two RISCs at a 32 B-aligned input column, each reads only its `r*C` half-rows from DRAM (no read amplification) |
| write | plain 32-bit L1 stores into the local shard - **no NOC writes** |
| fast path | 2-byte elements, `r` even, CHANNEL_MAJOR: a pixel's `r` columns from row `(c,dy)` are `r/2` aligned words landing at channel `c*r^2+dy*r` - also aligned words. Pure word copy, no shifts. |
| generic path | odd `r`, 4-byte elements, SPATIAL_MAJOR: element copies |
| pipeline | reads one image row ahead of the gather (depth 2), plain barriers, no transaction ids |
| validation | HEIGHT_SHARDED L1 output, shard width == padded_channels, `padded_channels*datum` a multiple of the L1 alignment, shards cover all pixels |
| golden test | `tests/ttnn/unit_tests/operations/data_movement/test_pixel_unshuffle_channels_last.py`: 11 cases bit-exact - both channel orders, r = 2/3/4, bf16 and fp32, batch 2, shards that cut image rows |

The suite caught one bug on the first run: the pair-gather stepped the second pixel by `r`
words instead of `r/2`, skipping 3 of every 4 pixels (every fast-path case failed at ~75 %,
every element-path case passed). A 4-pixel gather was also tried: 139 vs 137 us, no gain -
the kernel is at the per-core L1 scalar ceiling for its 14 memory ops per pixel (12 data
words + 2 zero-pad words), which is also why it is 15 % slower than the 12-op NCHW kernel.

### 7.3 Result

Generated report: `ops_perf_results_evo50_final_2026_09_23_12_56_59.csv` (all three tests in one run).

| | baseline (`%1`..`%6`) | S2D, doc form | **S2D + channels_last + tuned convs** |
| --- | ---: | ---: | ---: |
| stem device ops | 14 | 8 | **3** |
| stem kernel time | 2,109.2 us | 778.3 us | **284.4 us** |
| stem vs baseline | - | -63.1 % | **-86.5 %** |
| grouped conv -> output | 499.6 us | 483.9 us | **382.9 us** (act_block_h 832, act dbuf, no halo realloc) |
| whole block, ops | 21 | 14 | **9** |
| whole block, kernel time | 2,608.8 us | 1,262.2 us | **667.3 us** |
| whole block vs baseline | - | -51.6 % | **-74.4 %** |
| PCC vs torch golden | 0.999978 | 0.999980 | **0.999980** |

Stem, op by op:

| # | op | in -> out | cores | ns |
| ---: | --- | --- | ---: | ---: |
| 0 | `PixelUnshuffle` | DRAM_INTERLEAVED -> L1_HEIGHT_SHARDED | 64 | 137,286 |
| 1 | `Halo` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 15,913 |
| 2 | `Conv2d` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 131,162 |
| | **stem** | | | **284,361** |

Whole block, op by op:

| # | op | in -> out | cores | ns |
| ---: | --- | --- | ---: | ---: |
| 0 | `PixelUnshuffle` | DRAM_INTERLEAVED -> L1_HEIGHT_SHARDED | 64 | 137,286 |
| 1 | `Halo` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 15,913 |
| 2 | `Conv2d` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 131,162 |
| 3 | `Halo` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 24,246 |
| 4 | `Conv2d` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 123,271 |
| 5 | `Transpose` | L1_HEIGHT_SHARDED -> L1_HEIGHT_SHARDED | 64 | 6,667 |
| 6 | `Transpose` | L1_HEIGHT_SHARDED -> L1_INTERLEAVED | 64 | 130,073 |
| 7 | `Copy` | L1_INTERLEAVED -> DRAM_INTERLEAVED | 64 | 39,734 |
| 8 | `Untilize` | DRAM_INTERLEAVED -> DRAM_INTERLEAVED | 48 | 58,908 |
| | **total** | | | **667,260** |

### 7.4 What is left

- `Conv2d` 254.4 us
- `PixelUnshuffle` 137.3 us
- `Transpose` 136.7 us
- `Untilize` 58.9 us
- `Halo` 40.2 us
- `Copy` 39.7 us

The stem is now three ops at the floor of each: one read of the image (`PixelUnshuffle`),
the conv's halo, and the conv. Of the remainder, `Transpose` (125 us, the NHWC -> NCHW
output permute) and `Untilize` (59 us, the runtime's output conversion) are layout costs of
the block's *output* contract, not of the stem, and would go away if the consumer accepted
NHWC / tiled data. The grouped 3x3 conv and its halo (~148 us) are the compute floor.

### 7.5 Run

```bash
pytest -svv tests/ttnn/unit_tests/operations/data_movement/test_pixel_unshuffle_channels_last.py   # op golden, 11 cases
pytest -svv tests/ttnn/unit_tests/operations/conv/test_evo50_input_preproc.py                       # baseline, s2d, s2d_nhwc
python -m tracy -r -v -o generated/profiler/reports/evo50_final -n evo50_final \
  -m pytest tests/ttnn/unit_tests/operations/conv/test_evo50_input_preproc.py
```
