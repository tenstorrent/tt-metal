# Conv2d DRAM Bandwidth Bottleneck — Root Cause Analysis & Resolution Approaches

**GitHub Issue:** https://github.com/tenstorrent/tt-metal/issues/46831
**Key Comment:** https://github.com/tenstorrent/tt-metal/issues/46831#issuecomment-4691303095
**Test file:** `tests/ttnn/unit_tests/operations/conv/test_conv2d_pointwise.py`
**Hardware:** Wormhole N150 (single chip, 64 tensix cores, 288 GB/s peak DRAM BW)

---

## 1. Problem Statement

A 1×1 pointwise conv2d with `in_channels=3, out_channels=3` on large spatial inputs
(1536×1536 and 1280×2304) runs at **~96% of peak DRAM bandwidth** while achieving only
**0.008% FPU utilisation**. The kernel is completely memory-bound: it spends almost all
its cycles waiting for DRAM reads rather than doing compute.

The root cause is a **tile-padding inflation** in the NHWC activation tensor that
multiplies the effective DRAM read volume by **10.7×** over the real data size.

---

## 2. Root Cause: Tile-Padding Inflation

### 2.1 The Conv2d Input Chain (as compiled by Forge ONNX)

```
torch input [1, 3, H, W] — NCHW, host, bfloat16
    │
    ├─ from_torch → DRAM ROW_MAJOR  [1, 3, H, W]   17.7 MB
    ├─ to_layout → DRAM TILE        [1, 3, H, W]   17.7 MB   (C=3 in Z-dim, no waste)
    ├─ permute [0,2,3,1] → DRAM TILE [1, H, W, 3] 188.7 MB  ← INFLATION POINT
    ├─ reshape → DRAM TILE [1, 1, H×W, 3]          188.7 MB
    └─ conv2d (matmul path) ←─────────────────────────────── reads 188.7 MB
```

### 2.2 Why the Permute Creates 10.7× Inflation

When `C=3` moves from the Z-dimension (NCHW) to the X-dimension (NHWC), it becomes
the innermost (column) dimension of each tile.

A TT tile is always **32 × 32 elements**. With `C=3`:

```
Tile column layout for one pixel row:
  [v  v  v  0  0  0  0  0  0  0  0  0 ... 0]   (3 valid, 29 zeros)
  ←────────────────── 32 elements ──────────────→
```

- **Valid data per tile-row:** 3 × 2 bytes = 6 bytes
- **Tile-row width on DRAM:** 32 × 2 bytes = 64 bytes
- **Inflation factor:** 64 / 6 = **10.7×**

### 2.3 Measured Impact (Tracy Profiler)

#### Full chain — `test_conv2d_dram_bottleneck`

| Op | Kernel Duration | Cores | FPU Util | PM Required Input BW |
|----|----------------|-------|----------|----------------------|
| TilizeDeviceOperation | 318,139 ns | 49 | 0.0% | — |
| PermuteDeviceOperation (NCHW→NHWC) | 6,854,590 ns | 64 | 8.5% | **24.3 GB/s** |
| **MatmulDeviceOperation (conv2d)** | **6,429,082 ns** | **64** | **0.008%** | **277 GB/s** |
| PermuteDeviceOperation (NHWC→NCHW) | 938,630 ns | 64 | 56.7% | 277 GB/s |
| CopyDeviceOperation | 112,029 ns | 64 | 0.001% | — |

*Config 2 (1280×2304): conv2d kernel = 7,923,795 ns, FPU = 0.008%, PM BW = 277 GB/s*

#### Isolated conv2d — `test_conv2d_only`

| Config | Kernel Duration | FPU Util | PM Required Input BW |
|--------|----------------|----------|----------------------|
| 1×3×1536×1536 | 6,429,956 ns | 0.008% | **277 GB/s** |
| 1×3×1280×2304 | 7,922,834 ns | 0.008% | **277 GB/s** |

**277 GB/s = 96% of the 288 GB/s peak.** The kernel is saturating DRAM reads and sitting
idle for 99.992% of FPU cycles.

### 2.4 Why the Matmul Path Is Chosen

`conv2d_utils.cpp:524–535` — `use_matmul_for_1x1_conv()`:

```cpp
return kernel_size[0] == 1 && kernel_size[1] == 1
    && stride[0] == 1  && stride[1] == 1
    && padding[all] == 0
    && dilation[all] == 1
    && !is_width_sharded;
```

All conditions are met for this op (1×1 kernel, stride=1, zero padding, dilation=1,
not width-sharded), so conv2d unconditionally routes to `ttnn::linear()`.

The linear/matmul path **requires TILE layout** for its input matrix (`[1, 1, H×W, 3]`
tilized), which is where the 10.7× inflation is materialised before the kernel reads it.

### 2.5 Channel Alignment: Matmul vs Regular Conv

`conv2d_utils.cpp:92–123` — `get_input_channels_alignment()`:

| Path | Input layout | Alignment | C=3 padded to |
|------|-------------|-----------|---------------|
| **Matmul (1×1)** | TILE | 32 (TILE_WIDTH) | **32** (10.7× inflation) |
| Regular conv | ROW_MAJOR | 8 (L1 alignment / 2B) | **8** (2.7× inflation) |
| Regular conv | ROW_MAJOR, sharded width | 16 | **16** (5.3× inflation) |

The regular conv path has a fundamentally looser alignment rule and can work with
ROW_MAJOR input — avoiding materialising the TILE-padded tensor at all.

---

## 3. The Three Proposed Approaches

Comment by `peter941221` on the GitHub issue analyses three directions, from narrowest
to broadest scope. Below is an explanation of each and an analysis of feasibility.

---

### Method 1 — Opt 1×1 Small-C Out of the Matmul Lowering (Short Term)

#### What It Proposes

Change `use_matmul_for_1x1_conv()` so that small channel counts (where `C < TILE_WIDTH`)
route to the regular conv path instead of the matmul path. The regular conv path reads
activations in **ROW_MAJOR** with **8-element alignment**, dramatically reducing the
effective DRAM read volume.

#### Code Location

```
ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp   line 524–535
ttnn/cpp/ttnn/operations/conv/conv2d/conv2d.cpp         line 257–367
```

#### How It Would Work

- `use_matmul_for_1x1_conv()` adds a guard: if `in_channels < TILE_WIDTH` return `false`.
- conv2d then takes the `ttnn::prim::conv2d(...)` branch (regular conv).
- The regular conv path uses the halo op internally, but for 1×1 / stride=1 / pad=0
  this is a trivial pass-through (no padding, no sliding window accumulation).
- Input can remain ROW_MAJOR in DRAM. Alignment = 8 → C=3 pads to 8, not 32.

#### DRAM Read Volume Comparison

| Path | Bytes per pixel | H×W pixels | Total DRAM reads |
|------|----------------|-----------|-----------------|
| Current (matmul, TILE) | 32 × 2 = 64 B | 2,359,296 | **188.7 MB** |
| Regular conv (ROW_MAJOR, align=8) | 8 × 2 = 16 B | 2,359,296 | **37.7 MB** |
| **Reduction** | | | **5× fewer bytes** |

#### Feasibility Assessment

**Confidence: High.** The change is localised and the regular conv path already handles
all relevant input variants. Key considerations:

| Concern | Assessment |
|---------|-----------|
| Halo overhead for 1×1 trivial case | Halo for `kernel=1×1, stride=1, pad=0` is effectively a no-op (identity reshape). Very low overhead. |
| Reader kernel correctness | The regular conv reader (`activation_reader_width_sharded.cpp`) already handles ROW_MAJOR with alignment=8 correctly. |
| The padded channel count | Regular conv still pads C=3→8, not 3→32. DRAM read savings are immediate. |
| Performance model accuracy | `PM REQ I BW` would drop from 277 GB/s to ~69 GB/s (277/4), putting the op in a more compute-bound regime. FPU util would rise. |
| Risk | Low — only 1×1/stride=1/pad=0/dilation=1 ops with `C < 32` are affected. Existing matmul path is unchanged for all other cases. |

**The main open question the commenter flags:** does the regular conv path produce
exactly the same result for the 1×1 case as the matmul path? Static code review
suggests yes (they compute the same matrix product), but a numerical A/B test is
needed to confirm. This is why the commenter labels it an "experiment."

---

### Method 2 — Small-C Representation Path (Longer Term)

#### What It Proposes

Rather than re-routing to an existing path, introduce a small-C-specific representation
that avoids materialising the `C=3 → 32` padded NHWC tensor entirely. The kernel-stride
folding mechanism (already in the codebase for stride > 1 with small channels like RGB)
provides a template for this kind of pre-lowering rewrite.

#### Code Location

```
ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_nanobind.cpp   line 429–441   (folding docs)
ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp      line 1384–1402 (fold_tensor)
ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp      line 1467–1470 (folded params)
```

#### How Kernel-Stride Folding Works (Existing Mechanism)

For stride > 1 convolutions with small channel counts (e.g., C=3, stride=2):

```
Before folding: [N, H, W, 3]  stride=2   →  im2col matrix [N, H/2, W/2, 3×2×2=12]
After folding:  [N, H/2, W/2, 12]  stride=1  →  effectively 12 channels instead of 3
```

This folds the stride into the channel dimension, converting a strided 3-channel op
into a stride-1 12-channel op. The 12-channel tensor aligns much better to tile
boundaries (12 < 32 but only 20/32 padding vs 29/32 for C=3).

#### The Proposed Extension

For `stride=1` (where folding currently does nothing), a small-C specific path could:
- **Group multiple adjacent spatial positions** into one "super-pixel" with more channels.
  e.g., 11 adjacent pixels × 3 channels = 33 channels per "super-pixel" — fits neatly
  in one tile-row (33 rounds up to 64 with just 31/64 = 48% waste vs 91% waste for C=3).
- Or: **re-order the inner loop** at the kernel level so the innermost read unit spans
  multiple pixels rather than padded channels.

#### Feasibility Assessment

**Confidence: Medium.** This is more invasive than Method 1 but follows established
patterns in the codebase.

| Concern | Assessment |
|---------|-----------|
| Consistent with existing style | Yes — folding already rewrites channel/stride/kernel before lowering (`conv2d_utils.cpp:1391–1411`). |
| Scope of change | Moderate — requires changes to `compute_kernel_stride_folding_params`, the weight preparation, and potentially the reader kernel. |
| Weight matrix compatibility | Weights must be re-ordered to match the new channel layout (analogous to how folded weights are prepared now). |
| Interaction with other configs | Must not affect non-small-C cases. Guard condition `C < TILE_WIDTH` isolates the path. |
| Test coverage | Existing fold tests (`tests/ttnn/unit_tests/operations/conv/data_movement/test_fold_op.py`) provide a template. |

---

### Method 3 — Avoid Materialising the Inflated NHWC Tensor in DRAM (Higher Level)

#### What It Proposes

The 188.7 MB inflated TILE NHWC tensor should never land in DRAM at all. Instead, the
conv2d op should accept the original NCHW tensor (or a lightweight ROW_MAJOR NHWC
tensor) and perform the layout conversion internally, on-chip, using L1 as a staging
buffer.

#### Code Location

```
ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp   line 833–838  (flatten to [1,1,NHW,C])
ttnn/cpp/ttnn/operations/conv/conv2d/conv2d.cpp         line 820–870  (DRAM sliced reshape)
ttnn/cpp/ttnn/operations/data_movement/fold/fold.cpp    line 327–343  (fold: reshape→halo→reshape)
```

#### Why the Inflation Exists

In `conv2d_utils.cpp:833–838` the DRAM path explicitly flattens to `[1, 1, N*H*W, C]`
before sharding. This flattened TILE tensor is what the matmul reads from DRAM. If the
permute (NCHW→NHWC) upstream of conv2d produces a TILE tensor and writes it to DRAM,
that 188.7 MB must be read back in full.

#### Precedent in the Codebase

The production fold path (`fold.cpp:327–343`) already demonstrates that a
reshape → halo → reshape chain can be executed without materialising intermediate
tensors in DRAM. Similarly, the test fold path uses reshape → permute → reshape
(`test_fold_op.py:75–77`), keeping intermediates in L1.

#### The Proposed Fix

1. Accept NCHW ROW_MAJOR input directly in conv2d (currently the entry point requires
   NHWC, documented in `conv2d_nanobind.cpp:371–375`).
2. Perform the NCHW → NHWC transposition **inside the conv reader kernel** on-the-fly,
   reading along the C-stride dimension instead of expecting contiguous NHWC channels.
   Or: perform it via halo with L1 buffering, never writing back to DRAM.
3. Result: the 188.7 MB TILE NHWC tensor is never written to DRAM. DRAM reads drop
   to the ~17.7 MB of original NCHW data.

#### Feasibility Assessment

**Confidence: Lower (but highest potential impact).**

| Concern | Assessment |
|---------|-----------|
| API change | `conv2d` publicly documents NHWC input. Accepting NCHW requires an API extension or a separate code path guarded on layout. |
| Reader kernel complexity | The activation reader would need to stride across channels (non-contiguous reads) for NCHW — adds scatter-gather complexity. |
| Halo op dependency | Halo currently expects NHWC. Extending it to NCHW is non-trivial. |
| L1 buffering | For large spatial inputs (H×W = 2.36 M pixels), staging in L1 requires careful slicing to fit per-core allocation. The DRAM slicing path (`conv2d.cpp:820–870`) already handles this but assumes NHWC TILE input. |
| Potential gain | If NCHW ROW_MAJOR input (17.7 MB) replaces TILE NHWC (188.7 MB), DRAM reads drop by ~10.7×, bringing the kernel from 96% BW-bound to ~9% BW-bound. This would unlock the FPU for actual compute. |

---

## 4. Summary Comparison

| | Method 1 | Method 2 | Method 3 |
|---|---|---|---|
| **Scope** | Narrow: change 1 predicate | Moderate: new small-C lowering | Broad: change input contract |
| **DRAM savings** | ~5× (align=8 vs align=32) | ~4× (spatial grouping) | ~10.7× (no TILE inflation) |
| **FPU util after** | ~0.04% → DRAM still dominant | ~0.03% → still DRAM-bound | ~0.08% → compute-limited |
| **Risk** | Low | Medium | High |
| **Effort** | 1–2 days | 1–2 weeks | 3–6 weeks |
| **Confidence** | High (A/B experiment) | Medium | Lower |

*FPU utilisation estimates assume the only bottleneck is DRAM reads. In practice the
gain from Method 1 alone (5× fewer reads) would still leave the op DRAM-bound because
the theoretical compute demand for a 3-in/3-out 1×1 conv on 2.36 M pixels is tiny
(~14 MFLOP) compared to available throughput.*

---

## 5. Recommended Next Steps

### Immediate (Method 1 — 1–2 days)

Modify `use_matmul_for_1x1_conv()` to exclude small-C cases:

```cpp
// conv2d_utils.cpp:524
bool use_matmul_for_1x1_conv(..., uint32_t in_channels, ...) {
    bool is_width_sharded = ...;
    bool is_small_c = (in_channels < tt::constants::TILE_WIDTH);   // NEW GUARD
    return kernel_size[0] == 1 && kernel_size[1] == 1
        && stride[0] == 1   && stride[1] == 1
        && padding[all] == 0 && dilation[all] == 1
        && !is_width_sharded
        && !is_small_c;                                             // NEW GUARD
}
```

Then run the two test configs with Tracy to measure the new `PM REQ I BW` and kernel
duration. Compare against current 277 GB/s / 6.4 ms baseline. Expected: DRAM reads drop
to ~55–75 GB/s, kernel shortens to ~1.2–1.5 ms.

### Numerical Validation

After Method 1, run a golden comparison test to confirm the regular conv path and the
matmul path produce identical output values for 1×1/stride=1/pad=0/dilation=1 cases
across a range of `in_channels` values.

### Longer Term (Method 2 + 3)

Evaluate whether the remaining DRAM pressure from Method 1 (C=3→8 alignment still
wastes 2.7×) justifies Method 2 or 3 based on the measured speedup from Method 1.

---

## 6. Files to Change for Method 1

| File | Change |
|------|--------|
| `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp` | Add `in_channels` parameter and small-C guard to `use_matmul_for_1x1_conv()` |
| `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.hpp` | Update function signature |
| `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d.cpp` | Update call-site to pass `in_channels` |
| `tests/ttnn/unit_tests/operations/conv/test_conv2d_pointwise.py` | Add `enable_kernel_stride_folding=False` guard for the new path; re-run Tracy to measure |

---

## 7. Reference: Profiler Numbers

Reports location: `profiler_output/bottleneck_block_{A,C}/` and
`profiler_output/conv2d_only_block_{A,C}/`

### Config 1 — 1×3×1536×1536 (2.36 M pixels)

| Metric | Value |
|--------|-------|
| Real input data size (NCHW) | 17.7 MB |
| Permuted TILE NHWC size in DRAM | **188.7 MB** |
| Inflation factor | **10.7×** |
| Conv2d (MatmulDeviceOperation) kernel duration | 6,429 µs |
| Conv2d FPU utilisation | **0.008%** |
| Conv2d required input bandwidth | **277 GB/s** |
| Peak N150 DRAM bandwidth | 288 GB/s |
| DRAM utilisation | **96%** |

### Config 2 — 1×3×1280×2304 (2.95 M pixels)

| Metric | Value |
|--------|-------|
| Real input data size (NCHW) | 22.1 MB |
| Permuted TILE NHWC size in DRAM | **236.1 MB** |
| Inflation factor | **10.7×** |
| Conv2d (MatmulDeviceOperation) kernel duration | 7,923 µs |
| Conv2d FPU utilisation | **0.008%** |
| Conv2d required input bandwidth | **277 GB/s** |
| DRAM utilisation | **96%** |
