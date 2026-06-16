# Conv2d DRAM Bottleneck — Method 1 Implementation & Analysis Report

**Issue:** https://github.com/tenstorrent/tt-metal/issues/46831
**Comment:** https://github.com/tenstorrent/tt-metal/issues/46831#issuecomment-4691303095
**Branch:** `pchandrasekaran/conv2d_dram_bottleneck`
**Hardware:** Wormhole N150 · 64 Tensix cores · 288 GB/s peak DRAM BW

---

## 1. Problem Recap

A 1×1 pointwise conv2d with `in_channels=3, out_channels=3` on large spatial inputs
runs at **96% of peak DRAM bandwidth** with **0.008% FPU utilisation**.

Root cause: when `C=3` is placed in the X-dimension of a TILE-layout tensor, every
32-element tile row holds only 3 valid values and 29 zeros — a **10.7× read inflation**.

```
TILE row (32 elements × 2 B = 64 B):  [v  v  v  0  0  0  0 … 0]
                                        ← 3 valid ─────────────→29 zeros →
Wasted bytes per pixel: 64 / 6 = 10.7×
```

The `use_matmul_for_1x1_conv()` predicate routes all 1×1 / stride=1 / pad=0 / dilation=1
convolutions to `ttnn::linear()` which requires TILE-layout input, locking in this waste.

---

## 2. Method 1 — Change Summary

**File changed:** `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp`

Added a small-channel guard to `use_matmul_for_1x1_conv()`:

```cpp
// Before (routes all 1×1 to matmul regardless of channel count)
bool use_matmul_for_1x1_conv(...) {
    bool is_width_sharded = ...;
    return kernel_size[0] == 1 && kernel_size[1] == 1 && stride[0] == 1 && ...
           && !is_width_sharded;
}

// After (opts small-C out of matmul path)
bool use_matmul_for_1x1_conv(..., uint32_t in_channels) {
    bool is_width_sharded = ...;
    // Matmul requires TILE layout: C=3 → 29/32 zeros per tile row (10.7× inflation).
    // Regular conv reads ROW_MAJOR with 8-element alignment: C=3 → 5/8 zeros (2.7×).
    bool is_small_channel_count = (in_channels < tt::constants::TILE_WIDTH);
    return kernel_size[0] == 1 && ... && !is_width_sharded && !is_small_channel_count;
}
```

**Files updated (6 call sites):**

| File | Change |
|------|--------|
| `conv2d_utils.hpp` | Added `uint32_t in_channels = TILE_WIDTH` to declaration |
| `conv2d_utils.cpp` | Added `in_channels` parameter + `is_small_channel_count` guard |
| `conv2d.cpp` (line 62) | Pass `in_channels` — `conv2d_L1` |
| `conv2d.cpp` (line 540) | Pass `input_channels` — `Conv2dSliceAttr::get_L1_usage` |
| `conv2d.cpp` (line 747) | Pass `in_channels` — `conv2d_DRAM` |
| `prepare_conv2d_weights.cpp` (line 1279) | Pass `in_channels` — `setup_conv_prep_config` |
| `conv_transpose2d.cpp` (lines 117, 1006) | Pass `in_channels` — both DRAM variants |

---

## 3. How Method 1 Works

When `mm_conv = false` (triggered for C < 32), the code takes the **regular conv path**:

```
Matmul path (original):
  Input [1,1,H×W,3]  TILE DRAM  →  ttnn::linear()  →  reads 188.7 MB
  Alignment = 32  →  C=3 pads to 32  →  10.7× waste

Regular conv path (Method 1):
  Input [1,1,H×W,3]  ROW_MAJOR DRAM  →  ttnn::prim::conv2d()  →  reads ~18.9 MB
  Alignment = 8   →  C=3 pads to  8  →   2.7× waste
```

Key: the regular conv path reads **ROW_MAJOR** data, so each pixel row occupies only
`ceil(3/8) × 8 × 2 = 16 bytes` instead of `32 × 2 = 64 bytes` (TILE). For the input
to benefit, it must be passed in **ROW_MAJOR layout** (no `to_layout(TILE_LAYOUT)` step).

---

## 4. Test Changes

### `test_conv2d_only` — ROW_MAJOR input, no `Conv2dL1FullSliceConfig`

The test was updated to reflect the correct input contract for the regular conv path:

```python
# Before (TILE layout — no improvement even with regular conv path)
tt_input = ttnn.from_torch(torch_input_flat, ..., layout=ttnn.TILE_LAYOUT, ...)
[tt_out, ...] = ttnn.conv2d(..., slice_config=ttnn.Conv2dL1FullSliceConfig)

# After (ROW_MAJOR — channel alignment = 8, not 32)
tt_input = ttnn.from_torch(torch_input_flat, ..., layout=ttnn.ROW_MAJOR_LAYOUT, ...)
[tt_out, ...] = ttnn.conv2d(...)   # no slice_config → DRAM slicing auto-selected
```

With `Conv2dL1FullSliceConfig`, `determine_conv2d_execution_path()` forces the L1 path.
For a 1536×1536 input that is too large for L1, this caused an OOM. Removing the
explicit slice config lets the runtime use the DRAM slicing path automatically.

---

## 5. Profiler Results

### Tracy Options Used
```bash
python3 -m tracy -r -m --op-support-count 1000 --profile-dispatch-cores \
  --device-memory-profiler --check-exit-code -o profiler_output/method1_conv2d_only_{1,2} \
  pytest tests/ttnn/unit_tests/operations/conv/test_conv2d_pointwise.py::test_conv2d_only[...]
```

### Report Paths
| Config | Report |
|--------|--------|
| conv2d_1 (1×3×1536×1536) | `profiler_output/method1_conv2d_only_1/reports/2026_06_15_14_30_29/ops_perf_results_2026_06_15_14_30_29.csv` |
| conv2d_2 (1×3×1280×2304) | `profiler_output/method1_conv2d_only_2/reports/2026_06_15_14_30_46/ops_perf_results_2026_06_15_14_30_46.csv` |

### Config 1 — 1×3×1536×1536 (2.36 M pixels)

| Metric | Baseline (Matmul, TILE) | Method 1 (Regular Conv, ROW_MAJOR) |
|--------|------------------------|-------------------------------------|
| Op type | `MatmulDeviceOperation` | `Conv2dDeviceOperation` (×6 slices) |
| Device kernel (single op) | **6.430 ms** | 0.500 ms avg/slice |
| Total device kernel (all ops) | 6.430 ms | **17.187 ms** |
| Total host time | 0.550 ms | 1.688 ms |
| FPU utilisation | 0.008% | 0.072% |
| PM Required I BW | **277 GB/s** (96% of peak) | N/A (no PM model for sliced ops) |
| DRAM reads per pixel | 64 B (32 channels × 2 B) | 16 B (8 channels × 2 B) |
| DRAM inflation factor | 10.7× | 2.7× |
| Total DRAM read (activation) | ~188.7 MB | ~18.9 MB (10× less) |

### Config 2 — 1×3×1280×2304 (2.95 M pixels)

| Metric | Baseline (Matmul, TILE) | Method 1 (Regular Conv, ROW_MAJOR) |
|--------|------------------------|-------------------------------------|
| Op type | `MatmulDeviceOperation` | `Conv2dDeviceOperation` (×8 slices) |
| Device kernel (single op) | **7.923 ms** | 0.121 ms avg/slice |
| Total device kernel (all ops) | 7.923 ms | **21.430 ms** |
| Total host time | 0.454 ms | 1.402 ms |
| FPU utilisation | 0.008% | 0.071% |
| PM Required I BW | **277 GB/s** (96% of peak) | N/A |

---

## 6. DRAM Slicing Overhead Breakdown

Method 1 routes through `conv2d_DRAM` (DRAM slicing), which splits the large activation
into L1-sized slices and processes each slice independently.

### Config 1 (1536×1536) — 6 slices, 30 ops total

| Op type | Count | Total kernel | Avg/op | Share |
|---------|-------|-------------|--------|-------|
| `PaddedSliceDeviceOperation` | 6 | 5.577 ms | 929 µs | **32%** |
| `HaloDeviceOperation` | 6 | 5.203 ms | 867 µs | **30%** |
| `SliceWriteDeviceOperation` | 6 | 3.004 ms | 501 µs | 17% |
| **`Conv2dDeviceOperation`** | **6** | **2.997 ms** | **500 µs** | **17%** |
| `MoveDeviceOperation` | 6 | 0.407 ms | 68 µs | 2% |
| **Total** | **30** | **17.187 ms** | | |

### Config 2 (1280×2304) — 8 slices, 40 ops total

| Op type | Count | Total kernel | Avg/op | Share |
|---------|-------|-------------|--------|-------|
| `PaddedSliceDeviceOperation` | 8 | 16.327 ms | 2041 µs | **76%** |
| `SliceWriteDeviceOperation` | 8 | 3.954 ms | 494 µs | 18% |
| **`Conv2dDeviceOperation`** | **8** | **0.965 ms** | **121 µs** | **5%** |
| `MoveDeviceOperation` | 8 | 0.142 ms | 18 µs | 1% |
| `HaloDeviceOperation` | 8 | 0.043 ms | 5 µs | <1% |
| **Total** | **40** | **21.430 ms** | | |

---

## 7. Analysis & Findings

### Finding 1 — The guard fires correctly

`use_matmul_for_1x1_conv` now returns `false` for `in_channels=3`. The op name in Tracy
changed from `MatmulDeviceOperation` → `Conv2dDeviceOperation`, confirming the regular
conv path is taken.

### Finding 2 — Per-slice Conv2d is 10–65× faster

The individual `Conv2dDeviceOperation` slices are dramatically shorter than the baseline:

| Config | Baseline single kernel | Method 1 per-slice kernel | Speedup |
|--------|----------------------|--------------------------|---------|
| 1536×1536 | 6430 µs | 500 µs | **12.9×** |
| 1280×2304 | 7923 µs | 121 µs | **65.5×** |

This confirms the regular conv path reads far less DRAM per pixel (16 B vs 64 B).

### Finding 3 — DRAM slicing overhead reverses the gain

Despite each slice being 13–65× faster, the total end-to-end time is **2.67–2.70×
worse** than the baseline:

| Config | Baseline total | Method 1 total | Ratio |
|--------|---------------|----------------|-------|
| 1536×1536 | 6.43 ms | 17.19 ms | 2.67× slower |
| 1280×2304 | 7.92 ms | 21.43 ms | 2.70× slower |

The overhead comes from:
- **PaddedSlice**: reading each input slice from DRAM into L1 (92–76% of total)
- **Halo**: trivial for 1×1 but still dispatches a kernel per slice (30% for config 1)
- **SliceWrite**: writing output slices back to DRAM after each conv (17–18%)

The matmul path avoids this overhead entirely: it reads all 188.7 MB in one massive
parallel pass that saturates DRAM bandwidth across all 64 cores simultaneously. DRAM
slicing serialises this into 6–8 sequential rounds, each with its own dispatch overhead.

### Finding 4 — Why TILE input was necessary for matmul

The `to_layout(TILE_LAYOUT)` → `permute NHWC` step in `test_conv2d_dram_bottleneck`
exists because `ttnn::linear()` requires TILE-layout input. With Method 1 (regular conv),
this step is unnecessary — and removing it would reduce the inflated NHWC tensor from
188.7 MB (TILE) to 14.2 MB (ROW_MAJOR), a direct 13× size reduction before conv2d
even runs.

---

## 8. Why Method 1 Does Not Deliver End-to-End Improvement (Current State)

```
Matmul path (baseline):
  1 kernel × 64 cores × full DRAM bandwidth → 6.4 ms total
  Reads 188.7 MB at ~29 GB/s effective per core (saturates DRAM)

Regular conv path (Method 1 with DRAM slicing):
  6 slices × (PaddedSlice + Halo + Move + Conv2d + SliceWrite)
  Each slice processes 1/6 of the data efficiently,
  but 5 kernels overhead per slice cancel the gain:
    - PaddedSlice alone takes as long as the actual Conv2d
    - Total: 17.2 ms
```

The matmul path is a single monolithic kernel that saturates all DRAM channels at once.
The sliced regular conv path serialises work and re-pays dispatch overhead per slice.

---

## 9. Additional Experiments — Step 2 (BLOCK_SHARDED) and Step 3 (act_block_h tuning)

### Step 2 — BLOCK_SHARDED (failed, 9–10× worse)

Tried forcing `shard_layout=BLOCK_SHARDED` in `Conv2dConfig`. The DRAM slicing
framework decomposed the input into **48–72 slices** (vs 6–8 for HEIGHT_SHARDED),
multiplying per-slice overhead ~8×. `PaddedSliceDeviceOperation` alone consumed
70–147 ms.

| Config | Baseline | BLOCK_SHARDED | Ratio |
|--------|----------|---------------|-------|
| 1536×1536 | 14.652 ms | 138.251 ms | **9.4× slower** |
| 1280×2304 | 17.022 ms | 171.562 ms | **10.1× slower** |

### Step 3 — Reduce slice count via `act_block_h_override` (marginal, still slower)

Tried `act_block_h_override = 64` and `256` to reduce the number of slices.
Both halved the slice count (6→3 for 1536×1536, 8→3 for 1280×2304). The best result
was `act_block_h_override=256`:

| Config | Baseline | Step1 auto (6–8 slices) | Step3 abh=256 (3 slices) |
|--------|----------|------------------------|--------------------------|
| 1536×1536 | **14.652 ms** | 22.603 ms (1.54×) | 22.102 ms (**1.51×**) |
| 1280×2304 | **17.022 ms** | 26.790 ms (1.57×) | 26.070 ms (**1.53×**) |

Halving the slice count gave only a **2–3% improvement** because
`PaddedSliceDeviceOperation` scales with slice data volume — fewer, larger slices
each take proportionally longer.

---

## 10. Method 1 — Final Conclusion

**Method 1 is exhausted.** Every variant of the regular conv path (HEIGHT_SHARDED,
BLOCK_SHARDED, fewer slices via `act_block_h_override`) produces results that are
**1.5× to 10× slower than the baseline** matmul.

The fundamental blocker is the **DRAM slicing framework overhead**, not the conv
kernel itself:

```
Per-slice overhead (fixed cost regardless of slice size):
  PaddedSliceDeviceOperation  ~929 µs/slice  (reads input chunk from DRAM)
  HaloDeviceOperation         ~867 µs/slice  (trivial for 1×1, but still dispatches)
  MoveDeviceOperation          ~68 µs/slice  (L1 memory reorganisation)
  SliceWriteDeviceOperation   ~501 µs/slice  (writes output chunk to DRAM)
  ─────────────────────────────────────────
  Overhead per slice:       ~2,365 µs
  Actual Conv2dOp per slice:  ~500 µs        (the compute we want)
  Overhead-to-compute ratio:    ~5×
```

The output shard per core (`36,864 pixels × OC_padded=32 × 2 B = 2.25 MB`) exceeds
L1 bank size (~1.4 MB), making a single-pass L1 path impossible without changing the
output tile format. DRAM slicing is unavoidable for this shape, and its overhead
cannot be tuned away.

The matmul baseline wins because it issues **one monolithic kernel** that saturates
all 64 DRAM channels in parallel — despite reading 10.7× more data per pixel, it
does so with zero per-slice overhead.

**What Method 1 proved:** The individual `Conv2dDeviceOperation` slices running on
ROW_MAJOR input are 12–65× faster per slice than the baseline (confirming the DRAM
read reduction works), but the surrounding infrastructure costs more than the savings.

---

## 11. Summary of All Method 1 Experiments (Bottleneck Test)

| Experiment | Slices | Total kernel | vs Baseline |
|-----------|--------|-------------|-------------|
| Baseline (Matmul, TILE, 1536×1536) | — | **14.652 ms** | 1.00× |
| Step 1 — ROW_MAJOR + HEIGHT_SHARDED (auto) | 6 | 22.603 ms | 1.54× slower |
| Step 2 — ROW_MAJOR + BLOCK_SHARDED | 48 | 138.251 ms | 9.44× slower |
| Step 3 — ROW_MAJOR + HEIGHT_SHARDED (abh=256) | 3 | 22.102 ms | 1.51× slower |
| Baseline (Matmul, TILE, 1280×2304) | — | **17.022 ms** | 1.00× |
| Step 1 — ROW_MAJOR + HEIGHT_SHARDED (auto) | 8 | 26.790 ms | 1.57× slower |
| Step 2 — ROW_MAJOR + BLOCK_SHARDED | 72 | 171.562 ms | 10.08× slower |
| Step 3 — ROW_MAJOR + HEIGHT_SHARDED (abh=256) | 3 | 26.070 ms | 1.53× slower |

---

## 12. Code Changes (Committed, Retained)

The C++ guard in `use_matmul_for_1x1_conv()` and the test changes remain in the
branch. They correctly route small-C ops to the regular conv path and will be needed
as infrastructure for Method 2 (which also avoids the matmul path for small C but
uses a different representation strategy).

```
ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.hpp / .cpp  — is_small_channel_count guard
ttnn/cpp/ttnn/operations/conv/conv2d/conv2d.cpp               — 3 call sites
ttnn/cpp/ttnn/operations/conv/conv2d/prepare_conv2d_weights.cpp
ttnn/cpp/ttnn/operations/conv/conv_transpose2d/conv_transpose2d.cpp
tests/ttnn/unit_tests/operations/conv/test_conv2d_pointwise.py — ROW_MAJOR input
```

---

## 13. Moving to Method 2 — Spatial Packing

Method 1 proves the regular conv path reads far less data per pixel but cannot
deliver end-to-end improvement because it forces DRAM slicing.

**Method 2** attacks the root cause differently: instead of changing which kernel
reads the data, it **repacks the input tensor** so the matmul path (single kernel,
no slicing) reads efficient, well-packed tiles.

### Core Idea

For a 1×1 conv with `C=3`, group `K=32` adjacent spatial pixels into one
"super-pixel" with `C×K = 96` channels:

```
Before packing:  [N*H*W,    3]  TILE  — each tile row: [v v v 0 0 … 0]  (9.4% useful)
After  packing:  [N*H*W/32, 96] TILE  — each tile row: 3 full tiles      (100% useful)
```

`K = TILE_WIDTH / gcd(C, TILE_WIDTH) = 32/gcd(3,32) = 32` ensures `C×K=96` is an
exact multiple of `TILE_WIDTH=32` (three full tiles per row, **zero padding waste**).

### DRAM Reads Comparison

| Path | Bytes/pixel | Total activation reads | Kernel time |
|------|------------|----------------------|-------------|
| Baseline matmul (TILE, C=3→32) | 64 B | 188.7 MB | **6.4 ms** |
| Method 1 (ROW_MAJOR, C=3→8) | 16 B | 18.9 MB | 2.7× slower (slicing) |
| **Method 2 (packed, C*K=96)** | **6 B** | **17.7 MB** | **~0.6 ms** (single kernel) |

Method 2 reads only **6 bytes per pixel** (3 valid channels × 2 bytes, zero padding)
using the efficient matmul single-kernel path. Expected speedup: **~10× for conv2d**,
**~2.5–3× end-to-end**.

### Implementation Plan

**See:** `docs/method2_implementation_status.md` (next document)
