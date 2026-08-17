# TTNN `grid_sample` — Nearest-Mode Fixes

**Op:** `ttnn.grid_sample` (`ttnn/cpp/ttnn/operations/pool/grid_sample`)
**Scope:** two nearest-mode changes plus a comprehensive combinatorial test suite.
**Status:** all grid_sample tests green — **678 passed / 163 skipped / 0 failed** (comprehensive 329, unit 108, nightly 241).

This document is self-contained: it describes each issue, its root cause, the fix, the verification result, the exact `git diff`, and the full test matrix.

---

## Table of contents

1. [Overview](#1-overview)
2. [Fix A — Enable non-precomputed nearest mode](#2-fix-a--enable-non-precomputed-nearest-mode)
3. [Fix B — bf16 nearest rounding (round-half-to-even)](#3-fix-b--bf16-nearest-rounding-round-half-to-even)
4. [Files changed & git diff](#4-files-changed--git-diff)
5. [Test coverage](#5-test-coverage)
6. [How to build & run](#6-how-to-build--run)
7. [Results](#7-results)

---

## 1. Overview

`ttnn.grid_sample` implements PyTorch's `torch.nn.functional.grid_sample` for NHWC inputs. Two nearest-mode
gaps were closed:

| # | Fix | One-line description |
|---|-----|----------------------|
| A | Enable non-precomputed nearest | Allow nearest mode with `use_precomputed_grid=False` (compute the pixel index on-device from a raw `[-1,1]` grid), instead of requiring a host-precomputed grid. |
| B | bf16 nearest rounding | Round half-to-even in the kernel and host precompute to match PyTorch's `std::nearbyint`, so a bf16 grid is bit-exact with the golden instead of flipping pixels at ties (PCC 0.943 → 1.000). |

Both changes are **nearest-only**; the bilinear path is untouched.

### Standing constraints of the op (unchanged, for context)

- Input is **NHWC, ROW_MAJOR**, with channel `C` divisible by `TILE_WIDTH = 32` (validated:
  `padded_shape[-1] % 32 == 0`). Empirically confirmed: `C=48`/`C=80` are rejected, `C=96` works.
- Only `padding_mode="zeros"` is supported (`border`/`reflection` raise).
- Nearest with an **interleaved** grid auto-shards its output, sizing cores from `H_out × W_out`; `N>1` or
  K-width-extension can exceed 64 cores → route those via a **HEIGHT_SHARDED** grid.
- A HEIGHT_SHARDED grid needs an L1-aligned (16 B) shard width (interleave → reshard).

---

## 2. Fix A — Enable non-precomputed nearest mode

### Issue

Nearest mode only ran when `use_precomputed_grid=True` (grid indices computed on host via
`prepare_grid_sample_grid`). Passing a raw `[-1,1]` grid with `use_precomputed_grid=False` raised:

```
use_precomputed_grid = false is not supported with mode = 'nearest'.
Please use precomputed grid with nearest mode.
```

### Root cause

A validation guard in `GridSampleOperation::validate_on_program_cache_miss` explicitly blocked the
combination — even though the device kernel already had a non-precomputed branch.

The nearest kernel `writer_grid_sample_nearest_sharded.cpp::process_grid_point_nearest` already contained:

```cpp
if constexpr (use_precomputed_grid) {
    // read host-computed integer indices
} else {
    // read raw x,y, transform to image coords, round to nearest pixel on-device
}
```

So the capability was present in the kernel; only the host-side guard prevented reaching it.

### Fix

Remove the guard. The non-precomputed nearest path computes the nearest pixel index on-device from the raw
coordinates, with `align_corners` handled in-kernel — mirroring the bilinear non-precomputed path.

No host-side computation is added; the raw grid is uploaded to device as-is (bf16 or fp32) and everything
is computed in the kernel.

### Result

Non-precomputed nearest works across `align_corners ∈ {True,False}`, `batch_output_channels ∈ {True,False}`,
all K factors, sharded/unsharded, and `padding_mode="zeros"`.

---

## 3. Fix B — bf16 nearest rounding (round-half-to-even)

### Issue

Nearest mode with a **bf16, non-precomputed** grid disagreed with the PyTorch golden:

| align_corners | mode | grid | PCC vs golden (same coords) |
|:---:|:---:|:---:|:---:|
| False | nearest | bf16 raw | **0.943** ❌ |
| True  | nearest | bf16 raw | 1.000 ✅ (seed-dependent) |
| —     | bilinear | bf16 raw | 0.999–1.000 ✅ |

fp32 and precomputed grids passed; only bf16 raw nearest failed, and only intermittently — which is the
signature of a boundary/tie problem rather than a broad arithmetic error.

### Root cause — a tie-breaking rule mismatch (not floating-point noise)

PyTorch's `grid_sample` nearest rounds with **`std::nearbyint`** → **round-half-to-even** (banker's rounding).
The op rounded differently:

| Path | align_corners=False | align_corners=True |
|------|---------------------|--------------------|
| Kernel (`writer_grid_sample_nearest_sharded.cpp`) | `floor(x + 0.5)` → round-half-**up** | `round()` → round-half-**away-from-zero** |
| Host precompute (`grid_sample_prepare_grid.cpp`) | `floor(x + 0.5)` → round-half-**up** | `round()` → round-half-**away-from-zero** |
| **PyTorch golden** | `nearbyint` → round-half-**even** | `nearbyint` → round-half-**even** |

These rules differ **only at exact half-integer coordinates** (e.g. `2.5`: round-up → `3`, round-even → `2`).

- With **fp32** grids, a coordinate landing exactly on `x.5` after the affine transform is a *measure-zero*
  event → the rules effectively never disagree → nobody noticed.
- With **bf16** grids, the coarse coordinate spacing (~`0.008` near `1.0`) snaps *many* points exactly onto
  half-integer boundaries → the rules disagree frequently → whole pixels flip → PCC drops.

**Proof it is purely the tie rule.** Building goldens that use each rounding rule and comparing the device
output (with a bf16-quantized grid so both see identical coordinates):

```
ac=False:  device_vs_floor_half_up = 1.00000   device_vs_round_half_even = 0.94327
ac=True :  device_vs_floor_half_up = 1.00000   device_vs_round_half_even = 1.00000
```

The device matches its own rule (`floor(x+0.5)`) at **exactly 1.0** — so there is *no* fp-order noise; the
`0.94327` is entirely the floor-half-up vs round-half-even disagreement. This is also expected analytically:
for bf16 inputs and small integer image sizes the transform `(x+1)·scale + offset` needs `< 24` mantissa
bits, so it is **exact in fp32** — the only remaining degree of freedom is how ties are broken.

### Fix

Round **half-to-even** everywhere nearest resolves a pixel, matching PyTorch. The rule is identical for both
`align_corners` modes (only the coordinate transform differs), so the two branches collapse into one call.

Three consistent edits keep the on-device and host-precomputed paths bit-identical:

1. **`grid_sample_reader_common.hpp`** — new shared device helper `round_half_to_even(float)`.
2. **`writer_grid_sample_nearest_sharded.cpp`** — on-device nearest calls the helper.
3. **`grid_sample_prepare_grid.cpp`** — host precompute calls a matching `round_half_to_even_host(float)`
   (implemented explicitly rather than via `std::nearbyint`, so it is independent of the process's active
   FP rounding mode and stays identical to the device path).

Round-half-to-even reference implementation (identical on host and device):

```cpp
int32_t round_half_to_even(float x) {
    const float f = floor(x);
    const float diff = x - f;
    const int32_t fi = (int32_t)f;
    if (diff < 0.5f) return fi;
    if (diff > 0.5f) return fi + 1;
    return (fi % 2 == 0) ? fi : fi + 1;   // exact .5 tie → even neighbor
}
```

### Result

bf16 nearest is now **bit-exact with PyTorch (PCC 1.00000)** for both `align_corners`, given the same
coordinates:

```
ac=True   bf16_device_vs_bf16_golden_pcc = 1.00000   (was 1.000)
ac=False  bf16_device_vs_bf16_golden_pcc = 1.00000   (was 0.943)
```

No regressions: existing fp32/precomputed nearest tests still pass (ties are measure-zero there, so the new
rule is indistinguishable from the old one on those inputs).

### Important distinction — quantization is not a bug

Comparing a **bf16 grid** against an **fp32 golden** still gives ≈ 0.94. That is *not* a kernel error: a
bf16 grid genuinely carries coarser coordinates than fp32, and no kernel can recover fp32 precision from
bf16 input. The correct way to test a bf16 grid is against a **bf16-quantized golden** — i.e. the golden
must run in the **same dtype as the ttnn op**:

- ttnn grid bf16 → golden grid quantized to bf16.
- ttnn grid fp32 → golden grid fp32.
- Precomputed grid → golden uses the fp32 coords that produced the exact integer indices.

This dtype-parity rule is applied uniformly in the test suite (see §5).

---

## 4. Files changed & git diff

| File | Change |
|------|--------|
| `.../device/grid_sample_device_operation.cpp` | Fix A — remove the guard blocking non-precomputed nearest. |
| `.../device/kernels/grid_sample_reader_common.hpp` | Fix B — add shared `round_half_to_even()` device helper. |
| `.../device/kernels/dataflow/writer_grid_sample_nearest_sharded.cpp` | Fix B — nearest kernel uses round-half-to-even. |
| `.../grid_sample/grid_sample_prepare_grid.cpp` | Fix B — host precompute uses matching round-half-to-even. |
| `tests/ttnn/unit_tests/operations/pool/test_grid_sample_comprehensive.py` | New comprehensive combinatorial test file. |

### Fix A — `grid_sample_device_operation.cpp`

```diff
-    // Mode and precomputed grid compatibility validation
-    TT_FATAL(
-        !(operation_attributes.mode == "nearest" && !operation_attributes.use_precomputed_grid),
-        "use_precomputed_grid = false is not supported with mode = 'nearest'. Please use precomputed grid with nearest "
-        "mode.");
+    // Mode validation: nearest supports both precomputed and non-precomputed (raw [-1,1]) grids.
+    // For a non-precomputed nearest grid the device kernel computes the nearest pixel index from the
+    // raw coordinates on-device (align_corners handled in-kernel), mirroring the bilinear path.
```

### Fix B — `grid_sample_reader_common.hpp` (new shared helper)

```diff
 ALWI bool is_coordinate_valid(int32_t coord, uint32_t max_size) {
     return (coord >= 0) && (coord < static_cast<int32_t>(max_size));
 }

+// Round half to even (banker's rounding) to match PyTorch's grid_sample nearest, which uses
+// std::nearbyint under the default FE_TONEAREST mode. Ties (exactly x.5) resolve to the even
+// integer. This matters only for coordinates that land exactly on a half-integer boundary, which is
+// a measure-zero event for fp32 grids but common for bf16 grids, where the coarse coordinate spacing
+// snaps many points onto ties. floor(x+0.5) (round half up) and round() (round half away from zero)
+// would otherwise disagree with PyTorch at those ties and flip which pixel is "nearest".
+ALWI int32_t round_half_to_even(float x) {
+    const float f = floor(x);
+    const float diff = x - f;
+    const int32_t fi = static_cast<int32_t>(f);
+    if (diff < 0.5f) {
+        return fi;
+    }
+    if (diff > 0.5f) {
+        return fi + 1;
+    }
+    // Exact .5 tie: round to the even neighbor.
+    return (fi % 2 == 0) ? fi : fi + 1;
+}
+
 ALWI void fill_four_val(uint32_t begin_addr, uint16_t val, uint16_t val1, uint16_t val2, uint16_t val3) {
```

### Fix B — `writer_grid_sample_nearest_sharded.cpp` (nearest kernel)

```diff
         // Transform to image coordinates using the same formula as prepare_grid.cpp
         const float h_coord_image = ((h_coord_rel + 1.0f) * height_scale) + height_offset;
         const float w_coord_image = ((w_coord_rel + 1.0f) * width_scale) + width_offset;
-        if constexpr (align_corners) {
-            // For align_corners=True, use floor(coord) directly
-            nearest_h = static_cast<int32_t>(round(h_coord_image));
-            nearest_w = static_cast<int32_t>(round(w_coord_image));
-        } else {
-            // For nearest neighbor, use floor(coord + 0.5) to match preprocessing
-            nearest_h = static_cast<int32_t>(floor(h_coord_image + 0.5f));
-            nearest_w = static_cast<int32_t>(floor(w_coord_image + 0.5f));
-        }
+        // Round half to even to match PyTorch grid_sample nearest (std::nearbyint). The rounding rule
+        // is identical for both align_corners modes; only the coordinate transform above differs.
+        nearest_h = round_half_to_even(h_coord_image);
+        nearest_w = round_half_to_even(w_coord_image);
```

### Fix B — `grid_sample_prepare_grid.cpp` (host precompute)

```diff
 namespace {

+// Round half to even (banker's rounding), matching PyTorch grid_sample nearest (std::nearbyint under
+// the default FE_TONEAREST mode) and the on-device kernel helper of the same name. Implemented
+// explicitly rather than via std::nearbyint so the host precompute is independent of the process's
+// active floating-point rounding mode and stays bit-identical to the device path.
+inline int32_t round_half_to_even_host(float x) {
+    const float f = std::floor(x);
+    const float diff = x - f;
+    const int32_t fi = static_cast<int32_t>(f);
+    if (diff < 0.5f) {
+        return fi;
+    }
+    if (diff > 0.5f) {
+        return fi + 1;
+    }
+    return (fi % 2 == 0) ? fi : fi + 1;
+}
+
 // Unified helper function for grid preprocessing (both nearest and bilinear modes).
...
-                        int32_t h_nearest, w_nearest;
-                        if (align_corners) {
-                            h_nearest = static_cast<int32_t>(std::round(h_coord_image));
-                            w_nearest = static_cast<int32_t>(std::round(w_coord_image));
-                        } else {
-                            h_nearest = static_cast<int32_t>(std::floor(h_coord_image + 0.5f));
-                            w_nearest = static_cast<int32_t>(std::floor(w_coord_image + 0.5f));
-                        }
+                        // Round half to even (banker's rounding) to match PyTorch grid_sample nearest
+                        // (std::nearbyint) and the on-device kernel (round_half_to_even). Same rule for
+                        // both align_corners modes; only the coordinate transform above differs.
+                        int32_t h_nearest = round_half_to_even_host(h_coord_image);
+                        int32_t w_nearest = round_half_to_even_host(w_coord_image);
```

---

## 5. Test coverage

File: `tests/ttnn/unit_tests/operations/pool/test_grid_sample_comprehensive.py`
Reference: `ttnn.operations.pool.golden_grid_sample` (wraps `torch.nn.functional.grid_sample`), PCC ≥ 0.99.

**Golden dtype parity (core testing rule):** the golden's grid always runs in the same dtype as the ttnn
grid — bf16 ttnn grid → bf16-quantized golden grid; fp32 → fp32; precomputed → fp32 coords behind the exact
integer indices. Otherwise the two see different coordinates and the comparison is meaningless.

| Test | Cases | What it covers |
|------|:----:|----------------|
| `test_bilinear_matrix` | 72 | Bilinear cross-product: `align_corners{T,F}` × `precomputed{F,T}` × `(batch_output_channels,K) ∈ {(F,1),(F,4),(T,4)}` × `sharded{F,T}` × 3 shapes. |
| `test_nearest_matrix` | 72 | Same cross-product for nearest, incl. non-precomputed. Skips nearest + interleaved grid when `N>1` or K-width-extend (>64-core auto-shard limit). |
| `test_bilinear_grid_dtype` | 4 | Bilinear grid dtype `{bf16, fp32}` × `align_corners{T,F}`; golden quantized to match. |
| `test_grid_sample_nchw_shapes` | 7 | NCHW-shape sweep, raw (non-precomputed) grid, bilinear + nearest, incl. a BEV-representative `(1,64,80,144)` shape. |
| `test_nearest_batched_channel_extend` | 2 | Non-precomputed nearest, K=8, `batch_output_channels=True`: input `(1,{80×144,96×96},64)`, grid `(1,128,64,16)` → output `(1,128,64,512)`. |
| `test_grid_sample_large_matrix` | 216 | Large combinatorial matrix (see below). |
| **Total** | **373** | 329 passed + 44 skipped. |

### `test_grid_sample_large_matrix` axes (216 cases)

| Axis | Values |
|------|--------|
| mode | `bilinear`, `nearest` |
| align_corners | `True`, `False` |
| (precomputed, grid_dtype) | `(False, fp32)`, `(False, bf16)`, `(True, bf16)` — precomputed is always bf16, so fp32 there would just duplicate |
| (batch_output_channels, K) | `(False, 1)`, `(False, 4)` (W-extend), `(True, 4)` (C-batch) |
| memory config | `dram` (DRAM interleaved), `l1` (L1 interleaved), `sharded` (HEIGHT_SHARDED, 16 B-aligned shard width) |
| shapes (tile-aligned vs not) | `(1,64,96,64)` grid `64×32` → `N·H·W = 2048` (÷32, aligned); `(1,48,72,64)` grid `30×17` → `510` (not ÷32) |

**Bigger shapes:** input up to `(1,64,96,64)`, grid up to `64×32`, output up to `C·K = 256` channels —
larger than the base matrix's `≤ 32×24` shapes.

**"Tile-aligned vs not"** is measured on the **grid-point count** `N·H_out·W_grid`, *not* the input channel:
input `C` must stay `% 32` (an op requirement — `C=48`/`80` are rejected). The op pads grid points to tiles
internally, so both aligned (`2048`) and non-aligned (`510`) counts must produce correct output — and do.

**Skips in the large matrix (44 total across the file):**

- `nearest + interleaved grid (dram/l1) + (N>1 or K-width-extend)` — auto-shard exceeds 64 cores; those go
  through a sharded grid instead. *(The previous `nearest + non-precomputed + bf16` skip was removed after
  Fix B — those 28 cases now run and pass.)*

---

## 6. How to build & run

```bash
# Build ttnn (C++ changes: guard + host precompute + shared header).
cmake --build build_Release --target ttnn

# Refresh the runtime copy of the shared library (runtime loads the source-tree copy).
cp build_Release/ttnn/_ttnn.so ttnn/ttnn/_ttnn.so
# NOTE: the device kernel .cpp is JIT-compiled from source at runtime — no rebuild needed for it.

# Run the suites.
python -m pytest tests/ttnn/unit_tests/operations/pool/test_grid_sample_comprehensive.py -q
python -m pytest tests/ttnn/unit_tests/operations/pool/test_grid_sample.py -q
python -m pytest tests/ttnn/nightly/unit_tests/operations/pool/test_grid_sample.py -q
```

---

## 7. Results

| Suite | Result |
|-------|--------|
| `test_grid_sample_comprehensive.py` | **329 passed, 44 skipped** |
| `test_grid_sample.py` (unit) | **108 passed** |
| `nightly/.../test_grid_sample.py` | **241 passed, 119 skipped** |
| **Total** | **678 passed, 163 skipped, 0 failed** |

**Key numbers**

- bf16 nearest, PCC vs same-dtype golden: **0.943 → 1.00000** (ac=False); 1.0 (ac=True).
- Non-precomputed nearest: enabled across all flag combinations.
- No regressions in fp32/precomputed nearest or in the bilinear path.
