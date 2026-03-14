# Pre-existing Softmax Test Failures

All failures below are pre-existing (verified on parent commit). None are regressions from commit `7e9f51a71d4` (separate MAX/SUM scaler CBs).

Base branch merge point: `48879659c54` (pjosipovic/llk_helper_library).
Not yet verified whether these fail on the base branch or were introduced on this branch.

## 1. Program Cache PCC Failures (~0.899 / ~0.839)

```
test_softmax_stable_with_program_cache[in_dtype=DataType.BFLOAT8_B-fp32_acc_en=True-math_approx=True-skip_scale_mask=True-w=1024-h=32-batch_size=1]  PCC=0.8986
test_softmax_stable_with_program_cache[in_dtype=DataType.BFLOAT16-fp32_acc_en=True-math_approx=True-skip_scale_mask=True-w=1024-h=32-batch_size=1]   PCC=0.8986
test_softmax_sharded_stable_with_program_cache[in_dtype=DataType.BFLOAT8_B-fp32_acc_en=True-math_approx=True-skip_scale_mask=True-w=384-h=384-num_heads=4-batch_size=8]  PCC=0.8386
```

## ~~2. Ambiguous `reload_accumulator_if_needed` Compilation Error (REDUCE_COL path)~~ FIXED

Fixed: added `use_matmul` template param to declaration in `reduce_helpers_compute.hpp` to match definition in `.inl`. All 72 non-dim=-1 softmax tests now pass.

## 3. Softmax Accuracy ULP Failures (fp32_acc_en=True, shape 16384x256)

```
test_softmax_accuracy[shape=(1, 1, 16384, 256)-fp32_acc_en=True-math_approx_mode=False-expected_ulp=3-numeric_stable=True]   Max ULP Delta: 255.0
test_softmax_accuracy[shape=(1, 1, 16384, 256)-fp32_acc_en=True-math_approx_mode=True-expected_ulp=9-numeric_stable=False]   Max ULP Delta: 616.0
test_softmax_accuracy[shape=(1, 1, 16384, 256)-fp32_acc_en=True-math_approx_mode=True-expected_ulp=9-numeric_stable=True]    Max ULP Delta: 2160.0
```

## 4. Large Kernel Block Size PCC Failures (Wt=193..196)

```
test_softmax_large_kernel_block_size[Wt=193]  PCC=0.9900
test_softmax_large_kernel_block_size[Wt=194]  PCC=0.9913
test_softmax_large_kernel_block_size[Wt=195]  PCC=0.9924
test_softmax_large_kernel_block_size[Wt=196]  PCC=0.9907
```

---

## Root Cause Analysis: fp32_dest_acc_en + matmul-based reduce in softmax

### Common thread

All failing tests (categories 1, 3, 4) have `fp32_dest_acc_en=True`. The failures are **not** program-cache-related — they reproduce on the very first execution.

### Investigation

**1. Isolated the reduce sum op (`ttnn.sum`) — works correctly.**

Wrote `test_sum_row_fp32_acc` in `tests/ttnn/unit_tests/operations/reduce/test_sum.py`. With shape `(1,1,32,1024)`, bfloat16 input, `fp32_dest_acc_en=True`, `ttnn.sum(dim=-1)` produces PCC=1.0. The reduce LLK itself is not broken.

**2. Confirmed softmax output is inflated ~32x, not zeroed.**

Softmax row sums ≈ 32–57 instead of 1.0. The ratio `ttnn/torch` per element is ~32x. This pointed to the reduce sum inside softmax producing wrong results or the reciprocal being wrong.

**3. Identified that softmax uses `compute_kernel_lib::reduce()`, not standalone reduce ops.**

The reduce helper dispatches SUM+REDUCE_ROW to `matmul_tiles` (not `reduce_tile`). The `recip_tile` post-op computes 1/sum. Checked all three softmax compute kernels (`softmax.cpp`, `softmax_large_tensor.cpp`, `softmax_sharded.cpp`) — all use this path.

**4. Used `dprint_tensix_dest_reg()` to print DEST register contents.**

Printed DEST after the matmul-based sum, before `recip_tile`. Found that **all 16 columns** of DEST had non-zero values. A correct REDUCE_ROW matmul should produce values only in column 0 (the dot product). The matmul was computing a full matrix product instead of a column-0-only result.

**5. Verified scaler tile is correct in L1.**

Used `print_full_tile(cb, 0, true)` (untilized) from both UNPACK and PACK threads. The SUM scaler (c_13) has 1s in column 0, zeros elsewhere — correct for the matmul path. The MAX scaler (c_2) has 1s in row 0 of each face — correct for reduce_tile path.

**6. Confirmed the problem appears on the very first `matmul_tiles` call.**

Added DEST prints after each of the first 3 `matmul_tiles` iterations. Already after wt=0, all columns are populated. The matmul hardware is misinterpreting the scaler tile despite correct L1 data.

**7. Found the fix: `mm_init_short` is insufficient after `reduce_init`/`reduce_uninit`.**

The softmax compute kernel calls MAX reduce first (via `reduce_tile`), then SUM reduce (via `matmul_tiles`). The MAX reduce uses `reduce_init<MAX, REDUCE_ROW, enforce_fp32_accumulation=true>` followed by `reduce_uninit`. This modifies the unpacker hardware configuration. Then `mm_init_short` is called for the SUM reduce, but it does **not** call `llk_unpack_hw_configure<DST_ACCUM_MODE>` — so the unpacker remains in the state left by the reduce, causing it to misinterpret the scaler tile.

**8. Verified the fix.**

Adding `UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(scaler_cb, input_cb)))` before `mm_init_short` in `reduce_helpers_compute.inl` fixes the issue:
- Row sums: ~1.0
- PCC: 0.9996
- Verified with chip reset to rule out stale HW state

The full `mm_init` (which includes `llk_unpack_hw_configure`) also fixes it, but overrides packer state (`pack_dest_init`) which may cause issues for other callers.

### Debug tools used

| Tool | Purpose |
|------|---------|
| `print_full_tile(cb, tile, untilize=true)` | Print CB tile contents from UNPACK/PACK threads |
| `dprint_tensix_dest_reg(tile_id)` | Print DEST register from MATH thread (auto-detects fp32 format) |
| `TT_METAL_DPRINT_CORES="(0,0)"` | Enable DPRINT on specific core |
| `TT_METAL_DPRINT_ONE_FILE_PER_RISC=1` | Separate output per TRISC thread |
| `tt-smi -r` | Chip reset between runs to clear stale HW state |

### Fix location

`ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl`, in the `use_matmul` init path — add `llk_unpack_hw_configure` before `mm_init_short`.

### Status

Fix verified for the Cat 1 simplest test case. Full test suite (all categories) not yet run.
