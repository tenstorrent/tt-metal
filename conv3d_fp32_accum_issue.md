# Conv3D: Float32 intermediate matmul CB for multi-core reduction

## Goal

When conv3d splits input channels into `C_in_block` groups (`C_in_num_blocks > 1`), each core computes a matmul partial that gets reduced across cores via NOC. Previously `cb_matmul_interm_tiled` was bf16, causing truncation between partials. Making it Float32 eliminates the truncation and allows using faster blockings (1.5x speedup from PR #38579).

Related: https://github.com/tenstorrent/tt-metal/issues/40322

## Status: FIXED

Multi-core fp32 reduction now works: PCC=0.999996 vs torch.

**Results with aggressive blocking (C_in_block=32, C_in_num_blocks=3):**
- fp32 intermediates: PCC=0.999996, MAE=0.000775, MaxErr=0.005459
- bf16 intermediates (original): PCC=0.999957, MAE=0.002874, MaxErr=0.024772
- fp32 intermediates reduce MAE by 3.7x and max error by 4.5x

**Good vs Aggressive blocking comparison:**
- Both produce PCC > 0.999 vs torch
- Max difference between the two blockings is 1 bf16 ULP (0.007812) — inherent to bf16 output quantization since the computation order differs

## Device Perf (Blackhole P150, single chip)

Measured with `TT_METAL_DEVICE_PROFILER=1` using `DEVICE KERNEL DURATION [ns]` from `ttnn.ReadDeviceProfiler` / `ttnn.get_latest_programs_perf_data`. See `bench_conv3d_fp32_reduction.py` to reproduce.

### 192x192 k3, 32x32 (C_in_num_blocks: 1 vs 2)

| Config | Device Time | Speedup | PCC | MAE |
|---|---|---|---|---|
| Conservative C_in=192 | 192 μs | 1.0x | 0.999995 | 0.000822 |
| Aggressive C_in=96, bf16 interm | 122 μs | 1.57x | 0.999867 | 0.005515 |
| **Aggressive C_in=96, fp32 interm** | **122 μs** | **1.57x** | **0.999996** | **0.000731** |

### 96x96 k3, 32x32 (C_in_num_blocks: 1 vs 3)

| Config | Device Time | Speedup | PCC | MAE |
|---|---|---|---|---|
| Conservative C_in=96 | 91 μs | 1.0x | 0.999994 | 0.000852 |
| Aggressive C_in=32, bf16 interm | 53 μs | 1.70x | 0.999957 | 0.002871 |
| **Aggressive C_in=32, fp32 interm** | **66 μs** | **1.36x** | **0.999996** | **0.000768** |

### 384x384 k3, 90x160 (same C_in_block=128 for both, C_in_num_blocks=3)

| Config | Device Time | PCC | MAE |
|---|---|---|---|
| Conservative C_in=128 | 2288 μs | 0.999996 | 0.000779 |
| Aggressive C_in=128, bf16 interm | 2283 μs | 0.999804 | 0.007408 |
| **Aggressive C_in=128, fp32 interm** | **2293 μs** | **0.999996** | **0.000779** |

### 384x384 k3, 32x32 (conservative OOMs — must use aggressive)

| Config | Device Time | PCC | MAE |
|---|---|---|---|
| Conservative C_in=384 | **L1 OOM** | — | — |
| Aggressive C_in=128, bf16 interm | 306 μs | 0.999807 | 0.007132 |
| **Aggressive C_in=128, fp32 interm** | **312 μs** | **0.999996** | **0.000768** |

**Takeaway:** Aggressive blocking gives 1.4-1.7x device speedup on layers where it increases parallelism. The fp32 SFPU reduction adds ~25% overhead vs bf16 FPU add (66 vs 53 μs) but recovers full accuracy (MAE 0.0008 vs 0.003-0.007). For layers where conservative blocking OOMs, aggressive + fp32 is the only option with correct results.

## Root Cause (RESOLVED)

The PCC=0.789 bug was caused by a **missing unpacker reconfig** in the compute kernel's spatial loop.

### The bug

After the fp32 untilize step (which uses `UnpackAndPackReconfigure`), the unpacker's srcA was left configured for **Float32** (for reading `cb_matmul_interm_tiled`). On the next spatial block iteration, the tilize reads **bf16** data from `cb_vol2col_rm`, but the unpacker was still in Float32 mode. The tilize used `NoReconfigure`, so it didn't fix this.

**Why only the first spatial block was correct:** The unpacker starts in bf16 mode at the beginning of execution. The first spatial block's tilize reads bf16 correctly. After the first block's untilize switches the unpacker to Float32, all subsequent spatial blocks' tilize operations interpret bf16 vol2col data as Float32, producing garbage matmul inputs.

### The fix

One line added to `compute.cpp`, in the `use_fp32_partials` block before the tilize:

```cpp
if constexpr (use_fp32_partials) {
    pack_reconfig_data_format(cb_matmul_interm_tiled, cb_vol2col_tiled);
    reconfig_data_format_srca(cb_vol2col_rm);  // <-- THE FIX: reset unpacker from Float32 to bf16
}
```

### Debugging path

1. **DPRINT NOC read verification**: Confirmed raw bytes transferred correctly between worker and reducer cores (required mapping logical vs physical core coordinates — `TT_METAL_DPRINT_CORES` uses logical coords, writer runtime args use physical coords from `worker_core_from_logical_core`)
2. **Three add methods tested**: SFPU `add_binary_tile`, FPU `add_block_inplace`, and packer L1 accumulation all produced identical PCC=0.789 → proved the addition itself was correct
3. **Spatial error pattern**: Comparing output element-by-element revealed position [0,0,0] was correct but [0,16,16] was wrong → pointed to per-spatial-block state corruption
4. **Traced unpacker state**: The fp32 untilize (`UnpackAndPackReconfigure`) reconfigures srcA to Float32. The next tilize uses `NoReconfigure` and reads bf16 data with a Float32 unpacker → root cause

## Implementations

### Multi-core fp32 reduction path (WORKING)
- `use_fp32_partials=true` when `fp32_dest_acc_en && C_in_num_blocks > 1`
- Partials CB (`cb_matmul_interm_tiled`) and reduction CB (`cb_reduction_tiled`) use Float32 format
- Reduction uses SFPU `add_binary_tile` via `copy_tile` to load both operands to DST, then adds in SFPU
- Bias add uses `reconfig_data_format` for mixed Float32 partials + bf16 bias
- Untilize uses `UnpackAndPackReconfigure` for Float32 → bf16 conversion
- Writer NOC read uses `partials_tile_bytes` (4096 for fp32) instead of `tile_bytes` (2048 for bf16)

### packer_l1_acc path (WORKING but slow)
- `use_l1_acc=true`: all C_in blocks on one core, FIFO pointer save/reset for L1 accumulation
- PCC=0.999996 vs torch
- **6-13x slower** than multi-core reduction (serializes C_in work)
- Only useful if perf doesn't matter

## Files Modified

1. **`conv3d_program_factory.cpp`**:
   - `use_fp32_partials` flag and `partial_data_format` (Float32 when enabled)
   - fp32 partials and reduction CBs
   - Compile-time args for `use_fp32_partials` to compute (index 29), writer (not needed — uses CB tile size)
   - Writer NOC read uses `partials_tile_bytes` for fp32 tiles

2. **`compute.cpp`**:
   - `use_fp32_partials` compile-time arg
   - `pack_reconfig_data_format` before/after tilize to switch packer between bf16 (tilize) and fp32 (matmul)
   - **`reconfig_data_format_srca(cb_vol2col_rm)`** before tilize — THE KEY FIX
   - `mm_block_init_short_with_both_dt` after tilize to reconfigure matmul with proper data types
   - SFPU reduction: `copy_tile` + `add_binary_tile` for fp32-safe addition
   - `reconfig_data_format(cb_matmul_interm_tiled, cb_bias_tiled)` before bias add
   - Untilize with `UnpackAndPackReconfigure` for fp32 → bf16
   - L1 acc path: FIFO pointer save/reset, `pack_reconfig_l1_acc`, idle core guard

3. **`writer.cpp`**:
   - L1 acc path: loop reorder, idle core guard
   - `partials_tile_bytes` for fp32 NOC reads in reduction

4. **`reader_vol2col.cpp`**:
   - Flat iteration loop with `if constexpr (use_l1_acc)` for loop order
   - Compile-time args at indices 36-37

## Bugs Found and Fixed

### 1. Unpacker reconfig after fp32 untilize (ROOT CAUSE of PCC=0.789)
The fp32 untilize (`UnpackAndPackReconfigure`) left the unpacker in Float32 mode. The next tilize read bf16 vol2col data with a Float32 unpacker. Only the first spatial block was correct.
**Fix:** `reconfig_data_format_srca(cb_vol2col_rm)` before the tilize.

### 2. Idle Core Hang
Cores with `c_in_block_start == c_in_block_end` still had non-empty c_out/spatial ranges. The spatial loops ran with 0 c_in iterations, causing untilize to hang.
**Fix:** `if (c_in_block_start < c_in_block_end)` guard.

### 3. `cb_wait_front` Cumulative Counting
Multiple `cb_wait_front(cb, N)` without `cb_pop_front` requires increasing counts (N, 2N, 3N...).
**Fix:** `add_bias_inplace_no_bias_wait` for L1 acc path; proper wait/pop patterns.

### 4. fp32 NaN from Missing Unpacker Reconfig
Bias add with fp32 partials + bf16 bias needed `reconfig_data_format(partials_cb, bias_cb)`.
**Fix:** Added reconfig before bias add, gated by `if constexpr (use_fp32_partials)`.

### 5. pack_reconfig Breaks bf16 Path
Adding `pack_reconfig_data_format` unconditionally corrupted the bf16 path.
**Fix:** All reconfig calls gated by `if constexpr (use_fp32_partials)`.

## Test Commands

```bash
# Regression (original path, should pass)
pytest models/tt_dit/tests/models/wan2_2/test_vae_wan2_1.py -k "test_wan_conv3d and 1x1_h0_w1 and cache_none and bf16 and conv_0" -v -s --timeout=120

# Blocking comparison
pytest models/tt_dit/tests/models/wan2_2/test_conv3d_blocking_comparison.py -v -s --timeout=600
```
