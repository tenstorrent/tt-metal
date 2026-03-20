# Conv3D Blocking & FP32 Reduction

## Background

PR #38579 introduced aggressive Conv3D blockings for the WAN VAE decoder. PR #39043 reverted 4 blocking entries because the two configs produced different video hashes. This document explains why, and documents the fp32 intermediate CB fix (tested on BH, not yet tested on WH).

Related: https://github.com/tenstorrent/tt-metal/issues/40322

## The 4 Reverted Blocking Entries

```
Key: (in_channels, out_channels, kernel_size) → (C_in_block, C_out_block, T_out_block, H_out_block, W_out_block)

                              GOOD (current)          AGGRESSIVE (reverted)
(32,  384, (3,3,3)):         (32,  384, 1, 8,  8)    (32,  96, 1, 2, 32)
(192, 384, (3,3,3)):         (96,  128, 1, 32, 1)    (64, 128, 1, 8,  4)
(384, 384, (3,3,3)):         (128, 128, 1, 8,  2)    (96,  96, 1, 8,  4)
(384, 768, (3,3,3)):         (128, 128, 1, 16, 2)    (96,  96, 1, 8,  4)
```

Only `C_in_block` affects numerical output. `C_out_block` and spatial blocks only affect memory layout and parallelism.

| Entry | C_in_block change | Decoder layers affected | Impact |
|-------|-------------------|------------------------|--------|
| `(32, 384, 3x3x3)` | 32 → 32 (**same**) | 1 (conv_in) | **Zero** — only C_out_block differs |
| `(384, 768, 3x3x3)` | N/A | **0** | **Dead entry** — time_conv has kernel (3,1,1), not (3,3,3) |
| `(192, 384, 3x3x3)` | 96 → 64 | 1 (up_blocks.1.resnets.0.conv1) | Minor |
| `(384, 384, 3x3x3)` | 128 → 96 | **15** layers (mid_block + up_blocks 0,1) | **Dominant** |

**Only 16 of the decoder's ~30 conv layers are affected, and only via C_in_block.**

## Why C_in_block Changes the Hash

### The Conv3D matmul with C_in blocking

When `C_in_block < C_in`, the conv3d splits the input channels into groups and runs one matmul per group. Each group's partial result is accumulated via multi-core NOC reduction.

For `(384, 384, (3,3,3))` — the dominant entry:

```
GOOD (C_in_block=128):
  384 / 128 = 3 groups
  Each matmul: [num_patches × 3456] @ [3456 × C_out_block]
  (3456 = 27 spatial × 128 channels)

AGGRESSIVE (C_in_block=96):
  384 / 96 = 4 groups
  Each matmul: [num_patches × 2592] @ [2592 × C_out_block]
  (2592 = 27 spatial × 96 channels)
```

### Why different group counts produce different results

**fp32 addition is not associative.** Different group boundaries mean different subsets of products are accumulated together, producing different rounding at each step.

On WH (HiFi2/TF32): each bf16 input is truncated to 10-bit mantissa before multiply, amplifying the sensitivity significantly.

On BH (HiFi4): full 23-bit mantissa is preserved, making rounding differences ~2^13 times smaller. However, when partial sums are truncated to bf16 between groups (in the reduction CB), the truncation reintroduces C_in_block sensitivity regardless of math fidelity.

### Measured per-layer error (CPU simulation with real weights, WH)

Single pixel, single layer (decoder.mid_block.resnets.0.conv1):
```
GOOD vs fp32 ref:  max=1.97e-03, mean=4.52e-04
AGGR vs fp32 ref:  max=1.97e-03, mean=4.52e-04  (equally accurate!)
GOOD vs AGGR:      max=2.62e-06, mean=2.53e-07
Non-zero diffs:    342/384 output channels
```

**Both configs are equally far from fp32 reference.** Neither is "more correct."

## Amplification: 16 Layers + Temporal Cache

### Spatial cascade (within one frame)

The decoder has this structure:
```
conv_in → mid_block (4 affected conv layers)
        → up_block_0 (6 affected conv layers)
        → up_block_1 (6 affected conv layers + 1 (192→384) layer)
        → up_block_2 (no affected layers — uses (192,192))
        → up_block_3 (no affected layers — uses (96,96))
        → norm_out → conv_out
```

Error accumulation:
1. Each affected conv introduces ~1e-3 mean error between configs
2. **RMSNorm** normalizes per-position over channels — different inputs → different RMS → different scaling
3. **Residual connections** (`h(x) + conv(norm(conv(norm(x))))`) preserve and accumulate errors additively
4. **SiLU** is smooth — does not amplify errors significantly
5. After 16 affected layers: ~0.002 mean error (measured at T=1)

### Temporal cascade (across frames)

The decoder processes T input frames one-at-a-time. Each frame's conv outputs are stored in `feat_cache` (window size `CACHE_T=2`). The next frame reads this cache as temporal context via the causal conv3d's temporal kernel.

```
Frame 0: conv produces output_0 (with accumulated blocking error ε₀)
         → cache stores output_0
Frame 1: conv reads cache (output_0 + ε₀) as temporal context
         → produces output_1 with error ε₁ that INCLUDES contribution from ε₀
         → cache stores output_1
Frame N: error includes contributions from ALL previous frames
```

The temporal upsample structure creates a **4-frame repeating PCC pattern**:

```
Frame  0: PCC=0.99996  (fresh input frame)
Frame  1: PCC=0.99994  (first temporal upsample — from cached data)
Frame  2: PCC=0.99994  (second temporal upsample)
Frame  3: PCC=0.99995  (spatial-only upsample)
Frame  4: PCC=0.99996  (new input frame — cache refreshed)
...
Frame 36: PCC=0.99994  (accumulated temporal drift)
```

## Measured Impact on Video Output (WH Loud Box, bf16 intermediates)

### Test configuration
- Platform: WH Loud Box (8-chip, 2x4 mesh)
- Resolution: 480p (H=60, W=104 in latent space → 480×832 output)
- Frames: T=10 input → 37 output frames
- Dtype: bfloat16
- Weights: Real (Wan-AI/Wan2.2-T2V-A14B-Diffusers)

### Results

| Comparison | T=1 PCC | T=10 PCC |
|------------|---------|----------|
| GOOD vs Torch fp32 | 0.99994 | 0.99977 |
| AGGRESSIVE vs Torch fp32 | 0.99994 | 0.99973 |
| GOOD vs AGGRESSIVE | 0.99996 | 0.99955 |

### Error distribution (GOOD vs AGGRESSIVE, T=10)

| Percentile | Error ([-1,1] space) | Pixel values ([0,255]) |
|------------|---------------------|----------------------|
| Median (P50) | 0.00195 | 0.25 — below rounding threshold |
| P90 | 0.00391 | 0.50 — at rounding boundary |
| P95 | 0.00586 | 0.75 — likely flips pixel |
| P99 | 0.00879 | 1.12 — flips pixel value |
| P99.9 | 0.01490 | 1.90 — flips by 2 values |
| Max | 0.500 | 63.8 — 64 pixel values! |

### Hash impact

Per frame (480 × 832 × 3 = 1,198,080 pixel values):
- ~1% of pixels (11,981) differ by ≥ 1 pixel value
- ~5% of pixels (59,904) at the rounding boundary (may flip)
- ~0.1% of pixels (1,198) differ by ≥ 2 pixel values

Over 37 frames: **~443,000 pixel values change by ≥ 1** → deterministic video encoder produces a different hash.

## Complete Causal Chain

```
C_in_block differs (128 vs 96)
  │
  ├─ Different matmul K dimension (3456 vs 2592 elements)
  ├─ Different number of accumulation groups (3 vs 4)
  │
  └─ bf16 truncation between groups + non-associative fp32 add
       │
       └─ ~2.6e-06 max error per output channel, per pixel, per layer
            │
            ├─ × 16 affected layers (cascade through mid_block + up_blocks 0,1)
            ├─ + RMSNorm rescaling (amplifies when signal is small)
            ├─ + Residual connections (additive error accumulation)
            │
            └─ ~0.002 mean error per pixel at decoder output (T=1)
                 │
                 ├─ × 37 frames with temporal cache contamination
                 │
                 └─ ~1% of pixels differ by ≥ 1 value per frame
                      │
                      └─ Video hash is different
```

## FP32 Intermediate CB Fix

### Goal

Eliminate C_in_block sensitivity by keeping partial sums in Float32 through the multi-core reduction, only truncating to bf16 at the final untilize. This removes the bf16 truncation between C_in groups that is the source of blocking sensitivity.

### Status

- **Tested on BH**: Working (PCC=0.999996 vs torch)
- **Not yet tested on WH**: The approach uses different techniques than the earlier failed WH attempts (see below). It may work on WH but needs verification.

### What changed (3 kernel files)

**`conv3d_program_factory.cpp`**:
- `use_fp32_partials` flag when `fp32_dest_acc_en && C_in_num_blocks > 1`
- `cb_matmul_interm_tiled` and `cb_reduction_tiled` use Float32 format (4096-byte tiles)
- Compile-time arg `use_fp32_partials` passed to compute kernel

**`compute.cpp`**:
- `reconfig_data_format_srca(cb_vol2col_rm)` before tilize — resets unpacker from Float32 (left by previous untilize) back to bf16. **This was the key bug fix.**
- `pack_reconfig_data_format` around tilize to switch packer between bf16 (tilize) and fp32 (matmul)
- `mm_block_init_short_with_both_dt` after tilize to reinit matmul unpackers
- SFPU reduction: `copy_tile` + `add_binary_tile` (FPU `add_tiles` can't unpack Float32 from L1)
- `reconfig_data_format(cb_matmul_interm_tiled, cb_bias_tiled)` before bias add for mixed Float32+bf16
- Untilize with `UnpackAndPackReconfigure` for Float32 → bf16

**`writer.cpp`**:
- `partials_tile_bytes = get_tile_size(cb_matmul_interm_tiled)` for fp32-aware NOC read sizes

### Root cause of PCC=0.789 (the bug that was fixed)

The fp32 untilize uses `UnpackAndPackReconfigure` which sets the unpacker to Float32. On the next spatial block, the tilize reads bf16 `cb_vol2col_rm` data with a Float32 unpacker, producing garbage. The first spatial block was correct because the unpacker starts in bf16 mode.

Debugging path:
1. DPRINT verified NOC read was correct (raw bytes matched between worker and reducer)
2. Three different add methods (SFPU, FPU, L1 acc pack) all gave identical PCC=0.789 → issue upstream of add
3. Element-by-element comparison showed position [0,0,0] correct but [0,16,16] wrong → spatial-dependent corruption
4. Traced unpacker state: untilize's `UnpackAndPackReconfigure` left srcA in Float32, tilize's `NoReconfigure` didn't reset it

### Earlier failed WH attempts (different approaches)

These used different techniques from our current fix and all failed on WH:

| # | Approach | Failure Mode |
|---|----------|-------------|
| 1 | fp32 `cb_matmul_interm` + `NoReconfigure` untilize | `pack_untilize` requires same page size for input/output CBs |
| 2 | fp32 `cb_matmul_interm` + typecast CB before untilize | `mm_init` with fp32 output CB corrupts matmul on WH |
| 3 | Separate fp32 accum CB (matmul stays bf16) | Packer format switch mid-kernel leaves stale config |
| 4 | Same as #3 + `reconfig_data_format_srca` | Unpacker reconfig mid-matmul corrupts state |

Our current fix is architecturally different: the matmul directly outputs to a Float32 CB (not a bf16→fp32 typecast), and uses SFPU `copy_tile`+`add_binary_tile` for reduction (not FPU `add_tiles`). This avoids the FPU format-mismatch issues. Whether this approach also works on WH is untested.

## Accuracy (BH)

### Per-op accuracy with aggressive blocking (C_in_block=32, C_in_num_blocks=3)

| Intermediate format | PCC vs torch | MAE | Max Error |
|---|---|---|---|
| bf16 (original) | 0.999957 | 0.002874 | 0.024772 |
| **fp32 (this fix)** | **0.999996** | **0.000775** | **0.005459** |

fp32 intermediates reduce MAE by 3.7x and max error by 4.5x.

### Good vs Aggressive blocking comparison

Both produce PCC > 0.999 vs torch. Max difference is 1 bf16 ULP (0.007812) — inherent to bf16 output quantization since the computation order differs. They will NOT be bit-identical.

## Device Perf (BH P150, single chip)

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

### 384x384 k3, 90x160 (same C_in_block=128, C_in_num_blocks=3)

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

### Perf summary

- Aggressive blocking gives **1.4-1.7x device speedup** on layers where it increases parallelism
- fp32 SFPU reduction adds ~25% overhead vs bf16 FPU add (66 vs 53 μs) but recovers full accuracy
- For layers where conservative blocking OOMs, aggressive + fp32 is the only option with correct results

## Test Commands

```bash
# Regression (original bf16 path, should pass)
pytest models/tt_dit/tests/models/wan2_2/test_vae_wan2_1.py -k "test_wan_conv3d and 1x1_h0_w1 and cache_none and bf16 and conv_0" -v -s --timeout=120

# Blocking comparison (good vs aggressive)
pytest models/tt_dit/tests/models/wan2_2/test_conv3d_blocking_comparison.py -v -s --timeout=600

# Device perf benchmark
TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
  python models/tt_dit/tests/models/wan2_2/bench_conv3d_fp32_reduction.py
```
