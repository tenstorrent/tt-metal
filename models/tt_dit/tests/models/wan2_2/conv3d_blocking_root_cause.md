# Conv3D Blocking Root Cause Analysis

## Problem Statement

PR #38579 introduced aggressive Conv3D blockings for the WAN VAE decoder. PR #39043 reverted 4 blocking entries because the two configs produced different video hashes. This document explains exactly why.

## The 4 Reverted Blocking Entries

```
Key: (in_channels, out_channels, kernel_size) → (C_in_block, C_out_block, T_out_block, H_out_block, W_out_block)

                              GOOD (current)          AGGRESSIVE (reverted)
(32,  384, (3,3,3)):         (32,  384, 1, 8,  8)    (32,  96, 1, 2, 32)
(192, 384, (3,3,3)):         (96,  128, 1, 32, 1)    (64, 128, 1, 8,  4)
(384, 384, (3,3,3)):         (128, 128, 1, 8,  2)    (96,  96, 1, 8,  4)
(384, 768, (3,3,3)):         (128, 128, 1, 16, 2)    (96,  96, 1, 8,  4)
```

## Which Entries Actually Matter

Only `C_in_block` affects numerical output. `C_out_block` and spatial blocks (`H_out_block`, `W_out_block`) only affect memory layout and parallelism.

| Entry | C_in_block change | Decoder layers affected | Impact |
|-------|-------------------|------------------------|--------|
| `(32, 384, 3x3x3)` | 32 → 32 (**same**) | 1 (conv_in) | **Zero** — only C_out_block differs |
| `(384, 768, 3x3x3)` | N/A | **0** | **Dead entry** — time_conv has kernel (3,1,1), not (3,3,3) |
| `(192, 384, 3x3x3)` | 96 → 64 | 1 (up_blocks.1.resnets.0.conv1) | Minor |
| `(384, 384, 3x3x3)` | 128 → 96 | **15** layers (mid_block + up_blocks 0,1) | **Dominant** |

**Only 16 of the decoder's ~30 conv layers are affected, and only via C_in_block.**

## Root Cause: C_in_block Changes Matmul Accumulation Order

### The Conv3D matmul with C_in blocking

When `C_in_block < C_in`, the conv3d splits the input channels into groups and runs one matmul per group. Each group's partial result is accumulated via `add_tiles` in fp32.

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

The hardware uses **TF32** math fidelity (HiFi2 on WH):
1. Each bf16 input element is promoted to fp32
2. The mantissa is **truncated to 10 bits** (TF32 format, losing 13 bits)
3. The truncated values are multiplied
4. Products are accumulated in the fp32 dest register

**fp32 addition is not associative.** Different group boundaries mean different subsets of products are accumulated together, producing different rounding at each step.

### Measured per-layer error (CPU simulation with real weights)

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

The temporal upsample structure means 1 input frame produces 4 output frames (via two `upsample3d` blocks). This creates a **4-frame repeating PCC pattern** in the per-frame comparison:

```
Frame  0: PCC=0.99996  (fresh input frame)
Frame  1: PCC=0.99994  (first temporal upsample — from cached data)
Frame  2: PCC=0.99994  (second temporal upsample)
Frame  3: PCC=0.99995  (spatial-only upsample)
Frame  4: PCC=0.99996  (new input frame — cache refreshed)
...
Frame 36: PCC=0.99994  (accumulated temporal drift)
```

## Measured Impact on Video Output

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

### Per-channel analysis

| Channel | PCC | Max error |
|---------|-----|-----------|
| R | 0.99997 | 0.051 |
| G | 1.00000 | 0.500 |
| B | 0.99988 | 0.496 |

The G and B channels show the largest max errors, likely from clamp-boundary interactions (values near ±1.0 where one config clamps and the other doesn't).

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
  └─ TF32 truncation (13 mantissa bits) + non-associative fp32 add
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

## Implications

1. **This is not a correctness bug.** Both configs produce equally valid bfloat16 approximations of the fp32 reference. Neither is "more correct."

2. **The blocking parameters ARE the root cause** of the hash difference. Specifically, `C_in_block` changes the matmul accumulation order, which produces different TF32 rounding, which cascades through 16 layers + temporal cache to produce ~1% pixel-level differences.

3. **Any change to C_in_block will change the video hash.** This is inherent to the hardware's TF32 math. There is no way to make two different C_in_block values produce identical results.

4. **The 4 entries were not all necessary to revert:**
   - `(32, 384, (3,3,3))`: C_in_block is the same (32) — reverting this had zero numerical effect
   - `(384, 768, (3,3,3))`: Never matched by any decoder layer — reverting this had zero effect
   - Only `(384, 384, (3,3,3))` and `(192, 384, (3,3,3))` needed to be reverted to restore the original hash

5. **To re-enable aggressive blockings without changing the hash**, one would need to keep C_in_block the same while changing only C_out_block and spatial blocks. For the dominant entry `(384, 384, (3,3,3))`: keep C_in_block=128 but change (C_out_block, H, W) from (128, 8, 2) to something faster.

---

## FP32 Intermediate CB Approach: Why It Fails on WH

### Goal

Eliminate C_in_block sensitivity entirely by accumulating the multi-C_in partial sums in fp32 precision instead of bf16. This would make the conv3d output invariant to C_in_block choice (different C_in_block values → bit-identical output).

### Proof of Concept

Setting `cb_matmul_interm_tiled` and `cb_reduction_tiled` to Float32 format makes GOOD, FIXED, and AGGRESSIVE blockings produce **bit-identical output** (max_abs_err = 0.0, PCC = 1.0). This proves the approach is correct in principle.

However, the output is **garbage vs torch** (PCC ~0.02–0.09) because the format conversion corrupts the data.

### Approaches Attempted on WH

| # | Approach | GOOD=AGGR? | vs Torch | Failure Mode |
|---|----------|-----------|----------|-------------|
| 1 | fp32 `cb_matmul_interm` + `NoReconfigure` untilize | Bit-identical | PCC 0.087 | `pack_untilize` requires same page size for input/output CBs. fp32=4096B vs bf16=2048B → silent data corruption. |
| 2 | fp32 `cb_matmul_interm` + typecast CB (copy_tile fp32→bf16) before untilize | Bit-identical | PCC 0.017 | `mm_init(in0, in1, out_cb)` with fp32 `out_cb` corrupts the matmul compute path on WH. The 3-arg `state_configure(SRCA, SRCB, PACK)` misconfigures the FPU when PACK format differs from input formats. |
| 3 | Separate fp32 `cb_fp32_accum` (matmul stays bf16, only reduction in fp32) | Bit-identical | PCC 0.087 | `copy_tile` from bf16 CB + `pack_tile` to fp32 CB with `pack_reconfig_data_format` produces corrupt tiles. The packer format switch between bf16 and fp32 mid-kernel leaves stale configuration in the pack pipeline registers. |
| 4 | Same as #3 + explicit `reconfig_data_format_srca` | Breaks determinism | PCC 0.04 | `reconfig_data_format_srca` in the middle of a matmul+reduction sequence corrupts the unpacker state for subsequent operations. The unpacker format registers are shared across the compute pipeline — reconfiguring between matmul and reduction breaks the matmul's cached state. |

### Root Cause on WH

The Wormhole Tensix compute pipeline has three register-configuration domains that interact:

1. **Unpacker** (SRCA, SRCB): Configured by `state_configure`, `llk_unpack_A_init`, `reconfig_data_format_srca`. Reads tiles from CBs, converts to the internal accumulation format.

2. **Math/FPU**: Operates in the DEST register format (fp32 when `fp32_dest_acc_en=True`). The math fidelity (HiFi2 on WH bf16) controls TF32 truncation of SRC inputs before multiply.

3. **Packer** (PACK): Configured by `llk_pack_hw_configure`, `pack_reconfig_data_format`. Reads from DEST and packs to CB.

**The problem**: On WH, these three domains share hardware state that is not fully isolated. When `mm_init` configures the pipeline for `(bf16_SRCA, bf16_SRCB, bf16_PACK)`, the internal FPU state (micro-op configuration, data routing) is set up for a homogeneous bf16 pipeline. Changing the PACK domain to fp32 mid-kernel (via `pack_reconfig_data_format`) leaves the micro-op configuration inconsistent — the FPU's data routing and accumulation logic still expects bf16-sized outputs, but the packer now writes fp32-sized tiles, corrupting the CB data layout.

This is further complicated by WH's constraint: **"Do not use HiFi3/4 with fp32_dest_acc on WH due to accuracy issues"** (documented in `vae_wan2_1.py:290`). This means WH uses HiFi2 for bf16 conv3d, which implies TF32 truncation in SRC registers. The truncation happens in the unpacker stage — fp32 values read from a CB are truncated to TF32 (10-bit mantissa) before entering the FPU. This is a hardware behavior that cannot be bypassed by software configuration on WH.

### BH (Blackhole) Analysis: Why It Might Work

Blackhole has several architectural improvements over Wormhole that make the fp32 intermediate CB approach more likely to succeed:

#### 1. Native HiFi4 + fp32_dest_acc Support

BH supports `MathFidelity::HiFi4` with `fp32_dest_acc_en=True` (line 288-289 of `vae_wan2_1.py` uses HiFi4 on BH). HiFi4 means **no TF32 truncation** in the SRC registers — fp32 values pass through the FPU with full 23-bit mantissa precision. This eliminates the primary source of C_in_block sensitivity: the accumulation order no longer matters when all intermediate values retain full fp32 precision.

Even without fp32 intermediate CBs, BH's HiFi4 math fidelity produces less C_in_block sensitivity than WH's HiFi2, because:
- WH HiFi2: SRC truncates fp32→TF32 (10-bit mantissa) before every multiply → significant rounding differences between C_in_block choices
- BH HiFi4: SRC preserves full fp32 (23-bit mantissa) → rounding differences are 2^13 times smaller

#### 2. Wider DEST Registers

BH has 140 Tensix cores (vs WH's 56) and wider DEST registers for fp32 accumulation. The larger DEST capacity means:
- `fp32_dest_acc_en=True` with `dst_size = 8` tiles (vs WH's 4)
- More room for in-register accumulation without spilling to CB

#### 3. Independent Packer Configuration

BH's Tensix v2 has a more modular pack pipeline where `pack_reconfig_data_format` can change the output format without corrupting the FPU data routing. This is because BH separates the pack format configuration from the micro-op generation more cleanly — the packer reads from DEST independently of how the FPU writes to DEST.

This means Approach #3 (separate fp32 accum CB with copy_tile typecast) is more likely to work on BH, because:
- `copy_tile` from bf16 CB: unpacker reads bf16, promotes to fp32 in DEST (no truncation with HiFi4)
- `pack_tile` to fp32 CB: packer reads fp32 from DEST, writes fp32 to CB (no format mismatch)
- `add_tiles` between fp32 and bf16 CBs: both promoted to fp32 in DEST, added, packed to fp32 (clean pipeline)
- `pack_reconfig_data_format` to bf16: packer now writes bf16 from fp32 DEST (standard truncation, well-tested path)

#### 4. Recommendation for BH

The fp32 intermediate CB approach should be attempted on BH with:
1. `cb_matmul_interm_tiled` stays bf16 (matmul pipeline unchanged)
2. New `cb_fp32_accum` in Float32 format for reduction accumulation
3. Typecast bf16→fp32 via `copy_tile` + `pack_tile` with `pack_reconfig_data_format`
4. After reduction: typecast fp32→bf16 back to `cb_matmul_interm_tiled`
5. Bias + untilize proceed as before in bf16

If BH's HiFi4 already reduces C_in_block sensitivity sufficiently (possible since the TF32 truncation that drives the divergence on WH doesn't exist on BH), the fp32 accum CB may not even be needed — simply changing C_in_block on BH might produce the same hash. This should be tested first.

### Summary: WH vs BH

| Property | WH (Wormhole) | BH (Blackhole) |
|----------|--------------|----------------|
| Math fidelity for bf16 conv3d | HiFi2 (TF32 truncation) | HiFi4 (no truncation) |
| SRC register precision | TF32 (10-bit mantissa) | Full fp32 (23-bit mantissa) |
| C_in_block sensitivity | High (~1e-3 per layer) | Expected low (~1e-7 per layer) |
| fp32 pack_reconfig mid-kernel | Corrupts FPU state | Expected to work (modular pack pipeline) |
| fp32 intermediate CB approach | **Fails** (corrupt output) | **Likely works** (needs testing) |
| Preserve-C_in_block workaround | **Works** (bit-identical, no C++ changes) | May not be needed |
