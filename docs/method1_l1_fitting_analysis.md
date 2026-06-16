# Why Method 1 (ROW_MAJOR, align=8) Still Cannot Fit in L1

**Issue:** https://github.com/tenstorrent/tt-metal/issues/46831
**Hardware:** Wormhole N150 · 64 Tensix cores · L1 bank ≈ 1,400 KB per core

---

## The Question

Method 1 routes `C=3` to the regular conv path with `ROW_MAJOR` input and
`8-element channel alignment`. This reduces the input shard from `36,864 × 32 × 2B`
(TILE) down to `36,864 × 8 × 2B` (ROW_MAJOR). That shard is only **576 KB**, which
comfortably fits inside the 1,400 KB L1 bank per core.

So why does the system still fall back to DRAM slicing?

---

## The Answer in One Line

**The input shard fits in L1. The output shard does not.**

The `calculate_L1_usage_for_conv_op` function accounts for both input and output.
Even with a perfectly efficient ROW_MAJOR input (576 KB), the output shard in
**TILE format** is locked to `OC_padded = round_up(OC, TILE_WIDTH) = 32` channels,
giving **2,304 KB per core** — 1.6× the entire L1 bank. This single factor forces
DRAM slicing regardless of the input format.

---

## Exact L1 Calculation (1536×1536, 64 cores)

```
Pixels per core = H × W / num_cores = 2,359,296 / 64 = 36,864 pixels
```

### Input Shard (benefits from ROW_MAJOR)

```cpp
// get_input_channels_alignment() for ROW_MAJOR, non-mm_conv → returns 8
input_channels_alignment = 8
IC_padded = round_up(IC=3, 8) = 8
```

```
Input shard = pixels_per_core × IC_padded × 2 B
            = 36,864 × 8 × 2 B
            = 589,824 B = 576 KB
            ✅ Fits in L1 (1,400 KB)
```

ROW_MAJOR with alignment=8 correctly reduces the input from 2,304 KB (TILE, IC→32)
down to 576 KB — a 4× reduction. **The input benefit is real.**

### Output Shard (always TILE, OC_padded=32 regardless of input format)

```cpp
// From conv2d_utils.cpp line 1000 — output always rounds to TILE_WIDTH
const uint32_t output_channels_padded = tt::round_up(out_channels, tt::constants::TILE_WIDTH);
// = round_up(3, 32) = 32  ← hard-coded to TILE width, independent of input layout
```

```
Output shard = pixels_per_core × OC_padded × 2 B
             = 36,864 × 32 × 2 B
             = 2,359,296 B = 2,304 KB
             ❌ Does NOT fit in L1 (1,400 KB)
```

The output tensor is always stored in TILE format (32 × 32 tiles). For `OC=3`, every
output pixel requires 32 tile-column slots — 29 of which are zeros. The output padding
inflates per-core allocation to 2,304 KB, which alone is **1.6× the total L1 capacity**.

### Total L1 Required

```
Component                    Size        Fits?
─────────────────────────────────────────────────────
Input shard  (ROW_MAJOR)     576 KB      ✅
Output shard (TILE, OC→32) 2,304 KB      ❌  ← blocker
CB overhead  (weight+out)      ~16 KB
─────────────────────────────────────────────────────
Total needed               2,896 KB
L1 available               1,400 KB
Result:                    ❌ DRAM slicing required
```

---

## Why the Output Cannot Use ROW_MAJOR

The regular conv path produces output in TILE layout because:

1. The downstream ops (`ttnn.permute`, `ttnn.reshape`, `ttnn.to_memory_config`) all
   expect TILE layout.
2. The conv compute kernel outputs to a TILE-format CB (`TILE_WIDTH = 32` per row).
3. `output_channels_padded` is computed as `round_up(OC, TILE_WIDTH)` at
   `conv2d_utils.cpp:1000` — this applies to ALL shard layouts, regardless of whether
   the input is ROW_MAJOR or TILE.

There is no existing path in the regular conv framework that produces ROW_MAJOR output
with channel alignment=8 for the output.

---

## The Fundamental Asymmetry

| Dimension | Input | Output | Ratio |
|-----------|-------|--------|-------|
| Per-pixel bytes | `IC_padded×2 = 16 B` (ROW_MAJOR, align=8) | `OC_padded×2 = 64 B` (TILE, OC→32) | 4× |
| Per-core shard | **576 KB** ✅ | **2,304 KB** ❌ | 4× |
| vs L1 (1,400 KB) | 41% of L1 | **164% of L1** | — |

ROW_MAJOR fixes the input waste (IC=3→8 instead of 3→32), but the output has the
**identical 10.7× tile-padding problem** (OC=3→32) that the input had before Method 1.
Method 1 only solves half the problem.

---

## Code Path That Enforces This

### Step 1 — `calculate_L1_usage_for_conv_op` estimates total L1

```cpp
// conv2d_utils.cpp:959–1126
core_count_and_size calculate_L1_usage_for_conv_op(...) {
    // Input alignment: respects ROW_MAJOR → alignment=8
    const uint32_t input_channels_alignment =
        get_input_channels_alignment(shard_layout, input_layout, false, is_mm_conv, std::nullopt);
    const uint32_t in_channels_aligned = round_up(in_channels, input_channels_alignment);
    // → in_channels_aligned = round_up(3, 8) = 8  ✅ correct for ROW_MAJOR

    // Output alignment: ALWAYS TILE_WIDTH, ignores input layout
    const uint32_t output_channels_padded = round_up(out_channels, TILE_WIDTH);
    // → output_channels_padded = round_up(3, 32) = 32  ❌ same 10.7× waste as baseline

    // total_size = CB allocation + tensor allocation + halo output size
    // "tensor allocation" includes the output shard in TILE format → 2,304 KB
    return { .total_size = l1_usage.CB_allocation_size
                         + l1_usage.tensor_allocation_size    // ← includes output TILE shard
                         + precise_input_size_per_core * input_datum_size };
}
```

### Step 2 — `determine_conv_config_for_auto_shard` picks HEIGHT_SHARDED

`calculate_L1_usage_for_conv_op` returns `total_size ≈ 2,896 KB`.
This exceeds L1 (1,400 KB), so `determine_conv_config_for_auto_shard` keeps trying
BLOCK_SHARDED and WIDTH_SHARDED. For `OC=3` neither helps — there are not enough output
channels to split meaningfully. All configurations report `total_size > L1`.

### Step 3 — `determine_conv2d_execution_path` returns DRAM

Because no auto-shard config fits in L1, and the input is DRAM INTERLEAVED (not L1),
`determine_conv2d_execution_path` always returns `Conv2dExecutionPath::DRAM`:

```cpp
Conv2dExecutionPath determine_conv2d_execution_path(
    bool input_is_in_L1,
    const std::optional<const Conv2dSliceConfig>& slice_config) {
    if (slice_config && slice_config->slice_type == L1_FULL)   return L1;
    if (!slice_config && input_is_in_L1)                       return L1;
    return DRAM;  // ← always taken: input is DRAM, no slice_config passed
}
```

### Step 4 — `conv2d_DRAM` with `mm_conv=false` runs the sliced path

Inside `conv2d_DRAM`, with the Method 1 guard making `mm_conv=false`:

```cpp
if (mm_conv) {
    return conv2d_L1(...);   // bypasses slicing (only for matmul path)
}
// mm_conv=false → always runs slicing
run_sliced_op(input_tensor_on_device, output_tensors, &slice_attr, dram_slice_config_);
```

The sliced path creates 6–8 slices, each with the full
`PaddedSlice + Halo + Move + Conv2d + SliceWrite` overhead sequence.

---

## Summary of the L1 Fitting Problem

```
Input  (Method 1, ROW_MAJOR):  IC=3 → IC_padded=8  →  576 KB  ✅ fits
Output (always TILE):          OC=3 → OC_padded=32 → 2304 KB  ❌ does not fit

The OUTPUT shard is the blocker, not the input.
ROW_MAJOR alignment=8 only fixes the input half of the tile-padding problem.
The output half (OC=3→32, same 10.7× inflation) remains and prevents L1 fitting.
```

---

## What Would Fix It

To allow a single-pass L1 HEIGHT_SHARDED conv for `OC=3`:

| Fix | How | Impact |
|-----|-----|--------|
| **Output ROW_MAJOR with align=8** | Allow output CB to use `OC_padded=8` (not 32) | Output shard drops from 2,304 KB to 576 KB → both fit! |
| **Reduce per-core spatial** | Not possible — already at 64 cores (max) | — |
| **Split OC across cores (BLOCK_SHARDED)** | `OC_padded/8_cores = 4 channels/core` | Output shard → 288 KB → fits, but BLOCK_SHARDED DRAM slicing creates 48× more slices |
| **Use Method 2 (spatial packing)** | `C×K=96` fills output tiles completely | Matmul path, no slicing, 10.4× speedup ✅ |

The cleanest fix is **Method 2**: by packing `K=32` pixels together so `OC×K=96`
channels fill output tiles completely (96 = 3 × TILE_WIDTH), the output shard becomes
`(36,864/32) × 96 × 2B = 220 KB` — well within L1. And since it stays on the matmul
path, there is zero per-slice overhead.
