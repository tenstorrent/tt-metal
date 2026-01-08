# Bias Addition Before Tilization - Implementation Plan

## Build & Test Commands

```bash
# Build
./build_metal.sh --release

# Test (with 15s timeout)
source python_env/bin/activate && \
TT_METAL_CLEAR_L1=0 \
TT_METAL_DPRINT_ONE_FILE_PER_RISC=1 \
TT_METAL_DPRINT_CORES="(0,0)" \
timeout 15 pytest tests/ttnn/nightly/unit_tests/operations/conv/test_conv2d.py::test_groups_vs_pool2 -v

# Recovery if hung
tt-smi -r 1
```

---

## Problem Statement

Currently, bias is added **after tilization**:
```
reduce → pack_untilize_dest → collect 32 sticks → tilize → bias_temp_cb
      → load from bias_temp_cb → add_bias (tiles) → activation → pack → out_cb
```

**Goal**: Move bias addition **before tilization** (before pack_untilize_dest):
```
reduce → add_bias (in DST) → activation → pack_untilize_dest → collect 32 sticks → tilize → out_cb
```

---

## Key Insight

**We can use regular `add_tiles` - no broadcast needed!**

After `reduce_tile_math()`:
- DST[0..in_ntiles_c-1] contain results
- Only the **first row (stick)** of each DST tile position has valid data
- `pack_untilize_dest` extracts just this first row

**Bias tile structure** (from `prepare_conv_bias`):
- Original bias: `[1, 1, 1, out_channels]` - single row
- Padded to: `[1, 1, 32, out_channels_padded]` - rows 1-31 are **zeros**
- Tilized: Row 0 has actual bias values, rows 1-31 are zeros

**For bias addition with regular `add_tiles`**:
```
DST[row 0] + bias_tile[row 0] = correct bias added!
DST[row 1] + bias_tile[row 1] = DST[row 1] + 0  (don't care, discarded)
...
DST[row 31] + bias_tile[row 31] = DST[row 31] + 0  (don't care, discarded)
```

**Result**: Use simple `add_tiles` instead of `add_tiles_bcast_rows`. Simpler and sufficient!

---

## Current Implementation Analysis

### Current Flow (compute_pool_2d.cpp)

```
1. reduce_tile_math() → results in DST (first row = stick)
2. pack_untilize_dest() → sticks go to pre_tilize_cb
3. Collect 32 sticks
4. fast_tilize_block() → tiles go to bias_temp_cb
5. cb_wait_front(bias_temp_cb) → load tiles back
6. add_tiles_bcast_rows() → add bias in DST
7. SFPU_OP_FUNC_ACTIVATION → activation
8. pack_tile() → pack to out_cb
```

### Proposed Flow

```
1. reduce_tile_math() → results in DST (first row = stick)
2. add_tiles() → add bias in DST (row 0 + bias, rows 1-31 + 0)
3. SFPU_OP_FUNC_ACTIVATION → activation
4. pack_untilize_dest() → extracts first row (stick) to pre_tilize_cb
5. Collect 32 sticks
6. fast_tilize_block() → tiles go directly to out_cb
```

**Benefits**:
- Eliminates bias_temp_cb entirely
- No load/pack cycle after tilization
- Simpler code path

---

## Detailed Implementation Plan

### Step 1: Modify Compute Kernel - Add Bias Before pack_untilize_dest

**File**: `compute_pool_2d.cpp`

**Location**: After `reduce_tile_math()` loop (around line 370), before activation check

**Current code** (lines 377-390):
```cpp
// Apply activation here ONLY if no bias - otherwise it's applied after bias addition (post-tilization)
#ifdef SFPU_OP_FUNC_ACTIVATION
            if constexpr (!has_bias) {
                if (last_c_block) {
                    for (uint32_t i = 0; i < partial_iter_output_tiles; ++i) {
                        SFPU_OP_FUNC_ACTIVATION
                    }
                } else {
                    for (uint32_t i = 0; i < max_tiles_per_iter; ++i) {
                        SFPU_OP_FUNC_ACTIVATION
                    }
                }
            }
#endif
```

**New code**:
```cpp
// Add bias to DST BEFORE pack_untilize_dest
// Only first row (stick) of DST has valid data
// Bias tile has values in row 0, zeros in rows 1-31
// So add_tiles gives: row0 + bias (correct!), rows1-31 + 0 (don't care)
if constexpr (has_bias) {
    // Use acc_to_dest=true: DST[dst] = cb0[i0] + cb1[i1] + DST[dst]
    // With clear_value_cb (zeros): DST[dst] = 0 + bias + DST[dst] = bias + DST
    add_tiles_init(clear_value_cb_id, bias_cb_id, /*acc_to_dest=*/true);

    // Add bias to each channel tile in DST
    for (uint32_t i = 0; i < tiles_to_reduce; ++i) {
        uint32_t bias_tile_idx = c_i * max_tiles_per_iter + i;
        uint32_t dst_idx = i;
        // DST[dst_idx] = 0 + bias_cb[bias_tile_idx] + DST[dst_idx]
        add_tiles(clear_value_cb_id, bias_cb_id, 0, bias_tile_idx, dst_idx);
    }
}

// Apply activation (now for ALL cases, bias or not)
#ifdef SFPU_OP_FUNC_ACTIVATION
    if (last_c_block) {
        for (uint32_t i = 0; i < partial_iter_output_tiles; ++i) {
            SFPU_OP_FUNC_ACTIVATION
        }
    } else {
        for (uint32_t i = 0; i < max_tiles_per_iter; ++i) {
            SFPU_OP_FUNC_ACTIVATION
        }
    }
#endif
```

**Key insight**: Use `acc_to_dest=true` with existing `clear_value_cb_id` (zeros) to achieve `DST += bias`.

### Step 2: Simplify Tilization Path

**File**: `compute_pool_2d.cpp`

**Location**: Lines 428-491 (the bias post-tilization block)

**Current code** (simplified):
```cpp
if constexpr (has_bias) {
    // Tilize into bias_temp_cb
    fast_tilize_block(pre_tilize_cb_id, in_ntiles_c, bias_temp_cb_id);

    // Wait and load back
    cb_wait_front(bias_temp_cb_id, in_ntiles_c);

    // Add bias
    add_tiles_bcast_rows(...);

    // Activation
    SFPU_OP_FUNC_ACTIVATION

    // Pack to out_cb
    pack_tile(..., out_cb_id);
} else {
    // Direct tilize to out_cb
    fast_tilize_block(pre_tilize_cb_id, in_ntiles_c, out_cb_id);
    cb_push_back(out_cb_id, in_ntiles_c);
}
```

**New code** (both paths identical):
```cpp
// With bias added pre-tilization, no special handling needed
// Tilize directly to out_cb for both cases
pack_reconfig_data_format(out_cb_id);
fast_tilize_init(pre_tilize_cb_id, in_ntiles_c, out_cb_id);
fast_tilize_block(pre_tilize_cb_id, in_ntiles_c, out_cb_id);
fast_tilize_uninit(pre_tilize_cb_id, out_cb_id);
cb_push_back(out_cb_id, in_ntiles_c);
```

### Step 3: Remove bias_temp_cb Allocation

**File**: `conv2d_op_depthwise_program_factory.cpp`

Remove the `bias_temp_cb_id` CB allocation since it's no longer needed.

Also remove passing `bias_temp_cb_id` to compute kernel compile-time args.

### Step 4: Update CB Reservations

**File**: `compute_pool_2d.cpp`

**Current** (lines 198-205):
```cpp
if (is_output_tiled && !tilize_stick_counter) {
    if constexpr (has_bias) {
        cb_reserve_back(bias_temp_cb_id, in_ntiles_c);
    } else {
        cb_reserve_back(out_cb_id, in_ntiles_c);
    }
}
```

**New**:
```cpp
if (is_output_tiled && !tilize_stick_counter) {
    // Always reserve out_cb directly (no intermediate bias_temp_cb)
    cb_reserve_back(out_cb_id, in_ntiles_c);
}
```

---

## Technical Details: add_tiles with acc_to_dest

### Why We Don't Need Broadcast

Current post-tilization code uses `add_tiles_bcast_rows`:
```cpp
// Current usage (line 453-455):
for (uint32_t i = 0; i < in_ntiles_c; ++i) {
    add_tiles_bcast_rows(bias_temp_cb_id, bias_cb_id, i, i, i);
}
```

But broadcast is only needed when you want to add a single row to ALL rows. Since we only care about row 0 of DST (the stick), regular `add_tiles` is sufficient:
- Row 0: gets bias[row 0] added (correct!)
- Rows 1-31: get bias[rows 1-31]=0 added (don't care, discarded by pack_untilize_dest)

### Solution: acc_to_dest=true

The `add_tiles_init` function has an `acc_to_dest` parameter:
```cpp
// From eltwise_binary.h:87
// | acc_to_dest | If true, operation = A + B + dst_tile_idx of add_tiles |

ALWI void add_tiles_init(uint32_t icb0, uint32_t icb1, bool acc_to_dest = false);
```

With `acc_to_dest=true`:
```cpp
add_tiles(cb0, cb1, i0, i1, dst)
// Result: DST[dst] = cb0[i0] + cb1[i1] + DST[dst]
```

### Using clear_value_cb as Zero Source

We already have `clear_value_cb_id` containing zeros! So:
```cpp
add_tiles_init(clear_value_cb_id, bias_cb_id, /*acc_to_dest=*/true);
add_tiles(clear_value_cb_id, bias_cb_id, 0, bias_tile_idx, dst_idx);
// Result: DST[dst] = 0 + bias + DST[dst] = bias + DST  ✓
```

**This is the cleanest solution - no scratch CB needed, just use existing zero CB!**

---

## Investigation Steps

Before implementing, verify:

1. **Check `add_tiles` API for DST + CB operation**
   - Can it add a CB tile to existing DST value?
   - Or does it require two CBs?

2. **Look at existing bias addition patterns**
   - How does conv_bmm_tilize.cpp handle bias?
   - Any other kernels that add to DST in-place?

3. **Verify mul_cb availability**
   - Is mul_cb free after the reduce loop?
   - Can we reuse it as scratch if needed?

4. **Bias CB lifetime**
   - cb_wait_front already called at kernel start
   - Verify bias tiles stay available throughout computation

---

## Files to Modify

| File | Changes |
|------|---------|
| `compute_pool_2d.cpp` | Move bias addition before pack_untilize_dest, remove post-tilization bias block |
| `conv2d_op_depthwise_program_factory.cpp` | Remove bias_temp_cb allocation |

---

## Implementation Checklist

- [x] Verify add_tiles_bcast_rows can work with DST as source
- [x] Add bias addition after reduce_tile_math loop
- [x] Move activation to happen for all cases (not just !has_bias)
- [x] Simplify tilization block (remove has_bias branch)
- [x] Remove bias_temp_cb CB allocation
- [x] Remove bias_temp_cb compile-time arg
- [x] Update cb_reserve_back to always use out_cb_id
- [ ] Test with bias + activation (e.g., ReLU)
- [ ] Test numerical accuracy against reference
- [ ] Test different sharding modes

---

## Expected Code Changes Summary

### compute_pool_2d.cpp

**After reduce loop (~line 370), ADD:**
```cpp
// Add bias to DST before pack_untilize_dest
// Use acc_to_dest=true with zero CB: DST[i] = 0 + bias + DST[i]
if constexpr (has_bias) {
    add_tiles_init(clear_value_cb_id, bias_cb_id, /*acc_to_dest=*/true);
    for (uint32_t i = 0; i < tiles_to_reduce; ++i) {
        uint32_t bias_tile_idx = c_i * max_tiles_per_iter + i;
        // DST[i] = clear_value(0) + bias_cb[bias_tile_idx] + DST[i]
        add_tiles(clear_value_cb_id, bias_cb_id, 0, bias_tile_idx, i);
    }
}
```

**Activation block (~line 377), CHANGE to:**
```cpp
// Activation now always happens here (bias already added above)
#ifdef SFPU_OP_FUNC_ACTIVATION
    if (last_c_block) {
        for (uint32_t i = 0; i < partial_iter_output_tiles; ++i) {
            SFPU_OP_FUNC_ACTIVATION
        }
    } else {
        for (uint32_t i = 0; i < max_tiles_per_iter; ++i) {
            SFPU_OP_FUNC_ACTIVATION
        }
    }
#endif
```

**Tilization block (~line 428-491), SIMPLIFY to:**
```cpp
// Both bias and no-bias paths are now identical
pack_reconfig_data_format(out_cb_id);
fast_tilize_init(pre_tilize_cb_id, in_ntiles_c, out_cb_id);
fast_tilize_block(pre_tilize_cb_id, in_ntiles_c, out_cb_id);
fast_tilize_uninit(pre_tilize_cb_id, out_cb_id);
cb_push_back(out_cb_id, in_ntiles_c);
```

**CB reservation (~line 198), SIMPLIFY to:**
```cpp
if (is_output_tiled && !tilize_stick_counter) {
    cb_reserve_back(out_cb_id, in_ntiles_c);
}
```

### conv2d_op_depthwise_program_factory.cpp

**Remove:**
- `bias_temp_cb_id` allocation
- `bias_temp_cb_id` compile-time arg to compute kernel
