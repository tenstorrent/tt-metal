# Pack Untilize Helper Library - Implementation Plan

## Overview
Create a unified header-only library for **pack_untilize operations**, following the same pattern as the untilize and tilize helpers. This library will consolidate pack_untilize patterns across the codebase into a single, templated function with zero runtime overhead.

**Scope:** 9 kernels using pack_untilize_block across 4 pattern groups
**Strategy:** Group-based migration for maximum efficiency

## Pattern Groups (for batch migration)

| Group | Pattern Type | Kernels | Complexity | Notes |
|-------|-------------|---------|------------|-------|
| 1 | Simple Row Assembly | 3 kernels | ⭐ Easy | Main pack_untilize kernels |
| 2 | Single Block | 3 kernels | ⭐⭐ Medium | KV cache operations |
| 3 | Multi-Block Index | 1 kernel | ⭐⭐ Medium | Paged cache with block index |
| 4 | Conditional/Mixed | 1 kernel | ⭐⭐⭐ Medium-Hard | SDPA with runtime choice |
| N/A | Incompatible | 1 kernel | ❌ Skip | Alternating output CBs |

---

## Background: Standard Untilize vs Pack Untilize

### Key Differences

| Feature | Standard Untilize | Pack Untilize |
|---------|------------------|---------------|
| Init params | `icb` only | `icb, ocb` |
| Template params | `block_ct_dim` | `block_ct_dim, full_ct_dim` |
| Block function | Row-based | Column-block assembly |
| Uninit param | `icb` | `ocb` |
| Block index | Not used | `block_c_index` for multi-block |

### When to Use Pack Untilize
- When width exceeds DEST limits (8 tiles half-sync, 4 tiles 32-bit mode)
- Need to assemble multiple block columns into a full row
- Better performance for certain configurations

---

## Pack Untilize API Analysis

### API from `pack_untilize.h`

```cpp
// Initialization
template <uint32_t block_ct_dim = 8, uint32_t full_ct_dim = block_ct_dim>
ALWI void pack_untilize_init(uint32_t icb, uint32_t ocb);

// Processing - loops over height internally
template <uint32_t block_ct_dim = 8, uint32_t full_ct_dim = block_ct_dim>
ALWI void pack_untilize_block(uint32_t icb, uint32_t block_rt_dim, uint32_t ocb, uint32_t block_c_index = 0);

// Cleanup (being deprecated per tt-metal#22904)
ALWI void pack_untilize_uninit(uint32_t ocb);
```

### Key Characteristics
- Template parameters control block dimensions at compile time
- `block_ct_dim`: Width of single block to process at once
- `full_ct_dim`: Total output width (for proper row assembly)
- `block_c_index`: Which column block is being processed (0, 1, 2, ...)
- Internal loop over `block_rt_dim` (height in tiles)
- Uninit takes **output CB**, not input CB

---

## Common Pack Untilize Patterns

### Pattern 1: Simple Row Assembly (Most Common)
```cpp
// Reserve full row, process blocks, push full row
compute_kernel_hw_startup(src_cb_id, out_cb_id);
pack_untilize_init<block_ct_dim, full_ct_dim>(src_cb_id, out_cb_id);

for (uint32_t r = 0; r < num_rows; ++r) {
    cb_reserve_back(out_cb_id, full_ct_dim);
    for (uint32_t b = 0; b < num_blocks_per_col; ++b) {
        cb_wait_front(src_cb_id, block_ct_dim);
        pack_untilize_block<block_ct_dim, full_ct_dim>(src_cb_id, 1, out_cb_id, b);
        cb_pop_front(src_cb_id, block_ct_dim);
    }
    cb_push_back(out_cb_id, full_ct_dim);
}
pack_untilize_uninit(out_cb_id);
```

**Used in:**
- `pack_untilize.cpp`
- `pack_untilize_wh.cpp`
- `pack_untilize_variable_num_blocks.cpp`

### Pattern 2: Single Block (No Assembly)
```cpp
// When Wt <= 8: block_ct_dim = full_ct_dim = Wt
pack_untilize_init<Wt>(in_cb, out_cb);

cb_wait_front(in_cb, Wt);
cb_reserve_back(out_cb, Wt);
pack_untilize_block<Wt>(in_cb, 1, out_cb);  // block_c_index defaults to 0
cb_push_back(out_cb, Wt);
cb_pop_front(in_cb, Wt);

pack_untilize_uninit(out_cb);
```

**Used in:**
- `kv_cache/update_cache.cpp` (single head iteration)
- `paged_fused_update_cache.cpp`
- `paged_row_major_fused_update_cache.cpp`

### Pattern 3: Loop with Reinit (KV Cache)
```cpp
// Init once, then reinit in loop with different CBs
pack_untilize_init<Wt>(in_cb, untilized_in_cb);

for (uint32_t h = 0; h < num_batched_heads; ++h) {
    // Process input
    cb_wait_front(in_cb, Wt);
    cb_reserve_back(untilized_in_cb, Wt);
    pack_untilize_block<Wt>(in_cb, 1, untilized_in_cb);
    cb_push_back(untilized_in_cb, Wt);
    cb_pop_front(in_cb, Wt);

    // Reinit for cache CB
    reconfig_data_format_srca(in_cb, cache_cb);
    pack_untilize_init<Wt>(cache_cb, untilized_cache_cb);

    // ... process cache ...

    pack_untilize_uninit(untilized_cache_cb);

    // Reinit for next iteration
    reconfig_data_format_srca(cache_cb, in_cb);
    pack_untilize_init<Wt>(in_cb, untilized_in_cb);
}
```

**Used in:**
- `kv_cache/update_cache.cpp`

### Pattern 4: Multi-Block with Block Index
```cpp
// Process multiple block columns into full row
pack_untilize_init<block_ct_dim, full_ct_dim>(cache_cb, untilized_cache_cb);

cb_reserve_back(untilized_cache_cb, Wt);
for (uint32_t b = 0; b < num_blocks_per_col; ++b) {
    cb_wait_front(cache_cb, block_ct_dim);
    pack_untilize_block<block_ct_dim, full_ct_dim>(cache_cb, 1, untilized_cache_cb, b);
    cb_pop_front(cache_cb, block_ct_dim);
}
cb_push_back(untilized_cache_cb, Wt);

pack_untilize_uninit(untilized_cache_cb);
```

**Used in:**
- `experimental/paged_cache/update_cache.cpp`

### Pattern 5: Conditional Selection (CANNOT FULLY ABSTRACT)
```cpp
// Runtime choice between pack_untilize and standard untilize
if constexpr (use_pack_untilize) {
    pack_untilize_init<out_chunk_tiles>(cb_in, cb_out);
    cb_wait_front(cb_in, out_chunk_tiles);
    cb_reserve_back(cb_out, out_chunk_tiles);
    pack_untilize_block<out_chunk_tiles>(cb_in, 1, cb_out);
    pack_untilize_uninit(cb_out);
    cb_pop_front(cb_in, out_chunk_tiles);
    cb_push_back(cb_out, out_chunk_tiles);
} else {
    compute_kernel_lib::untilize<true, true>(...);
}
```

**Used in:**
- `sdpa_flash_decode.cpp`

### INCOMPATIBLE: Alternating Output CBs
```cpp
// Cannot migrate - uses alternating output CBs based on block_idx
for (uint32_t block_idx = 0; block_idx < total_blocks; block_idx++) {
    const uint32_t out_cb_id = (block_idx % NUM_RISCV_DATA_MOVEMENT_CORES == 0)
                               ? out_cb_id0 : out_cb_id1;
    // ...
    pack_untilize_block<tiles_per_row>(src_cb_id, block_size, out_cb_id);
}
```

**File:** `sliding_window/halo/device/kernels/compute/pack_untilize.cpp`

---

## Proposed Helper Function

### Header File: `ttnn/cpp/ttnn/kernel_lib/pack_untilize_helpers.h`

```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/cb_api.h"

/**
 * @file pack_untilize_helpers.h
 * @brief Header-only kernel library for pack_untilize operations
 *
 * This library provides a unified function for pack_untilize operations.
 *
 * Key Features:
 * - Handles simple row assembly and single-block patterns
 * - Zero runtime overhead (all functions inlined)
 * - Template-based compile-time optimization
 * - Reduces code duplication across 8+ kernels
 *
 * IMPORTANT: Pack untilize functions require compute kernel hardware initialization.
 * You MUST call compute_kernel_hw_startup() before using any pack_untilize functions.
 *
 * Usage:
 *   #include "ttnn/cpp/ttnn/kernel_lib/pack_untilize_helpers.h"
 *
 *   // Initialize compute kernel hardware FIRST
 *   compute_kernel_hw_startup(cb_in, cb_out);
 *
 *   // Simple row assembly (multiple blocks per row)
 *   pack_untilize<block_ct_dim, full_ct_dim>(cb_in, cb_out, num_rows, num_blocks_per_col);
 *
 *   // Single block (Wt <= 8)
 *   pack_untilize<Wt, Wt>(cb_in, cb_out, num_iterations, 1);
 */

namespace compute_kernel_lib {

/**
 * @brief Unified pack_untilize function handling row assembly patterns
 *
 * This function handles:
 * - Simple row assembly (num_blocks_per_col > 1)
 * - Single block pattern (num_blocks_per_col = 1)
 * - Optional init/uninit control for loop reinit patterns
 *
 * IMPORTANT - HARDWARE INITIALIZATION REQUIREMENT:
 * Before calling this function, you MUST initialize the compute kernel hardware by
 * calling compute_kernel_hw_startup() at the start of your kernel.
 *
 * @tparam block_ct_dim Width of a single block in tiles (1 to 8)
 * @tparam full_ct_dim Width of full output row in tiles (divisible by block_ct_dim)
 * @tparam init If true, calls pack_untilize_init before processing (default: true)
 * @tparam uninit If true, calls pack_untilize_uninit after processing (default: true)
 *               Note: uninit is being deprecated per tt-metal#22904
 *
 * @param icb Input circular buffer ID (tiled data)
 * @param ocb Output circular buffer ID (row-major data)
 * @param num_rows Number of rows to process (outer loop iterations)
 * @param num_blocks_per_col Number of block columns per row (default: 1)
 *                           When > 1, assembles multiple blocks into each output row
 * @param block_rt_dim Height of each block in tiles (default: 1)
 *
 * @example
 *   // Simple row assembly (pack_untilize.cpp pattern)
 *   // block_ct_dim=2, full_ct_dim=8, 4 blocks per row
 *   pack_untilize<2, 8>(src_cb, out_cb, per_core_block_cnt, 4);
 *
 * @example
 *   // Single block pattern (kv_cache pattern)
 *   pack_untilize<Wt, Wt>(in_cb, out_cb, num_heads, 1);
 *
 * @example
 *   // Skip uninit for loop reinit patterns
 *   pack_untilize<Wt, Wt, true, false>(cache_cb, untilized_cache_cb, 1, 1);
 *   // ... other operations ...
 *   pack_untilize_uninit(untilized_cache_cb);  // Manual uninit later
 *
 * @example
 *   // Skip init when reinitializing in loop
 *   pack_untilize<Wt, Wt, false, true>(cache_cb, untilized_cache_cb, 1, 1);
 */
template <
    uint32_t block_ct_dim,
    uint32_t full_ct_dim = block_ct_dim,
    bool init = true,
    bool uninit = true>
ALWI void pack_untilize(
    uint32_t icb,
    uint32_t ocb,
    uint32_t num_rows,
    uint32_t num_blocks_per_col = 1,
    uint32_t block_rt_dim = 1)
{
    // Compile-time initialization
    if constexpr (init) {
        pack_untilize_init<block_ct_dim, full_ct_dim>(icb, ocb);
    }

    // Main processing loop
    for (uint32_t r = 0; r < num_rows; ++r) {
        cb_reserve_back(ocb, full_ct_dim);

        for (uint32_t b = 0; b < num_blocks_per_col; ++b) {
            cb_wait_front(icb, block_ct_dim);
            pack_untilize_block<block_ct_dim, full_ct_dim>(icb, block_rt_dim, ocb, b);
            cb_pop_front(icb, block_ct_dim);
        }

        cb_push_back(ocb, full_ct_dim);
    }

    // Compile-time cleanup (being deprecated)
    if constexpr (uninit) {
        pack_untilize_uninit(ocb);
    }
}

}  // namespace compute_kernel_lib
```

---

## Implementation Plan

### Phase 1: Create Infrastructure

#### Tasks:
- [ ] Create `ttnn/cpp/ttnn/kernel_lib/pack_untilize_helpers.h`
- [ ] Add header to `ttnn/CMakeLists.txt` JIT API:
  ```cmake
  FILES
      ...
      cpp/ttnn/kernel_lib/pack_untilize_helpers.h
      ...
  ```
- [ ] Implement helper function with all template parameters
- [ ] Verify compilation

---

### Phase 2: GROUP 1 - Simple Row Assembly (3 kernels)

**Kernels:**
- `pack_untilize.cpp`
- `pack_untilize_wh.cpp`
- `pack_untilize_variable_num_blocks.cpp`

**Migration Example:**
```cpp
// Before:
pack_untilize_init<block_ct_dim, full_ct_dim>(src_cb_id, out_cb_id);
for (uint32_t r = 0; r < per_core_block_cnt; ++r) {
    cb_reserve_back(out_cb_id, full_ct_dim);
    for (uint32_t b = 0; b < num_blocks_per_col; ++b) {
        cb_wait_front(src_cb_id, block_ct_dim);
        pack_untilize_block<block_ct_dim, full_ct_dim>(src_cb_id, 1, out_cb_id, b);
        cb_pop_front(src_cb_id, block_ct_dim);
    }
    cb_push_back(out_cb_id, full_ct_dim);
}
pack_untilize_uninit(out_cb_id);

// After:
compute_kernel_lib::pack_untilize<block_ct_dim, full_ct_dim>(
    src_cb_id, out_cb_id, per_core_block_cnt, num_blocks_per_col);
```

**Tests:**
```bash
pytest tests/ttnn/unit_tests/operations/test_untilize.py -v
pytest tests/ttnn/unit_tests/operations/test_untilize.py -k "wh" -v
```

**Success Criteria:**
- [ ] All 3 kernels compile without warnings
- [ ] All untilize tests pass
- [ ] No performance regression

---

### Phase 3: GROUP 2 - Single Block Pattern (3 kernels)

**Kernels:**
- `kv_cache/update_cache.cpp` (single block operations)
- `paged_fused_update_cache.cpp`
- `paged_row_major_fused_update_cache.cpp`

**Migration Example:**
```cpp
// Before:
pack_untilize_init<Wt>(in_cb, untilized_in_cb);
cb_wait_front(in_cb, Wt);
cb_reserve_back(untilized_in_cb, Wt);
pack_untilize_block<Wt>(in_cb, 1, untilized_in_cb);
cb_push_back(untilized_in_cb, Wt);
cb_pop_front(in_cb, Wt);

// After:
compute_kernel_lib::pack_untilize<Wt, Wt, true, false>(  // Skip uninit
    in_cb, untilized_in_cb, 1, 1);
```

**Note:** These kernels have complex loop structures with reinit. May need to use `init=false` or `uninit=false` templates and manual calls in some places.

**Tests:**
```bash
pytest tests/ttnn/unit_tests/operations/ -k "kv_cache" -v
pytest tests/ttnn/unit_tests/operations/ -k "paged" -v
pytest tests/ttnn/unit_tests/operations/ -k "update_cache" -v
```

**Success Criteria:**
- [ ] All 3 kernels compile without warnings
- [ ] KV cache tests pass
- [ ] Paged cache tests pass
- [ ] No performance regression

---

### Phase 4: GROUP 3 - Multi-Block Index (1 kernel)

**Kernel:**
- `experimental/paged_cache/update_cache.cpp`

**Migration:**
This kernel uses `num_blocks_per_col` with explicit block index tracking. Should map directly to the helper.

```cpp
// Before:
pack_untilize_init<block_ct_dim, full_ct_dim>(cache_cb, untilized_cache_cb);
cb_reserve_back(untilized_cache_cb, Wt);
for (uint32_t b = 0; b < num_blocks_per_col; ++b) {
    cb_wait_front(cache_cb, block_ct_dim);
    pack_untilize_block<block_ct_dim, full_ct_dim>(cache_cb, 1, untilized_cache_cb, b);
    cb_pop_front(cache_cb, block_ct_dim);
}
cb_push_back(untilized_cache_cb, Wt);
pack_untilize_uninit(untilized_cache_cb);

// After:
compute_kernel_lib::pack_untilize<block_ct_dim, full_ct_dim>(
    cache_cb, untilized_cache_cb, 1, num_blocks_per_col);
```

**Tests:**
```bash
pytest tests/ttnn/unit_tests/operations/ -k "paged_cache" -v
```

**Success Criteria:**
- [ ] Kernel compiles without warnings
- [ ] Paged cache tests pass

---

### Phase 5: GROUP 4 - Conditional/Mixed (1 kernel)

**Kernel:**
- `sdpa_flash_decode.cpp`

**Analysis:**
This kernel has a compile-time conditional between pack_untilize and standard untilize:
```cpp
if constexpr (use_pack_untilize) {
    // pack_untilize path
} else {
    compute_kernel_lib::untilize<true, true>(...);
}
```

**Migration Strategy:**
Keep the conditional structure, just replace the pack_untilize path:
```cpp
if constexpr (use_pack_untilize) {
    compute_kernel_lib::pack_untilize<out_chunk_tiles, out_chunk_tiles>(
        cb_out_accumulate_im, cb_out_final, 1, 1);
} else {
    compute_kernel_lib::untilize<true, true>(
        cb_out_accumulate_im, out_chunk_tiles, cb_out_final, out_chunk_tiles);
}
```

**Tests:**
```bash
pytest tests/ttnn/unit_tests/operations/ -k "sdpa" -v
pytest tests/ttnn/unit_tests/operations/ -k "flash" -v
```

**Success Criteria:**
- [ ] Kernel compiles without warnings
- [ ] SDPA tests pass
- [ ] Both pack_untilize and untilize paths work correctly

---

### SKIP: Incompatible Kernel

**Kernel:**
- `sliding_window/halo/device/kernels/compute/pack_untilize.cpp`

**Reason:** Uses alternating output circular buffers (`out_cb_id0` and `out_cb_id1`) based on `block_idx % NUM_RISCV_DATA_MOVEMENT_CORES`. This is incompatible with the helper's assumption of a single fixed output CB.

---

### Phase 6: Final Verification

**Quick Smoke Test:**
```bash
pytest tests/ttnn/unit_tests/operations/test_untilize.py::test_untilize_simple -v
pytest tests/ttnn/unit_tests/operations/ -k "kv_cache" --first-only -v
pytest tests/ttnn/unit_tests/operations/ -k "sdpa" --first-only -v
```

**Medium Test Suite:**
```bash
pytest tests/ttnn/unit_tests/operations/ \
    -k "untilize or kv_cache or paged or sdpa" \
    -v --tb=short
```

**Build Verification:**
```bash
./build_metal.sh
# Check for compiler warnings
```

---

## Success Criteria

- [ ] 8 of 9 pack_untilize kernels migrated successfully (1 incompatible)
- [ ] All pattern groups tested independently
- [ ] Quick smoke test passes
- [ ] Medium test suite passes
- [ ] No performance regression on benchmarks
- [ ] Code follows same patterns as `tilize_helpers.h` and `untilize_helpers.h`
- [ ] Zero compiler warnings
- [ ] Documentation complete with migration examples

---

## Migration Checklist

### GROUP 1: Simple Row Assembly (3 kernels)
- [ ] `pack_untilize.cpp`
- [ ] `pack_untilize_wh.cpp`
- [ ] `pack_untilize_variable_num_blocks.cpp`
- [ ] Tests passing

### GROUP 2: Single Block Pattern (3 kernels)
- [ ] `kv_cache/update_cache.cpp`
- [ ] `paged_fused_update_cache.cpp`
- [ ] `paged_row_major_fused_update_cache.cpp`
- [ ] Tests passing

### GROUP 3: Multi-Block Index (1 kernel)
- [ ] `experimental/paged_cache/update_cache.cpp`
- [ ] Tests passing

### GROUP 4: Conditional/Mixed (1 kernel)
- [ ] `sdpa_flash_decode.cpp`
- [ ] Tests passing

### SKIP: Incompatible
- [x] `halo/pack_untilize.cpp` - CANNOT MIGRATE (alternating output CBs)

### Final Verification
- [ ] Full regression suite
- [ ] Performance benchmarks
- [ ] Documentation
- [ ] Code review

---

## File Locations Reference

### Header File
```
ttnn/cpp/ttnn/kernel_lib/pack_untilize_helpers.h
```

### Kernel Files to Modify

**Data Movement:**
- `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize.cpp`
- `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize_wh.cpp`
- `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize_variable_num_blocks.cpp`

**KV Cache:**
- `ttnn/cpp/ttnn/operations/kv_cache/device/kernels/compute/update_cache.cpp`

**Experimental Paged Cache:**
- `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/compute/update_cache.cpp`
- `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/compute/paged_fused_update_cache.cpp`
- `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/compute/paged_row_major_fused_update_cache.cpp`

**Transformer:**
- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp`

### Build System
```
ttnn/CMakeLists.txt
```

---

## Notes

- Follow exact same style as `tilize_helpers.h` and `untilize_helpers.h` for consistency
- Keep all functions `ALWI` (always inline) for zero overhead
- Use `constexpr if` for compile-time branching
- Document the hardware initialization requirement
- Be aware that `uninit` is being deprecated (tt-metal#22904)
- Template parameters `block_ct_dim` and `full_ct_dim` are compile-time requirements

---

## References

- Tilize helpers: `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.h`
- Untilize helpers: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.h`
- Issue tracking uninit deprecation: tt-metal#22904
- API header: `tt_metal/include/compute_kernel_api/pack_untilize.h`
