# Task: Adding Missing Compute API Variants (Tier 1 Completion)

## Overview

This task completes the **Tier 1 Compute API** by adding missing API variants that conform to the Compute API Contract. The goal is to provide a complete, consistent API surface where function names clearly convey their scope and behavior.

**Reference**: [GitHub Issue #35739](https://github.com/tenstorrent/tt-metal/issues/35739)

**Current Scope**: Implement `*_dest`, `*_block`, and `pack_*_block` variants
**Out of Scope**: `*_tensor` variants (L1→L1 pipeline) - documented here for future reference only

## Compute API Contract

The Compute API Contract defines four variant types by suffix, each with well-defined characteristics:

### 1. `*_dest` (DEST → DEST)
- **Threads**: MATH only
- **Capacity**: Single tile
- **Pattern**: Operations on data already in DEST registers
- **DEST Sync**: N/A (manual management by caller)
- **SFPU Fusible**: Yes
- **Example**: `reduce_dest<PoolType, ReduceDim>(tile_idx)`

### 2. `*_block` (L1 → DEST)
- **Threads**: UNPACK + MATH
- **Capacity**: Up to DEST bank (multiple tiles)
- **Pattern**: Unpack from L1 CB → Math → Result stays in DEST
- **DEST Sync**: Manual (caller manages)
- **Packing**: No (result stays in DEST for further operations)
- **SFPU Fusible**: Yes
- **Template Args**: Block dimensions (Ht, Wt) as compile-time parameters
- **Example**: `reduce_block<PoolType, ReduceDim, Ht, Wt>(cb_in, ...)`

### 3. `pack_*_block` (DEST → L1)
- **Threads**: PACK only
- **Capacity**: Up to DEST bank
- **Pattern**: Pack from DEST → L1 CB
- **DEST Sync**: Manual
- **Example**: `pack_reduce_block<ReduceDim, Ht, Wt>(cb_out, ...)`

### 4. `*_tensor` (L1 → L1)
- **Threads**: UNPACK + MATH + PACK (full pipeline)
- **Capacity**: Unlimited (processes entire tensors)
- **Pattern**: Complete L1 → Unpack → Math → Pack → L1 pipeline
- **DEST Sync**: Automatic (handled internally)
- **Packing**: Automatic (handled internally)
- **SFPU Fusible**: No (already includes packing)
- **Special Cases**:
  - Reduce operations need `llk_math_pack_reduce` before final pack
  - Some ops need generic unpacking without math (e.g., untilize)
- **Example**: `reduce_tensor<PoolType, ReduceDim>(cb_in, cb_out, num_tiles, ...)`

## Current State

**Tier 1 APIs** (conform to contract): ~211 functions
- ~200 SFPU functions (natively conformant)
- ~11 non-SFPU functions

**Missing for Tier 1**: Various `*_dest`, `*_block`, and `pack_*_block` variants

**This task**: Add missing `*_dest`, `*_block`, and `pack_*_block` variants only
**Future work**: `*_tensor` variants (explained below but not implemented in this task)

## Operations to Add

### ✅ Completed Block Variants
1. **Element-wise Binary** (`eltwise_binary.h`)
   - `add_block<Ht, Wt>(...)` ✅
   - `sub_block<Ht, Wt>(...)` ✅
   - `mul_block<Ht, Wt>(...)` ✅

2. **Reduce** (`reduce_custom.h`)
   - `reduce_block<PoolType, ReduceDim, Ht, Wt>(...)` ✅

3. **Pack** (`pack.h`)
   - `pack_block<Ht, Wt>(...)` ✅

### 🔄 TODO: Remaining Block Variants
4. **Broadcast Operations** (`bcast.h`)
   - `add_tiles_bcast_block<BroadcastType, Ht, Wt>(...)` 🔄
   - `sub_tiles_bcast_block<BroadcastType, Ht, Wt>(...)` 🔄
   - `mul_tiles_bcast_block<BroadcastType, Ht, Wt>(...)` 🔄

5. **Transpose** (`transpose_wh.h`)
   - `transpose_wh_block<Ht, Wt>(...)` 🔄

### ✅ Already Exists
6. **Copy** (`tile_move_copy.h`)
   - `copy_block_matmul_partials(...)` ✅

## Architecture & Layer Separation

### Software Stack Layers
```
┌─────────────────────────────────┐
│      Compute API (our work)     │ ← Multi-threaded, hides programming model
├─────────────────────────────────┤
│          llk_api                │ ← Interacts with tt-metal (CBs, threads)
├─────────────────────────────────┤
│          llk_lib                │ ← Tensix hardware operations (LLKs)
└─────────────────────────────────┘
```

**Key Rule**: Only call existing llk_api/llk_lib primitives. Don't implement new low-level functions.

### TRISC Threading Model (Hidden by Compute API)
- **UNPACK Thread**: Unpacks data from L1 circular buffers
- **MATH Thread**: Performs mathematical operations on DEST registers
- **PACK Thread**: Packs results from DEST back to L1 circular buffers

Compute API coordinates these threads internally—users don't need to manage them.

## Implementation Approach

### **CRITICAL: Block Variants are For-Loops, Not New Operations**

**Key Principle**: Block variants (`*_block`) are **NOT** new operations. They are simply for-loops that call existing single-tile operations multiple times.

**DO NOT**:
- ❌ Create new `*_block_init()` functions
- ❌ Add new LLK calls specific to blocks
- ❌ Implement new hardware operations
- ❌ Add operation-specific block logic

**DO**:
- ✅ Use existing `*_tiles()` or `*_tile()` functions inside a for-loop
- ✅ Use existing init functions (call once before the loop)
- ✅ Reuse all existing single-tile infrastructure
- ✅ Add only the for-loop iteration logic with template parameters

### In Scope: DEST and Block Variants (`*_dest`, `*_block`, `pack_*_block`)

#### `*_block` Functions (L1 → DEST) - For-Loop Pattern
```cpp
/**
 * WORK IN PROGRESS - Use with caution
 *
 * L1 → DEST: Processes a block of Ht x Wt tiles by calling existing tile operations.
 * This is a for-loop wrapper around existing single-tile functions.
 * Result stays in DEST for SFPU fusion or further operations.
 * Conforms to Compute API Contract.
 *
 * @tparam Ht Block height in tiles (compile-time)
 * @tparam Wt Block width in tiles (compile-time)
 */
template<uint32_t Ht, uint32_t Wt>
ALWAYSINLINE void add_block(uint32_t cb_a, uint32_t cb_b,
                             uint32_t tile_a_start, uint32_t tile_b_start,
                             uint32_t dst_start) {
    static_assert(Ht * Wt <= 16, "Block exceeds DEST capacity");

    // Simple for-loop over existing add_tiles function
    for (uint32_t h = 0; h < Ht; h++) {
        for (uint32_t w = 0; w < Wt; w++) {
            uint32_t offset = h * Wt + w;
            add_tiles(cb_a, cb_b,
                     tile_a_start + offset, tile_b_start + offset,
                     dst_start + offset);
        }
    }
}
```

#### `pack_*_block` Functions (DEST → L1) - For-Loop Pattern
```cpp
/**
 * WORK IN PROGRESS - Use with caution
 *
 * DEST → L1: Packs a block by calling pack_tile multiple times.
 * This is a for-loop wrapper around existing pack_tile function.
 * Companion to *_block functions. Conforms to Compute API Contract.
 */
template<uint32_t Ht, uint32_t Wt>
ALWAYSINLINE void pack_block(uint32_t dst_start, uint32_t cb_out) {
    static_assert(Ht * Wt <= 16, "Block exceeds DEST capacity");

    // Simple for-loop over existing pack_tile function
    for (uint32_t h = 0; h < Ht; h++) {
        for (uint32_t w = 0; w < Wt; w++) {
            uint32_t offset = h * Wt + w;
            pack_tile(dst_start + offset, cb_out);
        }
    }
}
```

### Out of Scope: Tensor Variants (`*_tensor`) - Future Work

**Note**: Tensor variants are documented here for completeness and future reference, but are **NOT to be implemented** in this task per [issue #35739](https://github.com/tenstorrent/tt-metal/issues/35739).

#### Pattern (for reference only)
```cpp
/**
 * L1 → L1: Complete pipeline for tensor operations.
 * Handles DEST sync and packing automatically.
 * Conforms to Compute API Contract.
 *
 * NOT IMPLEMENTED IN THIS TASK - Future work
 */
template<PoolType pool_type, ReduceDim reduce_dim>
ALWAYSINLINE void reduce_tensor(uint32_t cb_in, uint32_t cb_out, uint32_t num_tiles) {
    // Would use: llk_unpack_* + llk_math_reduce +
    //            llk_math_pack_reduce + llk_pack
    // Would handle automatic DEST sync management
}
```

#### Special Considerations (for future implementation)
- **Reduce ops**: Would require `llk_math_pack_reduce` before `llk_pack`
- **Untilize ops**: May need generic unpack without math operation
- **Transpose ops**: May need special DEST manipulation

## Implementation Requirements

### Core Constraints
1. **Do NOT modify existing APIs** - Only add new conforming variants
2. **Use only existing llk primitives** - Call llk_api/llk_lib, don't implement new low-level code
3. **Architecture parity** - Implement for both Blackhole and Wormhole B0 simultaneously
4. **Template parameters** - Use compile-time args for block sizes, operation params where suitable
5. **DEST capacity** - Enforce with `static_assert(Ht * Wt <= MAX_DEST_TILES, "...")`
6. **WIP marking** - All new functions clearly marked "WORK IN PROGRESS - Use with caution"
7. **Comprehensive guards** - Both `static_assert` (compile-time) and `ASSERT` (runtime)
8. **Follow existing patterns** - Match code style, naming, structure in codebase

### Per-Function Checklist
- [ ] Blackhole implementation (`tt_metal/hw/ckernels/blackhole/metal/llk_api/`)
- [ ] Wormhole B0 implementation (`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/`)
- [ ] Compute API header (`tt_metal/include/compute_kernel_api/`)
- [ ] Template parameters for compile-time optimization
- [ ] `static_assert` guards for DEST capacity and template params
- [ ] Runtime `ASSERT` guards where needed
- [ ] WIP documentation with contract conformance noted
- [ ] Doc comments: data flow, parameters, constraints, example
- [ ] Test implementation (see below)

## Testing

**Location**: `tests/tt_eager/ops/` or `tests/tt_metal/tt_metal/unit_tests/`

**Coverage**: Basic functionality, edge cases (min/max block sizes), data types (fp32/fp16/bfp8), both architectures, DEST capacity boundaries

**Validation**: Compare block variants against multiple tile/dest ops, use PCC (Pearson Correlation Coefficient) for numerical accuracy

**Test only in-scope variants**: `*_dest`, `*_block`, and `pack_*_block` (no tensor tests)

```python
@pytest.mark.parametrize("Ht,Wt", [(1,1), (2,4), (4,8)])
def test_reduce_block(device, Ht, Wt):
    # Test *_block variant against reference
    pass
```

## File Locations

### Compute API Headers (Add public-facing APIs)
`tt_metal/include/compute_kernel_api/*.h`
- `reduce_custom.h`, `eltwise_binary.h`, `tilize.h`, etc.

### Architecture-Specific Implementation
- **Blackhole**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/*.h`
- **Wormhole B0**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/*.h`

### LLK Submodule (if low-level changes needed - rare)
- **Blackhole**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/*.h`
- **Wormhole B0**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/*.h`
- **Note**: Submodule changes require special workflow (see CLAUDE.md)

## Learning from Existing Code

**Study these patterns before implementing:**
1. Existing `*_dest` functions (single-tile DEST operations)
2. Existing `*_block` functions (if any) for block patterns
3. How existing APIs call llk primitives (llk_unpack_*, llk_math_*, llk_pack)
4. Template parameter usage for compile-time optimization
5. DEST sync management patterns

## Success Criteria

- [ ] Missing `*_dest`, `*_block`, and `pack_*_block` variants implemented
- [ ] All functions conform to Compute API Contract
- [ ] Both Blackhole and Wormhole B0 supported
- [ ] Comprehensive guards (static_assert + ASSERT)
- [ ] WIP documentation on all new functions
- [ ] Tests for all new variants
- [ ] All tests passing
- [ ] Pre-commit hooks pass (clang-format, etc.)
- [ ] `*_tensor` variants NOT implemented (out of scope)

## Development Workflow

1. **Build**: `cd /localdev/ncvetkovic/reconfig/tt-metal && ./build_metal.sh`
2. **Test**: `source python_env/bin/activate && pytest <test_path> -v`
3. **Commit**: Use `#35739: Description` format
4. **Branch**: `ncvetkovic/35739_add_missing_functions` (main repo and submodule)

## Key Design Notes

- **DST_ACCUM_MODE**: Compile-time macro for FP32 dest accumulation mode
- **Circular Buffers (CB)**: L1 memory interface for data transfer
- **DEST tiles**: Limited capacity, varies by architecture
- **Pre-commit hooks**: Auto-format (clang-format for C++, black for Python)
