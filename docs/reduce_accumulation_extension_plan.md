# Reduce Function Accumulation Extension Plan

## Executive Summary

This document outlines a plan to extend the existing `compute_kernel_lib::reduce()` function to support accumulation patterns, unblocking migration of 19 kernel files (12 Moreh + 4 Moreh Softmax partial + 3 non-Moreh).

**Goal:** Extend existing `reduce()` function, not create a new one.

**Critical Design Requirements:**
1. **Zero overhead when disabled** - Accumulation support must be compile-time optional with no runtime cost when not used
2. **Proper init sequencing** - After `copy_tile_init`, `reduce_init_short` must be called to restore SRCA configuration

---

## 1. Problem Analysis

### The Blocked Pattern

All 19 blocked kernels share a common pattern: **accumulator reload** between `tile_regs_acquire()` and `reduce_tile()`:

```cpp
// Current pattern that library cannot support:
tile_regs_acquire();
if (enable_reload) {
    copy_tile_init(cb_accumulator);
    copy_tile(cb_accumulator, 0, dst0);  // Load accumulator to DST[0]
}
reduce_init_short(...);  // CRITICAL: Must re-init after copy_tile_init!
reduce_tile(..., dst0);  // Reduces INTO existing DST[0] value
tile_regs_commit();
tile_regs_wait();
pack_tile(dst0, cb_accumulator);  // Store result back
tile_regs_release();
```

### Why Re-init is Required

The `copy_tile_init()` function reconfigures the SRCA unpacker for the copy operation. After the copy is complete, `reduce_init_short()` (or equivalent) must be called to reconfigure SRCA for the reduce operation. Without this, the reduce_tile will operate with incorrect unpacker settings, causing incorrect results or hangs.

### Real-World Examples

**moreh_dot.cpp** (lines 40-66):
- Iterates over blocks, accumulating into DST[0]
- First iteration: no reload
- Subsequent iterations: reload from `cb_25`, reduce, pack back to `cb_25`
- Final iteration: pack to output CB instead

**layernorm_large_tensor.cpp** (lines 142-179):
- Iterates over blocks computing variance
- First block: no reload
- Subsequent blocks: reload from `cb_accumulate`, reduce multiple tiles, pack back
- Final block: pack to `cb_ex2` (final variance output)

**moreh_softmax_h.cpp** (lines 53-69):
- Two-phase MAX reduction
- Phase 1: Reduce Ht-1 tiles (using library - already migrated)
- Phase 2: Load accumulated max, reduce one more tile (blocked)

---

## 2. Proposed Solution

### Design Principle
Extend the existing `reduce()` function with a **compile-time optional** accumulation feature that has **zero overhead when disabled** (default behavior).

### Zero Overhead Design

**Key insight:** Use **type-based dispatch** instead of a template `bool` parameter. When the default `NoAccumulation` type is used, all accumulation code is eliminated at compile-time via `if constexpr`. When `Accumulate` type is passed, accumulation is enabled.

```cpp
// When AccumT = NoAccumulation (default):
// - is_accumulate_v<AccumT> is false
// - All accumulation code eliminated via if constexpr
// - No copy_tile_init, copy_tile, or reduce_init_short calls generated
// - Identical codegen to non-accumulation reduce()

// When AccumT = Accumulate:
// - is_accumulate_v<AccumT> is true
// - Accumulation code compiled in
// - iteration index controls reload behavior (skip on iteration==0)
```

### New API Components

```cpp
namespace compute_kernel_lib {

/**
 * @brief Tag type indicating no accumulation (zero overhead)
 *
 * When this type is passed to reduce(), all accumulation code is
 * eliminated at compile-time via `if constexpr`.
 */
struct NoAccumulation {};

/**
 * @brief Configuration for accumulation-style reductions
 *
 * Holds the static configuration for accumulation (CB and DST index).
 * Does not hold iteration state - that's provided via Accumulate wrapper.
 */
struct AccumulationConfig {
    uint32_t cb_accumulator = 0;  // CB for accumulator
    uint32_t dst_index = 0;       // DST register for accumulation (default: 0)

    static constexpr AccumulationConfig with_cb(uint32_t cb, uint32_t dst = 0) { return {cb, dst}; }
};

/**
 * @brief Accumulation wrapper that carries config + iteration index
 *
 * This type enables type-based dispatch in reduce():
 * - When Accumulate is passed: accumulation code is compiled in
 * - When NoAccumulation (default): accumulation code is eliminated
 *
 * The iteration index determines reload behavior:
 * - iteration == 0: skip reload (first call, no accumulated value yet)
 * - iteration > 0: reload from accumulator CB before reducing
 */
struct Accumulate {
    AccumulationConfig config;
    uint32_t iteration = 0;

    constexpr Accumulate(AccumulationConfig cfg, uint32_t iter = 0) : config(cfg), iteration(iter) {}
    constexpr Accumulate(uint32_t cb, uint32_t iter = 0) : config{cb, 0}, iteration(iter) {}

    // Factory for concise call sites
    static constexpr Accumulate at(uint32_t cb, uint32_t iter, uint32_t dst = 0) {
        return Accumulate(AccumulationConfig{cb, dst}, iter);
    }

    // Convenience: check if this is first iteration (skip reload)
    constexpr bool is_first() const { return iteration == 0; }
};

}  // namespace compute_kernel_lib
```

### Extended `reduce()` Signature

```cpp
template <
    PoolType reduce_type,
    ReduceDim reduce_dim,
    ReduceInputMode input_mode = ReduceInputMode::STREAMING,
    ReduceDataFormatReconfig reconfig = ReduceDataFormatReconfig::BOTH,
    bool init = true,
    bool uninit = true,
    typename AccumT = NoAccumulation,  // Type-based dispatch (zero overhead when NoAccumulation)
    typename PostReduceOp = NoOp>
ALWI void reduce(
    uint32_t icb,
    uint32_t icb_scaler,
    uint32_t ocb,
    TileShape shape,
    TileLayout layout = {},
    AccumT accum = {},            // Accumulate enables accumulation; NoAccumulation disables
    PostReduceOp post_reduce_op = {});
```

---

## 3. Implementation Details

### Type Trait for Dispatch

```cpp
// Type trait to detect if AccumT is Accumulate (enables accumulation code)
template <typename T>
struct is_accumulate : std::false_type {};

template <>
struct is_accumulate<Accumulate> : std::true_type {};

template <typename T>
inline constexpr bool is_accumulate_v = is_accumulate<T>::value;
```

### Core Logic Changes

Inside `reduce()`, the accumulation logic uses type-based dispatch with `if constexpr`:

```cpp
// Compile-time flag derived from type
constexpr bool enable_accumulation = is_accumulate_v<AccumT>;

tile_regs_acquire();

// Accumulator reload logic - ONLY compiled when AccumT is Accumulate
reload_accumulator_if_needed<reduce_type, reduce_dim, AccumT, enforce_fp32_accumulation>(
    icb, icb_scaler, accum);

// Existing reduce_tile logic...
const uint32_t dst_idx = get_dst_index(accum);  // Returns 0 for NoAccumulation
for (...) {
    reduce_tile<reduce_type, reduce_dim, enforce_fp32_accumulation>(
        icb, icb_scaler, tile_idx, 0, dst_idx);
}
```

### Helper Functions

```cpp
// Safely extract dst_index (returns 0 for NoAccumulation)
template <typename AccumT>
ALWI constexpr uint32_t get_dst_index(const AccumT& accum) {
    if constexpr (is_accumulate_v<AccumT>) {
        return accum.config.dst_index;
    } else {
        return 0;
    }
}

// Reload accumulator if needed (zero overhead when NoAccumulation)
template <PoolType reduce_type, ReduceDim reduce_dim, typename AccumT, bool enforce_fp32_accumulation>
ALWI void reload_accumulator_if_needed(uint32_t icb, uint32_t icb_scaler, const AccumT& accum) {
    if constexpr (is_accumulate_v<AccumT>) {
        if (!accum.is_first()) {  // Reload on all iterations except first
            cb_wait_front(accum.config.cb_accumulator, 1);
            copy_tile_to_dst_init_short_with_dt(icb, accum.config.cb_accumulator);
            copy_tile(accum.config.cb_accumulator, 0, accum.config.dst_index);
            cb_pop_front(accum.config.cb_accumulator, 1);

            // CRITICAL: Re-init reduce after copy_tile corrupts SRCA config
            reduce_init_short_with_dt<reduce_type, reduce_dim, enforce_fp32_accumulation>(
                accum.config.cb_accumulator, icb, icb_scaler);
        }
    }
}
```

### Init Sequencing Detail

The sequence of init calls is critical:

```cpp
// Initial setup (once, outside loop or with init=true on first call)
reduce_init<reduce_type, reduce_dim, fp32>(icb, icb_scaler, ocb);

// Inside each iteration:
tile_regs_acquire();

// If reloading accumulator (iteration > 0):
if (!accum.is_first()) {
    copy_tile_init_with_dt(cb_accumulator);  // Corrupts SRCA!
    copy_tile(cb_accumulator, 0, dst_index);
    reduce_init_short<reduce_type, reduce_dim>(icb, icb_scaler);  // Restore SRCA for reduce
}

// Now reduce can proceed with correct SRCA config
reduce_tile<...>(...);
```

**Why `reduce_init_short` instead of full `reduce_init`?**
- `reduce_init` configures both SRCA (unpacker) and packer
- `reduce_init_short` only reconfigures SRCA (unpacker) - faster, sufficient here
- The packer config from initial `reduce_init` remains valid

### Iteration State Management

**Iteration index passed to Accumulate (no mutable state):**
```cpp
// Pass iteration index directly - no state mutation needed
for (uint32_t block = 0; block < num_blocks; ++block) {
    reduce<SUM, REDUCE_ROW>(
        ..., {},
        Accumulate::at(cb_accum, block));  // iteration controls reload
}
```

| block | iteration | `is_first()` | Reload? |
|-------|-----------|--------------|---------|
| 0     | 0         | true         | **No**  |
| 1     | 1         | false        | **Yes** |
| 2     | 2         | false        | **Yes** |

### Output CB Handling

The caller controls output CB per iteration:
- **Intermediate iterations:** Pass `ocb = cb_accumulator`
- **Final iteration:** Pass `ocb = cb_output`

This is simpler than adding a flag - the library always packs to `ocb`, caller decides where.

---

## 4. Migration Examples

### Before: moreh_dot.cpp

```cpp
for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
    bool last_out = block == (per_core_block_cnt - 1);

    // elemwise-mul (unchanged)
    ...

    // reduce-w
    ACQ();
    if (enable_reload) {
        cb_wait_front(cb_25, onetile);
        copy_tile_to_dst_init_short(cb_25);
        copy_tile(cb_25, 0, 0);
        cb_pop_front(cb_25, onetile);
    }
    cb_wait_front(cb_24, onetile);
    reduce_init<REDUCE_OP, REDUCE_DIM>(...);
    reduce_tile<REDUCE_OP, REDUCE_DIM>(...);
    cb_pop_front(cb_24, onetile);
    reduce_uninit();

    pack_tile(0, last_out ? cb_16 : cb_25);
    REL();
    enable_reload = true;
}
```

### After: moreh_dot.cpp

```cpp
for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
    bool is_last = (block == per_core_block_cnt - 1);

    // elemwise-mul (unchanged)
    ...

    // reduce-w with accumulation
    // Type-based dispatch: Accumulate type enables accumulation code paths
    // iteration index (block) controls reload: skip on first (block==0), reload after
    compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM,
        compute_kernel_lib::ReduceInputMode::STREAMING,
        compute_kernel_lib::ReduceDataFormatReconfig::NONE>(
        cb_24, cb_2, is_last ? cb_16 : cb_25,
        compute_kernel_lib::TileShape::single(),
        {},
        compute_kernel_lib::Accumulate::at(cb_25, block));
}
```

**Key improvements:**
- No mutable state (`enable_reload` variable eliminated)
- No explicit `enable_accumulation=true` template parameter
- Iteration index passed directly to `Accumulate::at()`
- Output CB selection inlined

### Before: layernorm_large_tensor.cpp (lines 142-179)

```cpp
for (auto block : generic::blocks(Wt, blk)) {
    tile_regs_acquire();
    if (!block.is_first()) {
        cb_wait_front(cb_accumulate, onetile);
        copy_tile_init(cb_accumulate);
        copy_tile(cb_accumulate, 0, dst0);
        cb_pop_front(cb_accumulate, onetile);
    }
    cb_wait_front(cb_xmm2, block.full_block_size());

    reduce_init<SUM, REDUCE_ROW, FP32>(...);
    for (auto i : block.local()) {
        reduce_tile<SUM, REDUCE_ROW, FP32>(..., dst0);
    }
    cb_pop_front(cb_xmm2, ...);

    const auto pack_cb = final_iter ? cb_ex2 : cb_accumulate;
    // ... scaling for final ...
    reduce_uninit();
    pack_tile(dst0, pack_cb);
    ...
}
```

### After: layernorm_large_tensor.cpp

```cpp
uint32_t block_idx = 0;
for (auto block : generic::blocks(Wt, blk)) {
    const bool is_final = (block.last() == Wt);
    auto output_cb = is_final ? cb_ex2 : cb_accumulate;

    // Note: Need PostReduceOp for final iteration's scaling
    auto post_op = is_final
        ? [W](uint32_t) {
            binop_with_scalar_tile_init();
            mul_unary_tile(0, generic::bit_cast<uint32_t>(1.0f / W));
          }
        : compute_kernel_lib::NoOp{};

    compute_kernel_lib::reduce<SUM, REDUCE_ROW,
        compute_kernel_lib::ReduceInputMode::STREAMING_BATCHED,
        compute_kernel_lib::ReduceDataFormatReconfig::BOTH>(
        cb_xmm2, cb_scaler, output_cb,
        compute_kernel_lib::TileShape::row(block.full_block_size()),
        {},
        compute_kernel_lib::Accumulate::at(cb_accumulate, block_idx++),
        post_op);
}
```

---

## 5. Implementation Phases

### Phase 1: Core Implementation
1. Add `AccumulationConfig` struct to `reduce_helpers.hpp`
2. Extend `reduce()` function signature with optional `accum` parameter
3. Implement accumulator reload logic for `REDUCE_ROW` and `REDUCE_SCALAR`
4. Add unit tests for new accumulation mode

### Phase 2: Migrate Simple Cases
1. **moreh_dot.cpp** - Single-tile accumulation pattern
2. **moreh_mean_h.cpp** - Row reduction with accumulation
3. **moreh_sum_h.cpp** - Same pattern as mean

### Phase 3: Migrate Moreh Layer Norm
1. **moreh_layer_norm_large.cpp**
2. **moreh_layer_norm_backward** (3 files)

### Phase 4: Migrate Complex Cases
1. **layernorm_large_tensor.cpp** - With post-reduce scaling
2. **moreh_norm** (3 files)
3. **moreh_bias_backward** (2 files)

### Phase 5: Complete Partial Migrations
1. **moreh_softmax_{h,w,h_large,w_large}.cpp** - MAX reduction with single reload
2. **layernorm_sharded.cpp** - Complete partial migration
3. **groupnorm_sharded_v2.cpp** - Complete partial migration

---

## 6. Testing Strategy

### Unit Tests
- Test `Accumulate::at(cb, 0)` skips reload (first iteration)
- Test `Accumulate::at(cb, n)` where n > 0 performs reload
- Test multi-iteration accumulation produces correct results
- Test final iteration outputs to correct CB
- Test `NoAccumulation` (default) has identical codegen to non-accumulation

### Integration Tests
- Run existing moreh tests after migration
- Run layernorm tests with large tensors
- Verify numerical correctness vs original implementations

### Performance Tests
- Ensure no regression in non-accumulation cases (verify zero overhead)
- Measure overhead of accumulation path

---

## 7. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| API complexity increases | Factory methods (`Accumulate::at()`) hide details; `NoAccumulation` default maintains backward compatibility |
| Iteration state bugs | Iteration index passed explicitly - no mutable state, no hidden magic |
| Data format reconfig conflicts | Document reconfig behavior with accumulation; test combinations |
| PostReduceOp interaction | Test PostReduceOp with accumulation; document DST register usage |
| Non-zero overhead when disabled | Type-based dispatch with `if constexpr`; verify codegen with objdump |
| SRCA corruption after copy_tile | Always call reduce_init_short after copy_tile in accumulation path |

### Zero Overhead Verification

To verify zero overhead when `AccumT = NoAccumulation` (default):

```bash
# Compare generated assembly for both cases
# 1. Compile kernel without Accumulate parameter (uses NoAccumulation default)
# 2. Compile kernel with explicit NoAccumulation{}
# 3. objdump -d and diff - should be identical
```

The `if constexpr (is_accumulate_v<AccumT>)` ensures the compiler completely eliminates:
- The `cb_wait_front` for accumulator
- The `copy_tile_init_with_dt` call
- The `copy_tile` call
- The `cb_pop_front` for accumulator
- The `reduce_init_short` re-init call

**Type-based dispatch benefits:**
- No explicit template `bool` parameter needed
- `AccumT` deduced from argument type
- Zero overhead guaranteed by `if constexpr` on type trait

---

## 8. Success Metrics

- **15 fully blocked files** become migratable
- **7 partially migrated files** can be fully migrated
- **0 regressions** in existing migrated kernels
- **API remains backward-compatible** - existing code unchanged

---

## 9. Appendix: Affected Files

### Blocked by Accumulation (15)

**Moreh (12):**
- `moreh_norm/moreh_norm_w_kernel.cpp`
- `moreh_norm/moreh_norm_ord_other_h_kernel.cpp`
- `moreh_norm/moreh_norm_ord_other_w_kernel.cpp`
- `moreh_layer_norm_backward/moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp`
- `moreh_layer_norm_backward/moreh_layer_norm_backward_input_grad_large_kernel.cpp`
- `moreh_layer_norm_backward/moreh_layer_norm_backward_input_grad_small_kernel.cpp`
- `moreh_layer_norm/moreh_layer_norm_large_kernel.cpp`
- `moreh_bias_backward/moreh_bias_backward_single_core_hw_kernel.cpp`
- `moreh_bias_backward/moreh_bias_backward_multi_core_h_kernel.cpp`
- `moreh_dot/moreh_dot.cpp`
- `moreh_mean/moreh_mean_h_kernel.cpp`
- `moreh_sum/moreh_sum_h_kernel.cpp`

**Non-Moreh (3):**
- `layernorm/layernorm_large_tensor.cpp`
- `layernorm/layernorm_sharded_post_allgather.cpp`
- `layernorm_distributed/layernorm_post_allgather.cpp`

### Partially Migrated (can be completed)

**Moreh Softmax (4):** MAX with single reload
- `moreh_softmax_h.cpp`
- `moreh_softmax_w.cpp`
- `moreh_softmax_h_large.cpp`
- `moreh_softmax_w_large.cpp`

**Layernorm (3):**
- `groupnorm_sharded_v2.cpp`
- `layernorm_sharded.cpp`
- `layernorm_sharded_pre_allgather.cpp`
