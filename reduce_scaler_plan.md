# Plan: Kernel-side scaler computation for reduce helper library

## Goal
Move scaler computation from host to device dataflow kernel. The kernel receives `PoolType`, `ReduceDim`, `dim_size`, and an optional `user_scalar`, then auto-computes the final scaler tile.

## Approach: Option C - Combine on the dataflow side

### 1. New dataflow API in `reduce_helpers_dataflow.hpp`

Add a new overload of `generate_reduce_scaler` templated on `PoolType`:

```cpp
template <PoolType pool_type, bool half_tile = false>
FORCE_INLINE void generate_reduce_scaler(
    const uint32_t cb_id,
    const uint32_t dim_size,       // number of elements in reduction dimension
    const uint32_t user_scaler     // float reinterpreted as uint32 (default: 1.0f = 0x3F800000)
);
```

Implementation logic:
- `AVG`: `final = user_scaler_float * (1.0f / dim_size)`
- `SUM/MAX/MIN`: `final = user_scaler_float`
- Convert final float to packed bf16, call existing `generate_reduce_scaler_impl`

Keep the old `generate_reduce_scaler(cb_id, packed_scaler)` for backward compatibility.

### 2. Host program factory changes

Files to modify:
- `reduce_op_multi_core_w_program_factory.cpp`
- `reduce_op_multi_core_h_program_factory.cpp`
- `reduce_op_single_core_hw_program_factory.cpp`

Change compile-time args from `packed_scaler_value` to:
```cpp
uint32_t dim_size = <compute from tensor shape and reduce dim>;
uint32_t user_scalar_bits = std::bit_cast<uint32_t>(operation_attributes.scaler);
std::vector<uint32_t> reader_compile_time_args = { dim_size, user_scalar_bits };
```

Remove the host-side `bfloat16::truncate` + `pack_two_bfloat16_into_uint32` scaler computation.

### 3. Reader kernel changes

File: `reader_unary_reduce_universal_start_id.cpp` (and similar readers)

Change from:
```cpp
constexpr uint32_t scaler = get_compile_time_arg_val(0);
dataflow_kernel_lib::generate_reduce_scaler(cb_id_in2, scaler);
```

To:
```cpp
constexpr uint32_t dim_size = get_compile_time_arg_val(X);
constexpr uint32_t user_scaler = get_compile_time_arg_val(Y);
dataflow_kernel_lib::generate_reduce_scaler<POOL_TYPE>(cb_id_in2, dim_size, user_scaler);
```

Where `POOL_TYPE` comes from a define (already set in compute defines, may need to propagate to reader).

### 4. Python/C++ API - no change needed

The existing API already has `float scalar = 1.0f` parameter. The only change is that `generic_reductions.cpp` no longer computes `scalar / reduced_volume` for Mean - it just passes through `scalar` and `ReduceOpMath` and lets the kernel handle the division.

Alternatively, keep `generic_reductions.cpp` logic as-is for now and only change the kernel-lib level API. Callers of the helper library (new ops using the kernel lib directly) get the new clean API.

### 5. Key design decisions

- Pass `user_scalar` as raw float bits (uint32), NOT pre-packed bf16. This allows float*float multiply on device before single bf16 truncation.
- Float division `1.0f / dim_size` happens once per kernel launch on RISC-V dataflow processor - negligible cost.
- Old overload `generate_reduce_scaler(cb_id, packed_scaler)` kept for backward compat.
- `dim_size` is the actual element count (e.g., `Wt * TILE_WIDTH` for row reduce), not tile count.

### Files involved

| File | Change |
|------|--------|
| `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` | New templated overload |
| `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.inl` | Implementation |
| `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_multi_core_w_program_factory.cpp` | Pass dim_size + user_scalar instead of packed_scaler |
| `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_multi_core_h_program_factory.cpp` | Same |
| `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_single_core_hw_program_factory.cpp` | Same |
| `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reader_unary_reduce_universal_start_id.cpp` | Use new API |
| `ttnn/cpp/ttnn/operations/reduction/generic/generic_reductions.cpp` | Optional: stop computing scalar/reduced_volume on host |
