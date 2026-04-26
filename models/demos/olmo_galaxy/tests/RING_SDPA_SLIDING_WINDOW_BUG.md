# Ring SDPA + Sliding Window Bug

## Summary
`ring_distributed_scaled_dot_product_attention` ignores `sliding_window_size` parameter.
The sliding window mask is never generated or applied in the RING path.

## Impact
- OLMo3 (which uses hybrid sliding window: 3 sliding + 1 full per 4 layers) cannot use ring SDPA
- Without ring SDPA, OLMo3 prefill is limited to ~32K ISL (L1 memory limit for standard SDPA)
- With ring SDPA + sliding window disabled: PCC = 0.939 at 64 layers (usable but degraded)
- With ring SDPA + sliding window enabled: PCC = 0.624 at 64 layers (broken)

## Root Cause

### 1. Compute kernel: `compute_common.hpp:1563`
The RING path does not check sliding_window_size in the `apply_mask` decision:
```cpp
if constexpr (sdpa_type == RING) {
    // BUG: no sliding_window_size check here
    apply_mask = (ring_iter_needs_global_n_mask && k_chunk == global_n_mask_chunk_id) ||
                 (local_n_needs_masking && k_chunk == local_n_mask_chunk_id) ||
                 (ring_iter_needs_joint_n_mask && (k_chunk - num_local_k_chunks) == joint_n_mask_chunk_id);
} else if constexpr (is_causal || sliding_window_size > 0) {
    // This path correctly handles sliding window, but only for STANDARD path
    apply_mask = (q_low_idx < k_high_idx) || (sliding_window_size > 0);
}
```

### 2. Writer kernel: `ring_joint_writer.cpp:189`
The ring writer hardcodes `sliding_window_size=0` in the `generate_mask` call:
```cpp
generate_mask<false, false, 0, true, cb_mask_in>(  // 0 = no sliding window
    Sq_chunk_t, Sk_chunk_t, q_chunk, 0,
    ring_iter_needs_global_n_mask || ring_iter_needs_local_n_mask,
    ring_iter_needs_joint_n_mask,
    ring_iter_needs_global_n_mask ? global_n_within_ring_iter : local_padded_N,
    L);
```
Compare with non-ring writer (`writer_interleaved.cpp:120`):
```cpp
generate_mask<is_causal, is_chunked, sliding_window_size, use_padded_mask, cb_mask_in>(...)
```

## Fix Required
1. **`ring_joint_writer.cpp`**: Pass `sliding_window_size` compile-time arg to `generate_mask` template, along with `is_causal=true` and `is_chunked=true`
2. **`compute_common.hpp`**: Add `sliding_window_size > 0` check to the RING `apply_mask` condition, integrating with the existing ring mask chunk ID system
3. **`ring_distributed_sdpa_program_factory.cpp`**: Ensure `sliding_window_size` is passed as compile-time arg to the ring writer kernel (currently only passed to non-ring writer at line 269)

## Test
`models/demos/olmo_galaxy/tests/test_olmo_ring_sdpa_64k.py::test_olmo_ring_vs_standard_sdpa_pcc`
- Compares ring vs standard SDPA at 8K ISL for 1 and 64 layers
- Currently: 1L PCC=0.968, 64L PCC=0.624 (with sliding window)
- Expected after fix: 64L PCC > 0.99

## Files
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_writer.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_distributed_sdpa_program_factory.cpp`
