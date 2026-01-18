# Data Movement Kernel Analysis: Other Operations

Analysis of data movement kernels in:
- ttnn/cpp/ttnn/operations/normalization/
- ttnn/cpp/ttnn/operations/eltwise/
- ttnn/cpp/ttnn/operations/conv/
- ttnn/cpp/ttnn/operations/pool/
- ttnn/cpp/ttnn/operations/kv_cache/
- ttnn/cpp/ttnn/operations/experimental/paged_cache/
- ttnn/cpp/ttnn/operations/experimental/ssm/
- ttnn/cpp/ttnn/operations/experimental/reduction/

## Summary

| Category | Count | Description |
|----------|-------|-------------|
| heavy_barrier | 12 | Correct location, wrong type - should use `noc_async_writes_flushed()` |
| barrier_location | 3 | Can move barrier outside loop AND use flush |
| CORRECT | 3 | Good patterns with flush in loop, barrier at end |

**Note on categorization:**
- **heavy_barrier**: CB wait/pop is INSIDE the loop. Synchronization IS required before each `cb_pop_front`, but should use `noc_async_writes_flushed()`.
- **barrier_location**: CB wait/pop is OUTSIDE the inner loop. Barrier can be moved outside AND should use flush.

---

## HEAVY_BARRIER Issues (Correct Location, Wrong Type)

### Issue #1 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/welford_writer_unary_gn_rm_gb.cpp`
**Lines**: 206-214
**Code**:
```cpp
for (uint32_t mt = 0; mt < out_block_h_actual; mt++) {
    for (uint32_t nt = 0; nt < per_core_N; ++nt) {
        cb_wait_front(cb_out, 1);  // CB wait INSIDE innermost loop
        const uint32_t l1_read_addr = get_read_ptr(cb_out);
        noc_async_write_tile(..., dst_a, l1_read_addr);
        noc_async_write_barrier();  // ISSUE: should be noc_async_writes_flushed()
        cb_pop_front(cb_out, 1);  // CB pop INSIDE innermost loop
    }
    mt_offset += num_channels_tiles;
}
```
**Explanation**: CB wait/pop is inside the innermost loop. Each pop requires synchronization, but `noc_async_writes_flushed()` suffices since there's no cross-core signaling.
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #2 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/writer_batch_norm.cpp`
**Lines**: 106-113
**Code**:
```cpp
for (uint32_t t = start_t; t < HtWt && num_tiles_written < num_tiles; ++t, ++num_tiles_written) {
    cb_wait_front(cb_id_dst, onetile);  // CB wait INSIDE loop
    uint32_t l1_read_addr = get_read_ptr(cb_id_dst);
    noc_async_write_tile(start_tile_id + num_tiles_written, dst, l1_read_addr);
    noc_async_write_barrier();  // ISSUE: should be flush
    cb_pop_front(cb_id_dst, onetile);  // CB pop INSIDE loop
}
```
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #3 - heavy_barrier (3 barriers per iteration!)
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/writer_running_statistics.cpp`
**Lines**: 65-114
**Code**:
```cpp
for (uint32_t t = start_t; t < HtWt && num_tiles_written < num_tiles; ++t, ++num_tiles_written) {
    if constexpr (old_running_mean_has_value) {
        cb_wait_front(...);
        noc_async_write_tile(...);
        noc_async_write_barrier();  // BARRIER #1
        cb_pop_front(...);
    }
    if constexpr (old_running_var_has_value) {
        cb_wait_front(...);
        noc_async_write_tile(...);
        noc_async_write_barrier();  // BARRIER #2
        cb_pop_front(...);
    }
    cb_wait_front(cb_id_dst, onetile);
    noc_async_write_tile(...);
    noc_async_write_barrier();  // BARRIER #3
    cb_pop_front(cb_id_dst, onetile);
}
```
**Explanation**: Up to 3 barriers per loop iteration (for mean, var, and dst writes). This is extremely inefficient.
**Suggested Fix**: Replace all `noc_async_write_barrier()` calls with `noc_async_writes_flushed()`.

---

### Issue #4 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
**Lines**: 32-39
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #5 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_wh.cpp`
**Lines**: 31-45
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #6 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/writer_interleaved_scalar.cpp`
**Lines**: 62-68
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #7 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast.cpp`
**Lines**: 57-67
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #8 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/writer_grid_sample_interleaved.cpp`
**Lines**: 25-45
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #9 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/writer_unary_interleaved_start_id_blocked.cpp`
**Lines**: 26-36
**Code**:
```cpp
for (uint32_t i = 0; i < num_tiles; i += blk) {
    cb_wait_front(cb_out, blk);  // CB wait INSIDE loop
    uint32_t l1_read_addr = get_read_ptr(cb_out);
    for (uint32_t j = 0; j < blk; j++) {
        noc_async_write_tile(tile_id, s, l1_read_addr);
        tile_id++;
        l1_read_addr += tile_bytes;
    }
    noc_async_write_barrier();  // ISSUE: should be flush
    cb_pop_front(cb_out, blk);  // CB pop INSIDE loop
}
```
**Note**: This is a better pattern - batches `blk` tiles before barrier. Just needs to use flush.
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #10 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/experimental/ssm/repeat_and_interleave_eltwise_mul/device/kernels/writer_ssm_eltwise_mul.cpp`
**Lines**: 22-31
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #11 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/device/kernels/writer_reduce_nc.cpp`
**Lines**: 30-41
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

## BARRIER_LOCATION Issues (Can Move Barrier Outside Loop)

### Issue #12 - barrier_location
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/writer_unary_gn_rm_gb.cpp`
**Lines**: 231-248
**Code**:
```cpp
for (uint32_t mt = 0; mt < out_block_h_actual; mt++) {
    for (uint32_t nt = 0; nt < block_w_curr; nt++) {
        noc_async_write_tile(...);  // Writes in inner loops
        l1_read_addr += single_tile_size_bytes;
    }
}
noc_async_write_barrier();  // Barrier OUTSIDE inner loops - GOOD location
cb_pop_front(cb_out, out_block_hw_normal);  // Pop OUTSIDE inner loops
```
**Explanation**: This is actually a better pattern - barrier is outside inner loops. However, this whole block is inside outer loops, so the barrier should use `noc_async_writes_flushed()`.
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #13 - barrier_location
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/writer_upsample_interleaved.cpp`
**Lines**: 58-69
**Code**:
```cpp
for (uint32_t b = start_block_id; b < end_block_id; b++) {
    cb_wait_front(cb_id_out0, num_tiles_per_block_row);  // Waits for batch of tiles
    // ... multiple writes in nested loops ...
    noc_async_write_barrier();  // Barrier per block - GOOD batching
    cb_pop_front(cb_id_out0, num_tiles_per_block_row);
}
```
**Note**: This is already a good pattern - batches many writes before barrier. Just needs to use flush.
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

## CORRECT Patterns (Reference - No Fix Needed)

### Pattern #14 - CORRECT (KV Cache)
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/kv_cache/device/kernels/dataflow/writer_update_cache_interleaved_start_id.cpp`
**Lines**: 56-77
**Code**:
```cpp
for (uint32_t g = 0; g < granularity; ++g) {
    cb_wait_front(cache_cb_id, Wt);
    for (uint32_t curr_cache_id = cache_id; curr_cache_id < cache_id + Wt; ++curr_cache_id) {
        noc_async_write_tile(curr_cache_id, s0, out_l1_read_addr);
        out_l1_read_addr += cache_tile_bytes;
    }
    noc_async_writes_flushed();  // CORRECT: flush inside loop
    cb_pop_front(cache_cb_id, Wt);
}
noc_async_write_barrier();  // CORRECT: barrier at end
```
**Assessment**: This is the OPTIMAL pattern - `noc_async_writes_flushed()` inside loop for CB management, `noc_async_write_barrier()` at kernel end.

---

### Pattern #15 - CORRECT (Paged Cache)
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/writer_update_cache_interleaved_start_id.cpp`
**Lines**: 101-116
**Assessment**: Same optimal pattern as KV cache - flush in loop, barrier at end.

---

### Pattern #16 - CORRECT (Deepseek)
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_grouped_gate/device/kernels/dataflow/writer_deepseek_grouped_gate.cpp`
**Lines**: 486-493
**Code**:
```cpp
for (uint32_t height_tile = start_height_tile; height_tile < end_height_tile; height_tile++) {
    noc_async_write_page(...);
    noc_async_write_page(...);
    noc_async_writes_flushed();  // CORRECT: flush inside loop
    cb_pop_front(...);
    cb_pop_front(...);
}
noc_async_write_barrier();  // CORRECT: barrier at end
```
**Assessment**: Good pattern - flush in loop, barrier at end.

---

## Summary

| Category | Count | Fix |
|----------|-------|-----|
| heavy_barrier | 11 | Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()` |
| barrier_location | 2 | Already batched, use `noc_async_writes_flushed()` |
| CORRECT | 3 | Optimal pattern - no fix needed |

**High Priority Fixes:**
1. **writer_running_statistics.cpp** - 3 barriers per loop iteration
2. **welford_writer_unary_gn_rm_gb.cpp** - barrier in innermost of nested loops
3. **writer_batch_norm.cpp** - single tile per barrier
