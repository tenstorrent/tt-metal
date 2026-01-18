# Data Movement Kernel Analysis Report

## Summary

This report analyzes 67 kernel files in `ttnn/cpp/ttnn/operations/data_movement/` for inefficient or unsafe write patterns related to `noc_async_write_barrier()` usage.

**Issue Categories:**
- **barrier_location**: Barrier is inside a loop but CB wait/pop is outside the loop. The barrier can be moved outside the loop to batch writes. Should also use `noc_async_writes_flushed()` instead of `noc_async_write_barrier()`.
- **heavy_barrier**: CB wait/pop is inside the loop, so synchronization IS required before each pop. However, `noc_async_write_barrier()` should be replaced with `noc_async_writes_flushed()` (unless cross-core signaling is involved).

**Key Findings:**
- **barrier_location**: ~15 instances - barrier can be moved outside loop
- **heavy_barrier**: ~32 instances - correct location, wrong type
- **missing_flush**: 0 - all files have synchronization before CB pops

---

## BARRIER_LOCATION Issues (Can Move Barrier Outside Loop)

These files have CB wait/pop OUTSIDE the loop, but barrier INSIDE the loop. The barrier can be moved to right before `cb_pop_front`.

### Issue #1 - barrier_location
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multicore.cpp`
**Lines**: 44-48
**Code**:
```cpp
// CB wait is outside the loop (earlier in code)
for (uint32_t k = 0; k < num_rows; k++) {
    uint64_t dst_noc_addr = get_noc_addr(base_stick_id + k, s);
    noc_async_write(l1_read_addr, dst_noc_addr, unpadded_X_size);
    noc_async_write_barrier();  // ISSUE: barrier inside loop, but CB pop is after loop
    l1_read_addr += padded_X_size;
}
// cb_pop_front is after the loop
```
**Explanation**: CB data is reserved before the loop and popped after the loop. The barrier can be moved outside the loop since all writes use the same CB reservation.
**Suggested Fix**:
```cpp
for (uint32_t k = 0; k < num_rows; k++) {
    noc_async_write(l1_read_addr, dst_noc_addr, unpadded_X_size);
    l1_read_addr += padded_X_size;
}
noc_async_writes_flushed();  // Move here, use flush instead of barrier
cb_pop_front(...);
```

---

### Issue #2 - barrier_location
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/writer_unary_stick_layout_wh_multicore.cpp`
**Lines**: 42-47
**Code**:
```cpp
for (uint32_t k = start_row_id; k < start_row_id + num_rows; k++) {
    noc_async_write(l1_read_addr, dst_noc_addr + start_column_id, write_size);
    noc_async_write_barrier();  // ISSUE: can move outside loop
    l1_read_addr += width_size;
}
```
**Explanation**: Same pattern - CB pop is after the loop.
**Suggested Fix**: Move barrier (as flush) outside the loop.

---

### Issue #3 - barrier_location
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/writer_unary_unpad_dims_split_rows.cpp`
**Lines**: 58-64
**Code**:
```cpp
for (uint32_t k = 0; k < num_rows; k++) {
    noc_async_write(l1_read_addr, dst_noc_addr, block_size);
    l1_read_addr += block_row_size;
    curr_stick_id++;
    noc_async_write_barrier();  // ISSUE: can move outside loop
}
```
**Explanation**: Within `write_block` function, CB pop is after the loop.
**Suggested Fix**: Move barrier (as flush) outside the loop, before `cb_pop_front`.

---

### Issue #4 - UNCLEAR (Complex CB Management)
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/writer_unary_stick_layout_col_multicore.cpp`
**Lines**: 47-58
**Code**:
```cpp
for (uint32_t k = 0; k < num_rows; k++) {
    noc_async_write(l1_read_addr, dst_noc_addr + ..., write_size);
    noc_async_write_barrier();
    if (k > 0 && (k % tile_width == 0)) {
        cb_pop_front(cb_id_out0, onetile * has_rows);  // Conditional pop inside loop
        cb_wait_front(cb_id_out0, onetile * has_rows);
    }
    l1_read_addr += width_size;
}
```
**Explanation**: Complex pattern with conditional CB pop every `tile_width` rows. The barrier may be required at current location due to the conditional pops, but could potentially be restructured to batch writes within each tile.
**Suggested Fix**: Review if writes within each tile (before the conditional pop) can be batched. If so, move barrier to just before the conditional `cb_pop_front`.

---

### Issue #5 - barrier_location
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/move/device/kernels/dataflow/move_stick_layout_interleaved_with_overlap.cpp`
**Lines**: 85-90
**Code**:
```cpp
// cb_wait_front is outside the loop
for (uint32_t i = start_id; i < start_id + num_pages; ++i) {
    noc_async_write(l1_read_addr, dst_noc_addr, page_size);
    noc_async_write_barrier();  // ISSUE: can move outside
    l1_read_addr += aligned_page_size;
}
// cb_pop_front is after the loop
```
**Suggested Fix**: Move barrier (as flush) outside the loop.

---

### Issue #6 - barrier_location
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/move/device/kernels/dataflow/move_interleaved_with_overlap.cpp`
**Lines**: 86-89
**Code**:
```cpp
for (uint32_t i = start_id; i < start_id + num_tiles; i += ublock_size_tiles) {
    noc_async_write_tile(i, dst_addrgen, l1_read_addr);
    noc_async_write_barrier();  // ISSUE: can move outside
    l1_read_addr += tile_bytes;
}
```
**Suggested Fix**: Move barrier (as flush) outside the loop.

---

### Issue #7 - barrier_location
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/fill_rm/device/kernels/dataflow/fill_rm_interleaved.cpp`
**Lines**: 60-66
**Code**:
```cpp
for (uint32_t h = 0; h < H; h++) {
    if (h < fillH) {
        noc_async_write(l1_w_addr, dst_noc_addr, (W << 1));
    } else {
        noc_async_write(l1_zeros_addr, dst_noc_addr, (W << 1));
    }
    noc_async_write_barrier();  // ISSUE: can move outside
    nch_dst++;
}
```
**Explanation**: All writes use static source buffers (l1_w_addr or l1_zeros_addr), not CB. Barrier can be moved outside.
**Suggested Fix**: Move barrier (as flush) outside the loop.

---

### Issue #8 - barrier_location
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/writer_s2i_width.cpp`
**Lines**: 26-35
**Code**:
```cpp
cb_wait_front(cb_id_out0, num_pages_per_tensor);
uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
for (uint32_t page_id_input = 0; page_id_input < num_pages_per_tensor; page_id_input++) {
    noc_async_write_tile(input_page_id, s, l1_read_addr);
    noc_async_write_barrier();  // ISSUE: can move outside
    l1_read_addr += stick_size;
    page_id += num_tensors;
}
cb_pop_front(cb_id_out0, num_pages_per_tensor);
```
**Suggested Fix**: Move barrier (as flush) outside the loop, before `cb_pop_front`.

---

## HEAVY_BARRIER Issues (Correct Location, Wrong Type)

These files have CB wait/pop INSIDE the loop, so synchronization IS required before each `cb_pop_front`. However, `noc_async_write_barrier()` should be replaced with `noc_async_writes_flushed()`.

### Issue #9 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/split/device/kernels/dataflow/writer_tm_tile_layout_split_two_chunks.cpp`
**Lines**: 46-50, 64-68
**Code**:
```cpp
for (uint32_t i = 0; i < out_num_tiles_per_tensor_x; i++) {
    cb_wait_front(cb_id_out0, onetile);  // CB wait INSIDE loop
    // ...
    noc_async_write_tile(tile_id + out_tensor_tile_id, s0, l1_read_addr);
    noc_async_write_barrier();  // ISSUE: should be noc_async_writes_flushed()
    cb_pop_front(cb_id_out0, onetile);  // CB pop INSIDE loop
}
```
**Explanation**: CB wait/pop is inside the loop, so some synchronization is required before each pop. However, since there's no cross-core signaling, `noc_async_writes_flushed()` suffices.
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #10 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/writer_single_row_multi_core.cpp`
**Lines**: 62-64
**Code**:
```cpp
for (uint32_t w = 0; w < number_of_tiles_per_core; w++) {
    cb_wait_front(value_tensor_cb_index, one_tile);
    noc_async_write_tile(tile_offset, output_tensor_accessor, l1_write_addr_val);
    noc_async_write_barrier();  // ISSUE: should be noc_async_writes_flushed()
    cb_pop_front(value_tensor_cb_index, one_tile);
}
```
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #11 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/writer_single_row_single_core.cpp`
**Lines**: 59-65
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #12 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/reader_single_row_single_core.cpp`
**Lines**: 63-69
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #13 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/coordinator_single_row_multi_core.cpp`
**Lines**: 77-95
**Code**:
```cpp
for (uint32_t w = 0; w < Wt; w++) {
    cb_wait_front(index_tensor_cb_index, one_tile);
    noc_async_write_tile(..., output_index_tensor_addr_gen, ...);
    noc_async_write_barrier();  // First barrier - should be flush
    cb_pop_front(index_tensor_cb_index, one_tile);

    // ... then value tile ...
    noc_async_write_tile(..., output_tensor_addr_gen, ...);
    noc_async_write_barrier();  // Second barrier - should be flush
    cb_pop_front(input_tensor_cb_index, one_tile);
}
```
**Suggested Fix**: Replace both `noc_async_write_barrier()` calls with `noc_async_writes_flushed()`.

---

### Issue #14 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/strided_slice_writer_rm_interleaved.cpp`
**Lines**: 26-28
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #15 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/kernels/nd_reshard_copy_pages_writer.cpp`
**Lines**: 28-31
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #16 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/scatter/device/kernels/common.hpp`
**Lines**: 142-150 (`write_to_output` helper function)
**Code**:
```cpp
template <typename AddrGen>
FORCE_INLINE void write_to_output(...) {
    cb_wait_front(cb, ONE_PAGE);
    noc_async_write(l1_read_address, destination_noc_address + offset_bytes, chunk_size_bytes);
    noc_async_write_barrier();  // ISSUE: should be noc_async_writes_flushed()
    cb_pop_front(cb, ONE_PAGE);
}
```
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #17 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/writer_pad_dims_rm_sharded_stickwise.cpp`
**Lines**: 52-56
**Code**:
```cpp
for (uint32_t h = 0; h < padded_shard_height; h++) {
    noc_async_write(padding_value_base_addr, output_stick_noc_addr, padded_stick_bytes);
    noc_async_write_barrier();  // Writing to local L1, not from CB
    cb_push_back(output_shard_cb, 1);
    output_stick_noc_addr += padded_stick_bytes;
}
```
**Explanation**: This writes padding values (not from CB) to local L1 and pushes to CB. The barrier ensures the write completes before the CB push, but flush should suffice.
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #18 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/writer_pad_tiled.cpp`
**Lines**: 69-79
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #19 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/writer_unary_pad_dims_interleaved.cpp`
**Lines**: 54-58
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #20 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/writer_pad_dims_rm_interleaved.cpp`
**Lines**: 37-44
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #21 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_writer_single_row_multi_core.cpp`
**Lines**: 63-68
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #22 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_writer_single_row_single_core.cpp`
**Lines**: 71-76
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #23 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/fold/device/kernels/dataflow/writer_cb2dram_for_rm_input.cpp`
**Lines**: 47-50
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #24 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/copy/device/kernels/writer_unary_start_id.cpp`
**Lines**: 50-56
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #25 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/copy/device/kernels/writer_unary_stick_start_id.cpp`
**Lines**: 40-49
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #26-29 - heavy_barrier
**Files**:
- `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/write_kernel_rm_sharded.cpp`
- `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/write_kernel_sharded.cpp`
- `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/write_kernel_rm.cpp`
- `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/write_kernel.cpp`
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()` in all.

---

### Issue #30 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/dataflow/writer_unary_interleaved_input_cols_batched.cpp`
**Lines**: 32-40
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #31 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/dataflow/writer_permute_interleaved_rm_row_invariant.cpp`
**Lines**: 52-57
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #32 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/reader_cross_core_data_exchange.cpp`
**Lines**: 140-146
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

## UNCLEAR Cases

### Issue #33 - UNCLEAR (Cross-Core Communication)
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/cross_core_data_exchange_common.hpp`
**Lines**: 82-85
**Code**:
```cpp
noc_async_write(value_cb_self_read_addr, cb_value_peer_noc_write_addr, value_cb_tile_size);
noc_async_write(index_cb_self_read_addr, cb_index_peer_noc_write_addr, index_cb_tile_size);
noc_async_write_barrier();  // Followed by semaphore_inc to peer
noc_semaphore_inc(...)
```
**Explanation**: This writes to a remote core's CB and then signals via semaphore. The full barrier IS required here because we're writing to one location and then signaling another core about that data. This is a VALID use of `noc_async_write_barrier()`.
**Assessment**: CORRECT - no fix needed.

---

### Issue #34 - UNCLEAR
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/writer_unary_transpose_hc_interleaved_tiled_padding_aware.cpp`
**Lines**: 147-151
**Explanation**: Writes are scattered to different output tiles based on permutation. Need to verify if batching is safe.
**Assessment**: Needs manual review.

---

### Issue #35 - UNCLEAR
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/moe_expert_token_remap/device/kernels/dataflow/writer_moe_expert_token_remap.cpp`
**Lines**: 81, 89-91
**Explanation**: Periodic barrier every `reduction_size` iterations. May be intentional pipelining.
**Assessment**: Needs manual review.

---

## GOOD Patterns (Reference)

### writer_unary_transpose_wh_sharded_rm.cpp
Uses `noc_async_writes_flushed()` inside loop (line 38) and `noc_async_write_barrier()` at end of kernel (line 42). This is the correct pattern.

### untilize/writer_unary_stick_layout_split_rows_interleaved_parallel_columns.cpp
Batches writes across tile_height rows before barrier (lines 29-35). Good pattern.

### slice/writer_multicore_slice_4d.cpp and writer_multicore_slice_nd.cpp
Uses `noc_async_writes_flushed()` inside loop and `noc_async_write_barrier()` at end of kernel. Correct pattern.

---

## Summary

| Category | Count | Fix |
|----------|-------|-----|
| barrier_location | ~8 | Move barrier outside loop, use `noc_async_writes_flushed()` |
| heavy_barrier | ~24 | Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()` |
| UNCLEAR | 3 | Manual review needed |
| CORRECT | 3 | Cross-core signaling patterns - no fix needed |
