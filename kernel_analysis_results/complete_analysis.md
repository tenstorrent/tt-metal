# Complete NOC Write Barrier Pattern Analysis

## Executive Summary

Analysis of ~250 data movement kernels in the tt-metal repository for inefficient or unsafe NOC/fabric write patterns.

### Issue Categories

| Category | Description | Fix |
|----------|-------------|-----|
| **heavy_barrier** | CB wait/pop is INSIDE loop. Synchronization IS required before `cb_pop_front`, but `noc_async_write_barrier()` is heavier than needed. | Replace with `noc_async_writes_flushed()` |
| **barrier_location** | CB wait/pop is OUTSIDE the loop but barrier is INSIDE. Barrier can be moved outside. | Move barrier outside loop, use `noc_async_writes_flushed()` |
| **missing_flush** (UNSAFE) | Missing synchronization before `cb_pop_front` after writes from CB. | Add `noc_async_writes_flushed()` before pop |
| **CORRECT** | Valid patterns - cross-core signaling, ring algorithms, mcast sync. | No fix needed |

### Overall Statistics

| Category | Count |
|----------|------:|
| heavy_barrier | ~95 |
| barrier_location | ~15 |
| missing_flush (UNSAFE) | 1 |
| CORRECT (no fix needed) | ~15 |
| UNCLEAR (needs manual review) | ~5 |

---

## Key Understanding

### When is `noc_async_write_barrier()` required?

1. **At the end of a kernel** - ensures all writes complete before kernel terminates
2. **Before signaling another core about data** - when writing to location A, then signaling core B about data at A

### When should `noc_async_writes_flushed()` be used instead?

**Every other case**, including:
- Before `cb_pop_front` to ensure CB memory isn't reused before write completes
- Inside loops where multiple writes occur sequentially

### The Optimal Pattern

```cpp
for (uint32_t i = 0; i < iterations; i++) {
    cb_wait_front(cb, batch_size);
    for (uint32_t j = 0; j < batch_size; j++) {
        noc_async_write_tile(...);
    }
    noc_async_writes_flushed();  // Lightweight - ensures writes are issued
    cb_pop_front(cb, batch_size);
}
noc_async_write_barrier();  // Full barrier only at kernel end
```

---

## Good Reference Implementations

These files demonstrate optimal patterns:

1. **KV Cache**: `kv_cache/device/kernels/dataflow/writer_update_cache_interleaved_start_id.cpp`
   - Uses `noc_async_writes_flushed()` in loop, `noc_async_write_barrier()` at end

2. **Paged Cache**: `experimental/paged_cache/device/kernels/dataflow/writer_update_cache_interleaved_start_id.cpp`
   - Same optimal pattern

3. **Deepseek**: `experimental/reduction/deepseek_grouped_gate/device/kernels/dataflow/writer_deepseek_grouped_gate.cpp`
   - Batches writes, flush in loop, barrier at end

4. **RMS Pre-Allgather**: `fused_distributed_rmsnorm/device/kernels/dataflow/rms_pre_allgather_writer.cpp`
   - Flush in loop, barrier at end

5. **Slice Writers**: `slice/writer_multicore_slice_4d.cpp`, `writer_multicore_slice_nd.cpp`
   - Flush in loop, barrier at end

---

## Recommendations

### Immediate Actions

1. **Fix UNSAFE issue**: `root_receive_writer_kernel.cpp` - add missing flushes

2. **Fix high-impact files**: Focus on files with multiple barriers per iteration:
   - `writer_running_statistics.cpp` (3 barriers)
   - `writer_moreh_adamw.cpp` (3-4 barriers)
   - `writer_moreh_adam.cpp` (3-4 barriers)

3. **Simple find-and-replace** for most files: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()` (except at kernel end or before cross-core signaling)

### Verification Checklist

Before changing any barrier to flush, verify:
- [ ] No cross-core signaling follows the barrier
- [ ] Not at kernel end (keep barrier at end)
- [ ] Not followed by reading from same location by another core
- [ ] The pattern is write → flush → cb_pop_front (not write → cb_pop_front → flush)

---

# Section 1: CCL/Fabric Operations

## Summary

| Category | Count |
|----------|-------|
| INEFFICIENT - heavy_barrier | 4 |
| UNSAFE - missing_flush | 1 |
| CORRECT - valid pattern | 2 |

---

## INEFFICIENT Patterns

### Issue CCL-1 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_async/device/kernels/local_copy_writer.cpp`
**Lines**: 44-47
**Code**:
```cpp
for (uint32_t iter = 0; iter < num_sticks_to_read; ++iter) {
    cb_wait_front(cb_output_id, 1);
    uint32_t l1_read_addr = get_read_ptr(cb_output_id);
    uint64_t dst_noc_addr = get_noc_addr(dst_stick_id, dst_accessor);
    noc_async_write(l1_read_addr, dst_noc_addr, stick_size);
    dst_stick_id++;
    noc_async_write_barrier();  // Line 46 - ISSUE: should be noc_async_writes_flushed()
    cb_pop_front(cb_output_id, 1);
}
```
**Explanation**: The barrier location is correct (it must be before `cb_pop_front` to ensure the write from CB memory completes before the CB slot is freed). However, `noc_async_write_barrier()` is heavier than necessary. Since this is writing to DRAM/memory without any subsequent cross-core signaling about that data, `noc_async_writes_flushed()` would suffice.

**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`:
```cpp
    noc_async_writes_flushed();  // Lighter weight - just ensures write is issued
    cb_pop_front(cb_output_id, 1);
```

---

### Issue CCL-2 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_async/device/kernels/minimal_default_writer.cpp`
**Lines**: 117-129, 141-151, 166-191
**Code**:
```cpp
for (uint32_t iter = 0; iter < num_sticks_to_read; ++iter) {
    cb_wait_front(cb_output_id, 1);
    uint32_t l1_read_addr = get_read_ptr(cb_output_id);
    for (uint32_t pad_id = 0; pad_id < padding; pad_id++) {
        uint64_t dst_noc_addr = get_noc_addr(dst_stick_id + pad_id * num_sticks_per_halo_dim, dst_accessor);
        noc_async_write(l1_read_addr, dst_noc_addr, stick_size);
    }
    dst_stick_id++;
    noc_async_write_barrier();  // Lines 128, 149, 190 - ISSUE: should be noc_async_writes_flushed()
    cb_pop_front(cb_output_id, 1);
}
```
**Explanation**: Same pattern as Issue CCL-1. The barrier location is correct (before `cb_pop_front`), but `noc_async_write_barrier()` is heavier than necessary. The writes are to output memory with no cross-core signaling, so `noc_async_writes_flushed()` is sufficient.

**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()` at all three locations (lines 128, 149, 190).

---

### Issue CCL-3 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/dim_zero_ring_reduce_scatter_minimal_async_writer.cpp`
**Lines**: 362-373
**Code**:
```cpp
while (tiles_read < tiles_to_read) {
    // ...
    cb_wait_front(cb_output_id, tile_granularity);
    size_t l1_read_addr = get_read_ptr(cb_output_id);
    for (uint32_t j = 0; j < tiles_to_read_in_current_direction; ++j) {
        uint32_t output_tile_id = output_tile_id_start + tiles_read;
        uint64_t local_noc_addr = get_noc_addr(output_tile_id, output_addrgen);
        noc_async_write(l1_read_addr, local_noc_addr, page_size);
        l1_read_addr += page_size;
        tiles_read++;
    }
    noc_async_write_barrier();  // Line 372 - ISSUE: should be noc_async_writes_flushed()
    cb_pop_front(cb_output_id, tile_granularity);
    // ...
}
```
**Explanation**: The barrier location is correct (before `cb_pop_front`). This is in the final slice processing path when writing to the output buffer. Since it's writing to output memory without subsequent cross-core signaling about that specific data, `noc_async_writes_flushed()` should suffice.

**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue CCL-4 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/ring_reduce_scatter_minimal_async_writer.cpp`
**Lines**: 412-423
**Code**:
```cpp
while (tiles_read < tiles_to_read) {
    // ...
    cb_wait_front(cb_output_id, tile_granularity);
    size_t l1_read_addr = get_read_ptr(cb_output_id);
    for (uint32_t j = 0; j < tiles_to_read_in_current_direction; ++j) {
        uint32_t output_tile_id = output_tile_id_start + tiles_read;
        uint64_t local_noc_addr = get_noc_addr(output_tile_id, output_addrgen);
        noc_async_write(l1_read_addr, local_noc_addr, page_size);
        l1_read_addr += page_size;
        tiles_read++;
    }
    noc_async_write_barrier();  // Line 422 - ISSUE: should be noc_async_writes_flushed()
    cb_pop_front(cb_output_id, tile_granularity);
    // ...
}
```
**Explanation**: Same pattern as Issue CCL-3. Writing to output buffer without cross-core signaling.

**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

## CORRECT Patterns (No Fix Needed)

### Pattern CCL-5 - CORRECT
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/line_reduce_scatter_minimal_async_writer.cpp`
**Lines**: 407-408 (within `do_final_reduction` block)
**Code**:
```cpp
while (tiles_read < tiles_to_read) {
    // ...
    cb_wait_front(cb_compute_output_id, tile_granularity);
    size_t l1_read_addr = get_read_ptr(cb_compute_output_id);
    for (uint32_t j = 0; j < num_pages_to_read; ++j) {
        uint32_t output_tile_id = output_tile_id_start + tiles_read;
        uint64_t local_noc_addr = get_noc_addr(output_tile_id, output_addrgen);
        noc_async_write(l1_read_addr, local_noc_addr, page_size);
        l1_read_addr += page_size;
        tiles_read++;
    }
    if (detail::do_forward_sync(is_forward)) {
        noc_async_write_barrier();  // Full barrier when signaling
    } else {
        noc_async_writes_flushed();  // Flush otherwise
    }
    cb_pop_front(cb_compute_output_id, tile_granularity);
    // ...
}
```
**Assessment**: This is a CORRECT pattern. When `do_forward_sync` is true, the full barrier is used because the kernel needs to signal the opposite direction worker about the written data. When not syncing, it correctly uses the lighter `noc_async_writes_flushed()`. This conditional approach is optimal.

---

### Pattern CCL-6 - CORRECT
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/dim_zero_line_reduce_scatter_minimal_async_writer.cpp`
**Lines**: 374-375 (within `do_final_reduction` block)
**Assessment**: Same as Pattern CCL-5 - correctly uses conditional barrier/flush based on synchronization needs.

---

## UNSAFE Patterns

### Issue CCL-7 - UNSAFE (missing_flush)
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/ccl/reduce_to_root/device/kernels/root_receive_writer_kernel.cpp`
**Lines**: 23-39
**Code**:
```cpp
inline void write_data(...) {
    cb_wait_front(cb_int_cb_l, input_num_tiles);
    uint32_t l1_read_addr = get_read_ptr(cb_int_cb_l);
    uint64_t dst_noc_addr = get_noc_addr(core_noc_x, core_noc_y, dst_addr_l, 0);
    noc_async_write(l1_read_addr, dst_noc_addr, input_num_tiles * page_bytes);
    cb_pop_front(cb_int_cb_l, input_num_tiles);  // UNSAFE: pop BEFORE any flush/barrier

    cb_wait_front(cb_int_cb_s, onetile);
    l1_read_addr = get_read_ptr(cb_int_cb_s);
    dst_noc_addr = get_noc_addr(core_noc_x, core_noc_y, dst_addr_s, 0);
    noc_async_write(l1_read_addr, dst_noc_addr, onetile * page_bytes);
    cb_pop_front(cb_int_cb_s, onetile);  // UNSAFE: pop BEFORE any flush/barrier

    cb_wait_front(cb_int_cb_m, onetile);
    l1_read_addr = get_read_ptr(cb_int_cb_m);
    dst_noc_addr = get_noc_addr(core_noc_x, core_noc_y, dst_addr_m, 0);
    noc_async_write(l1_read_addr, dst_noc_addr, onetile * page_bytes);
    noc_async_write_barrier();
    cb_pop_front(cb_int_cb_m, onetile);
}
```
**Explanation**: The first two `cb_pop_front` calls (for cb_int_cb_l and cb_int_cb_s) happen BEFORE any flush or barrier. This means the CB memory could be overwritten by a producer before the async writes complete. The single barrier at line 38 only protects the third write.

**Suggested Fix**: Add `noc_async_writes_flushed()` before each `cb_pop_front`:
```cpp
    noc_async_write(l1_read_addr, dst_noc_addr, input_num_tiles * page_bytes);
    noc_async_writes_flushed();  // ADD THIS
    cb_pop_front(cb_int_cb_l, input_num_tiles);

    // ... second write ...
    noc_async_write(l1_read_addr, dst_noc_addr, onetile * page_bytes);
    noc_async_writes_flushed();  // ADD THIS
    cb_pop_front(cb_int_cb_s, onetile);

    // ... third write (already has barrier, could change to flush) ...
```

---

# Section 2: Data Movement Operations

## Summary

| Category | Count |
|----------|-------|
| barrier_location | ~8 |
| heavy_barrier | ~24 |
| UNCLEAR | 3 |
| CORRECT | 3 |

---

## BARRIER_LOCATION Issues (Can Move Barrier Outside Loop)

These files have CB wait/pop OUTSIDE the loop, but barrier INSIDE the loop. The barrier can be moved to right before `cb_pop_front`.

### Issue DM-1 - barrier_location
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

### Issue DM-2 - barrier_location
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

### Issue DM-3 - barrier_location
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

### Issue DM-4 - UNCLEAR (Complex CB Management)
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

### Issue DM-5 - barrier_location
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

### Issue DM-6 - barrier_location
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

### Issue DM-7 - barrier_location
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

### Issue DM-8 - barrier_location
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

### Issue DM-9 - heavy_barrier
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

### Issue DM-10 - heavy_barrier
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

### Issue DM-11 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/writer_single_row_single_core.cpp`
**Lines**: 59-65
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue DM-12 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/reader_single_row_single_core.cpp`
**Lines**: 63-69
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue DM-13 - heavy_barrier
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

### Issue DM-14 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/strided_slice_writer_rm_interleaved.cpp`
**Lines**: 26-28
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue DM-15 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/kernels/nd_reshard_copy_pages_writer.cpp`
**Lines**: 28-31
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue DM-16 - heavy_barrier
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

### Issue DM-17 - heavy_barrier
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

### Issue DM-18 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/writer_pad_tiled.cpp`
**Lines**: 69-79
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue DM-19 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/writer_unary_pad_dims_interleaved.cpp`
**Lines**: 54-58
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue DM-20 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/writer_pad_dims_rm_interleaved.cpp`
**Lines**: 37-44
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue DM-21 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_writer_single_row_multi_core.cpp`
**Lines**: 63-68
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue DM-22 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_writer_single_row_single_core.cpp`
**Lines**: 71-76
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue DM-23 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/fold/device/kernels/dataflow/writer_cb2dram_for_rm_input.cpp`
**Lines**: 47-50
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue DM-24 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/copy/device/kernels/writer_unary_start_id.cpp`
**Lines**: 50-56
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue DM-25 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/copy/device/kernels/writer_unary_stick_start_id.cpp`
**Lines**: 40-49
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issues DM-26 to DM-29 - heavy_barrier
**Files**:
- `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/write_kernel_rm_sharded.cpp`
- `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/write_kernel_sharded.cpp`
- `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/write_kernel_rm.cpp`
- `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/write_kernel.cpp`
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()` in all.

---

### Issue DM-30 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/dataflow/writer_unary_interleaved_input_cols_batched.cpp`
**Lines**: 32-40
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue DM-31 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/dataflow/writer_permute_interleaved_rm_row_invariant.cpp`
**Lines**: 52-57
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue DM-32 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/reader_cross_core_data_exchange.cpp`
**Lines**: 140-146
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

## UNCLEAR/CORRECT Cases

### Issue DM-33 - CORRECT (Cross-Core Communication)
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/cross_core_data_exchange_common.hpp`
**Lines**: 82-85
**Code**:
```cpp
noc_async_write(value_cb_self_read_addr, cb_value_peer_noc_write_addr, value_cb_tile_size);
noc_async_write(index_cb_self_read_addr, cb_index_peer_noc_write_addr, index_cb_tile_size);
noc_async_write_barrier();  // Followed by semaphore_inc to peer
noc_semaphore_inc(...)
```
**Explanation**: This writes to a remote core's CB and then signals via semaphore. The full barrier IS required here because we're writing to one location and then signaling another core about that data.
**Assessment**: CORRECT - no fix needed.

---

### Issue DM-34 - UNCLEAR
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/writer_unary_transpose_hc_interleaved_tiled_padding_aware.cpp`
**Lines**: 147-151
**Assessment**: Needs manual review.

---

### Issue DM-35 - UNCLEAR
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/data_movement/moe_expert_token_remap/device/kernels/dataflow/writer_moe_expert_token_remap.cpp`
**Lines**: 81, 89-91
**Assessment**: Needs manual review.

---

# Section 3: Moreh/Reduction Operations

## Summary

| Category | Count |
|----------|-------|
| heavy_barrier | 54 |
| barrier_location | 2 |
| CORRECT | 1 |
| UNCLEAR | 2 |

---

## HEAVY_BARRIER Issues (Correct Location, Wrong Type)

### Issue MR-1 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_h_impl_kernels/writer_moreh_sum_h.cpp`
**Lines**: 22-28
**Code**:
```cpp
for (uint32_t i = start_id; i < end_id; ++i) {
    cb_wait_front(cb_id_out, onetile);  // CB wait INSIDE loop
    uint32_t l1_read_addr = get_read_ptr(cb_id_out);
    noc_async_write_tile(i, s, l1_read_addr);
    noc_async_write_barrier();  // ISSUE: should be noc_async_writes_flushed()
    cb_pop_front(cb_id_out, onetile);  // CB pop INSIDE loop
}
```
**Explanation**: CB wait/pop is inside the loop, so synchronization is required before each pop. However, since there's no cross-core signaling, `noc_async_writes_flushed()` suffices.
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue MR-2 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_w_impl_kernels/writer_moreh_int_sum_w.cpp`
**Lines**: 23-38
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue MR-3 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_w_impl_kernels/writer_moreh_sum_w.cpp`
**Lines**: 22-28
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue MR-4 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_h_impl_kernels/writer_moreh_int_sum_h.cpp`
**Lines**: 23-39
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue MR-5 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/kernels/writer_moreh_softmax_backward_c.cpp`
**Lines**: 25-40
**Code**:
```cpp
for (uint32_t i = 0; i < num_tiles; i += onetile) {
    for (uint32_t d = 0; d < dim_size; d++) {
        cb_wait_front(cb_out, onetile);  // CB wait INSIDE inner loop
        uint32_t l1_read_addr = get_read_ptr(cb_out);
        noc_async_write_tile(tile_idx, dst_out, l1_read_addr);
        noc_async_write_barrier();  // ISSUE: should be flush
        cb_pop_front(cb_out, onetile);  // CB pop INSIDE inner loop
        tile_idx += dim_stride;
    }
    curr_tile += 1;
}
```
**Explanation**: Double-nested loop with CB wait/pop inside inner loop. Each pop requires synchronization, but flush suffices.
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issues MR-6 to MR-10 - heavy_barrier (moreh_softmax_backward)
**Files**:
- `writer_moreh_softmax_backward_h.cpp` (Lines 24-37)
- `writer_moreh_softmax_backward_w.cpp` (Lines 24-34)
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issues MR-11 to MR-15 - heavy_barrier (moreh_softmax)
**Files**:
- `writer_moreh_softmax_c.cpp` (Lines 25-40)
- `writer_moreh_softmax_h.cpp` (Lines 25-40)
- `writer_moreh_softmax_w.cpp` (Lines 21-31)
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issues MR-16 to MR-18 - heavy_barrier (moreh_sum_nc)
**Files**:
- `writer_moreh_sum_nc.cpp` (Lines 23-31)
- `writer_moreh_int_sum_nc.cpp`
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issues MR-19 to MR-22 - heavy_barrier (moreh_mean)
**Files**:
- `writer_moreh_mean_h.cpp`
- `writer_moreh_mean_w.cpp`
- `writer_moreh_mean_nc.cpp`
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issues MR-23 to MR-26 - heavy_barrier (moreh_matmul)
**Files**:
- `writer_moreh_matmul.cpp` (Lines 24-30)
- `writer_moreh_bias_add.cpp`
- `writer_moreh_bias_add_single_tile.cpp`
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issues MR-27 to MR-30 - heavy_barrier (moreh_linear)
**Files**:
- `writer_moreh_linear.cpp`
- `writer_moreh_linear_backward.cpp`
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issues MR-31 to MR-35 - heavy_barrier (moreh_nll_loss/moreh_dot)
**Files**:
- `writer_moreh_nll_loss.cpp`
- `writer_moreh_nll_loss_backward.cpp`
- `writer_moreh_nll_loss_unreduced_backward.cpp`
- `writer_moreh_nll_loss_unreduced.cpp`
- `writer_moreh_dot.cpp` (Lines 22-28)
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issues MR-36 to MR-40 - heavy_barrier (HIGH PRIORITY: moreh_sgd/moreh_adam/moreh_adamw)
**Files**:
- `writer_moreh_sgd.cpp` (Lines 28-36) - **2 barriers per iteration**
- `writer_moreh_adam.cpp` (Lines 42-68) - **3-4 barriers per iteration**
- `writer_moreh_adamw.cpp` (Lines 43-69) - **3-4 barriers per iteration**

**Code Example (moreh_adamw)**:
```cpp
for (uint32_t i = start_id; i < end_id; ++i) {
    // param write + barrier
    cb_wait_front(cb_id_param, onetile);
    noc_async_write_tile(i, dst_param, get_read_ptr(cb_id_param));
    noc_async_write_barrier();  // BARRIER #1
    cb_pop_front(cb_id_param, onetile);

    // exp_avg write + barrier
    cb_wait_front(cb_id_exp_avg, onetile);
    noc_async_write_tile(i, dst_exp_avg, get_read_ptr(cb_id_exp_avg));
    noc_async_write_barrier();  // BARRIER #2
    cb_pop_front(cb_id_exp_avg, onetile);

    // exp_avg_sq write + barrier
    cb_wait_front(cb_id_exp_avg_sq, onetile);
    noc_async_write_tile(i, dst_exp_avg_sq, get_read_ptr(cb_id_exp_avg_sq));
    noc_async_write_barrier();  // BARRIER #3
    cb_pop_front(cb_id_exp_avg_sq, onetile);

    // max_exp_avg_sq write + barrier (optional)
    if constexpr (amsgrad) {
        cb_wait_front(cb_id_max_exp_avg_sq, onetile);
        noc_async_write_tile(i, dst_max_exp_avg_sq, get_read_ptr(cb_id_max_exp_avg_sq));
        noc_async_write_barrier();  // BARRIER #4
        cb_pop_front(cb_id_max_exp_avg_sq, onetile);
    }
}
```
**Note**: These files have MULTIPLE barriers per loop iteration (for param, exp_avg, exp_avg_sq, etc.). All should use `noc_async_writes_flushed()`.
**Suggested Fix**: Replace all `noc_async_write_barrier()` calls with `noc_async_writes_flushed()`.

---

### Issues MR-41 to MR-45 - heavy_barrier (moreh_cumsum/moreh_getitem)
**Files**:
- `writer_moreh_cumsum_nc.cpp`
- `writer_moreh_getitem.cpp` (Lines 26-35)
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issues MR-46 to MR-50 - heavy_barrier (moreh_layer_norm/moreh_group_norm)
**Files**:
- `writer_moreh_layer_norm.cpp` (Lines 26-86, also in helper functions)
- `writer_moreh_layer_norm_backward.cpp`
- `writer_moreh_layer_norm_backward_input_grad.cpp`
- `writer_moreh_group_norm.cpp` (Lines 52-117)
- `writer_moreh_group_norm_backward.cpp`
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issues MR-51 to MR-54 - heavy_barrier (moreh_clip_grad_norm)
**Files**:
- `writer_moreh_clip_grad_norm_step1.cpp`
- `writer_moreh_clip_grad_norm_step2.cpp`
- `writer_moreh_clip_grad_norm_step3.cpp`
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

## REDUCTION Operations Issues

### Issue MR-55 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/writer_argmax_interleaved_multicore.cpp`
**Lines**: 30-38
**Code**:
```cpp
for (uint32_t n = 0; n < num_tiles; n++) {
    cb_wait_front(cb_values, 1);  // CB wait INSIDE loop
    auto values_l1_addr = get_read_ptr(cb_values);
    noc_async_write_tile(start_id + n, values_addrgen, values_l1_addr);
    noc_async_write_barrier();  // ISSUE: should be flush
    cb_pop_front(cb_values, 1);  // CB pop INSIDE loop
}
```
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue MR-56 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/writer_reduce_interleaved_start_id.cpp`
**Lines**: 22-29
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue MR-57 - CORRECT (Cross-Core Communication)
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/writer_local_topk.cpp`
**Lines**: 48-73
**Code**:
```cpp
// Write to remote core's L1
noc_async_write(l1_read_values_addr, peer_cb_write_addr, tile_size);
noc_async_write(l1_read_indices_addr, peer_cb_index_write_addr, tile_size);
noc_async_write_barrier();  // CORRECT: followed by semaphore signal
noc_semaphore_inc(signal_semaphore_addr, 1);
```
**Assessment**: CORRECT - writing to remote core and then signaling. Full barrier is required.

---

### Issue MR-58 - UNCLEAR
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/reader_argmax_interleaved_multicore.cpp`
**Lines**: 367, 396-397, 420-421, 434, 454-455
**Explanation**: Multiple barriers for multicore reduction with complex synchronization. Some barriers appear necessary for cross-core communication (before semaphore operations).
**Assessment**: Needs manual review - some barriers may be correct for multicore sync, others may be convertible to flush.

---

## BARRIER_LOCATION Issues (Can Move Barrier Outside Loop)

### Issue MR-59 - barrier_location
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/writer_moreh_softmax_h.cpp`
**Lines**: 25-40
**Code**:
```cpp
for (uint32_t i = 0; i < N; i++) {
    cb_wait_front(cb_id_out, Ht);  // Waits for Ht tiles
    auto l1_read_addr = get_read_ptr(cb_id_out);
    for (uint32_t h = 0; h < Ht; h++) {
        noc_async_write_tile(tile_idx, s, l1_read_addr);
        l1_read_addr += tile_bytes;
        tile_idx += Wt;
    }
    noc_async_write_barrier();  // ISSUE: can be flush, location OK for this pattern
    cb_pop_front(cb_id_out, Ht);  // Pops Ht tiles
    curr_tile += 1;
}
```
**Explanation**: This is a better pattern - batches Ht writes before barrier. The barrier is at the correct location (before pop), but should use `noc_async_writes_flushed()`.
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue MR-60 - barrier_location
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/writer_moreh_softmax_w.cpp`
**Lines**: 21-31
**Explanation**: Similar pattern - batches tiles within inner loop.
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

# Section 4: Transformer/Matmul Operations

## Summary

| Category | Count |
|----------|-------|
| heavy_barrier | 12 |
| barrier_location | 1 |
| CORRECT | 4 |
| UNCLEAR | 1 |

---

## HEAVY_BARRIER Issues (Correct Location, Wrong Type)

### Issue TM-1 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/writer_bmm_tile_layout.cpp`
**Lines**: 57-58
**Code**:
```cpp
for (...batch...) {
    for (...subblock_h...) {
        for (...subblock_w...) {
            cb_wait_front(cb_id_out0, out_subblock_tile_count);  // CB wait INSIDE loop
            // ... write tiles ...
            noc_async_write_barrier();  // ISSUE: should be noc_async_writes_flushed()
            cb_pop_front(cb_id_out0, out_subblock_tile_count);  // CB pop INSIDE loop
        }
    }
}
```
**Explanation**: CB wait/pop is inside the innermost loop. Synchronization is required before each pop, but `noc_async_writes_flushed()` suffices since there's no cross-core signaling.
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue TM-2 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp`
**Lines**: 194-196
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue TM-3 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp`
**Lines**: 533-534
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue TM-4 - heavy_barrier (HIGH PRIORITY: 2 barriers per iteration)
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer/rotate_half/device/kernels/dataflow/writer_rotate_half_interleaved_start_id.cpp`
**Lines**: 33, 40
**Code**:
```cpp
for (uint32_t row = 0; row < num_rows; row++) {
    cb_wait_front(cb_id_out_no_mul, onetile);  // CB wait INSIDE loop
    noc_async_write_tile(out_no_mul_curr_id, s, out_no_mul_l1_read_addr);
    noc_async_write_barrier();  // ISSUE #1: should be flush
    cb_pop_front(cb_id_out_no_mul, onetile);  // CB pop INSIDE loop

    cb_wait_front(cb_id_out_mul, onetile);  // CB wait INSIDE loop
    noc_async_write_tile(out_mul_curr_id, s, out_mul_l1_read_addr);
    noc_async_write_barrier();  // ISSUE #2: should be flush
    cb_pop_front(cb_id_out_mul, onetile);  // CB pop INSIDE loop
}
```
**Explanation**: Two separate CB wait/pop pairs inside the loop, each with its own barrier.
**Suggested Fix**: Replace both `noc_async_write_barrier()` calls with `noc_async_writes_flushed()`.

---

### Issue TM-5 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/dataflow/writer_rotary_embedding_interleaved_start_id.cpp`
**Lines**: 60
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issues TM-6 to TM-9 - heavy_barrier (NLP QKV heads kernels)
**Files**:
- `nlp_create_qkv_heads_segformer/.../writer_tm_tile_layout_nlp_create_qkv_heads.cpp` (Lines 46-47, 61, 84, 107)
- `nlp_create_qkv_heads_vit/.../writer_tm_tile_layout_nlp_create_qkv_heads.cpp`
- `nlp_create_qkv_heads_falcon7b/.../writer_tm_tile_layout_nlp_create_qkv_heads_falcon7b.cpp` (Line 96)
- `nlp_create_qkv_heads_boltz/.../writer_tm_tile_layout_nlp_create_qkv_heads_boltz.cpp` (Lines 60, 83, 106)
- `nlp_create_qkv_heads/.../writer_tm_tile_layout_nlp_create_qkv_heads.cpp` (Lines 60, 83, 106)

**Note**: Some of these have multiple barriers per block iteration (one each for Q, K, V processing).
**Suggested Fix**: Replace all `noc_async_write_barrier()` calls with `noc_async_writes_flushed()`.

---

### Issue TM-10 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/device/kernels/dataflow/writer_unary_interleaved_start_id_blocked.cpp`
**Lines**: 34-35
**Code**:
```cpp
for (uint32_t i = 0; i < num_tiles; i += blk) {
    cb_wait_front(cb_out, blk);  // CB wait INSIDE loop
    // ... write blk tiles ...
    noc_async_write_barrier();  // ISSUE: should be flush
    cb_pop_front(cb_out, blk);  // CB pop INSIDE loop
}
```
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue TM-11 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/device/kernels/dataflow/writer_transformer_group_attn_matmul.cpp`
**Lines**: 144
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue TM-12 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/writer_interleaved.cpp`
**Lines**: 158
**Code**:
```cpp
for (uint32_t q_iter = 0; q_iter < q_chunks_per_core; q_iter++) {
    // ... writes with threshold-based flushing ...
    noc_async_write_barrier();  // ISSUE: should be flush
    cb_pop_front(cb_out, out_chunk_tiles);
}
```
**Note**: This kernel already uses `noc_async_writes_flushed()` for intermediate writes with a threshold, but uses full barrier before cb_pop.
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

## BARRIER_LOCATION Issues (Can Move Barrier Outside Loop)

### Issue TM-13 - barrier_location
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/dataflow/writer_rotary_embedding_llama_interleaved_start_id.cpp`
**Lines**: 39-40
**Code**:
```cpp
for (uint32_t seq = 0; seq < seq_tiles; seq++) {
    cb_wait_front(cb_id_out, Wt);  // Waits for Wt tiles
    uint32_t l1_read_addr = get_read_ptr(cb_id_out);
    for (uint32_t w = 0; w < Wt; w++) {
        noc_async_write_tile(tile_idx, s, l1_read_addr);
        l1_read_addr += tile_bytes;
        tile_idx++;
    }
    noc_async_write_barrier();  // ISSUE: should be flush; location is OK (before pop)
    cb_pop_front(cb_id_out, Wt);  // Pops Wt tiles
}
```
**Explanation**: This is already a better pattern - batches Wt writes before barrier. Location is correct, just needs to use flush.
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

## CORRECT Patterns (No Fix Needed)

### Pattern TM-14 - CORRECT (Cross-Core Signaling)
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp`
**Lines**: 436-437
**Code**:
```cpp
noc_async_write_barrier();  // CORRECT: followed by semaphore signal
noc_semaphore_inc(in0_sender_semaphore_noc_addr, 1);
```
**Assessment**: This barrier IS required - writing data then signaling another core. Full barrier ensures data lands before signal.

---

### Pattern TM-15 - CORRECT (Ring Algorithm)
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_writer.cpp`
**Lines**: 159
**Code**:
```cpp
noc_async_write_barrier();  // Ensure writes complete before next ring iteration
```
**Assessment**: CORRECT - ring algorithm requires writes to complete before next iteration reads same locations.

---

### Pattern TM-16 - CORRECT (Mcast Synchronization)
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/device/kernels/dataflow/reader_mcast_transformer_group_attn_matmul.cpp`
**Lines**: 270
**Code**:
```cpp
// Write barrier needed to make sure we finish sending mcast flag before we modify locally
noc_async_write_barrier();
```
**Assessment**: CORRECT - multicast synchronization requires barrier before local modification.

---

### Pattern TM-17 - CORRECT (Flush in Loop, Barrier at End)
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/device/kernels/dataflow/rms_pre_allgather_writer.cpp`
**Lines**: 35
**Code**:
```cpp
for (uint32_t row = 0; row < output_rows; row++) {
    // ... writes ...
    noc_async_writes_flushed();  // CORRECT: flush inside loop
    cb_pop_front(output_cb, output_tiles_per_row);
}
noc_async_write_barrier();  // CORRECT: barrier at kernel end
```
**Assessment**: This is the OPTIMAL pattern - flush in loop, barrier at end.

---

## UNCLEAR Cases

### Issue TM-18 - UNCLEAR (BH Workaround)
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/writer_decode_all.cpp`
**Lines**: 178
**Code**:
```cpp
noc_async_write_barrier();  // #19201 BH hang workaround
```
**Assessment**: This is noted as a "BH hang workaround". Consider making it conditional on architecture.

---

# Section 5: Other Operations (Normalization, Eltwise, Pool, etc.)

## Summary

| Category | Count |
|----------|-------|
| heavy_barrier | 11 |
| barrier_location | 2 |
| CORRECT | 3 |

---

## HEAVY_BARRIER Issues (Correct Location, Wrong Type)

### Issue OT-1 - heavy_barrier
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

### Issue OT-2 - heavy_barrier
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

### Issue OT-3 - heavy_barrier (HIGH PRIORITY: 3 barriers per iteration!)
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

### Issue OT-4 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
**Lines**: 32-39
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue OT-5 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_wh.cpp`
**Lines**: 31-45
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue OT-6 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/writer_interleaved_scalar.cpp`
**Lines**: 62-68
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue OT-7 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast.cpp`
**Lines**: 57-67
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue OT-8 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/writer_grid_sample_interleaved.cpp`
**Lines**: 25-45
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue OT-9 - heavy_barrier
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

### Issue OT-10 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/experimental/ssm/repeat_and_interleave_eltwise_mul/device/kernels/writer_ssm_eltwise_mul.cpp`
**Lines**: 22-31
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue OT-11 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/device/kernels/writer_reduce_nc.cpp`
**Lines**: 30-41
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

## BARRIER_LOCATION Issues (Can Move Barrier Outside Loop)

### Issue OT-12 - barrier_location
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

### Issue OT-13 - barrier_location
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

### Pattern OT-14 - CORRECT (KV Cache)
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

### Pattern OT-15 - CORRECT (Paged Cache)
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/writer_update_cache_interleaved_start_id.cpp`
**Lines**: 101-116
**Assessment**: Same optimal pattern as KV cache - flush in loop, barrier at end.

---

### Pattern OT-16 - CORRECT (Deepseek)
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

# Appendix: Summary Tables

## Issues by Category

| Category | Total Count |
|----------|-------------|
| heavy_barrier | ~95 |
| barrier_location | ~15 |
| missing_flush (UNSAFE) | 1 |
| CORRECT | ~15 |
| UNCLEAR | ~5 |

## High Priority Files (Multiple Barriers Per Iteration)

| File | Barriers/Iteration | Location |
|------|-------------------|----------|
| writer_running_statistics.cpp | 3 | batch_norm |
| writer_moreh_adamw.cpp | 3-4 | moreh_adamw |
| writer_moreh_adam.cpp | 3-4 | moreh_adam |
| writer_moreh_sgd.cpp | 2 | moreh_sgd |
| writer_rotate_half_interleaved_start_id.cpp | 2 | rotate_half |

## Deeply Nested Barriers

| File | Nesting Level | Location |
|------|---------------|----------|
| writer_interleaved_no_bcast.cpp | 6 levels | binary_ng |
| writer_unary_interleaved_start_id_wh.cpp | 3 levels | eltwise/unary |
| welford_writer_unary_gn_rm_gb.cpp | 2 levels | groupnorm |

---

# Files Generated

| File | Description |
|------|-------------|
| `ccl_fabric_analysis.md` | CCL/fabric kernel analysis (4 heavy_barrier, 1 unsafe, 2 correct) |
| `data_movement_analysis.md` | Data movement analysis (24 heavy_barrier, 8 barrier_location, 3 correct) |
| `moreh_reduction_analysis.md` | Moreh/reduction analysis (54 heavy_barrier, 2 barrier_location, 1 correct) |
| `transformer_matmul_analysis.md` | Transformer/matmul analysis (12 heavy_barrier, 1 barrier_location, 4 correct) |
| `other_ops_analysis.md` | Other operations analysis (11 heavy_barrier, 2 barrier_location, 3 correct) |
| `final_report.md` | Executive summary |
| `complete_analysis.md` | This comprehensive file with all details |

All files located at: `/Users/snijjar/work/tt-metal/kernel_analysis_results/`
