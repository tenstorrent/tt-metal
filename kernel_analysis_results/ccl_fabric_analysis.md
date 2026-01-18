# CCL Fabric Kernel Analysis Results

This document contains analysis of data movement kernels for inefficient or unsafe patterns in:
- `ttnn/cpp/ttnn/operations/ccl/`
- `ttnn/cpp/ttnn/operations/experimental/ccl/`

---

## Summary

| Category | Count |
|----------|-------|
| INEFFICIENT - heavy_barrier | 4 |
| UNSAFE - missing_flush | 1 |
| CORRECT - valid pattern | 2 |

---

## INEFFICIENT Patterns

### Issue #1 - INEFFICIENT
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_async/device/kernels/local_copy_writer.cpp`
**Lines**: 44-47
**Category**: `heavy_barrier`
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

### Issue #2 - INEFFICIENT
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_async/device/kernels/minimal_default_writer.cpp`
**Lines**: 117-129, 141-151, 166-191
**Category**: `heavy_barrier`
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
**Explanation**: Same pattern as Issue #1. The barrier location is correct (before `cb_pop_front`), but `noc_async_write_barrier()` is heavier than necessary. The writes are to output memory with no cross-core signaling, so `noc_async_writes_flushed()` is sufficient.

**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()` at all three locations (lines 128, 149, 190).

---

### Issue #3 - INEFFICIENT
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/dim_zero_ring_reduce_scatter_minimal_async_writer.cpp`
**Lines**: 362-373
**Category**: `heavy_barrier`
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

### Issue #4 - INEFFICIENT
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/ring_reduce_scatter_minimal_async_writer.cpp`
**Lines**: 412-423
**Category**: `heavy_barrier`
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
**Explanation**: Same pattern as Issue #3. Writing to output buffer without cross-core signaling.

**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

## CORRECT Patterns (No Fix Needed)

### Pattern #5 - CORRECT
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

### Pattern #6 - CORRECT
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/dim_zero_line_reduce_scatter_minimal_async_writer.cpp`
**Lines**: 374-375 (within `do_final_reduction` block)
**Assessment**: Same as Pattern #5 - correctly uses conditional barrier/flush based on synchronization needs.

---

## UNSAFE Patterns

### Issue #7 - UNSAFE
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/ccl/reduce_to_root/device/kernels/root_receive_writer_kernel.cpp`
**Lines**: 23-39
**Category**: `missing_flush`
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

## GOOD Patterns Observed (Reference)

The following kernels demonstrate good practices:

1. **ccl_send_reader_two_input.cpp** (lines 734-736): Correctly uses `noc_async_writes_flushed()` before `cb_pop_front`:
```cpp
noc_async_writes_flushed();
cb_pop_front(cmd_ctx.cb_id, cmd_ctx.packet_size_in_pages);
```

2. **broadcast_rm_writer.cpp** and **broadcast_tile_writer.cpp**: Use `noc_async_writes_flushed()` inside loops and `noc_async_write_barrier()` only at kernel end.

3. **writer_all_to_all_dispatch.cpp**: Uses `noc_async_writes_flushed()` for intermediate writes and `noc_async_write_barrier()` only when specifically required before signaling.

4. **writer_all_to_all_combine.cpp**: Properly uses `noc_async_writes_flushed()` inside loops and `noc_async_write_barrier()` at kernel end.

5. **line_reduce_scatter_minimal_async_writer.cpp**: Correctly uses conditional barrier/flush based on whether cross-core signaling is needed.

---

## Summary

| Issue | File | Problem | Fix |
|-------|------|---------|-----|
| #1 | local_copy_writer.cpp | heavy_barrier | Use `noc_async_writes_flushed()` |
| #2 | minimal_default_writer.cpp | heavy_barrier | Use `noc_async_writes_flushed()` |
| #3 | dim_zero_ring_reduce_scatter_minimal_async_writer.cpp | heavy_barrier | Use `noc_async_writes_flushed()` |
| #4 | ring_reduce_scatter_minimal_async_writer.cpp | heavy_barrier | Use `noc_async_writes_flushed()` |
| #5 | line_reduce_scatter_minimal_async_writer.cpp | CORRECT | No fix needed |
| #6 | dim_zero_line_reduce_scatter_minimal_async_writer.cpp | CORRECT | No fix needed |
| #7 | root_receive_writer_kernel.cpp | UNSAFE: missing_flush | Add `noc_async_writes_flushed()` before pops |
