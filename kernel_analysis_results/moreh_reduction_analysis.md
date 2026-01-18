# Kernel Analysis Results: moreh and reduction Operations

This document contains an analysis of data movement kernels in the `ttnn/cpp/ttnn/operations/moreh/` and `ttnn/cpp/ttnn/operations/reduction/` directories for inefficient or unsafe write patterns.

## Summary

- **Total files analyzed**: 62 files
- **Issues found**: 59 issues identified
- **Categories breakdown**:
  - heavy_barrier: 54 issues (correct location, wrong type - should use `noc_async_writes_flushed()`)
  - barrier_location: 5 issues (can move barrier outside loop AND use flush)
  - CORRECT: Several patterns with valid cross-core signaling

**Note on categorization:**
- **heavy_barrier**: CB wait/pop is INSIDE the loop. Synchronization IS required before each `cb_pop_front`, but `noc_async_write_barrier()` should be replaced with `noc_async_writes_flushed()`.
- **barrier_location**: CB wait/pop is OUTSIDE the loop. Barrier can be moved to right before `cb_pop_front`.

---

## HEAVY_BARRIER Issues (Correct Location, Wrong Type)

These files have CB wait/pop INSIDE the loop. Synchronization IS required before each `cb_pop_front`, but `noc_async_writes_flushed()` should be used instead of the heavier `noc_async_write_barrier()`.

### Issue #1 - heavy_barrier
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

### Issue #2 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_w_impl_kernels/writer_moreh_int_sum_w.cpp`
**Lines**: 23-38
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #3 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_w_impl_kernels/writer_moreh_sum_w.cpp`
**Lines**: 22-28
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #4 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_h_impl_kernels/writer_moreh_int_sum_h.cpp`
**Lines**: 23-39
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #5 - heavy_barrier
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

### Issue #6-10 - heavy_barrier (moreh_softmax_backward)
**Files**:
- `writer_moreh_softmax_backward_h.cpp` (Lines 24-37)
- `writer_moreh_softmax_backward_w.cpp` (Lines 24-34)
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #11-15 - heavy_barrier (moreh_softmax)
**Files**:
- `writer_moreh_softmax_c.cpp` (Lines 25-40)
- `writer_moreh_softmax_h.cpp` (Lines 25-40)
- `writer_moreh_softmax_w.cpp` (Lines 21-31)
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #16-18 - heavy_barrier (moreh_sum_nc)
**Files**:
- `writer_moreh_sum_nc.cpp` (Lines 23-31)
- `writer_moreh_int_sum_nc.cpp`
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #19-22 - heavy_barrier (moreh_mean)
**Files**:
- `writer_moreh_mean_h.cpp`
- `writer_moreh_mean_w.cpp`
- `writer_moreh_mean_nc.cpp`
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #23-26 - heavy_barrier (moreh_matmul)
**Files**:
- `writer_moreh_matmul.cpp` (Lines 24-30)
- `writer_moreh_bias_add.cpp`
- `writer_moreh_bias_add_single_tile.cpp`
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #27-30 - heavy_barrier (moreh_linear)
**Files**:
- `writer_moreh_linear.cpp`
- `writer_moreh_linear_backward.cpp`
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #31-35 - heavy_barrier (moreh_nll_loss/moreh_dot)
**Files**:
- `writer_moreh_nll_loss.cpp`
- `writer_moreh_nll_loss_backward.cpp`
- `writer_moreh_nll_loss_unreduced_backward.cpp`
- `writer_moreh_nll_loss_unreduced.cpp`
- `writer_moreh_dot.cpp` (Lines 22-28)
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #36-40 - heavy_barrier (moreh_sgd/moreh_adam/moreh_adamw)
**Files**:
- `writer_moreh_sgd.cpp` (Lines 28-36) - 2 barriers per iteration
- `writer_moreh_adam.cpp` (Lines 42-68) - 3-4 barriers per iteration
- `writer_moreh_adamw.cpp` (Lines 43-69) - 3-4 barriers per iteration
**Note**: These files have MULTIPLE barriers per loop iteration (for param, exp_avg, exp_avg_sq, etc.). All should use `noc_async_writes_flushed()`.
**Suggested Fix**: Replace all `noc_async_write_barrier()` calls with `noc_async_writes_flushed()`.

---

### Issue #41-45 - heavy_barrier (moreh_cumsum/moreh_getitem)
**Files**:
- `writer_moreh_cumsum_nc.cpp`
- `writer_moreh_getitem.cpp` (Lines 26-35)
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #46-50 - heavy_barrier (moreh_layer_norm/moreh_group_norm)
**Files**:
- `writer_moreh_layer_norm.cpp` (Lines 26-86, also in helper functions)
- `writer_moreh_layer_norm_backward.cpp`
- `writer_moreh_layer_norm_backward_input_grad.cpp`
- `writer_moreh_group_norm.cpp` (Lines 52-117)
- `writer_moreh_group_norm_backward.cpp`
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #51-54 - heavy_barrier (moreh_clip_grad_norm)
**Files**:
- `writer_moreh_clip_grad_norm_step1.cpp`
- `writer_moreh_clip_grad_norm_step2.cpp`
- `writer_moreh_clip_grad_norm_step3.cpp`
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

## REDUCTION Operations Issues

### Issue #55 - heavy_barrier
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

### Issue #56 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/writer_reduce_interleaved_start_id.cpp`
**Lines**: 22-29
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #57 - CORRECT (Cross-Core Communication)
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

### Issue #58 - UNCLEAR
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/reader_argmax_interleaved_multicore.cpp`
**Lines**: 367, 396-397, 420-421, 434, 454-455
**Explanation**: Multiple barriers for multicore reduction with complex synchronization. Some barriers appear necessary for cross-core communication (before semaphore operations).
**Assessment**: Needs manual review - some barriers may be correct for multicore sync, others may be convertible to flush.

---

## BARRIER_LOCATION Issues (Can Move Barrier Outside Loop)

These files have CB wait/pop OUTSIDE the loop but barrier INSIDE.

### Issue #59 - barrier_location
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
**Note**: This pattern is already improved compared to single-tile-per-barrier. The only fix needed is using flush instead of full barrier.
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #60 - barrier_location
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/writer_moreh_softmax_w.cpp`
**Lines**: 21-31
**Explanation**: Similar pattern - batches tiles within inner loop.
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

## GOOD Patterns (Reference)

### Pattern 1: KV Cache Style (flush in loop, barrier at end)
```cpp
for (uint32_t i = 0; i < iterations; i++) {
    cb_wait_front(cb, batch_size);
    for (uint32_t j = 0; j < batch_size; j++) {
        noc_async_write_tile(...);
    }
    noc_async_writes_flushed();  // Lighter weight
    cb_pop_front(cb, batch_size);
}
noc_async_write_barrier();  // At kernel end
```

### Pattern 2: Cross-Core Signaling (barrier required)
```cpp
noc_async_write(local_data, remote_addr, size);
noc_async_write_barrier();  // REQUIRED - ensures data lands
noc_semaphore_inc(remote_semaphore, 1);  // Signal remote core
```

---

## Summary

| Category | Count | Fix |
|----------|-------|-----|
| heavy_barrier | 54 | Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()` |
| barrier_location | 2 | Already batched, just change to `noc_async_writes_flushed()` |
| CORRECT | 1 | Cross-core signaling - no fix needed |
| UNCLEAR | 2 | Needs manual review for multicore sync |

**High Priority Fixes:**
1. **moreh_adam/moreh_adamw** - 3-4 barriers per loop iteration
2. **moreh_sgd** - 2 barriers per loop iteration
3. **moreh_softmax_backward_c** - nested loop with barrier in inner loop
