# Data Movement Kernel Analysis: Transformer and Matmul Operations

This document analyzes kernel files in the transformer and matmul operations directories for inefficient or unsafe NOC write patterns.

## Summary

| Category | Count | Description |
|----------|-------|-------------|
| heavy_barrier | 12 | Correct location, wrong type - should use `noc_async_writes_flushed()` |
| barrier_location | 3 | Can move barrier outside inner loop |
| CORRECT | 4 | Valid patterns with cross-core signaling |

**Note on categorization:**
- **heavy_barrier**: CB wait/pop is INSIDE the loop. Synchronization IS required before each `cb_pop_front`, but should use `noc_async_writes_flushed()` instead of `noc_async_write_barrier()`.
- **barrier_location**: CB wait/pop is OUTSIDE the inner loop but barrier is INSIDE. Barrier can be moved outside.

---

## HEAVY_BARRIER Issues (Correct Location, Wrong Type)

### Issue #1 - heavy_barrier
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

### Issue #2 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp`
**Lines**: 194-196
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #3 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp`
**Lines**: 533-534
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #4 - heavy_barrier (2 barriers per iteration)
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

### Issue #5 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/dataflow/writer_rotary_embedding_interleaved_start_id.cpp`
**Lines**: 60
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #6-9 - heavy_barrier (NLP QKV heads kernels)
**Files**:
- `nlp_create_qkv_heads_segformer/.../writer_tm_tile_layout_nlp_create_qkv_heads.cpp` (Lines 46-47, 61, 84, 107)
- `nlp_create_qkv_heads_vit/.../writer_tm_tile_layout_nlp_create_qkv_heads.cpp`
- `nlp_create_qkv_heads_falcon7b/.../writer_tm_tile_layout_nlp_create_qkv_heads_falcon7b.cpp` (Line 96)
- `nlp_create_qkv_heads_boltz/.../writer_tm_tile_layout_nlp_create_qkv_heads_boltz.cpp` (Lines 60, 83, 106)
- `nlp_create_qkv_heads/.../writer_tm_tile_layout_nlp_create_qkv_heads.cpp` (Lines 60, 83, 106)

**Note**: Some of these have multiple barriers per block iteration (one each for Q, K, V processing).
**Suggested Fix**: Replace all `noc_async_write_barrier()` calls with `noc_async_writes_flushed()`.

---

### Issue #10 - heavy_barrier
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

### Issue #11 - heavy_barrier
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/device/kernels/dataflow/writer_transformer_group_attn_matmul.cpp`
**Lines**: 144
**Suggested Fix**: Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()`.

---

### Issue #12 - heavy_barrier
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

### Issue #13 - barrier_location
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

### Pattern #14 - CORRECT (Cross-Core Signaling)
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp`
**Lines**: 436-437
**Code**:
```cpp
noc_async_write_barrier();  // CORRECT: followed by semaphore signal
noc_semaphore_inc(in0_sender_semaphore_noc_addr, 1);
```
**Assessment**: This barrier IS required - writing data then signaling another core. Full barrier ensures data lands before signal.

---

### Pattern #15 - CORRECT (Ring Algorithm)
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_writer.cpp`
**Lines**: 159
**Code**:
```cpp
noc_async_write_barrier();  // Ensure writes complete before next ring iteration
```
**Assessment**: CORRECT - ring algorithm requires writes to complete before next iteration reads same locations.

---

### Pattern #16 - CORRECT (Mcast Synchronization)
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/device/kernels/dataflow/reader_mcast_transformer_group_attn_matmul.cpp`
**Lines**: 270
**Code**:
```cpp
// Write barrier needed to make sure we finish sending mcast flag before we modify locally
noc_async_write_barrier();
```
**Assessment**: CORRECT - multicast synchronization requires barrier before local modification.

---

### Pattern #17 - CORRECT (Flush in Loop, Barrier at End)
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

### Issue #18 - UNCLEAR (BH Workaround)
**File**: `/Users/snijjar/work/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/writer_decode_all.cpp`
**Lines**: 178
**Code**:
```cpp
noc_async_write_barrier();  // #19201 BH hang workaround
```
**Assessment**: This is noted as a "BH hang workaround". Consider making it conditional on architecture.

---

## Summary

| Category | Count | Fix |
|----------|-------|-----|
| heavy_barrier | 12 | Replace `noc_async_write_barrier()` with `noc_async_writes_flushed()` |
| barrier_location | 1 | Already batched, just use `noc_async_writes_flushed()` |
| CORRECT | 4 | Cross-core signaling, ring algorithm, mcast - no fix needed |
| UNCLEAR | 1 | BH workaround - make conditional on architecture |

**High Priority Fixes:**
1. **rotate_half** - 2 barriers per loop iteration
2. **NLP QKV heads kernels** - multiple barriers per block (Q, K, V processing)
