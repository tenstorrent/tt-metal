# cb_get_tile Migration Plan

## Background

GitHub Issue [#27979](https://github.com/tenstorrent/tt-metal/issues/27979) identified critical race conditions in the `cb_get_tile` API caused by Tensix semaphore limitations:

1. **Silent drops**: `sem_get` on empty or `sem_post` on full semaphore is silently dropped
2. **Simultaneous MMIO access**: Two accesses on same cycle causes one to be dropped
3. **L1 race condition**: Three TRISC threads sampling L1 at different times get inconsistent values

The solution is to migrate to direct mailbox-based communication where BRISC/UNPACK sends values to all compute threads, ensuring all threads receive identical values.

---

## Operations Requiring Migration

| Priority | Operation | File | Status |
|----------|-----------|------|--------|
| ✅ | Sparse MatMul | `bmm_large_block_zm_fused_bias_activation.cpp` | **DONE** |
| ✅ | SDPA Flash Decode | `sdpa_flash_decode.cpp` | **DONE** |
| ✅ | Embedding Backward | `embedding_backward.cpp` | **DONE** |
| ✅ | Normalization Utility | `memory.h` | **DONE** |

---

## Migration 1: SDPA Flash Decode ✅ COMPLETED

### Files Modified

- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp`

**Note:** The dataflow reader kernel runs on NCRISC (uses `ReaderDataMovementConfig`), so the UNPACK distribution approach was used instead of BRISC mailbox.

### Current Implementation (Lines 130-137)

```cpp
constexpr uint32_t cb_index_id = tt::CBIndex::c_8;
cb_wait_front(cb_index_id, 1);
volatile uint32_t* index_addr_ptr;
cb_get_tile(cb_index_id, 0, &index_addr_ptr);
uint32_t cb_get_tile_offset = 4;  // First 4 elements are metadata
cur_pos = index_addr_ptr[cb_get_tile_offset + (cur_batch / q_heads_parallel_factor)];
cb_release_tile(cb_index_id);
```

### Problem

- `cb_get_tile` distributes L1 pointer to UNPACK/MATH/PACK threads via semaphores
- Threads may sample L1 at different times, causing race conditions
- Can cause hangs on Wormhole and Blackhole

### Proposed Solution

**Option A: Direct mailbox from dataflow kernel (Preferred)**

Requires the dataflow reader to be on BRISC (not NCRISC).

**Dataflow kernel changes:**
```cpp
// After reading cur_pos data into CB
uint32_t cur_pos_value = /* compute cur_pos for this batch */;

// Send to all compute threads via mailbox
ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, cur_pos_value);
ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, cur_pos_value);
ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, cur_pos_value);
```

**Compute kernel changes:**
```cpp
// Replace cb_get_tile block with:
uint32_t cur_pos;
UNPACK(cur_pos = mailbox_read(ckernel::ThreadId::BriscThreadId);)
MATH(cur_pos = mailbox_read(ckernel::ThreadId::BriscThreadId);)
PACK(cur_pos = mailbox_read(ckernel::ThreadId::BriscThreadId);)
```

**Option B: UNPACK reads and distributes (if dataflow is on NCRISC)**

```cpp
uint32_t cur_pos;
UNPACK({
    cb_wait_front(cb_index_id, 1);
    uint32_t operand_id = get_operand_id(cb_index_id);
    uint32_t base_address = get_local_cb_interface(operand_id).fifo_rd_ptr - 1;
    volatile uint32_t* index_addr_ptr = reinterpret_cast<volatile uint32_t*>(base_address);
    uint32_t cb_get_tile_offset = 4;
    cur_pos = index_addr_ptr[cb_get_tile_offset + (cur_batch / q_heads_parallel_factor)];

    // Distribute to MATH and PACK via mailbox
    ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, cur_pos);
    ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, cur_pos);

    cb_pop_front(cb_index_id, 1);
})
MATH(cur_pos = mailbox_read(ckernel::ThreadId::UnpackThreadId);)
PACK(cur_pos = mailbox_read(ckernel::ThreadId::UnpackThreadId);)
```

### Testing

```bash
# Run SDPA decode tests
pytest tests/ttnn/unit_tests/operations/transformer/test_sdpa_decode.py -v

# Run on both architectures
pytest tests/ttnn/unit_tests/operations/transformer/test_sdpa_decode.py -v  # WH
pytest tests/ttnn/unit_tests/operations/transformer/test_sdpa_decode.py -v  # BH
```

---

## Migration 2: Embedding Backward

### Files to Modify

- `ttnn/cpp/ttnn/operations/embedding_backward/device/kernels/compute/embedding_backward.cpp`
- Corresponding dataflow reader kernel

### Current Implementation

**Location 1: chunk_count (Lines 28-32)**
```cpp
// Get chunk_count from reader
volatile uint32_t* chunk_addr_ptr;
cb_get_tile(cb_chunk_count_scratch, 0, &chunk_addr_ptr);
uint32_t chunk_count = chunk_addr_ptr[4];  // Offset for BBE read ptr
cb_release_tile(cb_chunk_count_scratch);
```

**Location 2: mask address (Lines 36-45)**
```cpp
// get cb_index pointer from unpack to math thread
volatile uint* idx_addr_ptr;
uint32_t tile_to_get = 0;
cb_get_tile(cb_mask, tile_to_get, &idx_addr_ptr);
uint32_t idx_addr = reinterpret_cast<uint32_t>(idx_addr_ptr);
#if defined(ARCH_BLACKHOLE)
// Workaround for tt-metal issue #11816
asm volatile("fence");
#endif
```

### Proposed Solution

**For chunk_count (dataflow kernel on BRISC):**

**Dataflow kernel:**
```cpp
// After determining chunk_count
ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, chunk_count);
ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, chunk_count);
ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, chunk_count);
```

**Compute kernel:**
```cpp
uint32_t chunk_count;
UNPACK(chunk_count = mailbox_read(ckernel::ThreadId::BriscThreadId);)
MATH(chunk_count = mailbox_read(ckernel::ThreadId::BriscThreadId);)
PACK(chunk_count = mailbox_read(ckernel::ThreadId::BriscThreadId);)
```

**For idx_addr (UNPACK reads and distributes):**

```cpp
uint32_t idx_addr;
UNPACK({
    cb_wait_front(cb_mask, 1);
    uint32_t operand_id = get_operand_id(cb_mask);
    uint32_t base_address = get_local_cb_interface(operand_id).fifo_rd_ptr - 1;
    idx_addr = base_address;

    // Distribute address to MATH and PACK
    ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, idx_addr);
    ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, idx_addr);
})
MATH(idx_addr = mailbox_read(ckernel::ThreadId::UnpackThreadId);)
PACK(idx_addr = mailbox_read(ckernel::ThreadId::UnpackThreadId);)

// Note: fence workaround may still be needed on BH
#if defined(ARCH_BLACKHOLE)
asm volatile("fence");
#endif
```

### Testing

```bash
pytest tests/ttnn/unit_tests/operations/test_embedding_backward.py -v
```

---

## Migration 3: Normalization Utility (memory.h) ✅ COMPLETED

### Files Modified

- `ttnn/cpp/ttnn/operations/normalization/kernel_util/compute/memory.h`

### Affected Kernels

- `layernorm_large_tensor_welford.cpp:247`
- `layernorm_sharded_welford.cpp:207`
- `layernorm_welford.cpp:80`
- `welford_groupnorm.cpp:212`

### Implementation

The `get_pointer_to_cb_data` template function was updated to use mailbox-based synchronization:

```cpp
template <typename To>
ALWI auto get_pointer_to_cb_data(uint32_t cb_id, uint32_t tile_index) -> To* {
    // Offset to skip metadata (in uint32_t units)
    constexpr uint32_t cb_data_offset = 4;

    uint32_t address = 0;

    // UNPACK thread: read from CB and distribute address to MATH/PACK via mailbox
    UNPACK((llk_unpack_get_tile<false, false>(cb_id, tile_index, &address)))
    UNPACK(address = address + cb_data_offset * sizeof(uint32_t))
    UNPACK((mailbox_write(ckernel::ThreadId::MathThreadId, address)))
    UNPACK((mailbox_write(ckernel::ThreadId::PackThreadId, address)))

    // MATH and PACK threads: receive address from UNPACK via mailbox
    MATH(address = mailbox_read(ckernel::ThreadId::UnpackThreadId))
    PACK(address = mailbox_read(ckernel::ThreadId::UnpackThreadId))

    return reinterpret_cast<To*>(address);
}
```

### Testing

```bash
# LayerNorm tests (42 passed)
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_layernorm.py -v

# GroupNorm tests (85 passed, 6 skipped)
pytest tests/ttnn/unit_tests/operations/fused/test_group_norm.py -v

# Run full normalization test suite
pytest tests/ttnn/unit_tests/operations/normalization/ -v
```

---

## Removal of cb_get_tile API ✅ COMPLETED

The `cb_get_tile` and `cb_release_tile` functions have been **removed** from `tt_metal/include/compute_kernel_api/cb_api.h`.

These APIs had race conditions due to Tensix semaphore limitations (see GitHub issue #27979).
All usages have been migrated to use mailbox-based communication instead.

---

## Implementation Order (All Complete)

1. ✅ **Phase 1**: Migrate `sdpa_flash_decode.cpp`
2. ✅ **Phase 2**: Migrate `embedding_backward.cpp`
3. ✅ **Phase 3**: Migrate `memory.h` utility (affects multiple normalization ops)
4. ✅ **Phase 4**: Remove `cb_get_tile` and `cb_release_tile` APIs

---

## Reference Implementation

The matmul sparse kernel provides a working reference:

**Dataflow kernel** (`reader_bmm_tile_layout_in0_sender_padding.cpp:174-176`):
```cpp
ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, is_batch_valid);
ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, is_batch_valid);
ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, is_batch_valid);
```

**Compute kernel** (`bmm_large_block_zm_fused_bias_activation.cpp:147-149`):
```cpp
UNPACK(is_batch_valid = (bool)mailbox_read(ckernel::ThreadId::BriscThreadId);)
MATH(is_batch_valid = (bool)mailbox_read(ckernel::ThreadId::BriscThreadId);)
PACK(is_batch_valid = (bool)mailbox_read(ckernel::ThreadId::BriscThreadId);)
```

---

## Key Constraints

1. **BRISC-only mailbox writes**: Only BRISC can write to mailboxes, NCRISC cannot
2. **Ordering**: All `mailbox_write` calls must complete before corresponding `mailbox_read`
3. **Single value**: All threads must receive identical values (no L1 sampling variance)
4. **Thread IDs**: Use `ckernel::ThreadId::BriscThreadId`, `UnpackThreadId`, `MathThreadId`, `PackThreadId`

---

## Validation Checklist

For each migration:

- [ ] Identify dataflow kernel RISC type (BRISC vs NCRISC)
- [ ] Choose appropriate solution (BRISC mailbox vs UNPACK distribution)
- [ ] Update dataflow kernel with `mailbox_write` calls
- [ ] Update compute kernel with `mailbox_read` calls
- [ ] Remove `cb_get_tile` and `cb_release_tile` calls
- [ ] Test on Wormhole B0
- [ ] Test on Blackhole
- [ ] Verify no hangs without watcher enabled
- [ ] Run full CI test suite
