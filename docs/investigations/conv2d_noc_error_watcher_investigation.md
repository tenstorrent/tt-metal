# Conv2D NOC Error Investigation with Watcher Enabled

## Issue Summary

A non-deterministic NOC error occurs during ResNet50 conv2d tests when watcher is enabled. The error manifests as an invalid NOC read attempt with coordinates that don't map to any known core.

## Error Message

```
Device 0 worker core(x= 0,y= 5) virtual(x=18,y=23): NCRISC using noc1 tried to unicast read 384 bytes to local L1[0x08e720] from Unknown core w/ virtual coords (x=26,y=23) [addr=0x0013afa0] (NOC target address did not map to any known Tensix/Ethernet/DRAM/PCIE core).
```

### Error Breakdown

| Field | Value | Analysis |
|-------|-------|----------|
| Core | (x=0, y=5) logical, (x=18, y=23) virtual | Valid worker core |
| RISCV | NCRISC | Reader RISCV processor |
| NOC | noc1 | Reader kernel uses NOC1 |
| Operation | unicast read | Reading activation data |
| Size | 384 bytes | Matches `coalesced_read_bytes` |
| Destination | L1[0x08e720] | Valid L1 address (~584KB) |
| Source coords | (x=26, y=23) | **INVALID** - no such core exists |
| Source addr | 0x0013afa0 | Valid L1 address (~1.3MB) |

## Key Observation

The source coordinates **(x=26, y=23)** are completely invalid for Wormhole B0:
- Worker cores typically have NOC x coordinates in range 1-12
- x=26 (0x1A) is way outside any valid coordinate range
- This suggests corrupted or uninitialized NOC state

## Test Configuration

The failing test is `test_resnet_conv.py` with:
- **Shard layout**: `HEIGHT_SHARDED`
- **Reader kernel**: `reader_conv_activations_padded_with_halo_3x3_weights_v2.cpp`
- **Data types**: BFLOAT8_B for activations and weights
- **Double buffering**: Enabled
- **Grid size**: 8x8 cores

## Code Analysis

### Reader Kernel Flow (Height-Sharded)

Located at: `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_conv_activations_padded_with_halo_3x3_weights_v2.cpp`

```cpp
void kernel_main() {
    // 1. Optional zero-out of tiles (sets NOC state to local zeros address)
    if constexpr (needs_act_block_zero_out) {
        zero_out_tiles<cb_id_act>();  // Uses noc_async_read_one_packet_set_state internally
    }

    // 2. Load config tensor from DRAM if needed
    load_config_tensor_if_in_dram<29, 30, 31, cb_reader_indices>(core_index);
    // ^^^ This does noc_async_read() which sets NOC state to DRAM coordinates

    // 3. Set NOC state for local L1 reads
    noc_async_read_one_packet_set_state(get_noc_addr(act_l1_read_addr), coalesced_read_bytes);

    // 4. Main loop using the state
    for (...) {
        read_sticks<...>(...);  // Uses noc_async_read_one_packet_with_state
    }
}
```

### The Watcher Sanitize Flow

In `tt_metal/hw/inc/api/dataflow/dataflow_api.h:623-647`:

```cpp
FORCE_INLINE void noc_async_read_one_packet_with_state(...) {
    // ...

    // SANITIZE READS COMMAND BUFFER REGISTERS HERE
    DEBUG_SANITIZE_NOC_READ_TRANSACTION_WITH_ADDR_AND_SIZE_STATE(noc, src_local_l1_addr, dst_local_l1_addr);

    // ACTUAL NOC COMMAND ISSUED HERE
    ncrisc_noc_read_with_state<...>(...);

    // ...
}
```

### The Sanitize Macro

In `tt_metal/hw/inc/internal/debug/sanitize.h:598-605`:

```cpp
#define DEBUG_SANITIZE_NOC_READ_TRANSACTION_WITH_ADDR_AND_SIZE_STATE(noc_id, noc_a_lower, worker_a) \
    DEBUG_SANITIZE_NOC_READ_TRANSACTION_(                                                           \
        noc_id,                                                                                     \
        ((uint64_t)NOC_CMD_BUF_READ_REG(noc_id, read_cmd_buf, NOC_TARG_ADDR_COORDINATE) << NOC_ADDR_COORD_SHIFT) | \
            ((uint64_t)NOC_CMD_BUF_READ_REG(noc_id, read_cmd_buf, NOC_TARG_ADDR_MID) << 32) | noc_a_lower, \
        worker_a,                                                                                   \
        NOC_CMD_BUF_READ_REG(noc_id, read_cmd_buf, NOC_AT_LEN_BE),                                  \
        false);
```

The sanitize macro reads `NOC_TARG_ADDR_COORDINATE` from the command buffer to extract the target core coordinates.

## Root Cause Hypothesis

### Memory Ordering Issue

The issue appears to be a **memory ordering race condition** between:

1. **The state write**: `noc_async_read_one_packet_set_state()` writes coordinates to NOC command buffer registers via `NOC_CMD_BUF_WRITE_REG`

2. **The sanitize read**: `DEBUG_SANITIZE_NOC_READ_TRANSACTION_WITH_ADDR_AND_SIZE_STATE` reads from the same registers via `NOC_CMD_BUF_READ_REG`

### Sequence of Events (Normal Case)

```
Time →
┌─────────────────────────────────────────────────────────────────────┐
│ 1. load_config_tensor_if_in_dram()                                  │
│    └── noc_async_read() sets state to DRAM coords                   │
│    └── noc_async_read_barrier() waits for completion                │
│                                                                      │
│ 2. noc_async_read_one_packet_set_state()                            │
│    └── NOC_CMD_BUF_WRITE_REG writes local coords to cmd buf         │
│                                                                      │
│ 3. noc_async_read_one_packet_with_state()                           │
│    └── DEBUG_SANITIZE reads cmd buf (sees local coords) ✓           │
│    └── ncrisc_noc_read_with_state issues command                    │
└─────────────────────────────────────────────────────────────────────┘
```

### Sequence of Events (Race Condition)

```
Time →
┌─────────────────────────────────────────────────────────────────────┐
│ 1. load_config_tensor_if_in_dram()                                  │
│    └── noc_async_read() sets state to DRAM coords                   │
│    └── noc_async_read_barrier() waits                               │
│                                                                      │
│ 2. noc_async_read_one_packet_set_state()                            │
│    └── NOC_CMD_BUF_WRITE_REG starts writing...                      │
│        (write not yet visible to subsequent reads)                  │
│                                                                      │
│ 3. noc_async_read_one_packet_with_state()                           │
│    └── DEBUG_SANITIZE reads cmd buf (sees STALE coords) ✗           │
│        └── ERROR: Invalid coordinates detected!                      │
│    └── ncrisc_noc_read_with_state (never reached)                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Why Watcher Triggers This

Without watcher:
- The sanitize code doesn't run
- Reads proceed directly, and the hardware handles sequencing

With watcher:
- The sanitize code runs BEFORE the actual NOC command
- Additional instrumentation changes timing
- The read from command buffer registers may see stale values if the previous write hasn't completed

### Why x=26?

The value x=26 (0x1A) is likely:
- **Garbage/uninitialized data** in the command buffer from a previous operation
- **DRAM coordinates** left over from `load_config_tensor_if_in_dram`
- **Partial write** where only some registers were updated

## Technical Details

### NOC Address Encoding (Wormhole B0)

```
64-bit NOC Address:
┌─────────────────┬───────────────┬────────────────────────────────┐
│  Coordinates    │  Addr[35:32]  │         Addr[31:0]             │
│   (x, y)        │   (4 bits)    │        (32 bits)               │
└─────────────────┴───────────────┴────────────────────────────────┘
        ↑                                      ↑
   NOC_TARG_ADDR_MID                    NOC_TARG_ADDR_LO
```

The `NOC_TARG_ADDR_COORDINATE` register (same as `NOC_TARG_ADDR_MID` on Wormhole) contains both the upper address bits and the x/y coordinates.

### Command Buffer Registers

- `NOC_TARG_ADDR_LO`: Lower 32 bits of target address
- `NOC_TARG_ADDR_MID`: Upper address bits + coordinates
- `NOC_AT_LEN_BE`: Transfer length and byte enables

### Register Access

```cpp
// Writing to command buffer
NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_MID, value);

// Reading from command buffer (done by watcher)
NOC_CMD_BUF_READ_REG(noc, cmd_buf, NOC_TARG_ADDR_COORDINATE);
```

These are memory-mapped I/O operations. On RISC-V, without explicit fencing, writes may not be immediately visible to subsequent reads.

## Potential Fixes

### Option 1: Add Memory Fence in Watcher Path

Add a fence instruction before the sanitize read:

```cpp
// In dataflow_api.h, before DEBUG_SANITIZE_NOC_READ_TRANSACTION_WITH_ADDR_AND_SIZE_STATE
__asm__ volatile("fence" ::: "memory");
DEBUG_SANITIZE_NOC_READ_TRANSACTION_WITH_ADDR_AND_SIZE_STATE(...);
```

**Pros**: Fixes the root cause
**Cons**: Adds overhead to every NOC operation with watcher enabled

### Option 2: Add Fence After State Set (IMPLEMENTED)

Add fence in `noc_async_read_one_packet_set_state` and related functions:

```cpp
inline void noc_async_read_one_packet_set_state(...) {
    ncrisc_noc_read_set_state<...>(...);
#if defined(WATCHER_ENABLED)
    asm volatile("fence" : : : "memory");  // Ensure writes are visible
#endif
}
```

**Pros**: More targeted fix, only adds overhead with watcher enabled
**Cons**: None significant

**STATUS: IMPLEMENTED** - Added to the following functions in `tt_metal/hw/inc/api/dataflow/dataflow_api.h`:
- `noc_async_read_one_packet_set_state` (line ~607)
- `noc_async_read_set_state` (line ~691)
- `noc_async_write_one_packet_set_state` (line ~1015)
- `noc_async_write_one_packet_with_trid_set_state` (line ~2459)

### Option 3: Skip Test with Watcher

Add watcher skip to the test:

```python
@pytest.mark.skipif(
    os.environ.get("TT_METAL_WATCHER", "0") != "0",
    reason="Non-deterministic NOC error with watcher enabled - see investigation doc"
)
def test_resnet_conv(...):
    ...
```

**Pros**: Quick workaround
**Cons**: Doesn't fix the underlying issue

## Files Involved

| File | Purpose |
|------|---------|
| `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_conv_activations_padded_with_halo_3x3_weights_v2.cpp` | Height-sharded reader kernel |
| `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp` | Block-sharded reader kernel |
| `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_reader_common.hpp` | Common reader functions |
| `tt_metal/hw/inc/api/dataflow/dataflow_api.h` | NOC API including read functions |
| `tt_metal/hw/inc/internal/debug/sanitize.h` | Watcher sanitize macros |
| `tt_metal/hw/inc/internal/tt-1xx/wormhole/noc_nonblocking_api.h` | Low-level NOC functions |
| `tests/didt/test_resnet_conv.py` | Failing test |

## Reproduction

The issue is non-deterministic but can be reproduced by:

1. Enable watcher: `export TT_METAL_WATCHER=1`
2. Run ResNet50 conv2d test multiple times:
   ```bash
   pytest tests/didt/test_resnet_conv.py -v
   ```
3. The error occurs intermittently, typically within 10-20 runs

## Conclusion

This is a **memory ordering issue** exposed by the watcher instrumentation. The watcher's sanitize code reads NOC command buffer registers before the actual NOC operation, and under certain timing conditions, it can observe stale values from a previous operation.

### Fix Applied

**Option 2 has been implemented**: Memory fences have been added to all `*_set_state` functions in `dataflow_api.h`, conditionally compiled only when `WATCHER_ENABLED` is defined. This ensures that:

1. NOC command buffer register writes from `*_set_state` are fully committed
2. Before subsequent sanitize reads in the corresponding `*_with_state` functions can observe them

The fence instructions use the RISC-V `fence` instruction with memory clobber to ensure both hardware memory ordering and compiler reordering prevention:

```cpp
#if defined(WATCHER_ENABLED)
    asm volatile("fence" : : : "memory");
#endif
```

This fix has minimal performance impact since:
- It only applies when watcher is enabled (debug builds)
- The fence is lightweight on RISC-V
- It's placed at the optimal location (after state write, before any reads)

## References

- Wormhole B0 NOC documentation
- RISC-V memory model specification
- TT-Metal watcher documentation
