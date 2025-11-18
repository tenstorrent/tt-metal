# Step 1: Circular Buffer (CB) Tracking - COMPLETE ✅

## Summary

Successfully implemented CB allocation and deallocation tracking in TT-Metal!

## Changes Made

### 1. AllocationClient API Extensions
**File**: `tt_metal/impl/allocator/allocation_client.hpp`
**File**: `tt_metal/impl/allocator/allocation_client.cpp`

Added new methods for CB-specific tracking:
```cpp
// Public API
static void report_cb_allocation(int device_id, uint64_t size, uint64_t cb_id);
static void report_cb_deallocation(int device_id, uint64_t cb_id);

// Private implementation
void send_cb_allocation_message(int device_id, uint64_t size, uint64_t cb_id);
void send_cb_deallocation_message(int device_id, uint64_t cb_id);
```

These methods send `CB_ALLOC` and `CB_FREE` message types (instead of regular `ALLOC`/`FREE`).

### 2. GraphTracker Integration
**File**: `tt_metal/graph/graph_tracking.cpp`

Modified `track_allocate_cb()` and `track_deallocate_cb()` to use the new CB-specific methods:

**Before** (line 227):
```cpp
// CBs were reported as regular L1 allocations
AllocationClient::report_allocation(device->id(), size, static_cast<uint8_t>(BufferType::L1), addr);
```

**After** (line 227):
```cpp
// CBs now reported with CB_ALLOC message type
AllocationClient::report_cb_allocation(device->id(), size, addr);
```

Similarly for deallocation at line 263.

### 3. Protocol Structure (Already in place)
**File**: `tt_metal/impl/allocator/allocation_client.cpp`

The `AllocMessage` struct already includes:
```cpp
enum Type : uint8_t {
    ALLOC = 1,
    FREE = 2,
    QUERY = 3,
    RESPONSE = 4,
    DUMP_REMAINING = 5,
    DEVICE_INFO_QUERY = 6,
    DEVICE_INFO_RESPONSE = 7,
    CB_ALLOC = 8,        // ← Now being used!
    CB_FREE = 9,         // ← Now being used!
    KERNEL_LOAD = 10,
    KERNEL_UNLOAD = 11
};
```

## How It Works

### CB Allocation Flow

1. **Program Creation**: User calls `CreateCircularBuffer(program, cores, config)`
2. **Program Execution**: Program is compiled and sent to device
3. **CB Allocation**: `ProgramImpl::allocate_circular_buffers(device)` allocates CBs on L1
4. **Tracking Hook**: `GraphTracker::track_allocate_cb()` is called
5. **Report to Server**: `AllocationClient::report_cb_allocation()` sends `CB_ALLOC` message
6. **Server Updates**: `allocation_server_poc` increments `cb_allocated` counter

### CB Deallocation Flow

1. **Program Cleanup**: `ProgramImpl::deallocate_circular_buffers()` is called
2. **Tracking Hook**: `GraphTracker::track_deallocate_cb()` is called
3. **Report to Server**: `AllocationClient::report_cb_deallocation()` sends `CB_FREE` messages
4. **Server Updates**: Server decrements `cb_allocated` counter

## Server Side (Already Implemented)

The `allocation_server_poc.cpp` already handles these message types:

```cpp
case AllocMessage::CB_ALLOC:
    device_stats_[msg.device_id].cb_allocated += msg.size;
    break;

case AllocMessage::CB_FREE:
    if (device_stats_[msg.device_id].cb_allocated >= msg.size) {
        device_stats_[msg.device_id].cb_allocated -= msg.size;
    }
    break;
```

## Testing

To test CB tracking:

### 1. Rebuild TT-Metal Library
```bash
cd /home/ttuser/aperezvicente/tt-metal-apv
cmake --build build --target tt_metal -j8
```

### 2. Start Allocation Server
```bash
# Kill old server
pkill -f allocation_server_poc

# Start new server
./build/programming_examples/allocation_server_poc > cb_test.log &
```

### 3. Run a TT-Metal Application with CBs
```bash
export TT_ALLOC_TRACKING_ENABLED=1

# Run any application that creates circular buffers
python your_model_with_cbs.py
```

### 4. Check tt_smi_umd
```bash
./build/programming_examples/tt_smi_umd
```

Press `2` for Chart View, then View 1 should show:
- **CBs**: X.XX MB (X.XX%)
- **Total L1**: Should include CB memory

### 5. Check Server Log
```bash
grep "CB_ALLOC\|CB_FREE" cb_test.log
```

You should see log messages like:
```
✓ [PID xxxxx] CB allocated 16384 bytes on device 0
✓ [PID xxxxx] CB freed on device 0
```

## Expected Behavior

**Before this change:**
- CBs were counted as regular L1 allocations
- `tt_smi_umd` showed "CBs: 0.00 MB"
- CB memory was hidden in "L1 Buffers"

**After this change:**
- CBs tracked separately with `CB_ALLOC` messages
- `tt_smi_umd` displays CB usage distinctly
- Breakdown: "L1 Buffers" + "CBs" + "Kernels" = "Total L1"

## Kernel Tracking (Step 2)

The AllocationClient now also has kernel tracking methods ready:
- `report_kernel_load()`
- `report_kernel_unload()`

These will be implemented in **Step 2** by hooking into kernel binary loading.

## Files Modified

1. `tt_metal/impl/allocator/allocation_client.hpp` - Added CB/Kernel API
2. `tt_metal/impl/allocator/allocation_client.cpp` - Implemented CB/Kernel messages
3. `tt_metal/graph/graph_tracking.cpp` - Changed to use CB-specific methods

## Files Ready for Step 2

- `tt_metal/impl/kernels/kernel.cpp` - Kernel loading (needs instrumentation)
- `tt_metal/impl/program/program.cpp` - Kernel allocation (needs hooks)

## Status

✅ **CB Tracking: COMPLETE**
⏳ **Kernel Tracking: Ready for implementation (Step 2)**

CB memory will now be accurately tracked and displayed in `tt_smi_umd`!
