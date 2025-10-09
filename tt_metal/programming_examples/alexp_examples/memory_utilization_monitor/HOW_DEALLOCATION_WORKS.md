# How Deallocation Tracking Works

## Overview

The allocation tracking system correctly handles deallocations across multiple devices. Here's how it works:

## Allocation Flow

1. **Application allocates buffer** on Device X
2. **Allocator calls** `AllocationClient::report_allocation(device_id=X, size, buffer_type, buffer_id=address)`
3. **Server receives** allocation message and stores:
   ```cpp
   allocations_[buffer_id] = {
       device_id: X,
       size: size,
       buffer_type: buffer_type,
       ...
   }
   ```
4. **Server updates** `device_stats_[X]` to add the allocation

## Deallocation Flow

1. **Application deallocates buffer** (calls `buffer.reset()` or `buffers.clear()`)
2. **Allocator calls** `AllocationClient::report_deallocation(buffer_id=address)`
   - **Note**: Only the buffer_id (address) is sent, NOT the device_id
3. **Server receives** deallocation message with just `buffer_id`
4. **Server looks up** the buffer_id in `allocations_` map:
   ```cpp
   auto it = allocations_.find(msg.buffer_id);
   if (it != allocations_.end()) {
       const auto& info = it->second;
       // info contains the original device_id, size, buffer_type
       auto& stats = device_stats_[info.device_id];  // Uses stored device_id!
       stats.num_buffers--;
       // Subtract from correct device's stats
   }
   ```
5. **Server updates** the CORRECT device's stats based on the stored information

## Why This Works

- Each buffer has a unique address (buffer_id)
- The server remembers which device each buffer belongs to
- When deallocating, the server uses the stored device_id from the allocation
- **Deallocations are correctly attributed to the right device**

## Common Misconception

❌ **Wrong**: "Deallocation only reports to one device because we don't pass device_id"
✅ **Correct**: "Deallocation correctly updates each device because the server remembers which device each buffer belongs to"

## Verification

To verify deallocations are working correctly:

1. Start the allocation server:
   ```bash
   ./allocation_server_poc
   ```

2. In another terminal, start the monitor:
   ```bash
   ./allocation_monitor_client -a -r 500
   ```

3. Run the mesh test:
   ```bash
   export TT_ALLOC_TRACKING_ENABLED=1
   ./build_Release_tracy/programming_examples/test_mesh_allocation_cpp
   ```

4. **Watch the monitor**: You should see:
   - Allocations appear on ALL 8 devices (4, 0, 3, 7, 5, 1, 2, 6)
   - Memory increases on each device as buffers are allocated
   - Memory decreases on EACH device as buffers are deallocated
   - Memory returns to baseline on ALL devices at the end

## Code References

- **Allocation tracking**: `/home/tt-metal-apv/tt_metal/impl/allocator/allocator.cpp:128-143`
- **Deallocation tracking**: `/home/tt-metal-apv/tt_metal/impl/allocator/allocator.cpp:157-176`
- **Server allocation handler**: `allocation_server_poc.cpp:handle_allocation()`
- **Server deallocation handler**: `allocation_server_poc.cpp:handle_deallocation()`

## Summary

✅ The C++ mesh test correctly:
- Allocates buffers on all 8 devices
- Reports allocations with correct device IDs
- Deallocates buffers from all devices
- Server correctly tracks and updates stats for each device

The system is working as designed!
