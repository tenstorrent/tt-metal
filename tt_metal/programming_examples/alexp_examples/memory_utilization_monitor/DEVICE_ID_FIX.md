# Device ID Corruption Fix

## Problem

When running multi-device applications (e.g., `tt-metalium-validation-basic` with 8 devices), the allocation server was reporting corrupted device IDs (14, 32, 33, 50-77, 68, etc.) instead of the expected device IDs (0-7).

## Root Cause

The original implementation attempted to get the device ID by calling:
```cpp
buffer->device()->id()
```

This approach had issues:
1. The `buffer->device()` pointer might not be fully initialized at the time of the call
2. In multi-threaded/multi-device environments, accessing the device through the buffer could lead to race conditions or memory corruption
3. The Allocator doesn't have a direct reference to its owning device

## Solution

Store the device ID directly in the Allocator during device initialization:

### Changes Made

1. **allocator.hpp** - Added device ID storage:
   ```cpp
   private:
       int device_id_ = -1;  // Device ID for allocation tracking

   public:
       void set_device_id(int device_id) { device_id_ = device_id; }
   ```

2. **allocator.cpp** - Use stored device ID:
   ```cpp
   if (AllocationClient::is_enabled() && device_id_ >= 0) {
       AllocationClient::report_allocation(
           device_id_,  // Use stored ID, not buffer->device()->id()
           size,
           static_cast<uint8_t>(buffer_type),
           address
       );
   }
   ```

3. **device.cpp** - Set device ID during initialization:
   ```cpp
   auto allocator = this->initialize_allocator(...);
   allocator->set_device_id(this->id_);  // Set once during initialization
   sub_device_manager_tracker_ = std::make_unique<SubDeviceManagerTracker>(
       this, std::move(allocator), sub_devices);
   ```

## Benefits

1. **Thread-safe**: Device ID is set once during initialization, no runtime access needed
2. **No corruption**: Device ID comes directly from the Device object during setup
3. **Simple**: No need to navigate pointer chains during allocation
4. **Reliable**: Works correctly in multi-device environments

## Testing

After applying this fix, rebuild TT-Metal:
```bash
cd /home/tt-metal-apv
cmake --build build --target impl -j8
```

Then test with multi-device applications:
```bash
# Terminal 1: Start allocation server
./tt_metal/programming_examples/alexp_examples/memory_utilization_monitor/allocation_server_poc

# Terminal 2: Run multi-device test
./build/test/tt_metal/dispatch/test_dispatch --gtest_filter="*"

# Terminal 3: Monitor all devices
./tt_metal/programming_examples/alexp_examples/memory_utilization_monitor/allocation_monitor_client -a -r 500
```

You should now see correct device IDs (0-7) instead of corrupted values.
