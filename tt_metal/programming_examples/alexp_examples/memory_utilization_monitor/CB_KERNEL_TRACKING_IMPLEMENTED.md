# CB + Kernel Tracking - Implementation Complete! ✅

## What Was Implemented

We've successfully added **Circular Buffer and Kernel Code tracking** to the monitoring system!

### Changes Made

#### 1. **AllocMessage Protocol** - Extended with new message types
```cpp
enum Type : uint8_t {
    // ... existing types ...
    CB_ALLOC = 8,        // Circular buffer allocation
    CB_FREE = 9,         // Circular buffer free
    KERNEL_LOAD = 10,    // Kernel loaded
    KERNEL_UNLOAD = 11   // Kernel unloaded
};

// Added fields:
uint64_t cb_allocated;       // Circular buffer memory
uint64_t kernel_allocated;   // Kernel code memory
```

#### 2. **Allocation Server** - Now tracks CB and Kernel memory
- Added `cb_allocated` and `kernel_allocated` to `DeviceStats`
- Added handlers for `CB_ALLOC`, `CB_FREE`, `KERNEL_LOAD`, `KERNEL_UNLOAD`
- Reports CB and Kernel memory in query responses
- Logs CB/Kernel events with size information

#### 3. **tt_smi_umd** - Displays CB and Kernel memory
- Updated `DeviceInfo` struct with `used_cb` and `used_kernel` fields
- Enhanced memory breakdown display:
  ```
  L1 Memory:
    Buffers:   1.5MB     [░░░░░░░░░░░░░░] 0.5%
    CBs:       95.2MB    [████████░░░░░░] 31.1%
    Kernels:   15.3MB    [██░░░░░░░░░░░░]  5.0%
    Total:     112.0MB / 306.0MB [████████░░░░░░] 36.6%
  ```

#### 4. **Instrumented Helpers** - Easy-to-use wrappers
- `instrumented_helpers.hpp` - Ready-to-use wrapper functions
- `CreateCircularBufferWithTracking()` - Auto-reports CB allocations
- `AddKernelWithTracking()` - Auto-reports kernel loading
- Sends IPC messages to allocation server automatically

## How to Use

### Method 1: Using Instrumented Wrappers (Recommended)

In your application code, replace standard calls with instrumented versions:

```cpp
#include "instrumented_helpers.hpp"

// Instead of:
auto cb = CreateCircularBuffer(device, data_format_spec, config);

// Use:
auto cb = tt::tt_metal::instrumented::CreateCircularBufferWithTracking(
    device, data_format_spec, config
);

// Instead of:
auto kernel_handle = program.add_kernel(kernel_config, core_ranges);

// Use:
auto kernel_handle = tt::tt_metal::instrumented::AddKernelWithTracking(
    program, kernel_config, core_ranges
);
```

**That's it!** The wrappers automatically report to the allocation server.

### Method 2: Manual Reporting

If you want more control, send messages directly:

```cpp
#include "instrumented_helpers.hpp"

// Report a CB allocation
tt::tt_metal::instrumented::send_memory_event(
    tt::tt_metal::instrumented::MemoryEventType::CB_ALLOC,
    device_id,
    cb_size,
    cb_index
);

// Report kernel load
tt::tt_metal::instrumented::send_memory_event(
    tt::tt_metal::instrumented::MemoryEventType::KERNEL_LOAD,
    device_id,
    kernel_size,
    kernel_id
);
```

## Testing

### Step 1: Start the Allocation Server

```bash
./build/programming_examples/allocation_server_poc
```

Expected output:
```
🔍 Device detection (using TT-Metal APIs):
   Device 0: Wormhole_B0 (24GB DRAM, 306MB L1)
   ...
🚀 Allocation Server started
   Listening on: /tmp/tt_allocation_server.sock
```

### Step 2: Run Your Application with Instrumented Code

When your app creates CBs or loads kernels, you'll see in the server:

```
✓ [CB_ALLOC] Device 0: +32.0 MB (Total: 32.0 MB)
✓ [CB_ALLOC] Device 0: +16.5 MB (Total: 48.5 MB)
✓ [KERNEL_LOAD] Device 0: +2.5 MB (Total: 2.5 MB)
✓ [KERNEL_LOAD] Device 0: +1.8 MB (Total: 4.3 MB)
...
```

### Step 3: Monitor with tt_smi_umd

```bash
./build/programming_examples/tt_smi_umd -w
```

Expected output:
```
Memory Breakdown:

Device 0 (Wormhole_B0):
----------------------------------------------------------------------
  DRAM:     2.5GB    / 24.0GB    [███░░░░░░░░░░░░░░░░░░░░░]

  L1 Memory:
    Buffers:   1.5MB     [░░░░░░░░░░░░░░░░░░░░]  0.5%
    CBs:       95.2MB    [████████░░░░░░░░░░░░] 31.1%
    Kernels:   15.3MB    [██░░░░░░░░░░░░░░░░░░]  5.0%
    Total:     112.0MB / 306.0MB [████████░░░░░░░░░░░░░░] 36.6%
```

**Now you can see where all your L1 is going!** 🎯

## Example Application

Here's a minimal example showing how to use the instrumented helpers:

```cpp
#include <tt-metalium/host_api.hpp>
#include "instrumented_helpers.hpp"

int main() {
    // Create device
    auto device = tt::tt_metal::CreateDevice(0);

    // Create program
    tt::tt_metal::Program program = CreateProgram();

    // Create circular buffer with tracking
    std::map<uint8_t, tt::DataFormat> data_format_spec = {
        {0, tt::DataFormat::Float16_b}
    };

    tt::tt_metal::CircularBufferConfig cb_config =
        tt::tt_metal::CircularBufferConfig(
            1024 * 2048,  // 2MB
            data_format_spec
        ).set_page_size(0, 2048);

    auto cb = tt::tt_metal::instrumented::CreateCircularBufferWithTracking(
        device, data_format_spec, cb_config
    );

    // Add kernel with tracking
    tt::tt_metal::DataMovementConfig kernel_config = {
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
        .noc = tt::tt_metal::NOC::NOC_0,
        // ... kernel config ...
    };

    CoreRange core_range({0, 0}, {7, 9});  // 80 cores

    auto kernel_handle = tt::tt_metal::instrumented::AddKernelWithTracking(
        program, kernel_config, core_range
    );

    // Run your program
    tt::tt_metal::EnqueueProgram(device->command_queue(), program, false);
    tt::tt_metal::Finish(device->command_queue());

    // Memory is now tracked!
    // Check tt_smi_umd to see the breakdown

    tt::tt_metal::CloseDevice(device);
    return 0;
}
```

## What You'll See

### Before (Only Buffers):
```
Memory Breakdown:
  L1:       1.5MB    / 306.0MB   [░░░░░░░░░░░░░░░░░░░░]
```
**Question:** Where's the rest?! 😕

### After (Complete Picture):
```
L1 Memory:
  Buffers:   1.5MB     [░░░░░░░░░░░░░░░░░░░░]  0.5%
  CBs:       95.2MB    [████████░░░░░░░░░░░░] 31.1%
  Kernels:   15.3MB    [██░░░░░░░░░░░░░░░░░░]  5.0%
  Total:     112.0MB / 306.0MB [████████░░░░░░░░░░░░░░] 36.6%
```
**Answer:** Now you see everything! 🎯

## Benefits

✅ **Complete L1 visibility** - See buffers, CBs, and kernels
✅ **Real-time tracking** - Updates as memory is allocated/freed
✅ **Cross-process** - Works with allocation server architecture
✅ **Easy to use** - Just replace function calls with instrumented versions
✅ **No TT-Metal core changes** - All in your application code
✅ **Accurate sizes** - Reports actual CB and kernel sizes

## Files Modified

1. **`allocation_server_poc.cpp`** - Added CB/Kernel tracking
2. **`tt_smi_umd.cpp`** - Added CB/Kernel display
3. **`instrumented_helpers.hpp`** - New! Wrapper functions

## Next Steps

### To Use in Your Application:

1. Include `instrumented_helpers.hpp` in your code
2. Replace `CreateCircularBuffer` with `CreateCircularBufferWithTracking`
3. Replace `program.add_kernel` with `AddKernelWithTracking`
4. Rebuild and run
5. Watch `tt_smi_umd` show complete L1 breakdown!

### For Even More Detail:

See `FULL_L1_TRACKING_GUIDE.md` for hooking directly into TT-Metal core (automatic for all applications).

## Summary

**What we built:**
- ✅ Protocol extension (CB_ALLOC, KERNEL_LOAD, etc.)
- ✅ Server tracking (aggregates CB/Kernel from all processes)
- ✅ Client display (tt_smi_umd shows breakdown)
- ✅ Easy-to-use wrappers (just include and use)

**What you get:**
- 🎯 Complete L1 memory visibility
- 🎯 Real-time monitoring
- 🎯 Cross-process tracking
- 🎯 Answer to "where's my 306MB?!"

**The mystery is solved!** 🚀
