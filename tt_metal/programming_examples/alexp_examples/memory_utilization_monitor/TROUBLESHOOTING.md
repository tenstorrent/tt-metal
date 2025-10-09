# Allocation Tracking Troubleshooting Guide

## Problem: No Allocations Showing in Server

### Symptom
The `allocation_server_poc` is running but shows:
```
📊 Current Statistics:
  Active allocations: 0
```

Even when running applications that allocate memory.

### Root Cause
**Allocation tracking is DISABLED by default** and must be explicitly enabled via environment variable.

### Solution

**You MUST set this environment variable before running your application:**

```bash
export TT_ALLOC_TRACKING_ENABLED=1
```

## Complete Testing Workflow

### Step 1: Start the Allocation Server
```bash
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor
./allocation_server_poc
```

### Step 2: Start the Monitor (Optional)
In another terminal:
```bash
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor

# Monitor single device
./allocation_monitor_client -d 0 -r 500

# OR monitor multiple devices
./allocation_monitor_client -d 0 -d 1 -d 2 -r 500

# OR monitor all devices
./allocation_monitor_client -a -r 500
```

### Step 3: Run Your Application WITH Tracking Enabled

#### For Python Scripts:
```bash
# Set environment variable first!
export TT_ALLOC_TRACKING_ENABLED=1

# Then run your script
python your_script.py
```

#### For the nlp_concat_heads example:
```bash
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/nlp_concat_heads_boltz_example

# Use the helper script (tracking already enabled)
./run_with_tracking.sh

# OR manually:
export TT_ALLOC_TRACKING_ENABLED=1
python python_nlp_concat_heads_boltz.py
```

#### For C++ Tests:
```bash
export TT_ALLOC_TRACKING_ENABLED=1
./build/test/tt_metal/dispatch/test_dispatch
```

## Verification Checklist

✅ **Server is running**: `allocation_server_poc` shows "Listening on: /tmp/tt_allocation_server.sock"

✅ **Environment variable is set**:
```bash
echo $TT_ALLOC_TRACKING_ENABLED
# Should output: 1
```

✅ **TT-Metal is rebuilt**: After applying device ID fixes, you must rebuild:
```bash
cd /home/tt-metal-apv
cmake --build build --target impl -j8
```

✅ **Application is running**: Your Python/C++ application should be actively allocating memory

## Expected Output

When working correctly, the server should show:
```
✓ [PID 12345] Allocated 524288 bytes of DRAM on device 0 (buffer_id=2560032)

📊 Current Statistics:
  Device 0:
    Buffers: 1
    DRAM: 524288 bytes
    L1: 0 bytes
    Total: 524288 bytes
  Active allocations: 1
```

## Common Issues

### Issue 1: "No allocations" even with TT_ALLOC_TRACKING_ENABLED=1

**Possible causes:**
1. TT-Metal library not rebuilt after applying fixes
2. Server not running or crashed
3. Socket file `/tmp/tt_allocation_server.sock` has wrong permissions

**Solution:**
```bash
# Rebuild TT-Metal
cd /home/tt-metal-apv
cmake --build build --target impl -j8

# Clean socket file and restart server
rm -f /tmp/tt_allocation_server.sock
./allocation_server_poc
```

### Issue 2: Corrupted Device IDs (14, 32, 50-77, etc.)

**This was fixed** by storing device IDs in the Allocator. Make sure you:
1. Applied the changes to `allocator.hpp`, `allocator.cpp`, `device.cpp`, and `sub_device_manager.cpp`
2. Rebuilt the `impl` target

### Issue 3: Monitor shows nothing but server shows allocations

**Possible causes:**
1. Monitor is watching the wrong device
2. Monitor client not built with latest changes

**Solution:**
```bash
# Check what device IDs the server is reporting
# Look at server output for "device X"

# Monitor that specific device
./allocation_monitor_client -d X -r 500

# Or monitor all devices
./allocation_monitor_client -a -r 500
```

## Quick Test

Run this quick test to verify everything works:

```bash
# Terminal 1
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor
./allocation_server_poc

# Terminal 2 (in another window)
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor
export TT_ALLOC_TRACKING_ENABLED=1
python test_ttnn_allocations.py
```

You should see allocations appear in Terminal 1 within seconds.

## Environment Variable Details

The `TT_ALLOC_TRACKING_ENABLED` environment variable is checked in:
- **File**: `/home/tt-metal-apv/tt_metal/impl/allocator/allocation_client.cpp`
- **Line**: 43-49
- **Behavior**:
  - If set to "1", allocation tracking is enabled
  - If unset or any other value, tracking is disabled (silent, no overhead)

This design ensures zero performance impact when tracking is not needed.
