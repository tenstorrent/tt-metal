# Quick Start - Allocation Tracking

## What You Need to Know

### 1. The Fix is Applied ✅
- Device ID corruption is fixed in the C++ code
- `impl` target has been rebuilt
- POC server/client don't need changes

### 2. The Missing Step ⚠️
**Python ttnn bindings need to be rebuilt** to use the fixed C++ library.

## To Make Python Scripts Work

Run this command (it will take 10-15 minutes):
```bash
cd /home/tt-metal-apv
./build_metal.sh
```

## After Rebuild - How to Use

### Step 1: Start Server
```bash
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor
./allocation_server_poc
```

### Step 2: Enable Tracking & Run Your Script
```bash
# CRITICAL: Set this environment variable!
export TT_ALLOC_TRACKING_ENABLED=1

# Then run your Python script
python your_script.py
```

### Step 3: Monitor (Optional)
In another terminal:
```bash
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor

# Single device
./allocation_monitor_client -d 0 -r 500

# Multiple devices
./allocation_monitor_client -d 0 -d 1 -d 2 -r 500

# All devices
./allocation_monitor_client -a -r 500
```

## Example: Test the nlp_concat_heads Script

```bash
# Terminal 1: Server
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor
./allocation_server_poc

# Terminal 2: Your script
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/nlp_concat_heads_boltz_example
export TT_ALLOC_TRACKING_ENABLED=1
python python_nlp_concat_heads_boltz.py
```

You should see allocations appear in Terminal 1!

## Why It Wasn't Working Before

1. ❌ **Device ID corruption** - Fixed by storing device ID in allocator
2. ❌ **Sub-device allocators** - Fixed by setting device ID on all allocators
3. ❌ **Tracking disabled** - Must set `TT_ALLOC_TRACKING_ENABLED=1`
4. ❌ **Python not rebuilt** - Old bindings don't have the fixes

## Summary of Changes Made

| File | Change | Why |
|------|--------|-----|
| `allocator.hpp` | Added `device_id_` member | Store device ID directly |
| `allocator.cpp` | Use `device_id_` instead of `buffer->device()->id()` | Avoid corrupted pointer access |
| `device.cpp` | Call `allocator->set_device_id()` | Initialize device ID |
| `sub_device_manager.cpp` | Set device ID on sub-allocators | Fix sub-device tracking |
| `allocation_monitor_client.cpp` | Multi-device support | Monitor multiple devices at once |

## Files That DON'T Need Changes

- ✅ `allocation_server_poc.cpp` - Standalone, protocol unchanged
- ✅ `allocation_client_demo.cpp` - Standalone, protocol unchanged
- ✅ `allocation_client.py` - Protocol unchanged
- ✅ `allocation_client.hpp/cpp` - Already correct

## Next Steps

1. **Rebuild Python**: `cd /home/tt-metal-apv && ./build_metal.sh`
2. **Test**: Use the example above
3. **Verify**: You should see correct device IDs (0-7) and allocations in the server

## Need Help?

See `TROUBLESHOOTING.md` for detailed debugging steps.
