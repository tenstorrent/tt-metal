# Why TTNN Python Tests Don't Show Allocations (And How to Fix It)

## The Problem

When you run Python TTNN tests like `test_ttnn_allocations.py`, allocations don't show up in the allocation monitor, even though:
- ✅ `TT_ALLOC_TRACKING_ENABLED=1` is set
- ✅ The allocation server is running
- ✅ C++ tests like `test_tracking_cpp` work fine

## Root Cause

**Python bindings are NOT automatically rebuilt when C++ code changes!**

The allocation tracking code we added is in C++ files:
- `/home/tt-metal-apv/tt_metal/impl/allocator/allocator.cpp`
- `/home/tt-metal-apv/tt_metal/impl/allocator/allocation_client.cpp`
- `/home/tt-metal-apv/tt_metal/impl/device/device.cpp`

When Python calls `ttnn.from_torch()` or creates tensors, it goes through:
```
Python → _ttnn.so (Python bindings) → libtt_metal.so (C++ library) → Allocator
```

If `_ttnn.so` was built BEFORE we added allocation tracking, it's using the OLD version of `libtt_metal.so` that doesn't have our tracking code!

## The Solution

### Option 1: Full Rebuild (Recommended)
Rebuild the entire project including Python bindings:

```bash
cd /home/tt-metal-apv
./build_metal.sh
```

This will:
1. Rebuild all C++ libraries with allocation tracking
2. Rebuild Python bindings (`_ttnn.so`) to link against the new libraries
3. Ensure everything is in sync

### Option 2: Quick Python Rebuild
If you only changed allocator code and want a faster rebuild:

```bash
cd /home/tt-metal-apv
cmake --build build_Release_tracy --target _ttnncpp -j8
```

### Option 3: Check Which Build is Being Used
The system might be using libraries from a different build directory:

```bash
# Check which build directory has the Python bindings
ls -la build*/ttnn/_ttnncpp.so

# The one with the most recent timestamp is being used
# Make sure it matches where you're building!
```

## How to Verify It's Working

### 1. Start the allocation server:
```bash
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor
./allocation_server_poc
```

### 2. In another terminal, run a Python test:
```bash
cd /home/tt-metal-apv
source env_vars_setup.sh
export TT_ALLOC_TRACKING_ENABLED=1
cd tt_metal/programming_examples/alexp_examples/memory_utilization_monitor
python test_ttnn_allocations.py
```

### 3. Check the server output
You should see:
```
✓ [PID xxxxx] Allocated 104857600 bytes of DRAM on device 0 (buffer_id=...)
✓ [PID xxxxx] Allocated 10485760 bytes of L1 on device 0 (buffer_id=...)
...
✗ [PID xxxxx] Freed buffer ... on device 0 (...)
```

## Common Issues

### Issue 1: "No allocations showing"
**Cause**: Python bindings not rebuilt after C++ changes
**Fix**: Run `./build_metal.sh`

### Issue 2: "Wrong build directory"
**Cause**: Building in `/build/` but system uses `/build_Release_tracy/`
**Fix**:
```bash
# Find which directory has Python bindings
find /home/tt-metal-apv -name "_ttnncpp.so" -type f 2>/dev/null

# Build in the correct directory
cd /home/tt-metal-apv
cmake --build build_Release_tracy -j8
```

### Issue 3: "C++ tests work but Python doesn't"
**Cause**: C++ executables link directly to `libtt_metal.so`, but Python uses cached `_ttnncpp.so`
**Fix**: Rebuild Python bindings specifically:
```bash
cd /home/tt-metal-apv
cmake --build build_Release_tracy --target _ttnncpp -j8
```

## Why C++ Tests Work Immediately

C++ tests like `test_tracking_cpp` and `test_mesh_allocation_cpp` are **executables** that link directly to the libraries at build time. When you rebuild them with `cmake --build`, they immediately get the new code.

Python tests use **shared libraries** (`_ttnncpp.so`) that were built separately and may not be automatically rebuilt when you change C++ source files.

## Quick Reference

| Test Type | Requires Rebuild? | Command |
|-----------|------------------|---------|
| C++ (`test_tracking_cpp`) | Yes, but automatic | `cmake --build build_Release_tracy --target test_tracking_cpp` |
| Python (`test_ttnn_allocations.py`) | Yes, MANUAL! | `./build_metal.sh` or `cmake --build build_Release_tracy --target _ttnncpp` |

## Summary

**Always rebuild Python bindings after modifying C++ allocator code:**
```bash
cd /home/tt-metal-apv
./build_metal.sh
```

Then your Python TTNN tests will show allocations in the monitor!
