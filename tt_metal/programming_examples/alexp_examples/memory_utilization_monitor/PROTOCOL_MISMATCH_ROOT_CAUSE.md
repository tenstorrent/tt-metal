# Protocol Mismatch Root Cause & Fix

## Problem Identified

The allocation server was crashing and showing warnings like:
```
⚠️  Invalid device_id 2657333 - ignoring message
⚠️  Too many consecutive errors (10) - closing connection (likely protocol mismatch)
```

## Root Cause

**The client and server were using different `AllocMessage` structures!**

### What Happened

1. **Server was updated** to track CB (Circular Buffer) and Kernel memory:
   - Added `cb_allocated` (8 bytes)
   - Added `kernel_allocated` (8 bytes)
   - Added new message types: `CB_ALLOC`, `CB_FREE`, `KERNEL_LOAD`, `KERNEL_UNLOAD`

2. **Client was NOT updated** - it was still using the old structure:
   - Missing `cb_allocated` field
   - Missing `kernel_allocated` field
   - Missing new message type enums

3. **Result**: When the client sent messages, the server interpreted the bytes incorrectly:
   - Fields were misaligned
   - `device_id` contained garbage (often the process PID!)
   - Server validation correctly rejected these malformed messages

## Files That Needed Updating

### 1. Server (Already Fixed)
- `allocation_server_poc.cpp` ✅
  - Has CB/Kernel tracking
  - Has validation to reject bad messages

### 2. Client (FIXED NOW)
- `tt_metal/impl/allocator/allocation_client.cpp` ✅
  - Updated `AllocMessage` struct to match server
  - Added `cb_allocated` and `kernel_allocated` fields
  - Added new message type enums

## The Fix

Updated `/home/ttuser/aperezvicente/tt-metal-apv/tt_metal/impl/allocator/allocation_client.cpp`:

```cpp
struct __attribute__((packed)) AllocMessage {
    enum Type : uint8_t {
        ALLOC = 1,
        FREE = 2,
        QUERY = 3,
        RESPONSE = 4,
        DUMP_REMAINING = 5,
        DEVICE_INFO_QUERY = 6,
        DEVICE_INFO_RESPONSE = 7,
        CB_ALLOC = 8,        // ← NEW
        CB_FREE = 9,         // ← NEW
        KERNEL_LOAD = 10,    // ← NEW
        KERNEL_UNLOAD = 11   // ← NEW
    };

    // ... existing fields ...

    // Response fields
    uint64_t dram_allocated;
    uint64_t l1_allocated;
    uint64_t l1_small_allocated;
    uint64_t trace_allocated;
    uint64_t cb_allocated;       // ← NEW: 8 bytes
    uint64_t kernel_allocated;   // ← NEW: 8 bytes

    // ... device info fields ...
};
```

## How to Apply the Fix

### Step 1: Rebuild TT-Metal Library

```bash
cd /home/ttuser/aperezvicente/tt-metal-apv
cmake --build build --target tt_metal -j8
```

This will recompile the allocation client with the updated protocol.

### Step 2: Kill Old Processes

Any process that was running with the old client needs to be restarted:

```bash
# Find old processes
ps aux | grep python | grep ttuser

# Kill them (example PID)
kill 2657333
```

Or kill the specific pytest process:
```bash
pkill -f simple_text_demo.py
```

### Step 3: Restart the Allocation Server

```bash
# Kill old server
pkill -f allocation_server_poc

# Start new server (already updated)
./build/programming_examples/allocation_server_poc &
```

### Step 4: Run Your Applications

Now when you run TT-Metal applications with `TT_ALLOC_TRACKING_ENABLED=1`, they will use the correct protocol!

```bash
export TT_ALLOC_TRACKING_ENABLED=1
python your_model.py
```

## Verification

After the fix, you should see in the server output:

✅ **Clean allocations:**
```
✓ [PID 2685613] Allocated 16384 bytes of DRAM on device 0 (buffer_id=328467008)
✓ [PID 2685613] Allocated 4096 bytes of L1 on device 1 (buffer_id=328469056)
```

❌ **No more errors:**
```
⚠️  Invalid device_id 2657333 - ignoring message  ← GONE
⚠️  Too many consecutive errors (10)              ← GONE
```

## Why This Was Hard to Debug

1. **Process ID coincidence**: The invalid `device_id` (2657333) was actually a process PID, making it look like a real device ID
2. **Silent struct mismatch**: C++ doesn't validate struct compatibility between client and server
3. **Multiple files**: The protocol is defined in 3 places:
   - `allocation_server_poc.cpp` (server)
   - `allocation_client.cpp` (TT-Metal library)
   - `tt_smi_umd.cpp` (monitoring tool)

## Best Practices Going Forward

### 1. Protocol Version Number

Add a version field to detect mismatches:

```cpp
struct AllocMessage {
    uint32_t protocol_version;  // e.g., 2 for CB/Kernel tracking
    // ... rest of fields ...
};
```

### 2. Single Source of Truth

Consider moving the struct to a shared header:

```cpp
// tt_metal/impl/allocator/allocation_protocol.hpp
struct AllocMessage {
    // Definition here
};
```

Then both client and server include this header.

### 3. Static Assertions

Add compile-time checks:

```cpp
static_assert(sizeof(AllocMessage) == 112, "AllocMessage size mismatch!");
```

## Current Status

✅ Server has robust validation (won't crash on bad messages)
✅ Client struct updated to match server
⏳ Need to rebuild TT-Metal library
⏳ Need to restart old processes
⏳ Need to restart server

After these steps, the protocol mismatch will be resolved! 🎉
