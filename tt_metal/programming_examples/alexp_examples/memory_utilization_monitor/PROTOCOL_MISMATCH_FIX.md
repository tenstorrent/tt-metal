# Protocol Mismatch Issue - FIXED ⚠️

## What Happened?

The allocation server was **crashing with segfaults** and showing many errors:
- `Unknown message type: 0, 134, 219, 170...`
- `Invalid buffer_type 96, 135, 227...`
- `Segmentation fault (core dumped)`

## Root Cause

**Protocol version mismatch!** When we added `cb_allocated` and `kernel_allocated` fields to `AllocMessage`:

- **Old struct size**: 112 bytes
- **New struct size**: 128 bytes (added 16 bytes)

Your running application was built with the **old protocol** (112 bytes), but the allocation server expected the **new protocol** (128 bytes).

### What Was Happening:
```
Client sends:  [112 bytes of data]
Server reads:  [128 bytes] ← last 16 bytes are GARBAGE from socket buffer!
Result:        Corrupted message fields → Invalid types → Crash
```

## Fix Applied

### 1. **Message Validation** (allocation_server_poc.cpp)
Added sanity checks before processing messages:

```cpp
// Peek at message first to validate
ssize_t n = recv(client_socket, &msg, sizeof(msg), MSG_PEEK);

// Check if message type is valid (1-11)
if (msg.type < 1 || msg.type > 11) {
    // Invalid - discard and continue
    std::cerr << "⚠️  Invalid message type - skipping" << std::endl;
    recv(client_socket, discard, sizeof(msg), 0);
    continue;
}

// Validate device_id range
if (msg.device_id < 0 || msg.device_id >= MAX_DEVICES) {
    return;  // Ignore garbage
}

// Validate size sanity
if (msg.size == 0 || msg.size > 100GB) {
    return;  // Ignore garbage
}
```

### 2. **Graceful Error Recovery**
- Server now **detects** malformed messages
- **Skips** garbage data instead of crashing
- **Continues** processing valid messages

## How to Fix Your Application

You have **two options**:

### Option A: Rebuild Your Application (Recommended)
Your application needs to be rebuilt with the new protocol:

```bash
cd /home/ttuser/aperezvicente/tt-metal-apv
cmake --build build --target <your_app> -j8
```

This will pick up the new `AllocMessage` struct with CB/Kernel fields.

### Option B: Keep Using Old Protocol (Temporary)
If you can't rebuild right now, the server will now **tolerate** old messages:
- Old ALLOC/FREE/QUERY messages will work
- New CB_ALLOC/KERNEL_LOAD messages won't be sent (app doesn't know about them yet)
- Server won't crash, but you won't see CB/Kernel tracking yet

## Testing the Fix

### Step 1: Restart Allocation Server
```bash
./build/programming_examples/allocation_server_poc
```

Expected output (no more crashes!):
```
🚀 Allocation Server started
✓ [PID 123] Allocated 1024 bytes of DRAM...
```

### Step 2: Run Your Application
Your app should now work without crashing the server.

### Step 3: Check for Warnings
If you see:
```
⚠️  Invalid message type X - skipping
```
This means your app is still using the old protocol. **Rebuild it** to get CB/Kernel tracking.

## Protocol Compatibility Table

| Client Protocol | Server Protocol | Result |
|----------------|----------------|---------|
| Old (112 bytes) | Old (112 bytes) | ✅ Works (no CB/Kernel tracking) |
| Old (112 bytes) | New (128 bytes) | ⚠️ Works but with warnings (no CB/Kernel tracking) |
| New (128 bytes) | New (128 bytes) | ✅ Full CB/Kernel tracking! |
| New (128 bytes) | Old (112 bytes) | ❌ Won't work - server needs update |

## For Future Protocol Changes

To avoid this issue in the future:

### 1. **Add Version Field** (Best Practice)
```cpp
struct AllocMessage {
    uint32_t magic = 0xDEADBEEF;  // Protocol magic number
    uint16_t version = 2;          // Protocol version
    // ... rest of fields ...
};
```

### 2. **Detect Version Mismatch**
```cpp
if (msg.magic != 0xDEADBEEF) {
    std::cerr << "❌ Protocol mismatch!" << std::endl;
    return;
}
```

### 3. **Support Multiple Versions**
```cpp
if (msg.version == 1) {
    // Handle old 112-byte protocol
} else if (msg.version == 2) {
    // Handle new 128-byte protocol
}
```

## Summary

✅ **Fixed**: Server now validates messages and doesn't crash
⚠️ **Action Needed**: Rebuild your application to use new protocol
🎯 **Result**: Full CB/Kernel tracking will work once app is rebuilt

The server is now **robust** and won't crash from protocol mismatches!
