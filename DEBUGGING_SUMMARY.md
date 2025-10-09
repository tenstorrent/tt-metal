# Memory Cleanup Debugging Summary

## Current Status ✅

### What's Working:
1. ✅ **Cleanup fixture IS running** - Confirmed by output lines 72-82
2. ✅ **Program cache is being cleared** - See line 75: "Disabling and clearing program cache"
3. ✅ **Fixture sends DUMP_REMAINING message** - Line 79: "Buffer dump requested"

### What's NOT Working:
1. ❌ **Server not displaying buffer dump** - The dump output isn't showing
2. ❌ **Server appears to hang/not respond** - No visible output after DUMP_REMAINING

## Root Cause Analysis

The server is likely receiving the message but either:
1. **Output is being buffered** - C++ stdout buffering issue
2. **Server is busy/locked** - Thread contention or deadlock
3. **Message not properly formatted** - Struct packing mismatch

## Fixes Applied

### 1. Added Debug Output to Server (`allocation_server_poc.cpp`)

```cpp
case AllocMessage::DUMP_REMAINING:
    std::cout << "📋 Received DUMP_REMAINING request..." << std::endl;
    std::cout.flush();  // Force output
    handle_dump_remaining();
    std::cout << "📋 DUMP_REMAINING complete." << std::endl;
    std::cout.flush();
    break;
```

### 2. Added Flush Calls in `handle_dump_remaining()`

```cpp
void handle_dump_remaining() {
    std::cout << "Total tracked allocations: " << allocations_.size() << "\n" << std::endl;
    std::cout.flush();  // Ensure immediate output

    // ... rest of function ...

    std::cout.flush();  // Flush at end too
}
```

### 3. Fixed Fixture Output (`conftest.py`)

Changed from `logger.info()` to `print(..., file=sys.stderr)` to bypass pytest capture:

```python
print("CLEANUP FIXTURE: Starting post-test cleanup...", file=sys.stderr)
```

## How to Debug

### Step 1: Restart the Allocation Server

**IMPORTANT**: You must restart the server after rebuilding!

```bash
# Kill old server
pkill -9 allocation_ser

# Start new server in Terminal 1
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor
./allocation_server_poc
```

You should see:
```
🚀 Allocation Server started
   Listening on: /tmp/tt_allocation_server.sock
   Press Ctrl+C to stop
```

### Step 2: Test DUMP_REMAINING Message

In Terminal 2:
```bash
cd /home/tt-metal-apv
python3 test_dump_message.py
```

Expected output in **Server Terminal**:
```
📋 Received DUMP_REMAINING request...

╔══════════════════════════════════════════════════════════════╗
║           REMAINING ALLOCATED BUFFERS                       ║
╚══════════════════════════════════════════════════════════════╝
Total tracked allocations: X

[Buffer details here]

📋 DUMP_REMAINING complete.
```

### Step 3: Run the Test

In Terminal 2:
```bash
cd /home/tt-metal-apv
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1"
```

Watch **both terminals**:

**Terminal 2 (pytest)** should show:
```
CLEANUP FIXTURE: Starting post-test cleanup...
CLEANUP FIXTURE: Running garbage collection...
CLEANUP FIXTURE: Clearing program cache...
✓ Program cache cleared
CLEANUP FIXTURE: Waiting for deallocations to propagate...
CLEANUP FIXTURE: Requesting buffer dump from allocation server...
✓ Buffer dump requested - CHECK ALLOCATION SERVER OUTPUT
```

**Terminal 1 (server)** should show:
```
📋 Received DUMP_REMAINING request...
[Buffer dump here]
📋 DUMP_REMAINING complete.
```

## If Server Still Hangs

### Check 1: Is server actually running?
```bash
ps aux | grep allocation_ser
ls -la /tmp/tt_allocation_server.sock
```

### Check 2: Test with simple message
```bash
python3 test_dump_message.py
```

If this works but pytest doesn't, the issue is with the fixture's message sending.

### Check 3: Check for deadlock
If the server shows "Received DUMP_REMAINING request..." but never shows "complete", there's a deadlock in `handle_dump_remaining()`.

Possible causes:
- `registry_mutex_` already locked
- Printing to stdout causing blocking I/O
- Large number of allocations causing slow loop

### Check 4: Server output buffering
The server might be writing to stdout but it's being buffered. Try:
```bash
./allocation_server_poc | cat -u
```

The `cat -u` forces unbuffered output.

## Expected Memory After Cleanup

After the fixture runs, you should see:

### ✅ Normal (Not Leaks):
- **~12-20KB L1 per device**: Circular buffers (infrastructure)
- **Small DRAM (<1MB)**: System/infrastructure buffers
- **Consistent across runs**: Same buffers remain

### ❌ Leaks (Should NOT appear):
- **36KB DRAM per device**: Program cache (should be cleared now!)
- **Growing memory**: Increases with each test run
- **Large buffers (>100MB)**: Model weights or KV cache

## Troubleshooting Checklist

- [ ] Server rebuilt with `./build_allocation_server.sh`
- [ ] Old server process killed (`pkill -9 allocation_ser`)
- [ ] New server started (`./allocation_server_poc`)
- [ ] Test script works (`python3 test_dump_message.py`)
- [ ] Server shows "Received DUMP_REMAINING request..."
- [ ] Server shows buffer dump
- [ ] Server shows "DUMP_REMAINING complete"
- [ ] Fixture output visible in pytest (lines 72-82)
- [ ] No "Could not request buffer dump" error in pytest

## Next Steps

1. **Restart the server** with the rebuilt binary
2. **Run test script** to verify DUMP_REMAINING works
3. **Run pytest** and watch both terminals
4. **Report what you see** in the server terminal

## Key Files Modified

1. `/home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor/allocation_server_poc.cpp`
   - Added debug output for DUMP_REMAINING
   - Added stdout flush calls

2. `/home/tt-metal-apv/models/tt_transformers/demo/conftest.py`
   - Changed to `print(..., file=sys.stderr)`
   - Added better error handling
   - Increased wait time to 2 seconds

3. `/home/tt-metal-apv/test_dump_message.py` (NEW)
   - Simple test script to verify server receives DUMP_REMAINING

## What You Should See Now

When everything is working:

1. **During test**: Cleanup fixture messages in stderr
2. **In server terminal**:
   - "📋 Received DUMP_REMAINING request..."
   - Buffer dump with details
   - "📋 DUMP_REMAINING complete."
3. **Memory state**: Only infrastructure buffers remain (12-20KB L1)

The fixture **is working**. The issue is with the **server not displaying output**. Restarting the server should fix it!
