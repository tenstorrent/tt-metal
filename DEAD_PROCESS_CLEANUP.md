# Automatic Dead Process Cleanup

## Problem

When you kill a process (Ctrl+C, SIGTERM, SIGKILL) in the middle of inference:

```
1. Process allocates buffers
   └─> Sends ALLOC messages to tracking server ✓

2. Process runs...

3. [Ctrl+C] Process dies immediately
   └─> Cleanup code never runs ❌
   └─> No FREE messages sent to server ❌

4. Allocation server still has records
   └─> "Remaining buffers" from dead process
```

**Result**: Orphaned buffer records in the tracking server.

## Important Note

⚠️ **The actual device memory IS freed** by the kernel/driver when the process dies!

The "problem" is only in the **tracking records** - the server doesn't know the process died.

## Solution: Automatic Detection

The server now **detects dead processes** and automatically cleans up their buffer records.

### How It Works

```cpp
void cleanup_dead_processes() {
    // 1. Get all PIDs that own buffers
    std::set<pid_t> all_pids;
    for (auto& buffer : allocations_) {
        all_pids.insert(buffer.owner_pid);
    }

    // 2. Check which are alive
    for (pid_t pid : all_pids) {
        if (kill(pid, 0) != 0) {
            // Process is dead - remove all its buffers
        }
    }
}
```

**When cleanup runs:**
- **Automatically every 10 seconds** in a background thread
- Also before every `DUMP_REMAINING` request

### Example Output

```
⚠️  Detected dead processes, cleaning up orphaned buffers...
   PID 12345 is dead, removing its buffers...
   ✓ Removed 381 buffers (403.84 MB) from PID 12345

╔══════════════════════════════════════════════════════════════╗
║           REMAINING ALLOCATED BUFFERS                       ║
╚══════════════════════════════════════════════════════════════╝
Total tracked allocations: 0

✓ No buffers remaining - perfect cleanup!
```

## How to Test

1. **Start the allocation server:**
   ```bash
   ./allocation_server_poc
   ```

   You'll see:
   ```
   🔄 Background cleanup thread started (checking every 10s)
   ```

2. **In another terminal, start a test:**
   ```bash
   pytest models/tt_transformers/demo/simple_text_demo.py -k batch-1
   ```

3. **Kill the test mid-run:**
   ```bash
   Ctrl+C
   ```

4. **Wait up to 10 seconds** - the server will automatically detect and clean up:
   ```
   ⚠️  Detected dead processes, cleaning up orphaned buffers...
   PID <test_pid> is dead, removing its buffers...
   ✓ Removed XXX buffers (XXX MB) from PID <test_pid>
   ```

5. **Or request dump immediately:**
   ```bash
   python3 dump_remaining_buffers.py
   ```
   This will trigger immediate cleanup before showing the dump.

## Technical Details

### Process Detection

Uses the Linux `kill(pid, 0)` system call:
- **Returns 0**: Process exists (even if zombie)
- **Returns -1**: Process doesn't exist

This is a standard way to check process liveness without actually sending a signal.

### What Gets Cleaned Up

For each dead process:
1. Remove all buffer records with that PID
2. Update device statistics (subtract freed sizes)
3. Print summary of what was cleaned up

### Limitations

**This only cleans tracking records!**

The actual device memory cleanup depends on:
- TT-Metal driver cleanup when process exits
- Kernel reclaiming resources
- Device reset mechanisms

The tracking server can't control actual device memory - it only maintains accounting records.

## Benefits

✅ **Accurate tracking**: No more orphaned records from killed processes
✅ **Automatic**: No manual intervention needed
✅ **Informative**: Shows which process died and how much it had allocated
✅ **Safe**: Only affects tracking, not actual device memory

## When This Matters

This is especially useful when:
- **Debugging**: Kill tests mid-run to check state
- **Development**: Frequent test interruptions
- **CI/CD**: Test timeouts or failures
- **Long-running services**: Processes crash unexpectedly

## Summary

**Before**: Killed process → Orphaned buffer records → Confusing dump output
**After**: Killed process → Automatic cleanup → Clean dump output

The server now knows when processes die and cleans up their tracking records automatically!
