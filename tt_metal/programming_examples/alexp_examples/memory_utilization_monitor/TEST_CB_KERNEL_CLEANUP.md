# Testing CB/KERNEL Automatic Cleanup

## What We Just Fixed

Added **PID tracking** for CB and KERNEL allocations so the background cleanup thread can automatically remove them when processes die.

## Before the Fix

```
After process exits:
- Regular buffers: ✅ Cleaned up (had PID tracking)
- CBs: ❌ Still showing (~22 MB)
- Kernels: ❌ Still showing (~0.2 MB)
```

## After the Fix

```
After process exits (within 5 seconds):
- Regular buffers: ✅ Cleaned up
- CBs: ✅ Cleaned up (NEW!)
- Kernels: ✅ Cleaned up (NEW!)
```

## How to Test

### Step 1: Rebuild BOTH Server and Client

```bash
cd /home/ttuser/aperezvicente/tt-metal-apv

# Rebuild allocation server (has the PID tracking fix)
cmake --build build --target allocation_server_poc -j$(nproc)

# Rebuild tt_metal library (has the kernel size and unload fixes)
cmake --build build --target tt_metal -j$(nproc)
```

### Step 2: Stop Old Server and Start New One

```bash
# Kill old server (if running)
pkill -f allocation_server_poc

# Start new server with PID tracking
./build/programming_examples/allocation_server_poc > out_cleanup_test.log 2>&1 &

# Get the server PID
SERVER_PID=$!
echo "Server PID: $SERVER_PID"
```

### Step 3: Run a Test Program

```bash
# Enable tracking
export TT_ALLOC_TRACKING_ENABLED=1

# Run a quick test
cd /home/ttuser/aperezvicente/tt-metal-apv
source python_env/bin/activate
pytest models/tt_transformers/demo/simple_text_demo.py::test_perf_device_prefill_decode[test_perf_device_with_batch_size-1] -s

# Get the test PID while it's running (in another terminal)
# ps aux | grep pytest | grep -v grep
```

### Step 4: Check Allocations DURING Test

```bash
# While test is running, check allocations
tail -50 out_cleanup_test.log | grep -E "CB_ALLOC|KERNEL_LOAD"

# You should see:
# ✓ [CB_ALLOC] Device 0: +X.XX MB (Total: Y.YY MB)
# ✓ [KERNEL_LOAD] Device 0: +X.XX MB (Total: Y.YY MB)
```

### Step 5: Kill the Test (Simulate Crash)

```bash
# Find the pytest PID
TEST_PID=$(ps aux | grep "pytest.*simple_text" | grep -v grep | awk '{print $2}')
echo "Test PID: $TEST_PID"

# Kill it abruptly (simulate crash)
kill -9 $TEST_PID
```

### Step 6: Wait and Watch Cleanup (THE MAGIC!)

```bash
# Watch the server log - within 5 seconds you should see:
tail -f out_cleanup_test.log

# Expected output:
# ⚠️  Detected dead processes, cleaning up orphaned buffers...
#    PID 12345 is dead, removing its buffers...
#    ✓ Removed 150 buffers (25.5 MB) from PID 12345
#    ✓ Removed 800 CBs (22.4 MB) from PID 12345          ← NEW!
#    ✓ Removed 30 kernels (0.2 MB) from PID 12345        ← NEW!
```

### Step 7: Verify Complete Cleanup

```bash
# Check final stats - should be ZERO!
./build/programming_examples/tt_smi_umd

# Expected output:
# Device 0 (Blackhole):
#   DRAM: 0.0B / 31.9GB
#   L1 Memory:
#     Buffers: 0.0B     ← Should be 0
#     CBs: 0.0B         ← Should be 0 (was 22 MB before!)
#     Kernels: 0.0B     ← Should be 0 (was 204 KB before!)
#     Total: 0.0B
```

## What the Fix Does

### New Data Structures (allocation_server_poc.cpp)

```cpp
// Track CBs with PID
struct CBInfo {
    uint64_t cb_id;
    int device_id;
    uint64_t size;
    pid_t owner_pid;  // ← NEW: Can now identify owner!
    std::chrono::steady_clock::time_point alloc_time;
};
std::unordered_map<BufferKey, CBInfo, BufferKeyHash> cb_allocations_;

// Track Kernels with PID
struct KernelInfo {
    uint64_t kernel_id;
    int device_id;
    uint64_t size;
    pid_t owner_pid;  // ← NEW: Can now identify owner!
    std::chrono::steady_clock::time_point alloc_time;
};
std::unordered_map<BufferKey, KernelInfo, BufferKeyHash> kernel_allocations_;
```

### Enhanced Cleanup Logic

```cpp
void cleanup_dead_processes() {
    // ... (existing buffer cleanup) ...

    // NEW: Clean up CBs from dead processes
    for (auto cb_it = cb_allocations_.begin(); cb_it != cb_allocations_.end();) {
        if (cb_it->second.owner_pid == dead_pid) {
            device_stats_[cb_it->second.device_id].cb_allocated -= cb_it->second.size;
            cb_it = cb_allocations_.erase(cb_it);
        } else {
            ++cb_it;
        }
    }

    // NEW: Clean up Kernels from dead processes
    for (auto kernel_it = kernel_allocations_.begin(); kernel_it != kernel_allocations_.end();) {
        if (kernel_it->second.owner_pid == dead_pid) {
            device_stats_[kernel_it->second.device_id].kernel_allocated -= kernel_it->second.size;
            kernel_it = kernel_allocations_.erase(kernel_it);
        } else {
            ++kernel_it;
        }
    }
}
```

## Why 204 KB Still Shows

The **204 KB you're seeing now** is from:
1. An **old server** without PID tracking, OR
2. A **test that ran before rebuilding**, OR
3. Server needs to be **restarted** to load the new code

**Solution**: Follow Steps 1-2 above to restart with the new server!

## Troubleshooting

### If cleanup doesn't happen:

1. **Check server was rebuilt:**
   ```bash
   ls -lh build/programming_examples/allocation_server_poc
   # Should show recent timestamp
   ```

2. **Check server is the new version:**
   ```bash
   # Start server and immediately check for the new tracking maps
   # The new server should handle CB_ALLOC/KERNEL_LOAD differently
   ```

3. **Verify PID is being tracked:**
   ```bash
   # During test, check if PID is in the messages
   grep "PID" out_cleanup_test.log
   ```

4. **Check cleanup thread is running:**
   ```bash
   grep "Background cleanup" out_cleanup_test.log
   # Should see: "🔄 Background cleanup thread started (checking every 10s)"
   ```

## Success Criteria

✅ **Before process exit**: See CB_ALLOC and KERNEL_LOAD messages
✅ **After process kill**: Within 5 seconds, see cleanup message
✅ **In tt_smi_umd**: All counters show 0.0B
✅ **In server log**: "Removed X CBs" and "Removed X kernels" messages

## Summary

This fix ensures that **even if a program crashes or is killed**, the allocation server will automatically clean up all its tracked memory (buffers, CBs, AND kernels) within 5 seconds via the background cleanup thread!
