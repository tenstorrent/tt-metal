# Persistent Kernel Cleanup - Why ~0.2 MB Remains

## The Issue

After your program ends, you see:
```
Device 0: 0.199219 MB kernels
Device 1: 0.212891 MB kernels
...
```

These kernels persist in the allocation server even though the process has exited.

## Why This Happens

### 1. What Are These Kernels?

The ~0.2 MB of "persistent" kernels are **Fast Dispatch system infrastructure**:
- **Prefetch kernels**: Load program data from DRAM to L1
- **Dispatch kernels**: Execute programs on worker cores
- **Completion queue kernels**: Handle program completion notifications

These are **device-level**, not user-program-level. They're created when the device initializes Fast Dispatch and destroyed when the device closes.

### 2. Tracking Limitation

**CB and KERNEL allocations don't store PID information!**

Look at the server's `handle_client` function:
```cpp
case AllocMessage::CB_ALLOC:
    device_stats_[msg.device_id].cb_allocated += msg.size;  // No PID tracking!
    break;

case AllocMessage::KERNEL_LOAD:
    device_stats_[msg.device_id].kernel_allocated += msg.size;  // No PID tracking!
    break;
```

Compare with regular buffers:
```cpp
case AllocMessage::ALLOC:
    BufferInfo info;
    info.owner_pid = msg.process_id;  // ✅ PID tracked!
    allocations_[key] = info;
    break;
```

**Result**: When `cleanup_dead_processes()` runs, it can clean up buffers (which have PIDs), but NOT CB/KERNEL allocations (which don't)!

### 3. Why Deallocations Don't Arrive

When your program exits normally:
```
1. Python test exits
2. Device.__del__() called
3. Device.close() called
4.   → command_queue_programs_.clear()  // Destroys dispatch programs
5.     → Program destructors run
6.       → deallocate_kernel_buffers() called
7.         → BUT: Process is already exiting!
8.           → AllocationClient tries to send KERNEL_UNLOAD
9.             → Socket may be closed/disconnected
10.            → Message never reaches server!
```

The tracking messages get "lost" during process teardown.

## Solutions

### Solution 1: Wait for Background Cleanup (Current Behavior)

The server won't automatically clean CB/KERNEL allocations from dead processes because it doesn't track PIDs for them.

**Workaround**: Manually reset the server between test runs:
```bash
# Stop old server
pkill -f allocation_server_poc

# Start fresh server
./build/programming_examples/allocation_server_poc > out.log 2>&1 &
```

### Solution 2: Add PID Tracking for CB/KERNEL (Proper Fix)

Modify the server to track CB/KERNEL allocations with PIDs:

```cpp
// Add new tracking maps
struct CBAllocation {
    uint64_t cb_id;
    uint64_t size;
    int device_id;
    pid_t owner_pid;
};
std::unordered_map<uint64_t, CBAllocation> cb_allocations_;

struct KernelAllocation {
    uint64_t kernel_id;
    uint64_t size;
    int device_id;
    pid_t owner_pid;
};
std::unordered_map<uint64_t, KernelAllocation> kernel_allocations_;

// Modify CB_ALLOC handler
case AllocMessage::CB_ALLOC:
    cb_allocations_[msg.buffer_id] = {
        msg.buffer_id, msg.size, msg.device_id, msg.process_id
    };
    device_stats_[msg.device_id].cb_allocated += msg.size;
    break;

// Modify cleanup_dead_processes
for (pid_t dead_pid : dead_pids) {
    // Clean up CBs
    auto cb_it = cb_allocations_.begin();
    while (cb_it != cb_allocations_.end()) {
        if (cb_it->second.owner_pid == dead_pid) {
            device_stats_[cb_it->second.device_id].cb_allocated -= cb_it->second.size;
            cb_it = cb_allocations_.erase(cb_it);
        } else {
            ++cb_it;
        }
    }

    // Clean up kernels
    auto kernel_it = kernel_allocations_.begin();
    while (kernel_it != kernel_allocations_.end()) {
        if (kernel_it->second.owner_pid == dead_pid) {
            device_stats_[kernel_it->second.device_id].kernel_allocated -= kernel_it->second.size;
            kernel_it = kernel_allocations_.erase(kernel_it);
        } else {
            ++kernel_it;
        }
    }
}
```

### Solution 3: Add Server RESET Command (Simple Fix)

Add a command to manually reset all CB/KERNEL stats:

```cpp
case AllocMessage::RESET_STATS:  // New message type
    for (auto& [device_id, stats] : device_stats_) {
        stats.cb_allocated = 0;
        stats.kernel_allocated = 0;
    }
    std::cout << "✓ Reset all CB and KERNEL stats" << std::endl;
    break;
```

Then send this from `tt_smi_umd` or manually via `echo` to the socket.

## Is This a Problem?

### For Development: **Minor Annoyance**
- Persistent stats between test runs
- Solved by restarting the server
- Doesn't affect actual device memory

### For Production: **Not an Issue**
- Devices typically run one long-lived process
- Stats reset when server restarts
- Actual kernel cleanup happens correctly (RAII)

## Recommended Action

**For now: Restart the server between test runs**
```bash
pkill -f allocation_server_poc && \
./build/programming_examples/allocation_server_poc > out.log 2>&1 &
```

**For long-term: Implement Solution 2** (PID tracking for CB/KERNEL)

This ensures accurate cross-process tracking and automatic cleanup of orphaned allocations.

## Testing the Fix

After implementing Solution 2:

1. Start server
2. Run test → See CB/KERNEL allocations
3. Kill test (Ctrl+C or `kill -9`)
4. Wait 5 seconds for background cleanup
5. Check server stats → Should be 0!

Currently:
```
Before process exit: 6.4 MB kernels
After process exit:  0.2 MB kernels (persistent, not cleaned)
```

After fix:
```
Before process exit: 6.4 MB kernels
After process exit:  0 MB kernels (all cleaned, even persistent ones!)
```
