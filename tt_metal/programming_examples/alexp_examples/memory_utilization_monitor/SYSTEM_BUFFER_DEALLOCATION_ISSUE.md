# System Buffer Deallocation Issue

## Problem

**If you run your test multiple times, the allocation monitor shows increasing DRAM usage that never gets freed.**

This is a **REAL BUG** in the tracking system, not just a cosmetic issue!

## Root Cause Analysis

### What's Happening

```
Run 1: ~14MB DRAM allocated
Run 2: ~28MB DRAM allocated (14MB + 14MB)
Run 3: ~42MB DRAM allocated (14MB + 14MB + 14MB)
...
```

### Why It Happens

System buffers (command queues, dispatch infrastructure) are **NOT reporting deallocations** to the allocation server. Here's the call chain:

#### 1. Device Close Path
```cpp
// Device::close()
this->command_queues_.clear();  // ← Destroys HWCommandQueue objects
this->sysmem_manager_.reset();  // ← Destroys SystemMemoryManager
```

#### 2. HWCommandQueue Destructor
```cpp
// hardware_command_queue.cpp:205
HWCommandQueue::~HWCommandQueue() {
    // Only joins the completion queue thread
    // NO buffer deallocation!
    // NO tracking calls!
    this->set_exit_condition();
    this->completion_queue_thread_.join();
}
```

#### 3. The Missing Link

**System buffers are NOT `Buffer` objects!** They're managed by:
- `SystemMemoryManager` (host-side hugepages)
- Direct DRAM allocations (device-side)
- Ringbuffer cache managers
- Config buffer managers

None of these go through `Buffer::deallocate()` or `GraphTracker::track_deallocate()`!

## Why System Buffers Don't Track Deallocations

### 1. They're Not Standard Buffers

System buffers are created **outside** the normal `Buffer` class:

```cpp
// HWCommandQueue constructor
prefetcher_cache_manager_ = std::make_unique<RingbufferCacheManager>(...);
// ← This allocates DRAM but doesn't create Buffer objects!
```

### 2. No Explicit Deallocation

When `HWCommandQueue` is destroyed:
- Managers are destroyed (`unique_ptr` cleanup)
- Memory is implicitly freed by the allocator
- **But no tracking calls are made!**

### 3. The Allocator Doesn't Track Its Own Cleanup

```cpp
// Allocator::deallocate_buffer() is called
// But it doesn't call AllocationClient::report_deallocation()
// Because we removed that code (it was causing issues)
```

## The Real Architecture

```
┌─────────────────────────────────────────────────────┐
│ Application Buffers (Tensors)                       │
│ ✅ Allocation: GraphTracker::track_allocate()       │
│ ✅ Deallocation: GraphTracker::track_deallocate()   │
│ ✅ FULLY TRACKED                                    │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ System Buffers (Command Queues)                     │
│ ✅ Allocation: Tracked (via Buffer::create)         │
│ ❌ Deallocation: NOT TRACKED                        │
│ ⚠️  PARTIALLY TRACKED                               │
└─────────────────────────────────────────────────────┘
```

## Solutions

### Option 1: Track System Buffer Deallocations (HARD)

**Modify HWCommandQueue destructor:**

```cpp
// In hardware_command_queue.cpp
HWCommandQueue::~HWCommandQueue() {
    // Report deallocation of all system buffers
    if (AllocationClient::is_enabled()) {
        // Problem: We don't have buffer addresses or sizes!
        // System buffers are managed by multiple managers
        // No central registry of what was allocated
    }

    this->set_exit_condition();
    this->completion_queue_thread_.join();
}
```

**Why This Is Hard:**
- System buffers are scattered across multiple managers
- No central tracking of addresses/sizes
- Would require major refactoring of dispatch infrastructure

### Option 2: Reset Server Between Runs (EASY) ✅

**Just restart the allocation server:**

```bash
#!/bin/bash
# kill_and_restart_server.sh

pkill -f allocation_server_poc
sleep 1
./allocation_server_poc &
echo "Server restarted with clean state"
```

**Pros:**
- Simple, works immediately
- No code changes needed
- Clean baseline for each test

**Cons:**
- Manual step between tests
- Loses historical data

### Option 3: Server-Side "Reset" Command (MEDIUM)

**Add a reset message to the protocol:**

```cpp
// allocation_protocol.hpp
enum class MessageType : uint8_t {
    ALLOCATION = 1,
    DEALLOCATION = 2,
    QUERY = 3,
    RESET = 4,  // ← New message type
};

// allocation_server_poc.cpp
void handle_reset() {
    allocations_.clear();
    for (int i = 0; i < MAX_DEVICES; i++) {
        device_stats_[i] = DeviceStats{};
    }
    std::cout << "🔄 Server state reset" << std::endl;
}
```

**Usage:**
```python
# At the start of each test
import socket, struct
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect("/tmp/allocation_tracker.sock")
sock.send(struct.pack('B', 4))  # Send RESET message
sock.close()
```

### Option 4: Track "Baseline" Per Device (MEDIUM)

**Server remembers initial state:**

```cpp
class AllocationServer {
    std::unordered_map<int, DeviceStats> baseline_stats_;  // Initial state
    std::unordered_map<int, DeviceStats> current_stats_;   // Current state

    void set_baseline(int device_id) {
        baseline_stats_[device_id] = current_stats_[device_id];
    }

    DeviceStats get_delta(int device_id) {
        return current_stats_[device_id] - baseline_stats_[device_id];
    }
};
```

**Client shows delta:**
```
Device 0:
  DRAM: 18.5 MB (baseline: 14.2 MB, delta: +4.3 MB)  ← Your app's usage
  L1:   22.4 KB (baseline: 22.4 KB, delta: +0 KB)
```

## Recommended Solution

**Use Option 2 (restart server) for now**, then implement **Option 3 (reset command)** for convenience.

### Quick Fix Script

```bash
#!/bin/bash
# run_test_with_clean_server.sh

echo "🧹 Cleaning up old server..."
pkill -f allocation_server_poc
sleep 1

echo "🚀 Starting fresh server..."
./allocation_server_poc &
sleep 1

echo "📊 Starting monitor..."
./allocation_monitor_client -a -r 500 &
MONITOR_PID=$!

echo "🧪 Running test..."
export TT_ALLOC_TRACKING_ENABLED=1
python test_mesh_allocation.py

echo "⏸️  Waiting 5 seconds for cleanup..."
sleep 5

echo "✅ Test complete! Check monitor for final state."
echo "Press Ctrl+C to stop monitor..."
wait $MONITOR_PID
```

## Bottom Line

**The issue is real:** System buffers don't report deallocations.

**Why it happens:** System buffers bypass the normal `Buffer` deallocation path.

**Impact:**
- ✅ Your application's memory is tracked correctly
- ❌ System buffer cleanup is not tracked
- ⚠️  Repeated runs show cumulative system buffer allocations

**Solution:** Restart the server between test runs (or implement a reset command).

**Is this a bug in TT-Metal?** No - the buffers ARE being freed, they're just not reporting it to your monitoring system. This is a limitation of the monitoring approach, not a memory leak in TT-Metal itself.
