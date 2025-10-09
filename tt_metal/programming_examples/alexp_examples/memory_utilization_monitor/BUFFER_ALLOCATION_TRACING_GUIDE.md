# Buffer Allocation Tracing Guide

## Complete Flow: Application → Allocation Server

### Overview
Here's how buffer allocations are tracked from your application to the allocation server:

```
Application Code (e.g., ttnn::matmul)
         ↓
    Buffer Creation
         ↓
  buffer.cpp::allocate_impl()  ← Line 411-434
         ↓
GraphTracker::track_allocate()  ← Line 125-167 in graph_tracking.cpp
         ↓
AllocationClient::report_allocation()  ← Line 139-149 in allocation_client.cpp
         ↓
  send() → Unix Socket
         ↓
  Allocation Server (allocation_server_poc.cpp)
```

---

## 1. Entry Point: Buffer Allocation

**File:** `tt_metal/impl/buffers/buffer.cpp`

### Path A: New Buffer with Allocation
```cpp
void Buffer::allocate_impl() {
    // Lines 411-434
    if (GraphTracker::instance().hook_allocate(this)) {
        address_ = 0;
        hooked_allocation_ = true;
    } else {
        address_ = allocator_->allocate_buffer(this);  // ← Actual memory allocation
    }

    allocation_status_ = AllocationStatus::ALLOCATED;
    GraphTracker::instance().track_allocate(this);  // ← TRACKING HAPPENS HERE
}
```

###Path B: Pre-allocated Buffer (e.g., MeshBuffer device-local buffers)
```cpp
std::shared_ptr<Buffer> Buffer::create(..., DeviceAddr address, ...) {
    // Lines 317-344
    auto buffer = std::make_shared<Buffer>(...);
    buffer->address_ = address;  // Pre-assigned address
    buffer->allocation_status_ = AllocationStatus::ALLOCATED;

    GraphTracker::instance().track_allocate(buffer.get());  // ← TRACKING HAPPENS HERE
    return buffer;
}
```

---

## 2. Tracking Layer: GraphTracker

**File:** `tt_metal/graph/graph_tracking.cpp`

```cpp
void GraphTracker::track_allocate(const Buffer* buffer) {
    // Lines 125-167

    if (buffer->device() != nullptr) {
        // Skip MeshDevice backing buffers (they're internal)
        if (dynamic_cast<const distributed::MeshDevice*>(buffer->device()) != nullptr) {
            return;
        }

        // Report to allocation server (if enabled)
        if (AllocationClient::is_enabled()) {
            AllocationClient::report_allocation(
                buffer->device()->id(),      // Which device (0-7)
                buffer->size(),              // Size in bytes
                static_cast<uint8_t>(buffer->buffer_type()),  // DRAM/L1/etc
                buffer->address()            // Memory address (used as buffer_id)
            );
        }
    }
}
```

**Key Point:** `buffer->address()` is used as the `buffer_id` in tracking. This is why you see the same `buffer_id` reused - it's the same memory address being reused for different buffers!

---

## 3. Client: Sending to Server

**File:** `tt_metal/impl/allocator/allocation_client.cpp`

```cpp
void AllocationClient::report_allocation(
    int device_id,
    uint64_t size,
    uint8_t buffer_type,
    uint64_t buffer_id  // ← This is buffer->address()!
) {
    // Lines 139-149
    auto& inst = instance();
    if (inst.enabled_) {
        inst.send_allocation_message(device_id, size, buffer_type, buffer_id);
    }
}

void AllocationClient::send_allocation_message(...) {
    // Lines 95-119
    AllocMessage msg;
    memset(&msg, 0, sizeof(msg));
    msg.type = AllocMessage::ALLOC;
    msg.device_id = device_id;
    msg.size = size;
    msg.buffer_type = buffer_type;
    msg.process_id = getpid();
    msg.buffer_id = buffer_id;
    msg.timestamp = std::chrono::system_clock::now().time_since_epoch().count();

    // Non-blocking send (doesn't slow down application)
    send(socket_fd_, &msg, sizeof(msg), MSG_DONTWAIT);
}
```

**Enabled by:** `export TT_ALLOC_TRACKING_ENABLED=1`

---

## 4. Deallocation Flow

**File:** `tt_metal/impl/buffers/buffer.cpp`

```cpp
void Buffer::deallocate_impl() {
    // Lines 471-496
    if (allocation_status_ != AllocationStatus::ALLOCATED) {
        return;
    }

    if (device_->is_initialized() && size_ != 0) {
        GraphTracker::instance().track_deallocate(this);  // ← TRACKING HAPPENS HERE
        allocator_->deallocate_buffer(this);  // ← Actual memory deallocation
    }

    allocation_status_ = AllocationStatus::DEALLOCATED;
}
```

**Called from:** `Buffer::~Buffer()` destructor (line 498-504)

---

## Why "Unknown Deallocations" Happen

Based on your debug.log analysis:

### Issue 1: Buffer Address Reuse
```
Line 3945: ✓ Allocated 524288 bytes (buffer_id=25718784, device 2)
Line 3973: ✗ Freed 524288 bytes (FINAL)      ← Removed from tracking
Line 3981: ✓ Allocated 98304 bytes (buffer_id=25718784, device 2)  ← SAME ADDRESS, DIFFERENT SIZE
Line 4021: ✗ Freed 98304 bytes (FINAL)       ← Removed from tracking
Line 5300: ⚠ Deallocation for unknown buffer 25718784  ← Old pointer freed again!
```

**Root Cause:** The memory allocator reuses addresses. The server tracks by `{device_id, buffer_id}` but doesn't account for the fact that the same address can be used for different allocations over time.

### Issue 2: Missing Deallocations
```
Line 36539: Active allocations: 87
Line 36541: PID 314528 is dead, cleaning up orphaned buffers...
```

**Root Cause:** The process exited/crashed before calling destructors for all buffers. Possible reasons:
1. Exception thrown before cleanup
2. Process killed (SIGKILL)
3. Exit() called without cleanup
4. Forgotten to deallocate

---

## How to Trace Buffer Origins

### Option 1: Add Stack Traces to Allocation Client (C++)

**File:** `tt_metal/impl/allocator/allocation_client.cpp`

Add this to capture stack traces:

```cpp
#include <execinfo.h>
#include <cxxabi.h>
#include <sstream>

std::string get_stack_trace() {
    void* callstack[128];
    int frames = backtrace(callstack, 128);
    char** symbols = backtrace_symbols(callstack, frames);

    std::stringstream ss;
    for (int i = 0; i < frames; i++) {
        ss << symbols[i] << "\n";
    }
    free(symbols);
    return ss.str();
}

void AllocationClient::send_allocation_message(...) {
    // ... existing code ...

    // Log stack trace for debugging
    if (buffer_type == 1) {  // L1 only
        std::cerr << "L1 Allocation: device=" << device_id
                  << " size=" << size
                  << " addr=0x" << std::hex << buffer_id << std::dec << "\n";
        std::cerr << get_stack_trace() << "\n";
    }
}
```

Compile with: `-rdynamic` flag to get symbol names

### Option 2: Use Environment Variable for Detailed Logging

Add to `allocation_client.cpp`:

```cpp
void AllocationClient::send_allocation_message(...) {
    // Existing code...

    const char* detailed = std::getenv("TT_ALLOC_TRACKING_DETAILED");
    if (detailed && std::string(detailed) == "1") {
        std::ofstream log("/tmp/tt_alloc_detailed.log", std::ios::app);
        log << "ALLOC: pid=" << getpid()
            << " device=" << device_id
            << " size=" << size
            << " type=" << (int)buffer_type
            << " addr=0x" << std::hex << buffer_id << std::dec
            << " timestamp=" << msg.timestamp
            << "\n";
    }
}
```

Usage:
```bash
export TT_ALLOC_TRACKING_ENABLED=1
export TT_ALLOC_TRACKING_DETAILED=1
./your_app
```

### Option 3: Modify Server to Track Buffer History

Use the enhanced server (`allocation_server_enhanced.cpp`) which tracks:
- Full allocation/deallocation history per buffer
- Size changes when buffer IDs are reused
- Double-free detection
- Leak detection

---

## Debugging Workflow

### Step 1: Run with Tracking Enabled
```bash
# Terminal 1: Start allocation server
cd tt_metal/programming_examples/alexp_examples/memory_utilization_monitor
g++ -o allocation_server_enhanced allocation_server_enhanced.cpp -std=c++17 -pthread
./allocation_server_enhanced > server.log 2>&1

# Terminal 2: Run your application
export TT_ALLOC_TRACKING_ENABLED=1
python your_test.py
```

### Step 2: Analyze Remaining Buffers
```bash
# While app is running or after it exits:
python dump_remaining_buffers.py
```

The enhanced server will show:
```
Buffer 0x18c6a00 on device 0:
  History (allocs=3 frees=2):
    ALLOC size=524288 pid=12345 ref=1
    FREE size=524288 pid=12345 ref=0
    ALLOC size=262144 pid=12345 ref=1  ← Address reused!
    FREE size=262144 pid=12345 ref=0
    ALLOC size=131072 pid=12345 ref=1  ← Leaked!
```

### Step 3: Find Leak Source in Code

Look for patterns:
1. **Buffers never freed:** Missing destructor calls or leaked `shared_ptr`
2. **Buffers freed multiple times:** Double-free bugs, already fixed in your codebase
3. **Size mismatches:** Normal address reuse, server should handle it

---

## Common Leak Patterns

### Pattern 1: Exception Before Cleanup
```cpp
void process_data(Device* device) {
    auto buffer = Buffer::create(device, ...);

    do_work();  // ← Throws exception

    // Never reaches here - buffer leaked!
}
```

**Fix:** Use RAII (buffer is `shared_ptr`, so it auto-cleans on exception)

### Pattern 2: Circular References
```cpp
struct Node {
    std::shared_ptr<Buffer> data;
    std::shared_ptr<Node> next;
};

// Circular ref: A->B->A prevents cleanup
```

**Fix:** Use `weak_ptr` for back-pointers

### Pattern 3: Global/Static Buffers
```cpp
static std::shared_ptr<Buffer> global_buffer;  // Never destroyed until program exit
```

**Fix:** Explicit cleanup in shutdown function

---

## Summary

### To trace "unknown deallocations":
1. The server tracks by `{device_id, buffer_address}`
2. Same address can be reused for different buffers
3. Use enhanced server to see allocation history

### To trace "buffers not deallocated":
1. Add stack traces to see where allocations come from
2. Check if destructors are called (add logging to `Buffer::~Buffer()`)
3. Use sanitizers: `ASAN=1` build to detect leaks
4. Review exception handling paths

### Key Files to Instrument:
- `tt_metal/impl/buffers/buffer.cpp` - Add logging to constructors/destructors
- `tt_metal/graph/graph_tracking.cpp` - Add detailed allocation/deallocation logs
- `tt_metal/impl/allocator/allocation_client.cpp` - Add stack traces

### Next Steps:
1. Use `allocation_server_enhanced.cpp` (already created)
2. Add detailed logging to `Buffer::~Buffer()` to see which buffers aren't being destroyed
3. Run with ASAN to detect actual memory leaks: `build.sh --asan`
