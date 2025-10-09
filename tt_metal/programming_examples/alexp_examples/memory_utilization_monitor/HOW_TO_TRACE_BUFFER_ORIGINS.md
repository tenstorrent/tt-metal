# How to Trace Buffer Origins - Practical Guide

## Summary of Your Issues

Based on your debug.log analysis:

### 1. "Unknown Deallocations"
- **NOT** because server started late
- **Root cause:** Buffer address (buffer_id) is **reused** for different allocations
- Same memory address gets allocated, freed, then allocated again with different size
- When old pointer is freed, server says "unknown" because it tracks current state only

### 2. "Buffers Not Deallocated"
- **NOT** just waiting for cleanup
- **Root cause:** Application **never called** destructors for 87 buffers
- Only cleaned up when dead process detector found PID was killed
- Means: leaked references, exceptions, or missing cleanup

---

## Quick Start: Trace Your Application Now

### Step 1: Build Enhanced Server
```bash
cd /workspace/tt-metal-apv/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor

# Compile enhanced server with history tracking
g++ -o allocation_server_enhanced allocation_server_enhanced.cpp -std=c++17 -pthread

# Compile stack trace client
g++ -o allocation_client_stacktrace allocation_client_with_stacktrace.cpp -std=c++17 -pthread -rdynamic
```

### Step 2: Run Server
```bash
# Terminal 1
./allocation_server_enhanced > enhanced_server.log 2>&1
```

### Step 3: Run Your Application with Stack Traces
```bash
# Terminal 2
export TT_ALLOC_TRACKING_ENABLED=1
export TT_ALLOC_STACK_TRACE=1  # Enable stack trace logging

# Run your application
python your_test_script.py

# Stack traces will be written to /tmp/tt_alloc_stack_traces.log
```

### Step 4: Check Results

**Check server output:**
```bash
tail -f enhanced_server.log
```

You'll see:
- Allocations with ref_count tracking
- **Warnings for buffer reuse with size changes**
- **Warnings for double-frees with event history**
- Leaked buffers at end

**Check stack traces:**
```bash
less /tmp/tt_alloc_stack_traces.log
```

You'll see exact call stacks showing where each allocation came from!

---

## Understanding the Complete Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ YOUR APPLICATION (e.g., ttnn.matmul)                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Buffer::create() or Buffer::allocate_impl()                     │
│  File: tt_metal/impl/buffers/buffer.cpp                         │
│  - Allocates actual GPU memory                                  │
│  - Sets buffer->address_ (THIS BECOMES buffer_id)               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ GraphTracker::track_allocate(buffer)                            │
│  File: tt_metal/graph/graph_tracking.cpp:125                    │
│  - Intercepts ALL buffer allocations                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ AllocationClient::report_allocation()                           │
│  File: tt_metal/impl/allocator/allocation_client.cpp:139        │
│  - Sends message to server via Unix socket                      │
│  - Parameters:                                                  │
│    • device_id: 0-7                                             │
│    • size: bytes                                                │
│    • buffer_type: 0=DRAM, 1=L1, etc                             │
│    • buffer_id: buffer->address() ← IMPORTANT!                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼ Unix Socket
┌─────────────────────────────────────────────────────────────────┐
│ Allocation Server                                               │
│  File: allocation_server_poc.cpp or allocation_server_enhanced  │
│  - Receives AllocMessage                                        │
│  - Tracks by {device_id, buffer_id}                             │
│  - Updates statistics                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Key Insight: buffer_id IS the Memory Address

```cpp
// In graph_tracking.cpp, line 147:
AllocationClient::report_allocation(
    buffer->device()->id(),
    buffer->size(),
    static_cast<uint8_t>(buffer->buffer_type()),
    buffer->address()  // ← This is buffer_id!
);
```

**This means:**
- `buffer_id=25718784` is address `0x18862C0` in GPU memory
- When GPU memory allocator **reuses** this address for a new buffer
- The server sees allocation at the **same buffer_id**
- But it might be a completely different buffer with different size!

**Your log shows this clearly:**
```
3945: ✓ Allocated 524288 bytes (buffer_id=25718784)  ← First allocation
3973: ✗ Freed (FINAL)                                 ← Freed it
3981: ✓ Allocated 98304 bytes (buffer_id=25718784)   ← Reused address, DIFFERENT SIZE!
```

---

## Why Buffers Are Not Deallocated

### Check 1: Are Destructors Being Called?

Add logging to `Buffer::~Buffer()`:

**File:** `tt_metal/impl/buffers/buffer.cpp` (line 498)

```cpp
Buffer::~Buffer() {
    // ADD THIS:
    if (device_ && device_->id() == 0) {  // Log device 0 only
        std::ofstream log("/tmp/buffer_destructors.log", std::ios::app);
        log << "~Buffer() called: addr=0x" << std::hex << address_
            << " size=" << std::dec << size_
            << " type=" << (int)buffer_type_ << "\n";
    }

    LIGHT_METAL_TRACE_FUNCTION_ENTRY();
    LIGHT_METAL_TRACE_FUNCTION_CALL(CaptureBufferDelete, *this);
    if (this->allocation_status_ != AllocationStatus::DEALLOCATED) {
        this->deallocate();
    }
}
```

Run your app and check:
```bash
# Count allocations vs deallocations
grep "Allocated.*device 0" /tmp/tt_alloc_stack_traces.log | wc -l
grep "~Buffer" /tmp/buffer_destructors.log | wc -l

# If allocations > destructions, you have a leak!
```

### Check 2: Find What's Holding References

Buffers are `shared_ptr`, so if destructor isn't called, something is still holding a reference.

Common causes:
1. **Forgotten variables:** Local variables not going out of scope
2. **Circular references:** Two objects pointing to each other
3. **Global/static storage:** Stored in global variables
4. **Exception thrown:** Before reaching cleanup code
5. **Cached:** Stored in a cache that's never cleared

**Add reference counting debug:**
```cpp
// In buffer.cpp, track shared_ptr usage
std::shared_ptr<Buffer> Buffer::create(...) {
    auto buffer = std::make_shared<Buffer>(...);

    // DEBUG: Log ref count
    std::cerr << "Buffer created: addr=0x" << std::hex << buffer->address()
              << " ref_count=" << buffer.use_count() << "\n";

    return buffer;
}
```

---

## Advanced: Integrate Stack Traces Into TT-Metal

### Option 1: Modify allocation_client.cpp

**File:** `tt_metal/impl/allocator/allocation_client.cpp`

Add this at the top:
```cpp
#include <execinfo.h>
#include <cxxabi.h>
#include <fstream>

static std::string get_stack_trace() {
    void* callstack[64];
    int frames = backtrace(callstack, 64);
    char** symbols = backtrace_symbols(callstack, frames);

    std::stringstream ss;
    for (int i = 2; i < std::min(frames, 12); i++) {  // Skip first 2, show 10
        ss << symbols[i] << "\\n";
    }
    free(symbols);
    return ss.str();
}
```

Modify `send_allocation_message()`:
```cpp
void AllocationClient::send_allocation_message(...) {
    // Existing code to send message...

    // NEW: Log stack trace if detailed tracking enabled
    const char* trace_env = std::getenv("TT_ALLOC_STACK_TRACE");
    if (trace_env && std::string(trace_env) == "1") {
        std::ofstream log("/tmp/tt_alloc_traces.log", std::ios::app);
        log << "ALLOC device=" << device_id
            << " size=" << size
            << " addr=0x" << std::hex << buffer_id << std::dec << "\\n";
        log << get_stack_trace() << "\\n";
    }
}
```

**Recompile TT-Metal:**
```bash
cmake --build build
```

### Option 2: Use Address Sanitizer (ASAN)

ASAN will automatically detect leaks and show stack traces!

```bash
# Build with ASAN
./build.sh --asan

# Run your test
export ASAN_OPTIONS=detect_leaks=1:log_path=/tmp/asan.log
python your_test.py

# Check results
cat /tmp/asan.log.*
```

ASAN output will show:
```
==12345==ERROR: LeakSanitizer: detected memory leaks

Direct leak of 1024 byte(s) in 1 object(s) allocated from:
    #0 0x7f... in malloc
    #1 0x7f... in allocate_buffer
    #2 0x7f... in Buffer::allocate_impl()
    #3 0x7f... in your_function()
    #4 0x7f... in main
```

---

## Practical Debugging Session

### Scenario: Find Why 87 Buffers Leaked

**Step 1:** Run with enhanced server
```bash
./allocation_server_enhanced &
export TT_ALLOC_TRACKING_ENABLED=1
python your_app.py
```

**Step 2:** Dump remaining buffers
```bash
python dump_remaining_buffers.py
```

Output:
```
Device 0:
  L1: 23 buffers, 14.5 MB total
    - Buffer 0x1000000: 1024 KB (PID 12345, ref_count=1)
      History (allocs=1 frees=0):
        ALLOC size=1048576 pid=12345 ref=1
  DRAM: 10 buffers, 5.2 MB total
```

**Step 3:** Add logging to find where these were allocated
```bash
export TT_ALLOC_STACK_TRACE=1
python your_app.py
grep "0x1000000" /tmp/tt_alloc_stack_traces.log
```

**Step 4:** Fix the leak
- Look at the stack trace
- Find the code that allocated the buffer
- Check why the destructor wasn't called
- Add proper cleanup or fix reference counting

---

## Summary

### To Answer Your Questions:

**Q: How can we trace where buffers come from?**
A:
1. Use `allocation_server_enhanced.cpp` (shows history)
2. Enable `TT_ALLOC_STACK_TRACE=1` (shows call stacks)
3. Add logging to `Buffer::~Buffer()` (shows destructor calls)
4. Use ASAN (automatic leak detection with stacks)

**Q: Why are there unknown deallocations?**
A:
- Buffer addresses are **reused** by the GPU memory allocator
- Same address can hold different buffers over time
- Server tracks current state only
- Solution: Enhanced server tracks full history

**Q: Why are buffers not deallocated at the end?**
A:
- Destructors were **never called** for those buffers
- Check reference counting (`shared_ptr.use_count()`)
- Look for leaks: global variables, circular refs, exceptions
- Solution: Use ASAN or add destructor logging

### Files Created for You:
1. `allocation_server_enhanced.cpp` - Tracks buffer history
2. `allocation_client_with_stacktrace.cpp` - Example with stack traces
3. `BUFFER_ALLOCATION_TRACING_GUIDE.md` - Complete flow documentation
4. `HOW_TO_TRACE_BUFFER_ORIGINS.md` - This file

### Next Steps:
1. Run `allocation_server_enhanced` instead of `allocation_server_poc`
2. Enable `TT_ALLOC_STACK_TRACE=1` to capture origins
3. Check `/tmp/tt_alloc_stack_traces.log` for exact call stacks
4. Add destructor logging to find what's not being cleaned up
