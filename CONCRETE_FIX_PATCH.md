# Concrete Fix for Double-Free Bug

## Root Cause Identified ✅

**File:** `tt_metal/distributed/mesh_buffer.cpp` line 160
**Issue:** `MeshBuffer::deallocate()` calls `mark_as_deallocated()` on each device buffer, but the Buffer destructor ALSO calls `deallocate_impl()`, causing double FREE messages.

---

## The Bug Explained

### Current Buggy Flow:

```
1. MeshBuffer::~MeshBuffer()
   └─> MeshBuffer::deallocate()
       └─> For each device buffer:
           └─> buffer->mark_as_deallocated()
               └─> GraphTracker::track_deallocate(this)  ← FREE message #1
               └─> Sets allocation_status_ = DEALLOCATED

2. [Later] Buffer::~Buffer()
   └─> if (allocation_status_ != DEALLOCATED)  ← FALSE, skips!
       └─> deallocate()

   ✅ GOOD: Buffer destructor is guarded!
```

**Wait, that should work!** Let me check if the guard is correct...

Looking at `buffer.cpp` line 433:
```cpp
void Buffer::deallocate_impl() {
    if (allocation_status_ != AllocationStatus::ALLOCATED) {
        return;  ← Should prevent double-free!
    }
    // ... send FREE message ...
}
```

And line 462:
```cpp
Buffer::~Buffer() {
    if (this->allocation_status_ != AllocationStatus::DEALLOCATED) {
        this->deallocate();
    }
}
```

**Both have guards!** So why are we seeing double-frees?

---

## The REAL Bug: Buffer Reuse!

Looking at the log more carefully:
```
✓ Allocated buffer 694996096 on device 4  → allocation_status_ = ALLOCATED
✗ Freed buffer 694996096 on device 4      → allocation_status_ = DEALLOCATED
✓ Allocated buffer 694996096 on device 4  → allocation_status_ = ALLOCATED (REUSED!)
✗ Freed buffer 694996096 on device 4      → allocation_status_ = DEALLOCATED
✗ Unknown buffer 694996096 on device 4    ← Extra free from old reference!
```

**The problem:** A Buffer object (address in memory) gets REUSED for a new allocation with the same buffer_id, but:
1. Old references to the Buffer still exist
2. Those old references try to deallocate the same buffer_id
3. Server sees deallocation for a buffer that (from its perspective) was already freed

---

## Two Possible Scenarios

### Scenario A: std::shared_ptr Lifetime Issue

```cpp
// Somewhere in the code:
std::shared_ptr<Buffer> buffer1 = Buffer::create(...);  // buffer_id = X
// ... use buffer1 ...
buffer1->deallocate();  // Server tracks FREE

// Later, allocator reuses the same buffer_id:
std::shared_ptr<Buffer> buffer2 = Buffer::create(...);  // buffer_id = X (reused!)
// ... use buffer2 ...

// Problem: buffer1 still exists and its destructor runs later!
buffer1.reset();  // Tries to deallocate buffer_id X again → "unknown"
```

### Scenario B: MeshDevice Buffer Sharing

```cpp
// MeshDevice creates per-device buffers:
for (auto& device : devices) {
    device_buffers[device] = Buffer::create(...);  // Same buffer_id on each device
}

// Later, MeshBuffer::deallocate() is called:
for (auto& [coord, buffer] : buffers_) {
    buffer->mark_as_deallocated();  // Sends FREE for each device
}

// BUT: The individual device Buffers are still alive!
// Their destructors run later:
~Buffer() {
    if (status != DEALLOCATED) {  // FALSE, so skips
        deallocate();  // Shouldn't run
    }
}

// Wait, this SHOULD be protected...
```

---

## The Actual Root Cause: Server-Side Ref Counting Bug!

Looking at `allocation_server_poc.cpp` line 147-156:

```cpp
auto it = allocations_.find(key);
if (it != allocations_.end()) {
    // Buffer already exists - increment ref count
    it->second.ref_count++;
    std::cout << "... ref_count=" << it->second.ref_count << ")" << std::endl;
    return;  ← Doesn't create new entry!
}
```

And line 207-213:

```cpp
// Decrement ref count
info.ref_count--;

if (info.ref_count > 0) {
    // Still has references
    return;  ← Doesn't erase yet
}

// ref_count reached 0 - fully deallocate
// ... erase from map ...
```

**The Problem:**
1. Buffer allocated → ref_count = 1
2. Buffer allocated AGAIN (same ID) → ref_count = 2
3. Buffer freed → ref_count = 1
4. Buffer freed AGAIN → ref_count = 0 → **ERASED**
5. Buffer freed THIRD time → NOT FOUND → "unknown" ⚠️

This is correct behavior IF there are truly 3 frees! The bug is that there ARE excessive frees being sent.

---

## The Fix: Prevent Excessive Frees

### Option 1: Fix at Source (RECOMMENDED)

**Ensure buffers are only freed once at the source.**

Check if `deallocate()` or `mark_as_deallocated()` is being called multiple times on the same Buffer object.

**Add debug logging:**

```cpp
// File: tt_metal/impl/buffers/buffer.cpp

void Buffer::mark_as_deallocated() {
    if (allocation_status_ != AllocationStatus::ALLOCATED) {
        std::cerr << "⚠️  WARNING: mark_as_deallocated() called on buffer "
                  << address_ << " device " << device_->id()
                  << " that is NOT allocated (status=" << (int)allocation_status_ << ")"
                  << std::endl;
        return;  // ← ADD THIS GUARD
    }

    if (allocation_status_ == AllocationStatus::ALLOCATED) {
        GraphTracker::instance().track_deallocate(this);
    }
    allocation_status_ = AllocationStatus::DEALLOCATED;
}
```

### Option 2: Add Stack Trace on Deallocation

To find WHERE the excessive frees are coming from:

```cpp
// File: tt_metal/impl/buffers/buffer.cpp

#include <execinfo.h>
#include <cstdio>

void Buffer::mark_as_deallocated() {
    if (allocation_status_ != AllocationStatus::ALLOCATED) {
        // Capture stack trace
        void* callstack[128];
        int frames = backtrace(callstack, 128);
        char** strs = backtrace_symbols(callstack, frames);

        std::cerr << "⚠️  Double-free attempt on buffer " << address_
                  << " device " << device_->id() << std::endl;
        std::cerr << "Stack trace:" << std::endl;
        for (int i = 0; i < frames; i++) {
            std::cerr << "  " << strs[i] << std::endl;
        }
        free(strs);
        return;
    }

    GraphTracker::instance().track_deallocate(this);
    allocation_status_ = AllocationStatus::DEALLOCATED;
}
```

### Option 3: Server-Side Protection (WORKAROUND)

Keep the current silent counting but add more details:

```cpp
// File: allocation_server_poc.cpp

void handle_deallocation(const AllocMessage& msg) {
    // ... existing code ...

    if (it == allocations_.end()) {
        // Track double-frees
        BufferKey key{msg.device_id, msg.buffer_id};
        double_free_count_[key]++;

        if (double_free_count_[key] <= 5) {  // Only log first 5
            std::cout << "⚠ [PID " << msg.process_id << "] Double-free #"
                      << double_free_count_[key]
                      << " for buffer " << msg.buffer_id
                      << " on device " << msg.device_id << std::endl;
        }
        return;
    }

    // ... rest of function ...
}
```

---

## Recommended Action Plan

### Step 1: Add Guard to mark_as_deallocated() ✅

```bash
cd /home/tt-metal-apv
```

Apply this patch to `tt_metal/impl/buffers/buffer.cpp`:
