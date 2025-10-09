# Why matmul_multicore_reuse Allocations Are Not Tracked

## The Problem

When running `matmul_multicore_reuse.cpp`, the allocation server and monitor client **do not show any allocations**, even though the program clearly allocates DRAM and L1 buffers.

## Root Cause Analysis

### 1. **GraphTracker Hook Bypass**

The key issue is in `/home/tt-metal-apv/tt_metal/impl/buffers/buffer.cpp:385-392`:

```cpp
void Buffer::allocate_impl() {
    if (GraphTracker::instance().hook_allocate(this)) {
        address_ = 0;
        hooked_allocation_ = true;
        // ❌ ALLOCATOR IS COMPLETELY BYPASSED!
    } else {
        address_ = allocator_->allocate_buffer(this);  // ✅ Our tracking is here
    }

    allocation_status_ = AllocationStatus::ALLOCATED;
    GraphTracker::instance().track_allocate(this);
}
```

**What happens:**
1. When `Buffer::create()` is called, it calls `allocate_impl()`
2. `GraphTracker::instance().hook_allocate(this)` is checked **FIRST**
3. If GraphTracker has a hook installed, it returns `true` and the allocator is **completely bypassed**
4. Our `AllocationClient::report_allocation()` is in `Allocator::allocate_buffer()`, which is **never called**

### 2. **When Does GraphTracker Hook?**

GraphTracker hooks allocations when:
- Graph capture mode is active (for trace recording)
- A custom `IGraphHooks` implementation is registered
- The system is in a special profiling/tracing mode

### 3. **MeshBuffer Allocations**

For the matmul example, allocations go through:
```
distributed::MeshBuffer::create()
  → Buffer::create() (per device)
    → allocate_impl()
      → GraphTracker::hook_allocate() returns TRUE
        → Allocator is bypassed ❌
```

### 4. **Circular Buffer (L1) Allocations**

Circular buffers (L1) have a different path:
```
CreateCircularBuffer()
  → Program::allocate_circular_buffers()
    → CircularBufferAllocator::mark_address()
      → Does NOT go through Allocator::allocate_buffer() ❌
```

Circular buffers use a **separate allocation system** (`CircularBufferAllocator`) that doesn't go through the main `Allocator` at all!

## Why Our Tests Worked

Our simple tests (`test_tracking_cpp.cpp`, `test_mesh_allocation_cpp.cpp`) worked because:
1. They create buffers directly without GraphTracker hooks
2. They use simple `Buffer::create()` calls
3. GraphTracker is not active in these simple examples

## Solutions

### Option A: Hook GraphTracker (Recommended)

Add allocation tracking at the `Buffer::allocate_impl()` level, **before** the GraphTracker check:

```cpp
void Buffer::allocate_impl() {
    // Report allocation BEFORE any hooks
    if (AllocationClient::is_enabled()) {
        // Get size and type from buffer
        AllocationClient::report_allocation(
            device_->id(),
            size_,
            static_cast<uint8_t>(buffer_type_),
            0  // address not yet known
        );
    }

    if (GraphTracker::instance().hook_allocate(this)) {
        address_ = 0;
        hooked_allocation_ = true;
    } else {
        address_ = allocator_->allocate_buffer(this);
    }

    allocation_status_ = AllocationStatus::ALLOCATED;
    GraphTracker::instance().track_allocate(this);
}
```

**Problem:** Address is not known yet when hooked!

### Option B: Track in GraphTracker (Better)

Modify `GraphTracker::track_allocate()` to report to our allocation server:

```cpp
void GraphTracker::track_allocate(const Buffer* buffer) {
    // Report to allocation tracking server
    if (AllocationClient::is_enabled()) {
        AllocationClient::report_allocation(
            buffer->device()->id(),
            buffer->size(),
            static_cast<uint8_t>(buffer->buffer_type()),
            buffer->address()
        );
    }

    // Original tracking
    if (processors.empty()) {
        return;
    }
    for (auto& it : processors) {
        it->track_allocate(buffer);
    }
}
```

**Advantages:**
- Catches ALL buffer allocations (hooked or not)
- Address is already assigned
- Single point of tracking

### Option C: Track Circular Buffers Separately

For L1 circular buffers, add tracking in `Program::allocate_circular_buffers()`:

```cpp
void detail::ProgramImpl::allocate_circular_buffers(const IDevice* device) {
    // ... existing code ...

    for (const auto& circular_buffer : this->circular_buffers_) {
        // ... allocation logic ...

        // Report to tracking server
        if (AllocationClient::is_enabled()) {
            AllocationClient::report_allocation(
                device->id(),
                circular_buffer->size(),
                static_cast<uint8_t>(BufferType::L1),
                computed_addr
            );
        }
    }
}
```

## Recommended Approach

**Implement Option B (GraphTracker tracking)** because:
1. ✅ Catches all buffer allocations (DRAM, L1, system buffers)
2. ✅ Works with or without GraphTracker hooks
3. ✅ Single point of instrumentation
4. ✅ Address is already assigned
5. ✅ Minimal code changes

Then **add Option C** for circular buffers since they use a completely separate allocation path.

## Testing the Fix

After implementing the fix, test with:

```bash
# Terminal 1: Start server
./allocation_server_poc

# Terminal 2: Run matmul with tracking
cd /home/tt-metal-apv/build/programming_examples
export TT_ALLOC_TRACKING_ENABLED=1
./matmul_multicore_reuse

# Terminal 3: Monitor
./allocation_monitor_client -d 0 -r 500
```

You should now see:
- DRAM allocations for input/output buffers
- L1 allocations for circular buffers
- Real-time updates as the program runs

## Summary

The allocation tracking doesn't work for `matmul_multicore_reuse` because:
1. **GraphTracker hooks bypass the allocator** where our tracking is
2. **Circular buffers use a separate allocator** that we don't instrument
3. Our current tracking only works for **simple, direct buffer allocations**

The fix requires moving the tracking to a higher level (`GraphTracker::track_allocate()`) that sees ALL allocations, regardless of how they're made.
