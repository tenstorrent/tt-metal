# CRITICAL: Fixing Circular Buffer and System Buffer Memory Leaks

## Problem Confirmed

The **36KB DRAM + 22KB L1 ARE accumulating** across successive runs:
```
After RUN 1: Active allocations: 3
During RUN 2: Active allocations: 11 (increased by 8)
During RUN 2: Active allocations: 27 (increased by 16)
```

**These are TRUE MEMORY LEAKS, not just untracked deallocations.**

## The Leaking Buffers

### 1. Circular Buffers (L1) - 3 buffers on Device 0
- `buffer_id=101152`: 4096 bytes (4KB)
- `buffer_id=105248`: 4096 bytes (4KB)
- `buffer_id=109344`: 2048 bytes (2KB)
- **Total**: ~10KB L1

### 2. System/Dispatch Buffer (DRAM) - Multiple devices
- `buffer_id=1073737728`: 24576 bytes (24KB) each
- **Allocated**: 18 times across devices
- **Total**: ~430KB DRAM

## Root Causes

### Circular Buffer Leak
**Problem**: `track_deallocate_cb()` doesn't have CB addresses, so it can't track deallocations.
**But**: The CBs themselves are also NOT being deallocated from device memory!

### System Buffer Leak
**Problem**: Buffer `1073737728` is allocated but never deallocated.
**Likely cause**: This is a special buffer type that bypasses normal Buffer deallocation.

## Where to Look

### 1. Find where CBs are created and destroyed

```bash
cd /home/tt-metal-apv
grep -r "track_allocate_cb" tt_metal/ --include="*.cpp" | head -5
grep -r "track_deallocate_cb" tt_metal/ --include="*.cpp" | head -5
```

Find the code that creates CBs and ensure it also destroys them.

### 2. Find buffer 1073737728

This address `1073737728 = 0x40000000` is suspicious - it's a power of 2, suggesting a fixed address.

```bash
cd /home/tt-metal-apv
grep -r "0x40000000" tt_metal/ --include="*.cpp" --include="*.hpp" | head -10
```

### 3. Check program/device cleanup

These buffers might be associated with Programs or device initialization.
Check if:
- Programs are being properly destroyed
- Device cleanup (`CloseDevice`) is deallocating all buffers
- Mesh device cleanup is complete

## Immediate Action Items

### Step 1: Add CB Address Tracking

Modify `graph_tracking.cpp` to store CB addresses so they can be deallocated:

```cpp
// Add to GraphTracker class
private:
    struct CBAllocation {
        uint64_t addr;
        uint64_t size;
        int device_id;
    };
    std::unordered_map<const IDevice*, std::vector<CBAllocation>> cb_allocations_;
    std::mutex cb_mutex_;

public:
    void track_allocate_cb(...) {
        // Existing code...

        // Store for later deallocation
        std::lock_guard<std::mutex> lock(cb_mutex_);
        cb_allocations_[device].push_back({addr, size, device->id()});
    }

    void track_deallocate_cb(const IDevice* device) {
        std::lock_guard<std::mutex> lock(cb_mutex_);
        auto it = cb_allocations_.find(device);
        if (it != cb_allocations_.end()) {
            // Report all CBs for this device as deallocated
            for (const auto& cb : it->second) {
                if (AllocationClient::is_enabled()) {
                    AllocationClient::report_deallocation(cb.addr);
                }
            }
            cb_allocations_.erase(it);
        }
    }
```

### Step 2: Find and Fix Buffer 1073737728

1. Search for where it's created
2. Ensure it goes through proper deallocation
3. If it's a special buffer, add explicit tracking

### Step 3: Verify Program Cleanup

Check if programs are being properly destroyed when mesh device closes.

## Testing the Fix

After implementing fixes:

```bash
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/memory_utilization_monitor

./allocation_server_poc > leak_test.log 2>&1 &
export TT_ALLOC_TRACKING_ENABLED=1

# Run 3 times
python test_mesh_allocation.py && sleep 5
python test_mesh_allocation.py && sleep 5
python test_mesh_allocation.py && sleep 5

# Check results
grep "Active allocations:" leak_test.log | tail -10
```

**Expected after fix:**
```
Active allocations: 3
Active allocations: 3
Active allocations: 3
```

**Success criteria**: Number stays constant, doesn't increase.

## Priority

**HIGH** - This is a true memory leak that will cause problems in production use.

The buffers ARE being allocated on the device but NOT being freed when the mesh device/programs are destroyed.
