# ✅ Circular Buffer Deallocation Tracking - IMPLEMENTED

## Changes Made

### 1. Program Destructor (program.cpp)
Modified `Program::~Program()` to call `deallocate_circular_buffers()`:
```cpp
Program::~Program() noexcept {
    if (internal_) {
        internal_->deallocate_circular_buffers();
    }
}
```

### 2. Added Deallocation Method (program.cpp)
Implemented `ProgramImpl::deallocate_circular_buffers()`:
```cpp
void detail::ProgramImpl::deallocate_circular_buffers() {
    if (cb_device_ != nullptr) {
        tt::tt_metal::GraphTracker::instance().track_deallocate_cb(cb_device_);
        cb_device_ = nullptr;
    }
}
```

### 3. Store Device Pointer (program.cpp)
Modified `allocate_circular_buffers()` to store the device pointer:
```cpp
// At end of allocate loop:
this->cb_device_ = device;
```

### 4. Added Device Member (program_impl.hpp)
Added `cb_device_` member to ProgramImpl:
```cpp
const IDevice* cb_device_ = nullptr;
```

### 5. GraphTracker CB Tracking (graph_tracking.hpp)
Added CB allocation tracking structures:
```cpp
struct CBAllocation {
    uint64_t addr;
    uint64_t size;
};
std::mutex cb_mutex;
std::unordered_map<const IDevice*, std::vector<CBAllocation>> device_cb_allocations;
```

### 6. Store CB Addresses (graph_tracking.cpp)
Modified `track_allocate_cb()` to store addresses:
```cpp
std::lock_guard<std::mutex> lock(cb_mutex);
device_cb_allocations[device].push_back({addr, size});
```

### 7. Report CB Deallocations (graph_tracking.cpp)
Implemented proper `track_deallocate_cb()`:
```cpp
// Retrieve stored CBs
auto it = device_cb_allocations.find(device);
if (it != device_cb_allocations.end()) {
    cbs_to_deallocate = std::move(it->second);
    device_cb_allocations.erase(it);
}

// Report each deallocation
for (const auto& cb : cbs_to_deallocate) {
    AllocationClient::report_deallocation(cb.addr);
}
```

## What This Fixes

**Before:**
- Programs allocated circular buffers (~10KB L1 per program)
- Programs were destroyed without deallocating CBs
- Tracking server never received deallocation notifications
- Memory accumulated: 3 → 11 → 27 active allocations

**After:**
- Programs now call `track_deallocate_cb()` in destructor
- GraphTracker reports all CB deallocations using stored addresses
- Tracking server receives deallocation notifications
- Memory should return to baseline after each run

## Remaining Issue: Buffer 1073737728

**Still not fixed:** The DRAM buffer at address `1073737728` (0x40000000)
- Size: 24576 bytes (24KB)
- Allocated 18 times across devices
- Likely a dispatch/system buffer

This needs separate investigation in Step 3.

## Testing

After rebuild, test with:
```bash
cd /home/tt-metal-apv
./build_metal.sh
pip install -e .

cd tt_metal/programming_examples/alexp_examples/memory_utilization_monitor
./allocation_server_poc > test_fix.log 2>&1 &
export TT_ALLOC_TRACKING_ENABLED=1

# Run 3 times
python test_mesh_allocation.py && sleep 5
python test_mesh_allocation.py && sleep 5
python test_mesh_allocation.py && sleep 5

# Check results
grep "Active allocations:" test_fix.log | tail -10
```

**Expected:** L1 allocations should be freed now (no longer accumulating).
**Remaining:** DRAM buffer 1073737728 still needs investigation.

##Files Modified

1. `/home/tt-metal-apv/tt_metal/impl/program/program.cpp`
2. `/home/tt-metal-apv/tt_metal/impl/program/program_impl.hpp`
3. `/home/tt-metal-apv/tt_metal/graph/graph_tracking.cpp`
4. `/home/tt-metal-apv/tt_metal/api/tt-metalium/graph_tracking.hpp`

## Next Step

Investigate and fix buffer `1073737728` (0x40000000) - likely dispatch-related.
