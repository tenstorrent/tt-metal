# 36KB DRAM Leak Analysis

## Summary
The remaining **36KB DRAM per device** (and **12KB L1 on device 0**) are leaks from dispatch/system buffers that are not being properly deallocated.

## Findings

### DRAM Leak (36KB per device)
From the server logs in `verify_cleanup.log`:

```
Buffer: 0x3FFFF000 (1073737728) - 12KB allocations
Buffer: 0x3FFFF800 (1073739776) - 12KB allocations
```

**Pattern:**
- **Allocated**: 2 times per device × 8 devices = 16 allocations per buffer
- **Freed**: 1 time per device × 8 devices = 8 deallocations per buffer
- **Remaining**: 1 × 12KB per device × 3 buffers = 36KB per device

**Buffer Characteristics:**
- Located near **1GB boundary** (end of DRAM address space)
- These addresses are typical for **dispatch queue buffers** or **system infrastructure**
- Created during command queue initialization
- Associated with **Programs** (likely cached command queue programs)

### L1 Leak (12KB on Device 0)
- **Circular buffers** (CBs) from programs that persist
- Only on Device 0 because:
  - Programs are compiled and stored on a "reference" device (typically device 0)
  - Even when tensors are distributed across a mesh, the program kernels reside on device 0
  - Circular buffers are associated with these programs

## Root Causes

### 1. Program Caching
The system has a `program_cache` (see `device_impl.hpp:240`):
```cpp
program_cache::detail::ProgramCache program_cache_;
```

**Issue**: Programs (including command queue programs) may be:
- Created multiple times (e.g., on successive mesh device opens)
- Cached for reuse
- Their buffers are allocated but not all instances are tracked for deallocation

### 2. Command Queue Program Lifecycle
From `device.cpp:286`:
```cpp
void Device::init_command_queue_device() {
    this->command_queue_programs_.push_back(get_compiled_cq_program(this));
    // ...
}
```

**Issue**:
- `command_queue_programs_` is a `std::vector<std::unique_ptr<Program>>`
- Programs can be created multiple times during device initialization
- Buffers within these programs are allocated but may not all be freed

### 3. System Buffer Tracking Gap
System buffers are:
- Allocated at specific addresses (0x3FFFF000, 0x3FFFF800)
- Not regular user buffers (no Buffer object in many cases)
- May be allocated directly via allocator without going through Buffer::create
- Some may be **pre-allocated** and reused, but tracking doesn't account for multiple allocations at the same address

## Affected Code

### Buffer Addresses
```
0x3FFFF000 (1,073,737,728) - System/dispatch buffer
0x3FFFF800 (1,073,739,776) - System/dispatch buffer
```

### Key Files
1. **`/home/tt-metal-apv/tt_metal/impl/device/device.cpp`**
   - `init_command_queue_device()` - Creates command queue programs
   - `configure_command_queue_programs()` - Sets up dispatch infrastructure

2. **`/home/tt-metal-apv/tt_metal/impl/device/device_impl.hpp`**
   - `command_queue_programs_` - Stores CQ programs
   - `program_cache_` - Caches programs for reuse

3. **`/home/tt-metal-apv/tt_metal/impl/program/program.cpp`**
   - `Program` destructor - Deallocates program resources
   - May not be called for cached programs

4. **`/home/tt-metal-apv/tt_metal/impl/dispatch/...`**
   - Dispatch infrastructure that creates system buffers

## Potential Fixes

### Option 1: Track Program Buffers in Program Destructor
**Pros**: Natural ownership model
**Cons**: Programs may be destroyed after device, causing issues

### Option 2: Clear Program Cache on Device Close
```cpp
// In Device destructor or close()
program_cache_.clear();  // Force destruction of cached programs
```
**Pros**: Ensures all programs are destroyed
**Cons**: Cache is actually cleared (see `device.cpp:685`), but buffers may still persist

### Option 3: Track System Buffer Lifecycle
- Add explicit tracking for system/dispatch buffers
- Ensure they report allocation/deallocation even if they're reused
- Problem: They may be allocated without going through our tracking hooks

### Option 4: Fix Multiple Allocation Issue
The real problem is **why are these buffers allocated twice**?
- Is `get_compiled_cq_program()` called multiple times?
- Are programs being created fresh instead of being retrieved from cache?
- Is there a bug in the accumulation test that opens the mesh device multiple times?

## Recommended Next Steps

1. **Understand the double allocation**:
   - Add logging to `get_compiled_cq_program()` to see when it's called
   - Check if `test_accumulation.py` is opening mesh devices correctly
   - Verify if program cache is working as expected

2. **Fix Program destructor for system buffers**:
   - Ensure `Program::~Program()` deallocates all associated buffers
   - Currently it only handles circular buffers (CBs)
   - Need to add handling for system/dispatch buffers

3. **Add explicit system buffer deallocation**:
   - In `Device::close()` or destructor
   - Explicitly free command queue program buffers
   - Call `command_queue_programs_.clear()` before allocators are destroyed

## Test Case
The accumulation test (`test_accumulation.py`) demonstrates the issue:
```python
def run_single_test():
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 4))
    # ... use device ...
    ttnn.close_mesh_device(mesh_device)
```

Running this multiple times shows:
- **Run 1**: 36KB DRAM + 12KB L1 remain
- **Run 2**: 72KB DRAM + 24KB L1 remain (doubled!)
- **Run 3**: 108KB DRAM + 36KB L1 remain (tripled!)

This confirms buffers are accumulating across runs.
