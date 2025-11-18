# Fast Dispatch CB and Kernel Tracking Fix

## Problem Discovered

When running tests with Fast Dispatch, **NO CB_ALLOC or KERNEL_LOAD messages** were appearing in the allocation server log, even though regular buffer allocations were being tracked correctly.

### Root Causes

1. **Globally Allocated CBs Were Skipped**
   - Most circular buffers in modern workloads use "global allocation" (allocated via the standard allocator)
   - The tracking code had an early `continue` at line 900-902 that skipped globally allocated CBs
   - This meant ~100% of CBs were not being tracked!

2. **Kernel Tracking Was Only in Slow Dispatch Path**
   - Kernel tracking was added to `ProgramImpl::finalize_offsets()` (instance method)
   - Fast Dispatch uses `ProgramImpl::finalize_program_offsets()` (static method) which processes multiple programs
   - The instance method is ONLY called for Slow Dispatch single programs

## Solutions Implemented

### 1. Track Globally Allocated CBs (`program.cpp` lines 900-910)

```cpp
uint64_t base_cb_address = device->allocator()->get_base_allocator_addr(HalMemType::L1);
for (const auto& circular_buffer : this->circular_buffers_) {
    if (circular_buffer->globally_allocated()) {
        // ✅ NEW: Track globally allocated CBs too (they use L1 memory allocated via the allocator)
        for (const IDevice* dev : devices_to_track) {
            tt::tt_metal::GraphTracker::instance().track_allocate_cb(
                circular_buffer->core_ranges(),
                circular_buffer->address(),
                circular_buffer->size(),
                circular_buffer->globally_allocated(),
                dev);
        }
        continue;
    }
    // ... rest of locally allocated CB tracking ...
}
```

**Why this works:**
- Globally allocated CBs still consume L1 memory
- They have valid addresses from the allocator
- Now we track them just like locally allocated CBs

### 2. Add Kernel Tracking to Fast Dispatch Path (`program.cpp` lines 1876-1904)

```cpp
// In ProgramImpl::finalize_program_offsets() - the STATIC method used by Fast Dispatch
for (auto& program : programs) {
    program->kernel_bins_sizeB = state.kernel_text_size;
    max_program_sizeB = std::max(max_program_sizeB, state.kernel_text_size);

    // ✅ NEW: Track kernel load for Fast Dispatch (kernel_bins_sizeB now set)
    if (program->kernel_bins_sizeB > 0) {
        // Determine which devices to track
        std::vector<const IDevice*> devices_to_track;
        const tt::tt_metal::distributed::MeshDevice* mesh_device =
            dynamic_cast<const tt::tt_metal::distributed::MeshDevice*>(device);

        if (mesh_device != nullptr) {
            // Mesh device: track all sub-devices
            for (IDevice* sub_device : mesh_device->get_devices()) {
                devices_to_track.push_back(sub_device);
            }
        } else {
            // Single device
            devices_to_track.push_back(device);
        }

        // Use program pointer as kernel identifier
        uint64_t kernel_id = reinterpret_cast<uint64_t>(program);

        // Report kernel load for all tracked devices
        for (const IDevice* dev : devices_to_track) {
            tt::tt_metal::GraphTracker::instance().track_kernel_load(
                program->kernel_bins_sizeB,
                kernel_id,
                dev);
        }
    }
}
```

**Why this works:**
- `finalize_program_offsets()` is called by `MeshWorkloadImpl::finalize_offsets()` at line 416 of `mesh_workload.cpp`
- This happens during Fast Dispatch compilation for MeshWorkloads
- Tracks kernel L1 usage for ALL programs in the workload
- Supports both single devices and MeshDevices

## Dispatch Path Comparison

### Slow Dispatch (Single Program)
```
LaunchProgram()
  └─> CompileProgram()
  └─> finalize_offsets() [INSTANCE METHOD]  ← Kernel tracking HERE (line 1764-1792)
  └─> TrackKernelDispatch()                 ← Also tracks here (redundant but safe)
```

### Fast Dispatch (MeshWorkload)
```
EnqueueMeshWorkload()
  └─> MeshWorkloadImpl::compile()
       └─> compile_program() [for each device range]
            └─> allocate_circular_buffers()  ← CB tracking HERE (line 900-910)
       └─> MeshWorkloadImpl::finalize_offsets()
            └─> ProgramImpl::finalize_program_offsets() [STATIC]  ← Kernel tracking HERE (line 1876-1904)
  └─> load_binaries()
  └─> generate_dispatch_commands()
```

## Testing

After rebuilding with these fixes:
1. Start the allocation server: `./build/programming_examples/allocation_server_poc > out.log 2>&1 &`
2. Run a test with tracking enabled: `export TT_ALLOC_TRACKING_ENABLED=1`
3. Run any Fast Dispatch test (e.g., transformer demo with `DP-4`)
4. Check `out.log` for `CB_ALLOC` and `KERNEL_LOAD` messages

**Expected result:**
- ✅ CB_ALLOC messages for all circular buffers (including globally allocated ones)
- ✅ KERNEL_LOAD messages for all programs with kernel code
- ✅ Tracking works across all devices in MeshDevice workloads

## Files Modified

1. **`tt_metal/impl/program/program.cpp`**
   - Line 900-910: Added tracking for globally allocated CBs
   - Line 1876-1904: Added kernel tracking in static `finalize_program_offsets()`
   - Line 1764-1792: Kept existing kernel tracking in instance `finalize_offsets()` (for Slow Dispatch)

## Build Command

```bash
cd /home/ttuser/aperezvicente/tt-metal-apv
cmake --build build --target tt_metal -j$(nproc)
```

## Status

- ✅ Globally allocated CB tracking implemented
- ✅ Fast Dispatch kernel tracking implemented
- ✅ MeshDevice support for both CB and Kernel tracking
- ⏳ **Ready to rebuild and test**
