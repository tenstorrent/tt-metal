# Kernel Type Tracking - Final Implementation

## ✅ Solution: Program Metadata Approach

Clean, explicit, maintainable solution that stores kernel type directly in the Program object.

## What Was Implemented

### 1. Added Enum (program_impl.hpp)

```cpp
namespace tt::tt_metal::detail {

enum class ProgramKernelType : uint8_t {
    APPLICATION = 0,  // User application kernels (default)
    FABRIC = 1,       // Fabric routing/communication kernels
    DISPATCH = 2      // Command queue dispatch kernels
};

}
```

### 2. Added Field to ProgramImpl (program_impl.hpp)

```cpp
class ProgramImpl {
public:
    // Kernel type classification for tracking
    void set_kernel_type(ProgramKernelType type) { kernel_type_ = type; }
    ProgramKernelType get_kernel_type() const { return kernel_type_; }

private:
    // Kernel type classification (for memory tracking purposes)
    ProgramKernelType kernel_type_ = ProgramKernelType::APPLICATION;
};
```

### 3. Set Type for Fabric Kernels (device.cpp)

```cpp
void Device::configure_fabric() {
    // ...
    // Mark program as Fabric type (for kernel tracking)
    fabric_program_->impl().set_kernel_type(tt::tt_metal::detail::ProgramKernelType::FABRIC);
    fabric_program_->impl().finalize_offsets(this);
    // ...
}
```

### 4. Set Type for Dispatch Kernels (device.cpp)

```cpp
void Device::configure_command_queue_programs() {
    // ...
    // Run the cq program (Mark as Dispatch type for kernel tracking)
    command_queue_program.impl().set_kernel_type(tt::tt_metal::detail::ProgramKernelType::DISPATCH);
    command_queue_program.impl().finalize_offsets(this);
    // ...
}
```

### 5. Use Type in Kernel Tracking (program.cpp)

**Slow Dispatch (finalize_offsets):**
```cpp
void ProgramImpl::finalize_offsets(IDevice* device) {
    // ...
    // Get kernel type from program metadata
    uint8_t kernel_type = static_cast<uint8_t>(this->kernel_type_);

    for (const IDevice* dev : devices_to_track) {
        GraphTracker::instance().track_kernel_load(
            kernel_size, kernel_id, dev, kernel_type);
    }
}
```

**Fast Dispatch (finalize_program_offsets):**
```cpp
uint32_t ProgramImpl::finalize_program_offsets(...) {
    // ...
    for (auto& program : programs) {
        // Get kernel type from program metadata
        uint8_t kernel_type = static_cast<uint8_t>(program->kernel_type_);

        if (kernel_size > 0) {
            for (const IDevice* dev : devices_to_track) {
                GraphTracker::instance().track_kernel_load(
                    kernel_size, kernel_id, dev, kernel_type);
            }
        }
    }
}
```

## Files Modified

1. ✅ `tt_metal/impl/program/program_impl.hpp` - Added enum, getters/setters, field
2. ✅ `tt_metal/impl/program/program.cpp` - Use type in both dispatch paths
3. ✅ `tt_metal/impl/device/device.cpp` - Set type for Fabric and Dispatch programs
4. ✅ `tt_metal/api/tt-metalium/graph_tracking.hpp` - Removed thread-local guard
5. ✅ `tt_metal/graph/graph_tracking.cpp` - Removed thread-local implementation

## Benefits

✅ **Explicit**: Type is set exactly where the program is created
✅ **Persistent**: Stored in the program object itself, no hidden state
✅ **Debuggable**: Can inspect `program->get_kernel_type()` at any time
✅ **Clean**: No size heuristics, no thread-local magic
✅ **Type-safe**: Enum class prevents invalid values
✅ **Minimal**: Only 5 files, ~20 lines of actual code
✅ **Default-safe**: Application kernels are the default (0)

## How It Works

1. When creating a Fabric program → `set_kernel_type(FABRIC)`
2. When creating a Dispatch program → `set_kernel_type(DISPATCH)`
3. User programs → Default to `APPLICATION`
4. When tracking kernel → Read `program->get_kernel_type()`
5. Server receives kernel type in KERNEL_LOAD message
6. Server stores kernel type in `KernelInfo` struct
7. tt_smi_umd can now display different types (future feature)

## Next Steps

1. **Build:** `./build_lib.sh`
2. **Test:** Run test workload and check server log
3. **Verify:** See `kernel_type` field in KERNEL_LOAD messages
4. **Observe:** 56KB kernels → FABRIC, 46KB kernels → DISPATCH
5. **(Optional):** Update server to show kernel type in output

## Testing

```bash
# Start server
./build/programming_examples/allocation_server_poc &

# Run test
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and DP-4-b1"

# Check log - should see kernel types:
grep "KERNEL_LOAD" out.log
# Expected: Type=1 (Fabric) for 56KB, Type=2 (Dispatch) for 46KB
```

## What This Fixes

- ❌ **Before:** Hardcoded size checks (57344 → Fabric, 47104 → Dispatch)
- ✅ **After:** Explicit type stored in program metadata
- ❌ **Before:** Would break if kernel sizes change
- ✅ **After:** Works regardless of kernel size
- ❌ **Before:** Implicit, hard to debug
- ✅ **After:** Explicit, easy to inspect

Perfect generalist solution! 🎉
