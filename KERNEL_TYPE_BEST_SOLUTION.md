# Best Solution: Kernel Type in Program Metadata

## Why This Is The Best Approach

Instead of:
- ❌ Hardcoded size checks (fragile)
- ❌ Thread-local context (implicit, hard to debug)
- ❌ Adding parameters everywhere (invasive)

We do:
- ✅ **Store kernel type in the Program object itself**
- ✅ Set it once when creating the program
- ✅ Read it when tracking kernels
- ✅ Explicit, clean, maintainable

## Implementation

### 1. Add Enum to base_types.hpp

```cpp
// In tt_metal/impl/program/base_types.hpp or program_impl.hpp
namespace tt::tt_metal {

enum class ProgramKernelType : uint8_t {
    APPLICATION = 0,  // User kernels (default)
    FABRIC = 1,       // Fabric routing kernels
    DISPATCH = 2      // Command queue/dispatch kernels
};

}
```

### 2. Add Field to ProgramImpl

```cpp
// In program_impl.hpp, ProgramImpl class
private:
    ProgramKernelType kernel_type_ = ProgramKernelType::APPLICATION;

public:
    void set_kernel_type(ProgramKernelType type) { kernel_type_ = type; }
    ProgramKernelType get_kernel_type() const { return kernel_type_; }
```

### 3. Set Type When Creating System Programs

```cpp
// In device.cpp - compile_fabric()
bool Device::compile_fabric() {
    fabric_program_ = tt::tt_fabric::create_and_compile_fabric_program(this);
    if (fabric_program_) {
        fabric_program_->impl().set_kernel_type(ProgramKernelType::FABRIC);
    }
    return fabric_program_ != nullptr;
}

// In dispatch.cpp or wherever get_compiled_cq_program is
std::unique_ptr<Program> get_compiled_cq_program(Device* device) {
    auto program = std::make_unique<Program>();
    program->impl().set_kernel_type(ProgramKernelType::DISPATCH);
    // ... rest of setup ...
    return program;
}
```

### 4. Use Type in Kernel Tracking

```cpp
// In program.cpp - finalize_offsets()
void ProgramImpl::finalize_offsets(IDevice* device) {
    // ...
    uint8_t kernel_type = static_cast<uint8_t>(this->kernel_type_);

    for (const IDevice* dev : devices_to_track) {
        GraphTracker::instance().track_kernel_load(
            kernel_size,
            kernel_id,
            dev,
            kernel_type);
    }
}

// Same in finalize_program_offsets() for Fast Dispatch
```

## Benefits

✅ **Explicit**: Type is set where program is created
✅ **Persistent**: Stored in the program object, no magic
✅ **Debuggable**: Can inspect program->get_kernel_type() anytime
✅ **Clean**: No hidden state, no size heuristics
✅ **Minimal changes**: Only touch program creation and tracking
✅ **Type-safe**: Enum class prevents invalid values

## Files to Modify

1. `tt_metal/impl/program/program_impl.hpp` - Add enum and field
2. `tt_metal/impl/device/device.cpp` - Set type in `compile_fabric()`
3. `tt_metal/impl/device/dispatch.cpp` or similar - Set type in `get_compiled_cq_program()`
4. `tt_metal/impl/program/program.cpp` - Use type in `finalize_offsets()` and `finalize_program_offsets()`

That's it! Only 4 files, minimal changes, maximum clarity.
