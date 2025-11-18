# Kernel Type Tracking - Generalist Solution

## Problem
We need to track kernel types (Fabric, Dispatch, Application) without hardcoding size checks.

## Solution: Program Metadata
Add a `kernel_type_hint` field to the Program class that flows through the call chain.

### 1. Add Enum to tt_metal.hpp

```cpp
namespace tt::tt_metal {

enum class KernelTypeHint : uint8_t {
    APPLICATION = 0,  // User kernels (default)
    FABRIC = 1,       // Fabric routing kernels
    DISPATCH = 2      // Command queue/dispatch kernels
};

}
```

### 2. Add Field to Program Class

```cpp
// In program_impl.hpp
class ProgramImpl {
    ...
    KernelTypeHint kernel_type_hint_ = KernelTypeHint::APPLICATION;

public:
    void set_kernel_type_hint(KernelTypeHint hint) { kernel_type_hint_ = hint; }
    KernelTypeHint get_kernel_type_hint() const { return kernel_type_hint_; }
    ...
};
```

### 3. Set Hint When Creating System Programs

```cpp
// In device.cpp - compile_fabric()
bool Device::compile_fabric() {
    if (fabric_program_) {
        return true;
    }
    fabric_program_ = std::make_unique<Program>();
    fabric_program_->impl().set_kernel_type_hint(KernelTypeHint::FABRIC);  // Mark as Fabric
    ...
}

// In device.cpp - init_command_queue_device()
void Device::init_command_queue_device() {
    this->command_queue_programs_.push_back(get_compiled_cq_program(this));
    auto& cq_program = this->command_queue_programs_[0];
    cq_program->impl().set_kernel_type_hint(KernelTypeHint::DISPATCH);  // Mark as Dispatch
    ...
}
```

### 4. Pass Hint Through to track_kernel_load

```cpp
// In program.cpp - finalize_offsets()
void ProgramImpl::finalize_offsets(IDevice* device) {
    ...
    // Get kernel type hint from program
    uint8_t kernel_type = static_cast<uint8_t>(this->get_kernel_type_hint());

    // Track kernel load with type
    if (kernel_size > 0) {
        for (const IDevice* dev : devices_to_track) {
            GraphTracker::instance().track_kernel_load(
                kernel_size,
                kernel_id,
                dev,
                kernel_type);  // Pass the hint!
        }
    }
}
```

## Benefits

✅ **Generalist**: No hardcoded sizes
✅ **Explicit**: Type is set where program is created
✅ **Clean**: Flows naturally through existing call chain
✅ **Maintainable**: Easy to add new types if needed
✅ **Backward compatible**: Defaults to APPLICATION for existing code

## Implementation Steps

1. Add `KernelTypeHint` enum to `tt_metal.hpp` or `base_types.hpp`
2. Add `kernel_type_hint_` field to `ProgramImpl`
3. Set hint in `compile_fabric()` → FABRIC
4. Set hint in `init_command_queue_device()` / `get_compiled_cq_program()` → DISPATCH
5. Pass hint to `track_kernel_load()` in:
   - `ProgramImpl::finalize_offsets()`
   - `ProgramImpl::finalize_program_offsets()`
   - `detail::TrackKernelDispatch()`
6. Rebuild tt_metal library
7. Test

## Alternative: Environment Context

If modifying Program is too invasive, use thread-local or global context:

```cpp
// In tt_metal namespace
thread_local KernelTypeHint g_current_kernel_type_hint = KernelTypeHint::APPLICATION;

struct KernelTypeHintGuard {
    KernelTypeHint previous;
    KernelTypeHintGuard(KernelTypeHint hint) : previous(g_current_kernel_type_hint) {
        g_current_kernel_type_hint = hint;
    }
    ~KernelTypeHintGuard() {
        g_current_kernel_type_hint = previous;
    }
};

// Usage:
void Device::configure_fabric() {
    KernelTypeHintGuard guard(KernelTypeHint::FABRIC);
    fabric_program_->impl().finalize_offsets(this);  // Will see FABRIC hint
    ...
}

// In track_kernel_load:
void GraphTracker::track_kernel_load(...) {
    uint8_t kernel_type = static_cast<uint8_t>(g_current_kernel_type_hint);
    ...
}
```

This approach requires NO changes to Program class or function signatures!

## Recommendation

Use **Environment Context** (Alternative) approach:
- ✅ No API changes needed
- ✅ No struct changes needed
- ✅ Just add guard in `configure_fabric()` and `configure_command_queue_programs()`
- ✅ Auto-resets on scope exit
- ✅ Thread-safe with thread_local

Much simpler and cleaner!
