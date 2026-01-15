# Proxy Operation Elimination Guide

## Overview

Many TTNN operations have an unnecessary "proxy" layer that just forwards calls to `prim::` functions. This guide explains how to eliminate these proxies by moving the implementation directly into `ttnn::`.

### What is a Proxy Operation?

A proxy operation is a wrapper that does nothing except forward to a `prim::` function:

```cpp
// BEFORE: Unnecessary proxy pattern
// foo.hpp
struct Foo {
    static Tensor invoke(const Tensor& input, ...);
};
constexpr auto foo = register_operation<"ttnn::foo", Foo>();

// foo.cpp
Tensor Foo::invoke(const Tensor& input, ...) {
    return ttnn::prim::foo(input, ...);  // Just forwards!
}
```

### Why Eliminate Proxies?

1. **Double tracing**: `device_operation::launch` already traces operations
2. **Unnecessary code**: Extra namespace layer and forwarding
3. **Maintenance burden**: Two places to update for any signature change
4. **Compile time**: Extra template instantiations

## Migration Approach

**Key insight**: Instead of `using prim::func;`, we move the implementation from `prim` namespace directly into `ttnn::`.

### Before (3-layer structure)
```
foo.hpp         → struct Foo + register_operation
foo.cpp         → Foo::invoke() calls prim::foo()
device/foo_device_operation.hpp → prim::foo declaration
device/foo_device_operation.cpp → prim::foo implementation
```

### After (2-layer structure)
```
foo.hpp         → ttnn::foo declaration (function, not struct)
foo.cpp         → ttnn::foo implementation (calls device_operation::launch directly)
device/foo_device_operation.hpp → DeviceOperation struct only (no prim::)
device/foo_device_operation.cpp → DeviceOperation methods only (no prim::)
```

## Migration Steps

### Step 1: Update the Device Operation Header

Remove the `prim::` function declaration:

```cpp
// BEFORE: device/foo_device_operation.hpp
namespace ttnn::prim {
Tensor foo(const Tensor& input, ...);
}

// AFTER: device/foo_device_operation.hpp
// Remove the prim:: namespace entirely - just keep the DeviceOperation struct
```

### Step 2: Move Implementation to foo.hpp

Move the function declaration directly into `ttnn::`:

```cpp
// AFTER: foo.hpp
#pragma once

#include "device/foo_device_operation.hpp"

namespace ttnn {

Tensor foo(
    const Tensor& input,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    ...);

}  // namespace ttnn
```

### Step 3: Move Implementation to foo.cpp

Move the prim implementation into the `ttnn::` function:

```cpp
// AFTER: foo.cpp
#include "foo.hpp"

namespace ttnn {

Tensor foo(
    const Tensor& input,
    const std::optional<MemoryConfig>& memory_config,
    ...) {
    // Any preprocessing (value_or, validation, etc.)
    const auto resolved_mem_config = memory_config.value_or(input.memory_config());

    using OperationType = operations::foo::FooDeviceOperation;
    return device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{...},
        OperationType::tensor_args_t{.input = input});
}

}  // namespace ttnn
```

### Step 4: Remove prim:: from Device Operation .cpp

```cpp
// BEFORE: device/foo_device_operation.cpp
namespace ttnn::prim {
Tensor foo(...) {
    using OperationType = ...;
    return device_operation::launch<OperationType>(...);
}
}

// AFTER: device/foo_device_operation.cpp
// Remove the prim:: namespace block entirely
```

### Step 5: Update Nanobind Bindings

Use `mod.def()` to bind the function directly:

```cpp
// AFTER: foo_nanobind.cpp
mod.def(
    "foo",
    &ttnn::foo,
    doc,
    nb::arg("input"),
    ...);
```

### Step 6: Build and Test

```bash
./build_metal.sh -c -e --debug --build-all
```

## Complete Example: reshard Operation

### Before

**device/reshard_device_operation.hpp:**
```cpp
namespace ttnn::prim {
Tensor reshard(const Tensor& input, const MemoryConfig& mem_config, ...);
}
```

**device/reshard_device_operation.cpp:**
```cpp
namespace ttnn::prim {
Tensor reshard(const Tensor& input, const MemoryConfig& mem_config, ...) {
    using OperationType = ReshardDeviceOperation;
    return device_operation::launch<OperationType>(...);
}
}
```

**reshard.hpp:**
```cpp
struct ReshardOperation {
    static Tensor invoke(...);
};
constexpr auto reshard = register_operation<"ttnn::reshard", ReshardOperation>();
```

**reshard.cpp:**
```cpp
Tensor ReshardOperation::invoke(...) {
    return ttnn::prim::reshard(...);
}
```

### After

**device/reshard_device_operation.hpp:**
```cpp
// Just the DeviceOperation struct, no prim:: namespace
struct ReshardDeviceOperation { ... };
```

**device/reshard_device_operation.cpp:**
```cpp
// Just DeviceOperation methods, no prim:: implementation
```

**reshard.hpp:**
```cpp
#pragma once
#include "device/reshard_device_operation.hpp"

namespace ttnn {
Tensor reshard(
    const Tensor& input_tensor,
    const MemoryConfig& memory_config,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);
}
```

**reshard.cpp:**
```cpp
#include "reshard.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn {
Tensor reshard(
    const Tensor& input_tensor,
    const MemoryConfig& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    using OperationType = operations::data_movement::reshard::ReshardDeviceOperation;
    return device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{.output_mem_config = memory_config},
        OperationType::tensor_args_t{
            .input_tensor = input_tensor,
            .optional_output_tensor = optional_output_tensor});
}
}
```

**reshard_nanobind.cpp:**
```cpp
mod.def(
    "reshard",
    &ttnn::reshard,
    doc,
    nb::arg("input_tensor").noconvert(),
    nb::arg("output_memory_config"),
    nb::arg("output_tensor").noconvert() = nb::none());
```

## Handling Near-Proxies

When a proxy does minor preprocessing before calling prim, move that logic into the `ttnn::` function:

### Pattern 1: `.value_or()` on optional params

```cpp
// In ttnn::foo():
const auto resolved_mem_config = memory_config.value_or(input.memory_config());
```

### Pattern 2: Computed flags

```cpp
// In ttnn::typecast():
bool fp32_dest_acc_en = (input.dtype() == DataType::FLOAT32) || ...;
```

### Pattern 3: Validation

```cpp
// In ttnn::foo():
if (optional_output.has_value()) {
    TT_FATAL(dtype == optional_output->dtype(), "dtype mismatch");
}
```

### Pattern 4: Early returns

```cpp
// In ttnn::sharded_to_interleaved():
if (!input_tensor.shard_spec().has_value()) {
    return input_tensor;  // Already interleaved
}
```

## Convenience Wrappers (Keep Separate)

Some wrappers provide real value. Keep them as separate functions:

```cpp
// fill_ones_rm is a convenience wrapper for fill_rm with hardcoded values
namespace ttnn {
Tensor fill_rm(..., float val_hi, float val_lo, ...);  // Full function

// Convenience wrapper - keep as registered operation
namespace operations::data_movement {
struct FillOnesRMOperation {
    static Tensor invoke(...) {
        return ttnn::fill_rm(..., 1.0f, 0.0f, ...);  // Hardcoded values
    }
};
}
constexpr auto fill_ones_rm = register_operation<"ttnn::fill_ones_rm", FillOnesRMOperation>();
}
```

## Breaking Changes

When migrating:

1. **Python API**: `ttnn.foo.name` properties will no longer exist (function instead of class)
2. **C++ API**: `ttnn::foo` changes from `registered_operation_t<>` to a function

## Checklist

- [ ] Remove `prim::` declaration from `device/foo_device_operation.hpp`
- [ ] Remove `prim::` implementation from `device/foo_device_operation.cpp`
- [ ] Update `foo.hpp` with function declaration (not struct)
- [ ] Update `foo.cpp` with implementation (calls `device_operation::launch` directly)
- [ ] Move any preprocessing logic (value_or, validation) into `foo.cpp`
- [ ] Update nanobind to use `mod.def()` directly
- [ ] Build with `./build_metal.sh -c -e --debug --build-all`
- [ ] Run relevant tests

## List of Proxy Operations

See `proxy_elimination_todo.txt` for the full list of proxy operations to migrate.
