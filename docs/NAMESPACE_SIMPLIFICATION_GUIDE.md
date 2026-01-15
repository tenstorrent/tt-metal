# Namespace Simplification Guide

## Problem Statement

Currently, TTNN device operations use a complex, hierarchical namespace structure that includes sub-categorization (e.g., `data_movement`, `normalization`, `eltwise`, etc.). This creates unnecessary verbosity and complexity in the codebase.

### Current Pattern (Problematic)

```cpp
// In sharded_to_interleaved_device_operation_types.hpp
namespace ttnn::operations::data_movement {
    struct sharded_to_interleaved_operation_attributes_t { ... };
    struct sharded_to_interleaved_tensor_args_t { ... };
    using sharded_to_interleaved_spec_return_value_t = TensorSpec;
    using sharded_to_interleaved_tensor_return_value_t = Tensor;
}

// In sharded_to_interleaved_device_operation.hpp
namespace ttnn::operations::data_movement {
    struct ShardedToInterleavedDeviceOperation {
        using operation_attributes_t = ttnn::operations::data_movement::sharded_to_interleaved_operation_attributes_t;
        using tensor_args_t = ttnn::operations::data_movement::sharded_to_interleaved_tensor_args_t;
        using spec_return_value_t = ttnn::operations::data_movement::sharded_to_interleaved_spec_return_value_t;
        using tensor_return_value_t = ttnn::operations::data_movement::sharded_to_interleaved_tensor_return_value_t;
        // ...
    };
}

// Public API in same file
namespace ttnn::prim {
    ttnn::operations::data_movement::ShardedToInterleavedDeviceOperation::tensor_return_value_t
    sharded_to_interleaved(...);
}
```

**Problems:**
1. **Redundant namespace nesting**: The `ttnn::operations::<category>` namespace adds no value and creates long, verbose type names
2. **Inconsistent usage**: Device operation types are in `ttnn::operations::<category>`, but public APIs are in `ttnn::prim`
3. **Unnecessary sub-categorization**: Categories like `data_movement`, `normalization`, etc. don't provide meaningful organization at the namespace level
4. **Verbose type references**: Requires fully qualified names like `ttnn::operations::data_movement::sharded_to_interleaved_operation_attributes_t`

## Solution: Unified Prim Namespace

All device operation types, structs, and public APIs should use a simplified namespace structure:

- **`ttnn::prim`** - For stable, production-ready operations
- **`ttnn::experimental::prim`** - For experimental or work-in-progress operations

### Naming Convention

1. **No sub-categorization**: Remove all intermediate namespace levels like `operations::data_movement`, `operations::normalization`, etc.
2. **Direct placement**: All types go directly into `ttnn::prim` or `ttnn::experimental::prim`
3. **Consistent structure**: Device operation types, structs, and public APIs all use the same namespace

### Example: ShardedToInterleaved Operation Refactoring

#### After (Desired Pattern)

```cpp
// In sharded_to_interleaved_device_operation_types.hpp
namespace ttnn::prim {
    struct sharded_to_interleaved_operation_attributes_t {
        tt::tt_metal::MemoryConfig output_mem_config;
        tt::tt_metal::DataType output_dtype{};
        uint32_t num_slices = 1;
        uint32_t slice_index = 0;
    };

    struct sharded_to_interleaved_tensor_args_t {
        Tensor input_tensor;
        std::optional<Tensor> preallocated_output;
    };

    using sharded_to_interleaved_spec_return_value_t = TensorSpec;
    using sharded_to_interleaved_tensor_return_value_t = Tensor;
}

// In sharded_to_interleaved_device_operation.hpp
namespace ttnn::prim {
    struct ShardedToInterleavedDeviceOperation {
        using operation_attributes_t = sharded_to_interleaved_operation_attributes_t;
        using tensor_args_t = sharded_to_interleaved_tensor_args_t;
        using spec_return_value_t = sharded_to_interleaved_spec_return_value_t;
        using tensor_return_value_t = sharded_to_interleaved_tensor_return_value_t;
        // ...
    };

    // Public API in same namespace
    tensor_return_value_t sharded_to_interleaved(
        const Tensor& input_tensor,
        const tt::tt_metal::MemoryConfig& output_mem_config,
        const tt::tt_metal::DataType& output_dtype,
        const std::optional<Tensor>& preallocated_output = std::nullopt);
}
```

#### Benefits

1. **Simpler type names**: `ttnn::prim::sharded_to_interleaved_operation_attributes_t` instead of `ttnn::operations::data_movement::sharded_to_interleaved_operation_attributes_t`
2. **Consistent namespace**: All related types and functions in the same namespace
3. **Easier to use**: Shorter, cleaner type references
4. **Less verbose**: No need for fully qualified names within the same namespace

## Migration Strategy

### Step 1: Identify All Affected Files

For each operation, identify:
- `*_device_operation_types.hpp` - Contains type definitions
- `*_device_operation.hpp` - Contains DeviceOperation struct
- `*_device_operation.cpp` - Contains implementations
- Any other files referencing these types

### Step 2: Update Namespace Declarations

1. Change `namespace ttnn::operations::<category>` to `namespace ttnn::prim` (or `ttnn::experimental::prim`)
2. Remove all fully qualified references to the old namespace
3. Update type aliases in DeviceOperation structs to use unqualified names (since they're in the same namespace)

### Step 3: Update All References

Search for and update:
- Type references in implementation files
- Type references in program factories
- Type references in public API wrappers
- Any cross-references between operations

### Step 4: Verify Compilation

Ensure all files compile and tests pass after namespace changes.

## Decision Criteria: `ttnn::prim` vs `ttnn::experimental::prim`

- **`ttnn::prim`**: Use for stable, production-ready operations that are part of the core API
- **`ttnn::experimental::prim`**: Use for:
  - Operations under active development
  - Operations that may have API changes
  - Operations in the `experimental/` directory structure
  - Operations marked as experimental in documentation

## Examples of Current Namespace Patterns to Fix

- `ttnn::operations::data_movement` → `ttnn::prim`
- `ttnn::operations::normalization` → `ttnn::prim`
- `ttnn::operations::eltwise` → `ttnn::prim`
- `ttnn::operations::reduction` → `ttnn::prim`
- `ttnn::operations::experimental::*` → `ttnn::experimental::prim`
- `ttnn::operations::moreh::*` → `ttnn::prim` (or `ttnn::experimental::prim` based on stability)

## Notes

- This is a breaking change that will require updating all references across the codebase
- Consider doing this migration incrementally, operation by operation
- Update documentation and examples to reflect the new namespace structure
- Ensure Python bindings are updated to match the new C++ namespace structure
