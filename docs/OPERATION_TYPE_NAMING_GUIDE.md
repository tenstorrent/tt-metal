# Operation Type Naming Guide

## Problem Statement

In TTNN device operations, we have a redundant naming pattern where type aliases in `DeviceOperation` structs use the same names as the types defined in the namespace. This creates unnecessary verbosity and confusion.

### Current Pattern (Problematic)

```cpp
// In slice_device_operation_types.hpp
namespace ttnn::operations::data_movement::slice {
    struct operation_attributes_t { ... };
    struct tensor_args_t { ... };
}

// In slice_device_operation.hpp
struct SliceDeviceOperation {
    using operation_attributes_t = slice::operation_attributes_t;  // Redundant!
    using tensor_args_t = slice::tensor_args_t;                    // Redundant!
    // ...
};
```

The problem: The type name `operation_attributes_t` is identical to the namespace-qualified name `slice::operation_attributes_t`, making the alias pointless and the code less readable.

## Solution: Operation-Specific Type Names

Rename the types in the `*_device_operation_types.hpp` files to operation-specific names that reflect the operation's purpose.

### Naming Convention

1. **`operation_attributes_t` → `{Operation}Params`**
   - Contains configuration parameters and attributes for the operation
   - Example: `SliceParams`, `Conv2dParams`, `MatmulParams`

2. **`tensor_args_t` → `{Operation}Inputs`**
   - Contains input tensors and tensor-related arguments
   - Example: `SliceInputs`, `Conv2dInputs`, `MatmulInputs`

The `{Operation}` prefix should match the `DeviceOperation` struct name (e.g., `SliceDeviceOperation` → `SliceParams`/`SliceInputs`).

### Example: Slice Operation Refactoring

#### Before

```cpp
// slice_device_operation_types.hpp
namespace ttnn::operations::data_movement::slice {
    struct operation_attributes_t {
        ttnn::Shape slice_start;
        ttnn::Shape slice_end;
        // ...
    };

    struct tensor_args_t {
        Tensor input;
        // ...
    };
}

// slice_device_operation.hpp
struct SliceDeviceOperation {
    using operation_attributes_t = slice::operation_attributes_t;
    using tensor_args_t = slice::tensor_args_t;
    // ...
};
```

#### After

```cpp
// slice_device_operation_types.hpp
namespace ttnn::operations::data_movement::slice {
    struct SliceParams {
        ttnn::Shape slice_start;
        ttnn::Shape slice_end;
        // ...
    };

    struct SliceInputs {
        Tensor input;
        // ...
    };
}

// slice_device_operation.hpp
struct SliceDeviceOperation {
    using operation_attributes_t = SliceParams;
    using tensor_args_t = SliceInputs;
    // ...
};
```

## Migration Steps

When refactoring an operation:

1. **Rename types in `*_device_operation_types.hpp`**
   - `operation_attributes_t` → `{Operation}Params`
   - `tensor_args_t` → `{Operation}Inputs`

2. **Update type aliases in `*_device_operation.hpp`**
   - Change from namespace-qualified names to direct type names
   - Example: `using operation_attributes_t = slice::operation_attributes_t;` → `using operation_attributes_t = SliceParams;`

3. **Update all usages throughout the codebase**
   - Search for `namespace::operation_attributes_t` and replace with `{Operation}Params`
   - Search for `namespace::tensor_args_t` and replace with `{Operation}Inputs`
   - Update function signatures, variable declarations, and type references

4. **Files typically affected:**
   - `*_device_operation_types.hpp` - Type definitions
   - `*_device_operation.hpp` - Type aliases
   - `*_device_operation.cpp` - Implementation
   - `*_program_factory_*.hpp` - Program factory headers
   - `*_program_factory_*.cpp` - Program factory implementations
   - Any other files that reference these types

## Benefits

1. **Clearer Intent**: `SliceParams` is more descriptive than `operation_attributes_t`
2. **Better Readability**: No redundant namespace qualification needed
3. **Easier Discovery**: Operation-specific names make it easier to find related types
4. **Consistency**: Follows a clear, predictable naming pattern across operations

## Pattern for Other Operations

Apply this pattern consistently across all operations:

- `Conv2dDeviceOperation` → `Conv2dParams`, `Conv2dInputs`
- `MatmulDeviceOperation` → `MatmulParams`, `MatmulInputs`
- `PadDeviceOperation` → `PadParams`, `PadInputs`
- etc.

## Notes

- The `operation_attributes_t` and `tensor_args_t` type aliases in `DeviceOperation` structs are kept for compatibility with the device operation framework
- Only the underlying type names in the namespace are changed
- This refactoring is purely a naming improvement and does not change functionality
