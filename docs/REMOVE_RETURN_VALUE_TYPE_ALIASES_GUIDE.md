# Guide: Removing Return Value Type Aliases from Types Files

## Overview

This guide documents the process of removing `spec_return_value_t` and `tensor_return_value_t` type aliases from `*_device_operation_types.hpp` files and defining them directly in device operation structs.

## Current Pattern

**Types file** (`*_device_operation_types.hpp`):
```cpp
using spec_return_value_t = TensorSpec;  // or ttnn::TensorSpec
using tensor_return_value_t = Tensor;     // or ttnn::Tensor
```

**Device operation file** (`*_device_operation.hpp`):
```cpp
struct XDeviceOperation {
    using spec_return_value_t = namespace::spec_return_value_t;
    using tensor_return_value_t = namespace::tensor_return_value_t;
    // ...
};
```

## Target Pattern

**Types file** (`*_device_operation_types.hpp`):
- Remove `using spec_return_value_t = ...;`
- Remove `using tensor_return_value_t = ...;`
- Keep `operation_attributes_t` and `tensor_args_t` (used elsewhere)

**Device operation file** (`*_device_operation.hpp`):
```cpp
struct XDeviceOperation {
    using operation_attributes_t = namespace::operation_attributes_t;
    using tensor_args_t = namespace::tensor_args_t;
    using spec_return_value_t = TensorSpec;      // Direct definition
    using tensor_return_value_t = Tensor;         // Direct definition
    // ...
};
```

## Step-by-Step Process (Example: Slice Operation)

### Step 1: Identify the Files

From `tensor_return_value_tensor_matches.json`, identify:
- Types file: `ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_device_operation_types.hpp`
- Device operation file: `ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_device_operation.hpp`

### Step 2: Examine the Types File

Check what namespace qualifiers are used:
```cpp
// In slice_device_operation_types.hpp
using spec_return_value_t = TensorSpec;        // No namespace qualifier
using tensor_return_value_t = Tensor;           // No namespace qualifier
```

Note: Some files use `ttnn::TensorSpec` or `ttnn::Tensor` - preserve the original pattern.

### Step 3: Remove from Types File

Remove the two `using` declarations:
```cpp
// Remove these lines:
using spec_return_value_t = TensorSpec;
using tensor_return_value_t = Tensor;
```

**Before:**
```cpp
struct tensor_args_t {
    Tensor input;
    std::optional<Tensor> start_tensor;
    std::optional<Tensor> end_tensor;
    std::optional<Tensor> preallocated_output;
};

using spec_return_value_t = TensorSpec;
using tensor_return_value_t = Tensor;

}  // namespace ttnn::operations::data_movement::slice
```

**After:**
```cpp
struct tensor_args_t {
    Tensor input;
    std::optional<Tensor> start_tensor;
    std::optional<Tensor> end_tensor;
    std::optional<Tensor> preallocated_output;
};

}  // namespace ttnn::operations::data_movement::slice
```

### Step 4: Update Device Operation File

Replace the namespace aliases with direct definitions:

**Before:**
```cpp
struct SliceDeviceOperation {
    using operation_attributes_t = slice::operation_attributes_t;
    using tensor_args_t = slice::tensor_args_t;
    using spec_return_value_t = slice::spec_return_value_t;
    using tensor_return_value_t = slice::tensor_return_value_t;
    // ...
};
```

**After:**
```cpp
struct SliceDeviceOperation {
    using operation_attributes_t = slice::operation_attributes_t;
    using tensor_args_t = slice::tensor_args_t;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    // ...
};
```

### Step 5: Verify Includes

Ensure the device operation file has necessary includes:
- `#include "ttnn/tensor/tensor.hpp"` - provides `Tensor` and `TensorSpec` (via `ttnn::TensorSpec` alias)
- The types file should already include `ttnn/tensor/tensor.hpp` for `Tensor`

**Note:** `TensorSpec` is available as:
- `tt::tt_metal::TensorSpec` (original)
- `ttnn::TensorSpec` (alias defined in `ttnn/tensor/tensor.hpp`)

Since device operation files typically include `ttnn/tensor/tensor.hpp`, `TensorSpec` and `Tensor` are available without namespace qualifiers in most cases.

### Step 6: Check for References

Search for any code that might reference the namespace-qualified types:

```bash
# Check implementation files
grep -r "namespace::spec_return_value_t\|namespace::tensor_return_value_t" ttnn/cpp/ttnn/operations/data_movement/slice/

# Check program factory files
grep -r "namespace::spec_return_value_t\|namespace::tensor_return_value_t" ttnn/cpp/ttnn/operations/data_movement/slice/
```

**Good news:** Most code already uses `DeviceOperation::spec_return_value_t` or `DeviceOperation::tensor_return_value_t`, which will continue to work.

### Step 7: Compile and Test

Compile the project to verify changes:

```bash
./build_metal.sh -c -e --debug --build-all
```

Check for compilation errors related to:
- Missing type definitions
- Namespace resolution issues
- Include order problems

### Step 8: Verify Functionality

Run relevant tests to ensure the operation still works correctly.

## Important Notes

### Namespace Qualifiers

Preserve the original namespace qualifier pattern:
- If types file uses `TensorSpec` → use `TensorSpec` in device operation
- If types file uses `ttnn::TensorSpec` → use `ttnn::TensorSpec` in device operation
- Same for `Tensor` vs `ttnn::Tensor`

### Includes

The device operation file must include:
- `ttnn/tensor/tensor.hpp` - for `Tensor` and `TensorSpec`
- The types file header - for `operation_attributes_t` and `tensor_args_t`

### Backward Compatibility

Code using `DeviceOperation::tensor_return_value_t` will continue to work because:
- The type alias is now defined directly in the struct
- The fully qualified type `ttnn::operations::namespace::DeviceOperation::tensor_return_value_t` still resolves correctly

### Types File Still Needed

The types file (`*_device_operation_types.hpp`) is still required because:
- It defines `operation_attributes_t` and `tensor_args_t`
- These are used in program factories and other places
- Only the return value type aliases are removed

## Automation Script

For processing multiple files, use the provided script:

```bash
python3 scripts/remove_return_value_aliases.py \
    --matches tensor_return_value_tensor_matches.json \
    --batch-size 10 \
    --dry-run
```

The script will:
1. Read the matches JSON file
2. For each match, extract the namespace qualifier pattern
3. Remove aliases from types file
4. Update device operation file
5. Verify includes are present
6. Optionally compile after each batch

## Batch Processing Strategy

Process files in small batches (10-20 at a time):

1. **Batch 1:** Process first 10 files
2. **Compile:** Verify no errors
3. **Commit:** Git commit the batch
4. **Repeat:** Continue with next batch

This approach:
- Catches errors early
- Makes rollback easier
- Simplifies code review
- Reduces risk

## Troubleshooting

### Compilation Error: 'TensorSpec' was not declared

**Solution:** Ensure `#include "ttnn/tensor/tensor.hpp"` is present in the device operation file.

### Compilation Error: 'Tensor' was not declared

**Solution:** Ensure `#include "ttnn/tensor/tensor.hpp"` is present in the device operation file.

### Namespace Resolution Issues

**Solution:** Check if the original types file used namespace qualifiers (`ttnn::TensorSpec`) and match that pattern.

### Program Factory Errors

**Solution:** Verify program factory files include the device operation header, not just the types header.

## Example: Complete Refactoring

See the slice operation refactoring as a reference:
- Types file: `ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_device_operation_types.hpp`
- Device operation file: `ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_device_operation.hpp`

## Success Criteria

After refactoring, verify:
- [ ] Types file no longer contains `spec_return_value_t` or `tensor_return_value_t`
- [ ] Device operation struct defines these types directly
- [ ] Project compiles successfully
- [ ] Tests pass
- [ ] No references to `namespace::spec_return_value_t` or `namespace::tensor_return_value_t` remain

## Related Files

- `tensor_return_value_tensor_matches.json` - List of 74 files to refactor
- `scripts/find_tensor_return_value_types.py` - Script to find matches
- `scripts/remove_return_value_aliases.py` - Script to automate refactoring (to be created)
