# MeshTensor Migration Guide for Program Factories

This guide explains how to migrate program factory implementations from using raw `Tensor` and `Buffer*` APIs to the new `MeshTensor` API.

## Overview

The migration involves changing program factories to work with `MeshTensor` instead of directly accessing `Buffer*` pointers from `Tensor` objects. This provides a more uniform interface for multi-device tensor operations.

## Key API Changes

| Old API | New API |
|---------|---------|
| `tensor_args.input.buffer()` | `tensor_args.input.mesh_tensor()` |
| `tensor.device()` (returns `MeshDevice*`) | `mesh_tensor.device()` (returns `const MeshDevice&`) |
| `tensor.buffer()->address()` | `mesh_tensor.address()` |
| `TensorAccessorArgs(*buffer)` | `TensorAccessorArgs(mesh_tensor)` |
| `tensor.shard_spec()` | `mesh_tensor.legacy_shard_spec()` |
| `tensor.is_sharded()` | `mesh_tensor.is_sharded()` |
| `buffer != nullptr` | `mesh_tensor.is_allocated()` |

## Migration Steps

### Step 1: Update Tensor Access in `create()` Function

**Before:**
```cpp
const auto& input = tensor_args.input;
const auto& output = tensor_return_value;
tt::tt_metal::IDevice* device = input.device();
```

**After:**
```cpp
const auto& input = tensor_args.input.mesh_tensor();
const auto& output = tensor_return_value.mesh_tensor();
const auto& device = input.device();  // Now returns reference, not pointer
```

### Step 2: Update Device Access

Since `device` is now a reference instead of a pointer, change all `device->` to `device.`:

**Before:**
```cpp
device->arch()
device->compute_with_storage_grid_size()
device->worker_core_from_logical_core(core)
```

**After:**
```cpp
device.arch()
device.compute_with_storage_grid_size()
device.worker_core_from_logical_core(core)
```

### Step 3: Remove Buffer Variables, Use MeshTensor Directly

Delete all `Buffer*` variables. Replace buffer methods with MeshTensor methods:

| Old (using Buffer*) | New (using MeshTensor) |
|---------------------|------------------------|
| `buffer->address()` | `mesh_tensor.address()` |
| `buffer != nullptr` | `mesh_tensor.is_allocated()` |
| `TensorAccessorArgs(*buffer)` | `TensorAccessorArgs(mesh_tensor)` |

### Step 4: Update Runtime Args (SetRuntimeArgs)

**Before:**
```cpp
SetRuntimeArgs(program, kernel, core, {src_buffer->address(), ...});
```

**After:**
```cpp
SetRuntimeArgs(program, kernel, core, {input.address(), ...});
```

### Step 7: Update `override_runtime_arguments()` Function

Apply the same changes to `override_runtime_arguments()`:

**Before:**
```cpp
void MyProgramFactory::override_runtime_arguments(...) {
    const auto& input = tensor_args.input;
    auto* src_buffer = input.buffer();
    // Use src_buffer->address() in SetRuntimeArgs
}
```

**After:**
```cpp
void MyProgramFactory::override_runtime_arguments(...) {
    const auto& input = tensor_args.input.mesh_tensor();
    // Use input.address() in SetRuntimeArgs
}
```

## Handling Sharded Tensors

For operations that use sharded tensors with dynamic circular buffers, additional changes are required.

### Shard Spec Access

**Before:**
```cpp
auto num_tiles = tensor.shard_spec()
```

**After:**
```cpp
auto num_tiles = mesh_tensor.legacy_shard_spec()
```

## Helper Function Migration

If your program factory has helper functions that take `const Tensor&` parameters, update them to take `const tt::tt_metal::MeshTensor&`:

**Before:**
```cpp
void set_runtime_args(
    Program& program,
    const Tensor& input,
    const Tensor& output,
    ...) {
    auto* src_buffer = input.buffer();
    ...
}
```

**After:**
```cpp
void set_runtime_args(
    Program& program,
    const tt::tt_metal::MeshTensor& input,
    const tt::tt_metal::MeshTensor& output,
    ...) {
    // Use input.address() directly
    ...
}
```

Then update the callers to pass `.mesh_tensor()`:

```cpp
set_runtime_args(program,
    tensor_args.input.mesh_tensor(),
    tensor_return_value.mesh_tensor(),
    ...);
```

## Complete Example

See the following migrated files for reference:
- `ttnn/cpp/ttnn/operations/experimental/matmul/attn_matmul/device/attn_matmul_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/device/group_attn_matmul_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/experimental/dropout/device/dropout_program_factory.cpp`

## Build and Test Workflow

### Step 0: Find the Appropriate Tests

**Start every migration by finding the relevant tests.** Tests may exist as:

- **Python tests:** `tests/ttnn/**/test_*.py`
- **C++ tests:** `tests/ttnn/unit_tests/gtests/**/*.cpp`

Search for tests related to your operation:

```bash
# Find Python tests
find tests -name "*.py" | xargs grep -l "operation_name"

# Find C++ tests
find tests -name "*.cpp" | xargs grep -l "operation_name"
```

For example, for `attn_matmul`:
- Python: `tests/ttnn/nightly/unit_tests/operations/matmul/test_attn_matmul.py`

### Environment Setup

Activate the Python virtual environment before running tests:

```bash
source ./python_env/bin/activate
```

### Build Command

```bash
./build_metal.sh -b RelWithDebInfo --build-tests -e
```

### Testing Workflow

**IMPORTANT: Always run the relevant tests BEFORE and AFTER refactoring.**

1. **Find tests** for the operation you're migrating (see Step 0)
2. **Before refactoring:** Run tests to establish a baseline and ensure they pass
3. **After refactoring:** Run the same tests to verify the migration didn't break anything

```bash
# Python tests
source ./python_env/bin/activate
pytest "path/to/test_file.py::test_function_name" --no-header -q

# C++ tests (after building with --build-tests)
./build_RelWithDebInfo/tests/ttnn/unit_tests/gtests/test_binary_name
```

If tests fail after refactoring, the issue is likely:
- Missing `.mesh_tensor()` call
- Using `->` instead of `.` for device access
- Forgetting to update `override_runtime_arguments()`

## Checklist

- [ ] **Find tests** for the operation (Python and/or C++)
- [ ] **Run tests BEFORE refactoring** (establish baseline)
- [ ] Replace `tensor_args.xxx` with `tensor_args.xxx.mesh_tensor()`
- [ ] Replace `tensor_return_value` with `tensor_return_value.mesh_tensor()`
- [ ] Change device access from `->` to `.`
- [ ] Remove buffer pointer variables
- [ ] Update `TensorAccessorArgs` to use MeshTensor
- [ ] Update address access to use `mesh_tensor.address()`
- [ ] Update shard_spec access to use `legacy_shard_spec()`
- [ ] Update circular buffer APIs to use MeshTensor overloads
- [ ] Update `override_runtime_arguments()` similarly
- [ ] **While refactoring:** Note any patterns requiring `mesh_buffer()` or `get_reference_buffer()` (see below)
- [ ] Rebuild: `./build_metal.sh -b RelWithDebInfo --build-tests -e`
- [ ] **Run tests AFTER refactoring** (verify migration)
- [ ] **Report API improvements** (see "Identifying API Improvement Opportunities" below)

---

## Identifying API Improvement Opportunities

**This is a required part of every migration.** While refactoring, look for patterns that indicate the need for API improvements. Document and report any issues you find.

### What to Look For

**1. Any pattern requiring `mesh_buffer()` access**

If you encounter code that needs to call `mesh_tensor.mesh_buffer()`, this is a red flag. Report it.

```cpp
// RED FLAG - report this pattern
mesh_tensor.mesh_buffer()->some_method()
```

**2. Any pattern requiring `get_reference_buffer()` access**

This is an even stronger signal that an API overload is missing. Report it immediately.

```cpp
// RED FLAG - report this pattern
*mesh_tensor.mesh_buffer()->get_reference_buffer()
```

**3. Functions that accept `Buffer&` but should accept `MeshTensor`**

Look for tt_metal APIs that accept `const Buffer&` where you need to pass tensor data. These are candidates for MeshTensor overloads.

### What NOT to Propose

1. **Do not propose renaming `legacy_shard_spec()`** - The naming is intentional.
2. **Do not propose changes unrelated to MeshTensor** - Keep proposals scoped.

### Scope of Valid Proposals

Proposals must be strictly limited to:

- New methods on the `MeshTensor` class itself
- New overloads of existing functions to accept `MeshTensor` as a parameter
- Functions that currently require extracting `mesh_buffer()` or `get_reference_buffer()`

### How to Report

When you identify an improvement opportunity, document:

1. **The verbose pattern** you encountered
2. **The API that needs a MeshTensor overload**
3. **The simplified usage** after the overload would be added

**Example report format:**

```
Pattern found: *mesh_tensor.mesh_buffer()->get_reference_buffer()
Location: SomeFunction() in some_file.cpp
API needing overload: void SomeFunction(const Buffer&, ...)
Suggested overload: void SomeFunction(const MeshTensor&, ...)
```

---

## Migration Summary

A complete migration includes:

1. **Find tests** - Locate Python and/or C++ tests for the operation
2. **Run baseline tests** - Ensure tests pass before changes
3. **Refactor code** - Apply the MeshTensor migration steps
4. **Note improvement opportunities** - Document any `mesh_buffer()` or `get_reference_buffer()` patterns
5. **Rebuild and test** - Verify migration didn't break functionality
6. **Report improvements** - Submit any API improvement opportunities found

**The migration is not complete until improvement opportunities are documented and reported.**
