# Plan: Fix ttnn.gather and ttnn.scatter crashes on negative dim values

**Related to:** Issue #41248 (split negative dim fix)
**Branch:** TBD - suggest `fix-gather-scatter-negative-dim`
**Date:** 2026-04-07

## Problem Summary

The `ttnn.gather` and `ttnn.scatter` operations crash or behave incorrectly when using negative values for the `dim` parameter (e.g., `dim=-1`). In Python/PyTorch conventions, negative dimensions refer to dimensions from the end (-1 is last, -2 is second-to-last, etc.).

### Current Behavior

**gather.cpp:**
- Takes `int8_t dim` as parameter (line 156)
- Checks for `dim == -1 || dim == input_tensor_rank - 1` as special case for last dimension (line 178)
- Passes `dim` directly to `ttnn::prim::gather()` without normalization (lines 204-211)
- Also passes unnormalized `dim` to helper functions `pre_gather_transform_tensor()` and `post_gather_transform_tensor()`

**scatter.cpp:**
- Takes `int32_t dim` as parameter (line 274)
- Normalizes `dim` ONLY within `validate_inputs()` for validation checks (line 42: `const int32_t normalized_dim = (dim < 0) ? (dim + input_rank) : dim;`)
- Passes original (potentially negative) `dim` to:
  - `check_support()` (line 285)
  - `pre_scatter_transform_tensor()` (lines 303, 307, 310)
  - `ttnn::prim::scatter()` (line 319)
  - `post_scatter_transform_tensor()` (line 325)
- Checks for `dim == -1 || dim == input_tensor_rank - 1` as special case (line 295)

### Root Cause

Both operations use negative `dim` for special-case detection (checking if it's the last dimension) but don't normalize the dimension before passing it to downstream functions. This causes incorrect behavior in helper functions and device operations that expect normalized (non-negative) dimension indices.

## Solution

Normalize the `dim` parameter early in both `gather()` and `scatter()` functions, after validation but before any processing. This follows the same pattern successfully used in:
- `split.cpp` - Uses `get_normalized_index()`
- `sort.cpp` - Uses inline normalization: `dim < 0 ? rank + dim : dim`
- `concat.cpp` - Uses `get_normalized_index()`

## Files to Modify

### 1. ttnn/cpp/ttnn/operations/data_movement/gather/gather.cpp

**Function:** `Tensor gather(...)` (lines 154-215)

**Change Location:** After line 164 (after getting `input_tensor_rank`)

**Before:**
```cpp
Tensor gather(
    const Tensor& input_tensor,
    const int8_t dim,
    const Tensor& input_index_tensor,
    const bool sparse_grad,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    // Input tensor
    const ttnn::Shape& original_input_tensor_lshape = input_tensor.logical_shape();
    const auto input_tensor_rank = input_tensor.padded_shape().rank();

    // Index tensor
    const auto& original_index_tensor_lshape = input_index_tensor.logical_shape();
    const auto index_tensor_rank = input_index_tensor.padded_shape().rank();

    // Check for early exit for empty tensors tensors
    if (original_input_tensor_lshape == ttnn::Shape{}) {
        return input_tensor;
    }
    if (original_index_tensor_lshape == ttnn::Shape{}) {
        return input_index_tensor;
    }

    const bool input_tensor_is_dim_last_idx = (dim == -1 || dim == input_tensor_rank - 1);
```

**After:**
```cpp
Tensor gather(
    const Tensor& input_tensor,
    int8_t dim,  // Remove const to allow mutation
    const Tensor& input_index_tensor,
    const bool sparse_grad,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    // Input tensor
    const ttnn::Shape& original_input_tensor_lshape = input_tensor.logical_shape();
    const auto input_tensor_rank = input_tensor.padded_shape().rank();

    // Index tensor
    const auto& original_index_tensor_lshape = input_index_tensor.logical_shape();
    const auto index_tensor_rank = input_index_tensor.padded_shape().rank();

    // Check for early exit for empty tensors tensors
    if (original_input_tensor_lshape == ttnn::Shape{}) {
        return input_tensor;
    }
    if (original_index_tensor_lshape == ttnn::Shape{}) {
        return input_index_tensor;
    }

    // Normalize negative dimension to positive index
    dim = dim < 0 ? dim + input_tensor_rank : dim;

    const bool input_tensor_is_dim_last_idx = (dim == input_tensor_rank - 1);
```

**Note:** After normalization, we can simplify the last-dimension check since `dim` will always be positive.

### 2. ttnn/cpp/ttnn/operations/data_movement/scatter/scatter.cpp

**Function:** `Tensor scatter(...)` (lines 272-333)

**Change Location:** After line 281 (after getting `input_tensor_rank`)

**Before:**
```cpp
Tensor scatter(
    const Tensor& input_tensor,
    const int32_t& dim,
    const Tensor& index_tensor,
    const Tensor& source_tensor,
    const std::optional<MemoryConfig>& output_memory_config,
    const std::optional<std::string>& opt_reduction_string,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    const ttnn::Shape& original_input_tensor_lshape = input_tensor.logical_shape();
    const auto input_tensor_rank = input_tensor.padded_shape().rank();

    using namespace operations::data_movement::CMAKE_UNIQUE_NAMESPACE;

    check_support(input_tensor, index_tensor, source_tensor, dim);
    validate_inputs(input_tensor, index_tensor, source_tensor, dim, opt_reduction_string);

    const auto& original_index_tensor_lshape = index_tensor.logical_shape();
    if (original_input_tensor_lshape == ttnn::Shape{} || original_index_tensor_lshape == ttnn::Shape{}) {
        return input_tensor;
    }
    const auto original_layout = input_tensor.layout();

    // index and source tensors should have same rank as input tensor
    const bool input_tensor_is_dim_last_idx = (dim == -1 || dim == input_tensor_rank - 1);
```

**After:**
```cpp
Tensor scatter(
    const Tensor& input_tensor,
    int32_t dim,  // Remove const& to allow mutation
    const Tensor& index_tensor,
    const Tensor& source_tensor,
    const std::optional<MemoryConfig>& output_memory_config,
    const std::optional<std::string>& opt_reduction_string,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    const ttnn::Shape& original_input_tensor_lshape = input_tensor.logical_shape();
    const auto input_tensor_rank = input_tensor.padded_shape().rank();

    using namespace operations::data_movement::CMAKE_UNIQUE_NAMESPACE;

    // Validate inputs before normalization (validate_inputs handles negative dims internally)
    check_support(input_tensor, index_tensor, source_tensor, dim);
    validate_inputs(input_tensor, index_tensor, source_tensor, dim, opt_reduction_string);

    // Normalize negative dimension to positive index
    dim = dim < 0 ? dim + input_tensor_rank : dim;

    const auto& original_index_tensor_lshape = index_tensor.logical_shape();
    if (original_input_tensor_lshape == ttnn::Shape{} || original_index_tensor_lshape == ttnn::Shape{}) {
        return input_tensor;
    }
    const auto original_layout = input_tensor.layout();

    // index and source tensors should have same rank as input tensor
    const bool input_tensor_is_dim_last_idx = (dim == input_tensor_rank - 1);
```

**Note:** Validation happens before normalization since `validate_inputs()` already handles negative dims correctly with its own normalization.

### 3. Function: `scatter_add()` (lines 335+)

Apply the same fix to `scatter_add()` which is a convenience wrapper:

**Before:**
```cpp
Tensor scatter_add(
    const Tensor& input_tensor,
    const int32_t& dim,
    const Tensor& index_tensor,
    const Tensor& source_tensor,
    const std::optional<MemoryConfig>& output_memory_config,
    const std::optional<CoreRangeSet>& sub_core_grid) {
```

**After:**
```cpp
Tensor scatter_add(
    const Tensor& input_tensor,
    int32_t dim,
    const Tensor& index_tensor,
    const Tensor& source_tensor,
    const std::optional<MemoryConfig>& output_memory_config,
    const std::optional<CoreRangeSet>& sub_core_grid) {
```

And normalize `dim` before calling the main `scatter()` function.

### 4. Tests: tests/ttnn/nightly/unit_tests/operations/data_movement/test_gather.py

**Test file location:** Nightly test suite for comprehensive regression coverage

**Note:** These tests are placed in the nightly suite (similar to split tests from #41248) to provide thorough regression coverage across multiple tensor shapes, dimensions, and data types without impacting fast CI runs.

**Add new test function:**
```python
@pytest.mark.parametrize("shape", [(4, 128), (2, 3, 4), (1, 2, 3, 4)])
@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_gather_negative_dim(device, shape, dim, dtype):
    """
    Regression test for negative dimension support in ttnn.gather.
    Verifies that negative dimension values work correctly and match PyTorch behavior.
    """
    torch_input = torch.rand(shape, dtype=dtype)

    # Create index tensor for gathering
    index_shape = list(shape)
    index_shape[dim] = min(index_shape[dim], 2)  # Gather subset
    torch_index = torch.randint(0, shape[dim], index_shape, dtype=torch.int32)

    # PyTorch reference with negative dim
    torch_output_neg = torch.gather(torch_input, dim, torch_index.long())

    # Convert to ttnn
    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_index = ttnn.from_torch(torch_index, device=device, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT)

    # Test with negative dim
    ttnn_output = ttnn.gather(ttnn_input, dim, ttnn_index)
    output = ttnn.to_torch(ttnn_output)

    assert output.shape == torch_output_neg.shape, \
        f"Output shape {output.shape} does not match expected {torch_output_neg.shape}"
    assert_with_pcc(torch_output_neg, output, 0.999)


@pytest.mark.parametrize("shape,dim", [
    ((4, 128), -1),  # Last dimension
    ((4, 128), -2),  # First dimension
    ((2, 3, 4), -1), # 3D tensor, last dim
    ((2, 3, 4), -2), # 3D tensor, middle dim
    ((2, 3, 4), -3), # 3D tensor, first dim
])
def test_gather_negative_dim_equals_positive(device, shape, dim):
    """
    Verify that negative and positive dim produce identical results.
    """
    positive_dim = len(shape) + dim

    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    index_shape = list(shape)
    index_shape[dim] = min(index_shape[dim], 2)
    torch_index = torch.randint(0, shape[dim], index_shape, dtype=torch.int32)

    # Get results for both negative and positive dims
    torch_output_neg = torch.gather(torch_input, dim, torch_index.long())
    torch_output_pos = torch.gather(torch_input, positive_dim, torch_index.long())

    # Verify PyTorch results match
    assert torch.allclose(torch_output_neg, torch_output_pos)

    # Test ttnn
    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_index = ttnn.from_torch(torch_index, device=device, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT)

    ttnn_output_neg = ttnn.to_torch(ttnn.gather(ttnn_input, dim, ttnn_index))
    ttnn_output_pos = ttnn.to_torch(ttnn.gather(ttnn_input, positive_dim, ttnn_index))

    assert_with_pcc(ttnn_output_neg, ttnn_output_pos, 0.9999)
    assert_with_pcc(torch_output_neg, ttnn_output_neg, 0.999)
```

### 5. Tests: tests/ttnn/nightly/unit_tests/operations/data_movement/test_scatter.py

**Test file location:** Nightly test suite for comprehensive regression coverage

**Note:** These tests are placed in the nightly suite (similar to split tests from #41248) to provide thorough regression coverage across multiple tensor shapes, dimensions, and data types without impacting fast CI runs.

**Add new test function:**
```python
@pytest.mark.parametrize("shape", [(4, 128), (2, 3, 4), (1, 2, 3, 4)])
@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_scatter_negative_dim(device, shape, dim, dtype):
    """
    Regression test for negative dimension support in ttnn.scatter.
    Verifies that negative dimension values work correctly and match PyTorch behavior.
    """
    torch_input = torch.rand(shape, dtype=dtype)

    # Create index and source tensors for scattering
    index_shape = list(shape)
    index_shape[dim] = min(index_shape[dim], 2)  # Scatter subset
    torch_index = torch.randint(0, shape[dim], index_shape, dtype=torch.int32)
    torch_source = torch.rand(index_shape, dtype=dtype)

    # PyTorch reference with negative dim
    torch_output_neg = torch_input.clone()
    torch_output_neg.scatter_(dim, torch_index.long(), torch_source)

    # Convert to ttnn
    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn_index = ttnn.from_torch(torch_index, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn_source = ttnn.from_torch(torch_source, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Test with negative dim
    ttnn_output = ttnn.scatter(ttnn_input, dim, ttnn_index, ttnn_source)
    output = ttnn.to_torch(ttnn_output)

    assert output.shape == torch_output_neg.shape, \
        f"Output shape {output.shape} does not match expected {torch_output_neg.shape}"
    assert_with_pcc(torch_output_neg, output, 0.999)


@pytest.mark.parametrize("shape,dim", [
    ((4, 128), -1),  # Last dimension
    ((4, 128), -2),  # First dimension
    ((2, 3, 4), -1), # 3D tensor, last dim
    ((2, 3, 4), -2), # 3D tensor, middle dim
    ((2, 3, 4), -3), # 3D tensor, first dim
])
def test_scatter_negative_dim_equals_positive(device, shape, dim):
    """
    Verify that negative and positive dim produce identical results.
    """
    positive_dim = len(shape) + dim

    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    index_shape = list(shape)
    index_shape[dim] = min(index_shape[dim], 2)
    torch_index = torch.randint(0, shape[dim], index_shape, dtype=torch.int32)
    torch_source = torch.rand(index_shape, dtype=torch.bfloat16)

    # Get results for both negative and positive dims
    torch_output_neg = torch_input.clone()
    torch_output_neg.scatter_(dim, torch_index.long(), torch_source)

    torch_output_pos = torch_input.clone()
    torch_output_pos.scatter_(positive_dim, torch_index.long(), torch_source)

    # Verify PyTorch results match
    assert torch.allclose(torch_output_neg, torch_output_pos)

    # Test ttnn
    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn_index = ttnn.from_torch(torch_index, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn_source = ttnn.from_torch(torch_source, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    ttnn_output_neg = ttnn.to_torch(ttnn.scatter(ttnn_input, dim, ttnn_index, ttnn_source))

    ttnn_input_pos = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn_output_pos = ttnn.to_torch(ttnn.scatter(ttnn_input_pos, positive_dim, ttnn_index, ttnn_source))

    assert_with_pcc(ttnn_output_neg, ttnn_output_pos, 0.9999)
    assert_with_pcc(torch_output_neg, ttnn_output_neg, 0.999)
```

## Implementation Checklist

- [ ] Read and understand gather.cpp implementation
- [ ] Read and understand scatter.cpp implementation
- [ ] Modify `gather()` function to normalize `dim` parameter
- [ ] Modify `scatter()` function to normalize `dim` parameter
- [ ] Modify `scatter_add()` function to normalize `dim` parameter
- [ ] Add tests to nightly suite: `tests/ttnn/nightly/unit_tests/operations/data_movement/test_gather.py`
- [ ] Add `test_gather_negative_dim` test
- [ ] Add `test_gather_negative_dim_equals_positive` test
- [ ] Add tests to nightly suite: `tests/ttnn/nightly/unit_tests/operations/data_movement/test_scatter.py`
- [ ] Add `test_scatter_negative_dim` test
- [ ] Add `test_scatter_negative_dim_equals_positive` test
- [ ] Build the project: `source python_env/bin/activate && ./build_metal.sh`
- [ ] Run gather tests: `python -m pytest <gather_test_path> -v -k "negative_dim"`
- [ ] Run scatter tests: `python -m pytest <scatter_test_path> -v -k "negative_dim"`
- [ ] Run full gather test suite
- [ ] Run full scatter test suite
- [ ] Create reproducers to verify fixes
- [ ] Commit changes with message referencing this analysis
- [ ] Create PR targeting main branch

## Testing Commands

```bash
# Activate environment
source python_env/bin/activate

# Build
./build_metal.sh

# Find test files
find tests -name "*gather*.py" -path "*/operations/data_movement/*"
find tests -name "*scatter*.py" -path "*/operations/data_movement/*"

# Run only the new negative dim tests (in nightly suite)
python -m pytest tests/ttnn/nightly/unit_tests/operations/data_movement/test_gather.py -v -k "negative_dim"
python -m pytest tests/ttnn/nightly/unit_tests/operations/data_movement/test_scatter.py -v -k "negative_dim"

# Run all gather tests
python -m pytest tests/ttnn/nightly/unit_tests/operations/data_movement/test_gather.py -v

# Run all scatter tests
python -m pytest tests/ttnn/nightly/unit_tests/operations/data_movement/test_scatter.py -v
```

## Verification Reproducers

### Gather Reproducer
```python
import torch
import ttnn

device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1,1))

# Test gather with negative dim
input_tensor = torch.rand((4, 128), dtype=torch.bfloat16)
index_tensor = torch.randint(0, 128, (4, 32), dtype=torch.int32)

# PyTorch reference
torch_result = torch.gather(input_tensor, dim=-1, index=index_tensor.long())
print(f"PyTorch result shape: {torch_result.shape}")

# TTNN with negative dim
ttnn_input = ttnn.from_torch(input_tensor, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
ttnn_index = ttnn.from_torch(index_tensor, device=device, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT)

ttnn_result = ttnn.gather(ttnn_input, dim=-1, index=ttnn_index)
print(f"SUCCESS: gather with dim=-1 works!")
print(f"TTNN result shape: {ttnn_result.shape}")

ttnn.close_mesh_device(device)
```

### Scatter Reproducer
```python
import torch
import ttnn

device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1,1))

# Test scatter with negative dim
input_tensor = torch.zeros((4, 128), dtype=torch.bfloat16)
index_tensor = torch.randint(0, 128, (4, 32), dtype=torch.int32)
source_tensor = torch.ones((4, 32), dtype=torch.bfloat16)

# PyTorch reference
torch_result = input_tensor.clone()
torch_result.scatter_(dim=-1, index=index_tensor.long(), src=source_tensor)
print(f"PyTorch result shape: {torch_result.shape}")

# TTNN with negative dim
ttnn_input = ttnn.from_torch(input_tensor, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
ttnn_index = ttnn.from_torch(index_tensor, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
ttnn_source = ttnn.from_torch(source_tensor, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

ttnn_result = ttnn.scatter(ttnn_input, dim=-1, index=ttnn_index, src=ttnn_source)
print(f"SUCCESS: scatter with dim=-1 works!")
print(f"TTNN result shape: {ttnn_result.shape}")

ttnn.close_mesh_device(device)
```

## Reference Examples

Similar fixes in the codebase that normalize negative dimensions:

1. **split.cpp** (issue #41248): Uses `get_normalized_index()`
   ```cpp
   dim = input_shape.get_normalized_index(dim);
   ```

2. **concat.cpp:290**: Uses `get_normalized_index()`
   ```cpp
   dim = first_tensor.logical_shape().get_normalized_index(dim);
   ```

3. **sort.cpp:184**: Inline normalization
   ```cpp
   const int8_t normalized_dim = dim < 0 ? rank + dim : dim;
   ```

4. **narrow.cpp:13**: Uses `get_normalized_index()`
   ```cpp
   uint32_t dim = input_tensor_shape.get_normalized_index(narrow_dim);
   ```

## Notes

- Both operations already have special-case handling for checking if `dim` is the last dimension by testing `(dim == -1 || dim == positive_last_dim)`. After normalization, this can be simplified to just checking equality with the positive last dimension.

- The `validate_inputs()` function in scatter.cpp already normalizes `dim` internally for validation (line 42), which is good practice. However, the main `scatter()` function should also normalize before passing to other functions.

- Both operations use `dim` extensively in helper functions (`pre_*_transform_tensor`, `post_*_transform_tensor`) which expect normalized dimensions.

- The normalization pattern `dim < 0 ? rank + dim : dim` is simple and effective, used successfully in sort.cpp and chunk.cpp.

- Alternative: Could use `Shape::get_normalized_index()` like split.cpp, but inline normalization is clearer here since we already have the rank.

## Impact Assessment

**Risk Level:** Medium-Low
- Changes are localized to parameter normalization at function entry points
- No changes to core logic or device operations
- Pattern proven successful in multiple operations (split, sort, chunk, concat, etc.)
- Validation logic already handles negative dims correctly

**Compatibility:** Fully backward compatible
- Positive dimensions continue to work as before
- Negative dimensions now work instead of crashing
- No API changes (same function signatures)

**Testing Coverage:**
- New tests verify negative dim behavior
- New tests verify negative/positive equivalence
- Existing tests ensure no regressions for positive dims
