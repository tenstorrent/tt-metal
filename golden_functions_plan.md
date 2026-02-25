# Plan: Improve Pool Golden Functions Readability

## Problem Statement

Currently, pool tests have verbose and repetitive tensor creation/reshaping code:

```python
# Current pattern (repeated in every test)
torch_input = torch.randn((N, C, H, W), dtype=torch.bfloat16)  # NCHW format
torch_input_permuted = torch.permute(torch_input, (0, 2, 3, 1))  # N, H, W, C
ttnn_input_shape = (1, 1, N * H * W, C)
torch_input_reshaped = torch_input_permuted.reshape(ttnn_input_shape)  # 1, 1, NHW, C

# ... call TTNN op ...

# For golden comparison - REPEAT the reshaping
torch_input_formatted = torch_input.permute(0, 2, 3, 1).reshape(1, 1, N * H * W, C)
torch_output = golden_maxpool2d(input_tensor=torch_input_formatted, ...)
torch_output = torch_output.reshape(N, out_h, out_w, C).permute(0, 3, 1, 2)  # Back to NCHW
```

This is:
1. **Verbose** - 10+ lines just for tensor creation/reshaping
2. **Repetitive** - Same pattern in every test file
3. **Confusing** - Mixing NCHW (PyTorch) and NHWC/2D (TTNN) formats
4. **Error-prone** - Easy to make mistakes in permute/reshape

## Solution Overview

### 1. Add Helper Function for TTNN-Format Tensor Creation

Add to `ttnn/ttnn/operations/pool.py`:

```python
def create_pool_input_tensor(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    dtype=None,
    seed: int = 0,
):
    """
    Create a random tensor directly in TTNN 2D format (1, 1, N*H*W, C).

    This is the format expected by TTNN pool operations and their golden functions.

    Args:
        batch_size: Batch dimension N
        channels: Channel dimension C
        height: Height dimension H
        width: Width dimension W
        dtype: PyTorch dtype (default: torch.bfloat16)
        seed: Random seed for reproducibility

    Returns:
        torch.Tensor: Shape (1, 1, N*H*W, C) in NHWC flattened format
    """
    import torch
    if dtype is None:
        dtype = torch.bfloat16
    torch.manual_seed(seed)
    # Create directly in NHWC format and reshape to 2D TTNN format
    tensor = torch.randn((batch_size, height, width, channels), dtype=dtype)
    return tensor.reshape(1, 1, batch_size * height * width, channels)
```

### 2. Golden Functions Already Accept 2D Format

The golden functions already handle the conversion internally:
- **Input**: `(1, 1, N*H*W, C)` - 2D TTNN format
- **Internal**: Reshape to `(N, C, H, W)` for PyTorch ops
- **Output**: `(1, 1, N*out_H*out_W, C)` - 2D TTNN format

No changes needed to golden functions!

### 3. Simplified Test Pattern

**Before (current):**
```python
torch_input = randomize_torch_tensor(tensor_map, input_shape)  # NCHW
torch_input_permuted = torch.permute(torch_input, (0, 2, 3, 1))  # NHWC
ttnn_input_shape = (1, 1, in_n * in_h * in_w, in_c)
torch_input_reshaped = torch_input_permuted.reshape(ttnn_input_shape)
ttnn_input = ttnn.from_torch(torch_input_reshaped, ...)

# TTNN op
ttnn_output = ttnn.max_pool2d(input_tensor=ttnn_input, ...)

# Golden comparison - reshape AGAIN
torch_input_formatted = torch_input.permute(0, 2, 3, 1).reshape(1, 1, in_n * in_h * in_w, in_c)
torch_output = golden_maxpool2d(input_tensor=torch_input_formatted, ...)
torch_output = torch_output.reshape(in_n, out_h, out_w, in_c).permute(0, 3, 1, 2)

# Compare - need to reshape TTNN output too
ttnn_output_torch = ttnn.to_torch(ttnn_output)
ttnn_output_torch = ttnn_output_torch.reshape(in_n, out_h, out_w, in_c).permute(0, 3, 1, 2)
assert torch.equal(ttnn_output_torch, torch_output)
```

**After (simplified):**
```python
from ttnn.operations.pool import create_pool_input_tensor, golden_maxpool2d

# Create input directly in 2D TTNN format
input_tensor = create_pool_input_tensor(in_n, in_c, in_h, in_w)
ttnn_input = ttnn.from_torch(input_tensor, ...)

# TTNN op - output is (1, 1, N*out_H*out_W, C)
ttnn_output = ttnn.max_pool2d(input_tensor=ttnn_input, ...)

# Golden - same format in, same format out
golden_output = golden_maxpool2d(input_tensor=input_tensor, ...)

# Compare directly - both in same format!
ttnn_output_torch = ttnn.to_torch(ttnn_output)
assert torch.allclose(ttnn_output_torch, golden_output)
```

## Files to Modify

### 1. `ttnn/ttnn/operations/pool.py`
- Add `create_pool_input_tensor()` helper function

### 2. Test Files to Update

| File | Changes |
|------|---------|
| `tests/ttnn/nightly/.../test_maxpool2d.py` | Update `run_max_pool2d()` - use `create_pool_input_tensor()`, remove NCHW→2D reshaping |
| `tests/ttnn/nightly/.../test_avgpool2d.py` | Update `run_avg_pool2d()` - use `create_pool_input_tensor()`, remove NCHW→2D reshaping |
| `tests/ttnn/nightly/.../test_upsample.py` | Update test functions - create input directly in 2D format |
| `tests/ttnn/nightly/.../test_grid_sample.py` | Update test functions |
| `tests/sweep_framework/.../adaptive_pool2d_common.py` | Update `run_adaptive_pool2d()` |

### 3. `max_pool2d_with_indices_common.py`
Already updated to use `golden_maxpool2d` with `return_indices=True` and validation!

## Current Tensor Creation Patterns Found

### Pattern 1: Direct creation + permute + reshape
```python
torch_input = torch.randn(input_shape, dtype=torch.bfloat16)  # NCHW
torch_input_permuted = torch.permute(torch_input, (0, 2, 3, 1))  # NHWC
torch_input_reshaped = torch_input_permuted.reshape(ttnn_input_shape)  # 2D TTNN
```
**Files**: `test_upsample.py`, `test_grid_sample.py`

### Pattern 2: Cache map + permute + reshape
```python
torch_input = randomize_torch_tensor(tensor_map, input_shape)  # NCHW cached
torch_input_permuted = torch.permute(torch_input, (0, 2, 3, 1))
torch_input_reshaped = torch_input_permuted.reshape(ttnn_input_shape)
```
**Files**: `test_maxpool2d.py`, `test_avgpool2d.py`, `adaptive_pool2d_common.py`

## Implementation Steps

### Step 1: Add Helper Function
Add `create_pool_input_tensor()` to `pool.py`

### Step 2: Update Nightly Test Files

#### test_maxpool2d.py
- Replace `randomize_torch_tensor()` that returns NCHW
- Create new `randomize_pool_tensor()` that returns 2D TTNN format
- Update `run_max_pool2d()` to use 2D format throughout

#### test_avgpool2d.py
- Same changes as maxpool

#### test_upsample.py
- Update test functions to create 2D format directly
- Remove permute/reshape chains

### Step 3: Update Sweep Utils
- `adaptive_pool2d_common.py` - Update `run_adaptive_pool2d()`

## Benefits

1. **Cleaner tests** - Remove 5-10 lines of boilerplate per test
2. **Consistent format** - Always work in TTNN 2D format
3. **Less error-prone** - No manual permute/reshape
4. **Better documentation** - Helper function documents the format
5. **Easier maintenance** - Single place to change format if needed

## What's Already Done

1. ✅ `golden_maxpool2d` - Added `return_indices` support with `MaxPool2dWithIndicesResult` class
2. ✅ `golden_upsample` - Fixed missing `import torch`
3. ✅ `max_pool2d_with_indices_common.py` - Updated to use golden function with validation

## Verification

```bash
# Run pool tests to verify no regression
pytest tests/ttnn/unit_tests/operations/pool/test_maxpool2d.py -v -k "height_shard"
pytest tests/ttnn/unit_tests/operations/pool/test_avgpool2d.py -v
pytest tests/ttnn/unit_tests/operations/pool/test_mpwi.py -v -k "test_mpwi_small"
pytest tests/ttnn/unit_tests/operations/pool/test_upsample.py -v
```
