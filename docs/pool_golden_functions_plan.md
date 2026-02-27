# Plan: Add Golden Functions and Simplify Pool Tests

## Context
The current branch has added golden functions for pool operations (max_pool2d, avg_pool2d, rotate). However, `golden_upsample` and `golden_grid_sample` are being imported but don't exist yet. Additionally, the tests create tensors in NCHW format, then permute to NHWC, then reshape to (1,1,NHW,C), which involves unnecessary transformations.

The goal is to:
1. Add missing golden functions (`golden_upsample`, `golden_grid_sample`)
2. Adjust tests to create tensors directly in (1,1,NHW,C) format
3. Minimize reshapes and permutes by having golden functions handle format conversion internally
4. Ensure tests don't regress (no tolerance changes)

## Golden Function Interface Pattern
From existing golden functions in `ttnn/ttnn/operations/pool.py`:
- **Input**: tensor in `(1, 1, N*H*W, C)` format (matches ttnn)
- **Output**: tensor in `(1, 1, N*out_H*out_W, C)` format (matches ttnn)
- **Internal**: Convert to NCHW for torch operations, then convert back
- **`**_` parameter**: All golden functions should accept `**_` to ignore extra kwargs that may be passed by ttnn's golden function attachment mechanism

## Implementation Tasks

### Task 1: Add `golden_upsample` to pool.py
**File**: `ttnn/ttnn/operations/pool.py`

```python
def golden_upsample(
    input_tensor: ttnn.Tensor,
    batch_size: int,
    input_h: int,
    input_w: int,
    channels: int,
    scale_factor: Tuple[int, int],
    mode: str = "nearest",
    align_corners: bool = None,
    **_,
):
    # Reshape from (1, 1, N*H*W, C) to (N, H, W, C) then to (N, C, H, W)
    input_nchw = input_tensor.reshape(batch_size, input_h, input_w, channels).permute(0, 3, 1, 2)

    # Apply upsample
    output_nchw = torch.nn.functional.interpolate(
        input_nchw, scale_factor=scale_factor, mode=mode, align_corners=align_corners
    )

    # Convert back to (1, 1, N*H*W, C)
    N, C, H, W = output_nchw.shape
    output_tensor = output_nchw.permute(0, 2, 3, 1).reshape(1, 1, N * H * W, C)

    return output_tensor

ttnn.attach_golden_function(ttnn.upsample, golden_upsample)
```

**Notes**:
- `scale_factor` may be passed as a single int or a tuple; `torch.nn.functional.interpolate` handles both cases correctly.
- `align_corners` defaults to `None`, which is valid for "nearest" mode but must be explicitly set for other modes like "bilinear".

### Task 2: Add `golden_grid_sample` to pool.py
**File**: `ttnn/ttnn/operations/pool.py`

```python
def golden_grid_sample(
    input_tensor: ttnn.Tensor,
    grid: ttnn.Tensor,
    batch_size: int,
    input_h: int,
    input_w: int,
    channels: int,
    output_h: int,
    output_w: int,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
    **_,
):
    # Reshape input from (1, 1, N*H*W, C) to (N, C, H, W)
    input_nchw = input_tensor.reshape(batch_size, input_h, input_w, channels).permute(0, 3, 1, 2)

    # Reshape grid from (1, 1, N*H*W, 2) to (N, H, W, 2)
    grid_nhwc = grid.reshape(batch_size, output_h, output_w, 2)

    # Apply grid_sample
    output_nchw = torch.nn.functional.grid_sample(
        input_nchw.float(), grid_nhwc.float(),
        mode=mode, padding_mode=padding_mode, align_corners=align_corners
    )

    # Convert back to (1, 1, N*H*W, C)
    N, C, H, W = output_nchw.shape
    output_tensor = output_nchw.permute(0, 2, 3, 1).reshape(1, 1, N * H * W, C).to(input_tensor.dtype)

    return output_tensor

ttnn.attach_golden_function(ttnn.grid_sample, golden_grid_sample)
```

**Notes**:
- `output_h` and `output_w` are required parameters to correctly reshape the grid from `(1, 1, N*H*W, 2)` to `(N, H, W, 2)`. Tests must pass these explicitly.
- `.float()` conversion is used because `torch.nn.functional.grid_sample` often requires float32 inputs; the result is converted back to the original dtype.
- The grid tensor uses 2 channels (for x, y coordinates) rather than the input's channel count.

### Task 3: Simplify test_upsample.py
**File**: `tests/ttnn/unit_tests/operations/pool/test_upsample.py`

Current pattern:
```python
input = torch.rand(input_shapes, dtype=torch.bfloat16)  # NCHW
tt_input = input.permute(0, 2, 3, 1)  # NHWC
# Then reshape in golden call...
```

New pattern:
```python
# Create directly in ttnn format (1, 1, NHW, C)
torch.manual_seed(0)
input_tensor = torch.rand((1, 1, batch_size * height * width, channels), dtype=torch.bfloat16)

# Use golden directly (no reshape needed)
torch_result = golden_upsample(
    input_tensor=input_tensor,
    batch_size=batch_size,
    input_h=height,
    input_w=width,
    channels=channels,
    scale_factor=scale_factor,
    mode="nearest",
)

# For ttnn, just use the tensor as-is
ttnn_input = ttnn.from_torch(input_tensor, ...)
```

### Task 4: Simplify test_grid_sample.py
**File**: `tests/ttnn/unit_tests/operations/pool/test_grid_sample.py`

Same pattern: create input and grid directly in (1,1,NHW,C/2) format.

### Task 5: test_rotate.py - No changes needed
**File**: `tests/ttnn/unit_tests/operations/pool/test_rotate.py`

The rotate operation uses NHWC format (not 1,1,NHW,C) since ttnn.rotate operates on NHWC tensors directly. This is by design and will be kept as-is. No changes needed for rotate tests.

### Task 6: Update nightly test files similarly
- `tests/ttnn/nightly/unit_tests/operations/pool/test_upsample.py`
- `tests/ttnn/nightly/unit_tests/operations/pool/test_grid_sample.py`

## Files to Modify
1. `ttnn/ttnn/operations/pool.py` - Add golden_upsample and golden_grid_sample
2. `tests/ttnn/unit_tests/operations/pool/test_upsample.py` - Simplify tensor creation
3. `tests/ttnn/unit_tests/operations/pool/test_grid_sample.py` - Simplify tensor creation
4. `tests/ttnn/nightly/unit_tests/operations/pool/test_upsample.py` - Simplify tensor creation
5. `tests/ttnn/nightly/unit_tests/operations/pool/test_grid_sample.py` - Simplify tensor creation

**No changes needed**:
- `tests/ttnn/unit_tests/operations/pool/test_rotate.py` - Already uses NHWC which matches ttnn.rotate interface
- `tests/ttnn/unit_tests/operations/pool/test_maxpool2d.py` - Already uses golden function via run_max_pool2d
- `tests/ttnn/unit_tests/operations/pool/test_avgpool2d.py` - Already uses golden function via run_avg_pool2d

## Verification
1. Run unit tests: `pytest tests/ttnn/unit_tests/operations/pool/test_upsample.py -v`
2. Run unit tests: `pytest tests/ttnn/unit_tests/operations/pool/test_grid_sample.py -v`
3. Run nightly tests: `pytest tests/ttnn/nightly/unit_tests/operations/pool/test_upsample.py -v`
4. Run nightly tests: `pytest tests/ttnn/nightly/unit_tests/operations/pool/test_grid_sample.py -v`
5. Verify no tolerance changes were needed (tests pass with existing tolerances)
