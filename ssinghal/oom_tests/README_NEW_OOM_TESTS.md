# Out of Memory (OOM) Tests for YOLOv12 High-Resolution Operations

This directory contains OOM tests for PyTorch operations that are likely to fail with Out of Memory errors when using YOLOv12 high-resolution input shapes.

## Purpose

These tests serve three main purposes:

1. **Documentation**: Record which input shapes cause OOM failures for each operation
2. **Memory Analysis**: Calculate theoretical vs actual memory requirements and overhead factors
3. **Regression Testing**: Ensure OOM handling works correctly and doesn't crash unexpectedly

## New OOM Tests Created

### YOLOv12 High-Resolution OOM Tests

1. **test_oom_ones.py** - Tests `torch.ones` tensor creation with high-res shapes
2. **test_oom_sub.py** - Tests `torch.ops.aten.sub.Tensor` subtraction operations
3. **test_oom_silu_inplace.py** - Tests `torch.ops.aten.silu_` in-place SiLU activation
4. **test_oom_split_with_sizes.py** - Tests `torch.ops.aten.split_with_sizes` tensor splitting
5. **test_oom_upsample_nearest2d.py** - Tests `torch.ops.aten.upsample_nearest2d` upsampling
6. **test_oom_native_batch_norm.py** - Tests `torch.ops.aten.native_batch_norm` batch normalization
7. **test_oom_transpose.py** - Tests `torch.ops.aten.transpose.int` tensor transposition
8. **test_oom_unsqueeze.py** - Tests `torch.ops.aten.unsqueeze` dimension addition

## Test Structure

Each OOM test file contains two types of tests:

### 1. OOM Tests (`test_oom_<operation>`)
- **Expected Behavior**: These tests should be **SKIPPED** due to OOM conditions
- **Input Shapes**: YOLOv12 high-resolution shapes that are likely to cause OOM
- **Memory Config**: Uses `ttnn.L1_MEMORY_CONFIG` to trigger OOM faster
- **Error Handling**: Properly catches and skips on "Out of Memory" errors

### 2. Memory Estimation Tests (`test_memory_estimation_<operation>`)
- **Purpose**: Calculate theoretical memory requirements without running on device
- **Analysis**: Compares theoretical vs actual memory usage
- **Overhead Calculation**: Determines memory overhead factors for optimization
- **Always Passes**: These are analysis tests that never fail

## YOLOv12 High-Resolution Shapes Used

The OOM tests use shapes that represent real YOLOv12 high-resolution inference scenarios:

- **Ultra High-Res Input**: `[1, 3, 1280, 1280]` - 19.5 MB theoretical
- **High-Res Feature Maps**: `[1, 64, 1280, 800]` - 125.0 MB theoretical
- **Medium-Res Feature Maps**: `[1, 128, 640, 400]` - 65.0 MB theoretical
- **Detection Head Shapes**: `[1, 256, 320, 200]` - 32.5 MB theoretical

## Memory Analysis

Each test calculates:

- **Theoretical Memory**: Based on tensor size × element size (bfloat16 = 2 bytes)
- **Actual OOM Threshold**: Empirically determined memory limit
- **Overhead Factor**: Ratio of actual vs theoretical memory usage

### Typical Overhead Factors
- **Simple Operations** (ones, transpose): ~2-3x overhead
- **Binary Operations** (add, sub, mul): ~3-4x overhead  
- **Complex Operations** (batch_norm, upsample): ~4-6x overhead

## Running OOM Tests

```bash
# Run all OOM tests
pytest ssinghal/oom_tests/ -v

# Run specific OOM test
pytest ssinghal/oom_tests/test_oom_ones.py -v

# Run only memory estimation tests (no device needed)
pytest ssinghal/oom_tests/ -k "memory_estimation" -v
```

## Expected Results

- **OOM Tests**: Should be SKIPPED with "Expected OOM" messages
- **Memory Estimation Tests**: Should PASS and print memory analysis
- **Failures**: Only if unexpected errors occur (not OOM-related)

## Memory Optimization Insights

The OOM tests help identify:

1. **Memory Bottlenecks**: Which operations hit memory limits first
2. **Optimization Targets**: Operations with highest overhead factors
3. **Memory Planning**: Required memory for different resolution tiers
4. **Batching Strategies**: Maximum feasible batch sizes for high-res inference

## Integration with CI/CD

These tests can be used in CI/CD pipelines to:

- Monitor memory regression over time
- Validate memory optimizations
- Document memory requirements for different hardware configurations
- Guide memory allocation strategies for production deployments
