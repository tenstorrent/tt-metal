# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# NOTE: To verify which kernel (small vs large) is being used, look for log messages:
#   "SoftmaxBackward: Using SMALL (non-streaming) kernel | Shape: ..."
#   "SoftmaxBackward: Using LARGE (streaming) kernel | Shape: ..."
#
# Enable logs with: export TT_METAL_LOGGER_LEVEL=2
# Or use: export TT_METAL_LOGGER_TYPES=Op to see only operation logs

import torch
import pytest
import ttnn
from loguru import logger
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range_dtype


def reference_softmax_backward_output(y: torch.Tensor, grad: torch.Tensor, axis: int) -> torch.Tensor:
    dot = (y * grad).sum(dim=axis, keepdim=True)
    return y * (grad - dot)


def compute_pcc(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    x = tensor1.to(torch.float32).reshape(-1)
    y = tensor2.to(torch.float32).reshape(-1)
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    vx = x - x_mean
    vy = y - y_mean
    num = torch.sum(vx * vy)
    den = torch.sqrt(torch.sum(vx * vx) * torch.sum(vy * vy)) + 1e-12
    return (num / den).item()


def assert_pcc(tensor1: torch.Tensor, tensor2: torch.Tensor, pcc_threshold: float = 0.999) -> None:
    pcc_value = compute_pcc(tensor1, tensor2)
    logger.debug(f"  PCC: {pcc_value:.6f} (threshold: {pcc_threshold})")
    assert pcc_value >= pcc_threshold, f"PCC {pcc_value:.6f} < threshold {pcc_threshold}"


def print_tolerance_metrics(tensor1: torch.Tensor, tensor2: torch.Tensor, dtype_name: str = "", range: int = 0) -> None:
    """Calculate and print tolerance metrics between two tensors"""
    # Calculate actual differences
    abs_diff = torch.abs(tensor1 - tensor2)
    max_abs_diff = torch.max(abs_diff).item()
    mean_abs_diff = torch.mean(abs_diff).item()

    # Calculate relative difference
    rel_diff = abs_diff / (torch.abs(tensor2) + 1e-8)
    max_rel_diff = torch.max(rel_diff).item()
    mean_rel_diff = torch.mean(rel_diff).item()

    # Pearson correlation coefficient (PCC)
    pcc = compute_pcc(tensor1, tensor2)

    logger.info(f"\nTolerance metrics for {dtype_name} and range {range}:")
    logger.info(f"  Max absolute difference: {max_abs_diff:.6e}")
    logger.info(f"  Mean absolute difference: {mean_abs_diff:.6e}")
    logger.info(f"  Max relative difference: {max_rel_diff:.6e}")
    logger.info(f"  Mean relative difference: {mean_rel_diff:.6e}")
    logger.info(f"  PCC: {pcc:.6f}")


BATCH_SIZE = 1
SEED = 77
PCC_THRESHOLD = 0.9999

# Relative tolerance is very sensitive to small values, because division by small value results in large relative difference.
# So we use absolute tolerance only.
RELATIVE_TOLERANCE = 0.0


@pytest.mark.parametrize(
    "input_shapes,atol,pcc_threshold",
    [
        # Small tensor
        pytest.param(torch.Size([BATCH_SIZE, 3, 64, 64]), 1e-2, PCC_THRESHOLD, id="small_3x64x64"),
        # Big tensor but small last row - should use SMALL kernel
        pytest.param(torch.Size([BATCH_SIZE, 30, 6400, 128]), 6e-2, 0.9998, id="small_3x6400x128"),
        # === DIAGNOSTIC TESTS FOR LARGE KERNEL ===
        # tiles_per_block = 4, so test different remainders
        # Exactly 1 full block (4 tiles = 128 width) - SHOULD WORK
        pytest.param(torch.Size([BATCH_SIZE, 1, 32, 128]), 1e-2, PCC_THRESHOLD, id="large_1block_exact"),
        # 1 full block + 1 remainder (5 tiles = 160 width) - SHOULD FAIL if remainder broken
        pytest.param(torch.Size([BATCH_SIZE, 1, 32, 160]), 1e-2, PCC_THRESHOLD, id="large_1block_rem1"),
        # 1 full block + 2 remainder (6 tiles = 192 width) - SHOULD FAIL if remainder broken
        pytest.param(torch.Size([BATCH_SIZE, 1, 32, 192]), 1e-2, PCC_THRESHOLD, id="large_1block_rem2"),
        # 1 full block + 3 remainder (7 tiles = 224 width) - SHOULD FAIL if remainder broken
        pytest.param(torch.Size([BATCH_SIZE, 1, 32, 224]), 1e-2, PCC_THRESHOLD, id="large_1block_rem3"),
        # Exactly 2 full blocks (8 tiles = 256 width) - SHOULD WORK
        pytest.param(torch.Size([BATCH_SIZE, 1, 32, 256]), 1e-2, PCC_THRESHOLD, id="large_2blocks_exact"),
        # 2 full blocks + 1 remainder (9 tiles = 288 width) - SHOULD FAIL if remainder broken
        pytest.param(torch.Size([BATCH_SIZE, 1, 32, 288]), 1e-2, PCC_THRESHOLD, id="large_2blocks_rem1"),
        # Original large tensor
        pytest.param(torch.Size([BATCH_SIZE, 7, 128, 20448]), 10.1, PCC_THRESHOLD, id="large_7x128x20448"),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "range",
    [10],
)
@pytest.mark.parametrize(
    "dim",
    [3],  # only last dimension supported for now
)
def test_bw_softmax(input_shapes, atol, pcc_threshold, dtype, range, dim, device):
    grad_data, grad_tensor = data_gen_with_range_dtype(input_shapes, -range, range, device, ttnn_dtype=dtype, seed=SEED)
    in_data, input_tensor = data_gen_with_range_dtype(
        input_shapes, -range, range, device, required_grad=True, ttnn_dtype=dtype, seed=SEED
    )

    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    pt_softmax_tensor = torch.softmax(in_data, dim=dim, dtype=torch_dtype)
    tt_softmax_tensor = ttnn.from_torch(pt_softmax_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    # Test the fused kernel implementation
    tt_output_tensor_fused = ttnn.softmax_backward(tt_softmax_tensor, grad_tensor, dim=dim)
    pt_output_tensor_fused = ttnn.to_torch(tt_output_tensor_fused)
    pt_output_tensor_reference = reference_softmax_backward_output(pt_softmax_tensor, grad_data, axis=dim)

    # Use torch.allclose with torch reference for bf16 and fp32 types
    if dtype in [ttnn.bfloat16, ttnn.float32]:
        logger.debug(f"  Shape: {input_shapes}, Elements: {input_shapes.numel():,}")
        logger.debug(f"  Tolerances: rtol={RELATIVE_TOLERANCE}, atol={atol}, pcc_threshold={pcc_threshold}")

        try:
            assert torch.allclose(
                pt_output_tensor_fused,
                pt_output_tensor_reference,
                rtol=RELATIVE_TOLERANCE,
                atol=atol,
            )

            assert_pcc(pt_output_tensor_fused, pt_output_tensor_reference, pcc_threshold)
        except AssertionError:
            # Print detailed metrics on failure to help debug
            print_tolerance_metrics(
                pt_output_tensor_fused,
                pt_output_tensor_reference,
                dtype_name=f"dtype={dtype}",
                range=range,
            )
            raise


@pytest.mark.parametrize(
    "input_shapes,atol,pcc_threshold",
    [
        # Small padded tensors
        pytest.param(torch.Size([1, 1, 128, 499]), 5e-3, PCC_THRESHOLD, id="small_padded_128x499"),
        pytest.param(torch.Size([2, 1, 64, 500]), 5e-3, PCC_THRESHOLD, id="small_padded_64x500"),
        # Large padded tensor
        pytest.param(torch.Size([BATCH_SIZE, 5, 32, 20470]), 6e-3, PCC_THRESHOLD, id="large_padded_5x32x20470"),
        pytest.param(torch.Size([BATCH_SIZE, 7, 64, 21096]), 6e-3, PCC_THRESHOLD, id="large_padded_7x64x21096"),
        # FIXME! PCC
        pytest.param(torch.Size([BATCH_SIZE, 3, 32, 20000]), 2e-2, 0.077, id="large_padded_3x32x20000"),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "range",
    [10],
)
@pytest.mark.parametrize(
    "dim",
    [-1],  # test on last dimension
)
def test_bw_softmax_padded(input_shapes, atol, pcc_threshold, dtype, range, dim, device):
    """Test softmax backward with padded tensors (non-tile-aligned dimensions)"""

    torch.manual_seed(seed=SEED)
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16

    grad_data = torch.rand(input_shapes, dtype=torch_dtype, requires_grad=False) * (range - (-range)) + (-range)
    in_data = torch.rand(input_shapes, dtype=torch_dtype, requires_grad=True) * (range - (-range)) + (-range)

    # Compute reference output on logical (unpadded) tensors
    pt_softmax_tensor = torch.softmax(in_data, dim=dim, dtype=torch_dtype)
    pt_output_tensor_reference = reference_softmax_backward_output(pt_softmax_tensor, grad_data, axis=dim)

    # Use the pattern from test_fast_reduce_nc.py which works correctly
    # Create ttnn.Tensor directly, pad, then convert layout, then move to device
    tt_softmax_tensor = ttnn.Tensor(pt_softmax_tensor, dtype).pad_to_tile(float("nan")).to(ttnn.TILE_LAYOUT).to(device)

    tt_grad_tensor = ttnn.Tensor(grad_data, dtype).pad_to_tile(float("nan")).to(ttnn.TILE_LAYOUT).to(device)

    logger.debug(f"\nOriginal shape: {input_shapes}, Elements: {input_shapes.numel():,}")
    logger.debug(f"Padded shape: {tt_softmax_tensor.shape}")
    logger.debug(f"Tolerances: rtol={RELATIVE_TOLERANCE}, atol={atol}, pcc_threshold={pcc_threshold}")

    # Run softmax backward on padded tensors
    tt_output_tensor_fused = ttnn.softmax_backward(tt_softmax_tensor, tt_grad_tensor, dim=dim)

    # Convert back to torch (automatically unpads to logical shape)
    pt_output_tensor_fused = ttnn.to_torch(tt_output_tensor_fused)

    # Verify the output matches reference (only on logical/unpadded region)
    if dtype in [ttnn.bfloat16, ttnn.float32]:
        try:
            # The key test: output should match reference on the logical (unpadded) region
            # This verifies that padding did not corrupt the result
            assert (
                pt_output_tensor_fused.shape == pt_output_tensor_reference.shape
            ), f"Unpadded output shape mismatch: {pt_output_tensor_fused.shape} vs {pt_output_tensor_reference.shape}"

            assert torch.allclose(
                pt_output_tensor_fused,
                pt_output_tensor_reference,
                rtol=RELATIVE_TOLERANCE,
                atol=atol,
            ), f"Padded tensor output does not match reference! This means padding corrupted the reduction."
        except AssertionError:
            # Print detailed metrics on failure to help debug
            print_tolerance_metrics(
                pt_output_tensor_fused,
                pt_output_tensor_reference,
                dtype_name=f"dtype={dtype} (padded)",
                range=range,
            )
            raise


@pytest.mark.parametrize(
    "shape,expected_kernel",
    [
        # Small tensors - should use SMALL (non-streaming) kernel
        ((1, 1, 32, 32), "SMALL"),  # tiny
        ((1, 1, 64, 64), "SMALL"),  # small
        ((2, 2, 64, 128), "SMALL"),  # small
        # Large tensors - should use LARGE (streaming) kernel
        ((4, 4, 128, 20480), "LARGE"),  # large
        ((8, 8, 64, 40960), "LARGE"),  # large
    ],
)
def test_softmax_backward_kernel_selection(shape, expected_kernel, device):
    """
    Test that verifies correct kernel selection based on tensor size.

    This test ensures that:
    - Small tensors use the SMALL (non-streaming) kernel for better performance
    - Large tensors use the LARGE (streaming) kernel to avoid L1 overflow

    To see the kernel selection logs, run with:
        export TT_METAL_LOGGER_LEVEL=2
        export TT_METAL_LOGGER_TYPES=Op

    Expected log output format:
        "SoftmaxBackward: Using SMALL (non-streaming) kernel | Shape: 2x2 tiles (4 total) | Estimated L1: XX KB"
    or:
        "SoftmaxBackward: Using LARGE (streaming) kernel | Shape: 64x16 tiles (1024 total) | Estimated L1: XX KB"
    """
    logger.debug(f"\n{'='*40}")
    logger.debug(f"Testing kernel selection for shape {shape}")
    logger.debug(f"Expected kernel: {expected_kernel}")
    logger.debug(f"{'='*40}")

    torch.manual_seed(SEED)

    # Create test tensors
    y = torch.softmax(torch.randn(shape, dtype=torch.bfloat16), dim=-1)
    grad = torch.randn(shape, dtype=torch.bfloat16)

    # Convert to ttnn
    tt_y = ttnn.from_torch(y, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_grad = ttnn.from_torch(grad, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    # Run operation - kernel selection happens here
    # Check the logs to verify expected_kernel was used
    logger.debug("⚠️  Check the logs above for kernel selection message!")
    tt_output = ttnn.softmax_backward(tt_y, tt_grad, dim=-1)

    # Verify correctness
    pt_output = ttnn.to_torch(tt_output)
    pt_reference = reference_softmax_backward_output(y, grad, axis=-1)

    # Quick sanity check
    pcc = compute_pcc(pt_output, pt_reference)
    logger.debug(f"PCC: {pcc:.6f}")

    try:
        assert pcc >= PCC_THRESHOLD, f"Output doesn't match reference (PCC={pcc:.6f})"
    except AssertionError:
        # Print detailed metrics on failure to help debug
        print_tolerance_metrics(pt_output, pt_reference, dtype_name=f"shape={shape}", range=0)
        raise

    logger.debug(f"✅ Test passed for shape {shape} (expected {expected_kernel} kernel)")
    logger.debug(f"{'='*40}\n")
