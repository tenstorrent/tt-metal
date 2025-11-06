# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import torch
import pytest
import ttnn
from loguru import logger
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range_dtype


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
    logger.info(f"  PCC: {pcc_value:.6f} (threshold: {pcc_threshold})")
    assert pcc_value >= pcc_threshold, f"PCC {pcc_value:.6f} < threshold {pcc_threshold}"


def dump_tensors_to_files(pt_output_tensor_fused: torch.Tensor, pt_output_tensor_reference: torch.Tensor) -> None:
    if os.environ.get("TTNN_DUMP_TENSORS_TO_FILES", "0") == "1":
        torch.set_printoptions(threshold=1_000_000)

        # Write outputs to separate files for analysis
        with open("softmax_backward_fused_output.txt", "w") as f:
            f.write(f"pt_output_tensor_fused: {pt_output_tensor_fused}")

        with open("softmax_backward_reference_output.txt", "w") as f:
            f.write(f"pt_output_tensor_reference: {pt_output_tensor_reference}")

        with open("softmax_backward_diff.txt", "w") as f:
            f.write(f"diff (fused vs reference): {pt_output_tensor_fused - pt_output_tensor_reference}")


def print_tolerance_metrics(tensor1: torch.Tensor, tensor2: torch.Tensor, dtype_name: str = "", range: int = 0) -> None:
    if os.environ.get("TTNN_PRINT_TOLERANCES", "0") == "1":
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


PCC_THRESHOLD = 0.987
BATCH_SIZE = 1
SEED = 77


@pytest.mark.parametrize(
    "input_shapes",
    (
        # Llama2-7B tensor shapes:
        # (torch.Size([1, 1, 32, 2048])),    # actually should be (torch.Size([B, 2048])),
        # (torch.Size([1, BATCH_SIZE, 2048, 4096])),    # actually should be (torch.Size([B, 2048, 4096])),
        # (torch.Size([BATCH_SIZE, 32, 2048, 128])),
        (torch.Size([BATCH_SIZE, 32, 2048, 2048])),
        # (torch.Size([1, BATCH_SIZE, 2048, 32000])),  # actually should be (torch.Size([B, 2048, 32000])),
    ),
)
@pytest.mark.parametrize(
    "dtype",
    [
        # ttnn.float32, # not supported by softmax_backward kernel
        ttnn.bfloat16,
        # ttnn.bfloat8_b, # not supported by pytorch
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
def test_bw_softmax(input_shapes, dtype, range, dim, device):
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

    # Debug output (enable with TTNN_DUMP_TENSORS_TO_FILES=1 environment variable)
    dump_tensors_to_files(pt_output_tensor_fused, pt_output_tensor_reference)

    # Use torch.allclose with torch reference for bf16 and fp32 types
    if dtype in [ttnn.bfloat16, ttnn.float32]:
        relative_tolerance = 3e05
        absolute_tolerance = 6e-3
        logger.info(f"  Required rtol: {relative_tolerance}, atol: {absolute_tolerance}")

        # Debug output (enable with TTNN_PRINT_TOLERANCES=1 environment variable)
        print_tolerance_metrics(
            pt_output_tensor_fused,
            pt_output_tensor_reference,
            dtype_name=f"dtype={dtype}",
            range=range,
        )

        assert torch.allclose(
            pt_output_tensor_fused,
            pt_output_tensor_reference,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )

        assert_pcc(pt_output_tensor_fused, pt_output_tensor_reference, PCC_THRESHOLD)


@pytest.mark.parametrize(
    "input_shapes",
    (
        # Shapes with non-tile-aligned dimensions (require padding)
        (torch.Size([1, 1, 320, 1000])),  # width=1000, needs padding to 1024
        (torch.Size([2, 1, 64, 500])),  # width=500, needs padding to 512
    ),
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
def test_bw_softmax_padded(input_shapes, dtype, range, dim, device):
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

    logger.info(f"\nOriginal shape: {input_shapes}")
    logger.info(f"Padded shape: {tt_softmax_tensor.shape}")
    # logger.info(f"Logical shape: {tt_softmax_tensor.logical_shape()}")

    # Run softmax backward on padded tensors
    tt_output_tensor_fused = ttnn.softmax_backward(tt_softmax_tensor, tt_grad_tensor, dim=dim)

    # Convert back to torch (automatically unpads to logical shape)
    pt_output_tensor_fused = ttnn.to_torch(tt_output_tensor_fused)

    # Debug output
    dump_tensors_to_files(pt_output_tensor_fused, pt_output_tensor_reference)

    # Verify the output matches reference (only on logical/unpadded region)
    if dtype in [ttnn.bfloat16, ttnn.float32]:
        relative_tolerance = 3e05
        absolute_tolerance = 6e-3
        logger.info(f"  Required rtol: {relative_tolerance}, atol: {absolute_tolerance}")

        print_tolerance_metrics(
            pt_output_tensor_fused,
            pt_output_tensor_reference,
            dtype_name=f"dtype={dtype} (padded)",
            range=range,
        )

        # The key test: output should match reference on the logical (unpadded) region
        # This verifies that padding did not corrupt the result
        assert (
            pt_output_tensor_fused.shape == pt_output_tensor_reference.shape
        ), f"Unpadded output shape mismatch: {pt_output_tensor_fused.shape} vs {pt_output_tensor_reference.shape}"

        assert torch.allclose(
            pt_output_tensor_fused,
            pt_output_tensor_reference,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        ), f"Padded tensor output does not match reference! This means padding corrupted the reduction."

        # PCC assertion
        assert_pcc(pt_output_tensor_fused, pt_output_tensor_reference, PCC_THRESHOLD)
