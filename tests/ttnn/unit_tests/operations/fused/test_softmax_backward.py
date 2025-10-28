# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from numpy import absolute
import os
import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range_dtype


def reference_softmax_backward_output(y: torch.Tensor, grad: torch.Tensor, axis: int) -> torch.Tensor:
    dot = (y * grad).sum(dim=axis, keepdim=True)
    return y * (grad - dot)


def moreh_softmax_backward_reference(
    softmax_output: ttnn.Tensor, grad: ttnn.Tensor, dim: int, device: ttnn.Device
) -> torch.Tensor:
    """Reference using moreh's softmax_backward implementation"""
    # moreh supports only insigned int for dim, so normalize it here
    if dim < 0:
        dim = len(softmax_output.shape) + dim
    # Use moreh softmax_backward operation
    tt_output_moreh = ttnn.operations.moreh.softmax_backward(softmax_output, grad, dim)
    return ttnn.to_torch(tt_output_moreh)


def dump_tensors_to_files(
    pt_output_tensor_fused: torch.Tensor, pt_output_tensor_reference: torch.Tensor, pt_output_tensor_moreh: torch.Tensor
) -> None:
    if os.environ.get("TTNN_DUMP_TENSORS_TO_FILES", "0") == "1":
        torch.set_printoptions(threshold=1_000_000)

        # Write outputs to separate files for analysis
        with open("softmax_backward_fused_output.txt", "w") as f:
            f.write(f"pt_output_tensor_fused: {pt_output_tensor_fused}")

        with open("softmax_backward_reference_output.txt", "w") as f:
            f.write(f"pt_output_tensor_reference: {pt_output_tensor_reference}")

        with open("softmax_backward_moreh_output.txt", "w") as f:
            f.write(f"pt_output_tensor_moreh: {pt_output_tensor_moreh}")

        with open("softmax_backward_diff.txt", "w") as f:
            f.write(f"diff (fused vs reference): {pt_output_tensor_fused - pt_output_tensor_reference}")
            f.write(f"\ndiff (fused vs moreh): {pt_output_tensor_fused - pt_output_tensor_moreh}")
            f.write(f"\ndiff (reference vs moreh): {pt_output_tensor_reference - pt_output_tensor_moreh}")


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

        print(f"\nTolerance metrics for {dtype_name} and range {range}:")
        print(f"  Max absolute difference: {max_abs_diff:.6e}")
        print(f"  Mean absolute difference: {mean_abs_diff:.6e}")
        print(f"  Max relative difference: {max_rel_diff:.6e}")
        print(f"  Mean relative difference: {mean_rel_diff:.6e}")


BATCH_SIZE = 1


@pytest.mark.parametrize(
    "input_shapes",
    (
        # Don't waste time on small shapes
        # (torch.Size([1, 1, 32, 32])),
        # (torch.Size([1, 1, 320, 384])),
        (torch.Size([10, 3, 320, 384])),  # 3.6M items, doesn't fit in L1 cache
        # Llama2-7B tensor shapes:
        # (torch.Size([1, 1, 32, 2048])),    # actually should be (torch.Size([B, 2048])),
        # (torch.Size([1, BATCH_SIZE, 2048, 4096])),    # actually should be (torch.Size([B, 2048, 4096])),
        # (torch.Size([BATCH_SIZE, 32, 2048, 128])),
        # (torch.Size([BATCH_SIZE, 32, 2048, 2048])),
        # (torch.Size([1, BATCH_SIZE, 2048, 32000])),    # actually should be (torch.Size([B, 2048, 32000])),
    ),
)
@pytest.mark.parametrize(
    "dtype",
    [
        # ttnn.float32,
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "range",
    [20, 200],
)
@pytest.mark.parametrize(
    "dim",
    [-1, 3],  # only last dimension supported for now
)
def test_bw_softmax(input_shapes, dtype, range, dim, device):
    grad_data, grad_tensor = data_gen_with_range_dtype(input_shapes, -range, range, device, ttnn_dtype=dtype)
    in_data, input_tensor = data_gen_with_range_dtype(
        input_shapes, -range, range, device, required_grad=True, ttnn_dtype=dtype
    )

    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    pt_softmax_tensor = torch.softmax(in_data, dim=dim, dtype=torch_dtype)
    tt_softmax_tensor = ttnn.from_torch(pt_softmax_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    # Test the fused kernel implementation
    tt_output_tensor_fused = ttnn.softmax_backward(tt_softmax_tensor, grad_tensor, dim=dim)
    pt_output_tensor_fused = ttnn.to_torch(tt_output_tensor_fused)
    pt_output_tensor_reference = reference_softmax_backward_output(pt_softmax_tensor, grad_data, axis=dim)

    # Test moreh reference implementation
    pt_output_tensor_moreh = moreh_softmax_backward_reference(tt_softmax_tensor, grad_tensor, dim, device)

    # Debug output (enable with TTNN_DUMP_TENSORS_TO_FILES=1 environment variable)
    dump_tensors_to_files(pt_output_tensor_fused, pt_output_tensor_reference, pt_output_tensor_moreh)

    # Use torch.equal with moreh reference for bf8 and bf16 types
    if dtype in [ttnn.bfloat8_b, ttnn.bfloat16]:
        # Debug output (enable with TTNN_PRINT_TOLERANCES=1 environment variable)
        print_tolerance_metrics(
            pt_output_tensor_fused,
            pt_output_tensor_moreh,
            dtype_name=f"dtype={dtype}",
            range=range,
        )
        assert torch.equal(pt_output_tensor_fused, pt_output_tensor_moreh)

    # Use torch.allclose with torch reference for bf16 and fp32 types
    if dtype in [ttnn.bfloat16, ttnn.float32] and os.environ.get("TTNN_DEBUG_USE_TORCH_REFERENCE", "0") == "1":
        # TODO: tolerance is huge and unacceptable!
        relative_tolerance = 2.62e08
        absolute_tolerance = 3.15
        print(f"  Required rtol: {relative_tolerance}, atol: {absolute_tolerance}")

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
