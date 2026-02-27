# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Tests corner cases of reduction operations.
# These tests are not meant to exhaustively sweep over all parameter combinations.
# Many parameters are exposed to make it easy to add new tests, but are currently
# set to a single value.

import math
import pytest
import torch
import ttnn
import sys

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import comp_allclose_and_pcc
from loguru import logger
from models.common.utility_functions import torch_random

# Test a 0D, 1D, 2D, 3D, 4D, 5D, and a 0-volume tensor
# @pytest.mark.parametrize("input_shape", [(), (2,), (3, 10), (6, 3, 60), (1, 11, 67, 77), (3, 6, 40, 63, 20), (6, 0, 32)])
# @pytest.mark.parametrize("dim", [None, 0])
# @pytest.mark.parametrize("keepdim", [True, False])
# @pytest.mark.parametrize("dtype", [ttnn.bfloat16])
# def test_prod(device, input_shape, dim, keepdim, dtype):
#     torch.manual_seed(0)

#     rank = len(input_shape)
#     if dim is not None and (dim < -rank or dim > rank - 1):
#         pytest.skip("Dimension not applicable for input shape")

#     torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
#     # tensor.size, which is called by torch.prod, doesn't accept dim=None,
#     # so we need to handle it separately.
#     # See https://github.com/pytorch/pytorch/issues/127882
#     if dim is None:
#         torch_output_tensor = torch.prod(torch_input_tensor)
#         if keepdim:
#             # torch.prod does not support keepdim=True for dim=None,
#             # so we need to reshape to match the input tensor.
#             new_shape = [1] * torch_input_tensor.dim()
#             torch_output_tensor = torch_output_tensor.reshape(new_shape)
#     else:
#         torch_output_tensor = torch.prod(torch_input_tensor, dim=dim, keepdim=keepdim)

#     input_tensor = ttnn.from_torch(
#         torch_input_tensor,
#         layout=ttnn.TILE_LAYOUT,
#         device=device,
#         memory_config=ttnn.L1_MEMORY_CONFIG,
#         dtype=dtype,
#         pad_value=1.0,
#     )

#     output_tensor = ttnn.prod(input_tensor, dim=dim, keepdim=keepdim, memory_config=ttnn.L1_MEMORY_CONFIG)
#     output_tensor = ttnn.from_device(output_tensor)

#     output_tensor = ttnn.to_torch(output_tensor, dtype=torch.bfloat16)
#     assert len(output_tensor.shape) == len(torch_output_tensor.shape)
#     assert output_tensor.shape == torch_output_tensor.shape

#     rtol = atol = 0.1
#     passing, output_pcc = comp_allclose_and_pcc(torch_output, tt_output_cpu, pcc=0.999, rtol=rtol, atol=atol)

#     logger.info(f"Out passing={passing}")
#     logger.info(f"Output pcc={output_pcc}")

#     assert passing


# @pytest.mark.parametrize("tensor_shape", [(), (2,), (3, 10), (6, 3, 60), (1, 11, 67, 77), (3, 6, 40, 63, 20), (6, 0, 32)])
@pytest.mark.parametrize("tensor_shape", [(6, 0, 32)])
@pytest.mark.parametrize("dim", [None, 0])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("op", ["mean", "sum", "max", "min", "prod"])
def test_reduction_ops(device, tensor_shape, dim, keepdim, dtype, layout, op):
    """
    Test the compatibility of the torch and ttnn output for the given operation and different
    tensor shapes, keepdim, and dim values.
    Checks for the exactness of shape, values, and dtype of the output tensors.
    Some operations raise exceptions in torch, we check if the same behavior is observed in ttnn.
    Note: We do not enforce the same exception type or message.
    """
    rank = len(tensor_shape)

    if dim is not None and (dim < -rank or dim > rank - 1):
        pytest.skip("Dimension not applicable for input shape")

    torch_tensor = torch.randn(tensor_shape, dtype=dtype)
    pad_value = 1.0 if op == "prod" else None
    ttnn_tensor = ttnn.from_torch(torch_tensor, layout=layout, device=device, pad_value=pad_value)

    torch_op, ttnn_op = getattr(torch, op), getattr(ttnn, op)

    # Run on both and flag exceptions
    torch_errored = False
    try:
        # tensor.size, which is called by various torch reduction ops, doesn't accept dim=None,
        # so we need to handle it separately.
        # See https://github.com/pytorch/pytorch/issues/127882
        if dim is None:
            torch_result = torch_op(torch_tensor)
            if keepdim:
                # torch.prod does not support keepdim=True for dim=None,
                # so we need to reshape to match the input tensor.
                new_shape = [1] * torch_tensor.dim()
                torch_result = torch_result.reshape(new_shape)
        else:
            torch_result = torch_op(torch_tensor, dim=dim, keepdim=keepdim)
    except (IndexError, TypeError, RuntimeError):
        torch_errored = True

    ttnn_errored = False
    try:
        ttnn_result = ttnn_op(ttnn_tensor, dim=dim, keepdim=keepdim)
    except RuntimeError:
        ttnn_errored = True

    if op == "min" or op == "max":
        assert (torch_errored and not ttnn_errored) or (
            torch_errored == ttnn_errored
        ), f"torch: {torch_errored}, ttnn: {ttnn_errored}"
    else:
        assert torch_errored == ttnn_errored, f"torch_errored: {torch_errored}, ttnn_errored: {ttnn_errored}"

    # Skip the rest of the test if an exception was raised in both
    if torch_errored:
        return

    # torch.min/max does not return just a tensor like ttnn.min/max.
    # It returns a small named tuple type (torch.return_types.min or torch.return_types.max) with:
    # .values – the min/max values (tensor of reduced values)
    # .indices – the indices where those values occur (equivalent to ttnn.argmin/argmax)
    # To make comparison with ttnn meaningful, extract the values only
    if isinstance(torch_result, (torch.return_types.min, torch.return_types.max)):
        torch_result = torch_result.values

    logger.info(f"Before: torch_result: {torch_result}")
    logger.info(f"Before: torch_result shape: {torch_result.shape}")
    logger.info(f"Before: ttnn_result: {ttnn_result}")
    logger.info(f"Before: ttnn_result shape: {ttnn_result.shape}")

    ttnn_result = ttnn.to_torch(ttnn.from_device(ttnn_result))

    logger.info(f"After: torch_result: {torch_result}")
    logger.info(f"After: torch_result shape: {torch_result.shape}")
    logger.info(f"After: ttnn_result: {ttnn_result}")
    logger.info(f"After: ttnn_result shape: {ttnn_result.shape}")

    atol = rtol = 0.1
    pcc = 0.999
    # There is a scale factor difference between torch and ttnn for std and var
    # But for other operations, it should be close. Issue #19478
    # if op == "std":
    #     atol, rtol = sys.maxsize, 0.1 + math.sqrt(2)
    # elif op == "var":
    #     atol, rtol = sys.maxsize, 0.1 + 2

    # if torch_result.numel() == 0:
    #     # Can't call comp_allclose for 0-volume tensors because it hits this Pythorch issue:
    #     # https://github.com/pytorch/pytorch/issues/71629
    #     assert torch.equal(ttnn_result, torch_result), f"ttnn: {ttnn_result}, torch: {torch_result}"
    #     # assert len(ttnn_result.shape) == len(torch_result.shape)
    #     # assert ttnn_result.shape == torch_result.shape
    #     # assert_with_pcc(torch_result, ttnn_result, pcc=pcc)
    # else:
    passing, output_pcc = comp_allclose_and_pcc(torch_result, ttnn_result, pcc=pcc, rtol=rtol, atol=atol)
    assert passing, f"{output_pcc}, torch: {torch_result}, ttnn: {ttnn_result}"
