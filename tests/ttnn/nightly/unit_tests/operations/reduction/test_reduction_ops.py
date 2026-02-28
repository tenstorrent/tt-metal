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


# Test a 0D, 1D, 5D, and a 0-volume tensor
@pytest.mark.parametrize("tensor_shape", [(), (2,), (3, 6, 40, 63, 20), (6, 0, 32)])
@pytest.mark.parametrize("dim", [None, 0])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
# std and var will be handled separately. Issue #25100
@pytest.mark.parametrize("op", ["mean", "sum", "max", "min", "prod"])
def test_reduction_ops(device, tensor_shape, dim, keepdim, dtype, layout, op):
    """
    Test the compatibility of the torch and ttnn output for the given operation and different
    tensor shapes, keepdim, and dim values.
    Checks that resulting tensors are within a certain tolerance of PyTorch outputs.
    Some operations raise exceptions in torch, we check if the same behavior is observed in ttnn.
    Note: We do not enforce the same exception type or message.
    """
    torch.manual_seed(0)
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
    except (IndexError, TypeError, RuntimeError):
        ttnn_errored = True

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

    ttnn_result = ttnn.to_torch(ttnn.from_device(ttnn_result))

    atol = rtol = 0.1
    pcc = 0.999
    passing, output_pcc = comp_allclose_and_pcc(torch_result, ttnn_result, pcc=pcc, rtol=rtol, atol=atol)
    assert passing, f"{output_pcc}, torch: {torch_result}, ttnn: {ttnn_result}"


# @pytest.mark.parametrize("tensor_shape", [(60, 0, 32)])
@pytest.mark.parametrize("tensor_shape", [(), (170,), (3, 6, 40, 63, 20), (60, 0, 32)])
@pytest.mark.parametrize("dim", [0, None])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
# @pytest.mark.parametrize("k", [50])
@pytest.mark.parametrize("k", [50, 1, 0])
def test_topk(device, tensor_shape, dim, dtype, layout, k):
    """
    Test the compatibility of the torch and ttnn topk output for different tensor shapes.
    topk returns a tuple of (values, indices). We compare values via PCC and validate
    indices semantically by gathering from the original tensor and checking cosine similarity,
    since torch and ttnn may break ties differently in bfloat16.
    """
    torch.manual_seed(0)
    rank = len(tensor_shape)

    if dim is not None and (dim < -rank or dim > rank - 1):
        pytest.skip("Dimension not applicable for input shape")

    torch_tensor = torch.randn(tensor_shape, dtype=dtype)
    ttnn_tensor = ttnn.from_torch(torch_tensor, layout=layout, device=device)

    torch_errored = False
    try:
        torch_values, torch_indices = torch.topk(torch_tensor, k, dim=dim)
    except (IndexError, TypeError, RuntimeError):
        torch_errored = True

    ttnn_errored = False
    try:
        ttnn_result = ttnn.topk(ttnn_tensor, k, dim=dim)
    except (IndexError, TypeError, RuntimeError):
        ttnn_errored = True

    if torch_errored and ttnn_errored:
        logger.info(f"Both PyTorch and TTNN errored")
    assert torch_errored == ttnn_errored, f"torch_errored: {torch_errored}, ttnn_errored: {ttnn_errored}"

    if torch_errored:
        return

    ttnn_values = ttnn.to_torch(ttnn.from_device(ttnn_result[0]))
    ttnn_indices = ttnn.to_torch(ttnn.from_device(ttnn_result[1]))

    pcc = 0.999
    passing, output_pcc = comp_allclose_and_pcc(torch_values, ttnn_values, pcc=pcc, rtol=0.1, atol=0.1)
    assert passing, f"Values: {output_pcc}, torch: {torch_values}, ttnn: {ttnn_values}"

    if (
        torch_values.numel() == 0
        and ttnn_values.numel() == 0
        and torch_indices.numel() == 0
        and ttnn_indices.numel() == 0
    ):
        logger.info(f"Both PyTorch and TTNN returned 0-volume tensors")
        # Cosine similarity cannot be computed for empty tensors.
        # comp_allclose_and_pcc already validated that shapes match, so we can return.
        return

    ttnn_indices = ttnn_indices.to(torch.int32)
    # Indices can come back as negative values from ttnn (stored as unsigned in bfloat16).
    # This is fixed by adding 2^16 to negative values.
    ttnn_indices = torch.where(ttnn_indices < 0, ttnn_indices + 65536, ttnn_indices)

    cosine_sim_target = 0.99
    ttnn_gather_from_indices = torch.gather(torch_tensor, dim, ttnn_indices.to(torch.int64))
    cosine = torch.nn.CosineSimilarity(dim=dim)
    cosine_sim = torch.mean(cosine(torch_values, ttnn_gather_from_indices)).float()
    assert (
        cosine_sim >= cosine_sim_target
    ), f"Cosine similarity between topk values and gather from indices is {cosine_sim} which is less than {cosine_sim_target}"
