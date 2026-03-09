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


# Helper function that calls topk with preallocated output tensors, whose shapes are
# determined by the ttnn_result obtained from a previous run of topk without preallocated output tensors.
def _run_topk_with_preallocated(ttnn_tensor, k, dim, device, ttnn_result):
    """Re-runs topk with preallocated output tensors and returns (values, indices) as torch tensors."""
    prealloc_values = ttnn.empty(
        list(ttnn_result[0].shape),
        dtype=ttnn_result[0].dtype,
        layout=ttnn_result[0].layout,
        device=device,
        memory_config=ttnn_result[0].memory_config(),
    )
    prealloc_indices = ttnn.empty(
        list(ttnn_result[1].shape),
        dtype=ttnn_result[1].dtype,
        layout=ttnn_result[1].layout,
        device=device,
        memory_config=ttnn_result[1].memory_config(),
    )
    ttnn_result_prealloc = ttnn.topk(ttnn_tensor, k, dim=dim, output_tensor=(prealloc_values, prealloc_indices))
    return (
        ttnn.to_torch(ttnn.from_device(ttnn_result_prealloc[0])),
        ttnn.to_torch(ttnn.from_device(ttnn_result_prealloc[1])),
    )


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
                # Various torch ops don't support keepdim=True for dim=None,
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
        if ttnn_errored and not torch_errored:
            raise

    if torch_errored and ttnn_errored:
        logger.info(f"Both PyTorch and TTNN errored")
    assert torch_errored == ttnn_errored, f"torch_errored: {torch_errored}, ttnn_errored: {ttnn_errored}"

    # Skip the rest of the test if an exception was raised in both
    if torch_errored:
        return

    ttnn_values = ttnn.to_torch(ttnn.from_device(ttnn_result[0]))
    ttnn_indices = ttnn.to_torch(ttnn.from_device(ttnn_result[1]))

    if (
        torch_values.numel() == 0
        and ttnn_values.numel() == 0
        and torch_indices.numel() == 0
        and ttnn_indices.numel() == 0
    ):
        logger.info(f"Both PyTorch and TTNN returned 0-volume tensors")
        assert (
            torch_values.shape == ttnn_values.shape
        ), f"Shape mismatch on values: torch: {torch_values.shape}, ttnn: {ttnn_values.shape}"
        assert (
            torch_indices.shape == ttnn_indices.shape
        ), f"Shape mismatch on indices: torch: {torch_indices.shape}, ttnn: {ttnn_indices.shape}"

        # Repeat the test with preallocated output tensors.
        prealloc_values, prealloc_indices = _run_topk_with_preallocated(ttnn_tensor, k, dim, device, ttnn_result)
        # The two methods should produce identical results.
        assert torch.equal(
            prealloc_values, ttnn_values
        ), f"Preallocated values differ from non-preallocated: {prealloc_values} vs {ttnn_values}"
        assert torch.equal(
            prealloc_indices, ttnn_indices
        ), f"Preallocated indices differ from non-preallocated: {prealloc_indices} vs {ttnn_indices}"

        # Other checks are not meaningful for empty tensors.
        return

    atol = rtol = 0.1
    pcc = 0.999
    passing, output_pcc = comp_allclose_and_pcc(torch_values, ttnn_values, pcc=pcc, rtol=rtol, atol=atol)
    assert passing, f"Values: {output_pcc}, torch: {torch_values}, ttnn: {ttnn_values}"

    ttnn_indices_adjusted = ttnn_indices.to(torch.int32)
    # Indices can come back as negative values from ttnn (stored as unsigned in bfloat16).
    # This is fixed by adding 2^16 to negative values.
    ttnn_indices_adjusted = torch.where(ttnn_indices_adjusted < 0, ttnn_indices_adjusted + 65536, ttnn_indices_adjusted)

    cosine_sim_target = 0.99
    # Use ttnn's returned indices to gather values from the original input tensor.
    # The result is "the values that ttnn thinks are the top-k."
    ttnn_gather_from_indices = torch.gather(torch_tensor, dim, ttnn_indices_adjusted.to(torch.int64))
    cosine = torch.nn.CosineSimilarity(dim=dim)
    # Comparing indices directly may not be a good measure because when there are ties
    # (duplicate values), both implementations may return different but equally valid
    # index positions.
    # Compare PyTorch's top-k values against the values gathered using ttnn's indices.
    # If ttnn returned correct indices, then gathering from the original tensor at those
    # index positions should yield the same (or very similar) values as PyTorch's top-k values.
    cosine_sim = torch.mean(cosine(torch_values, ttnn_gather_from_indices)).float()
    assert (
        cosine_sim >= cosine_sim_target
    ), f"Cosine similarity between topk values and gather from indices is {cosine_sim} which is less than {cosine_sim_target}"

    # Repeat the test with preallocated output tensors.
    prealloc_values, prealloc_indices = _run_topk_with_preallocated(ttnn_tensor, k, dim, device, ttnn_result)

    # The two methods should produce identical results.
    assert torch.equal(
        prealloc_values, ttnn_values
    ), f"Preallocated values differ from non-preallocated: {prealloc_values} vs {ttnn_values}"
    assert torch.equal(
        prealloc_indices, ttnn_indices
    ), f"Preallocated indices differ from non-preallocated: {prealloc_indices} vs {ttnn_indices}"


@pytest.mark.parametrize("tensor_shape", [(), (2,), (3, 6, 40, 63, 20), (6, 0, 32)])
@pytest.mark.parametrize("dim", [None, 0])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_argmax(device, tensor_shape, dim, keepdim, dtype, layout):
    """
    Test the compatibility of the torch and ttnn argmax output for different tensor shapes.
    argmax returns indices (UINT32). We validate semantically by checking that the values at
    the returned indices match the actual maximum values from torch.max, which is robust against
    tie-breaking differences between torch and ttnn.
    """
    torch.manual_seed(0)
    rank = len(tensor_shape)

    # Skip known ttnn.argmax limitations, but only for non-zero-volume tensors.
    # 0-volume tensors take a separate early-return path in ttnn and should be
    # tested for error parity with torch.
    is_zero_volume = 0 in tensor_shape
    if not is_zero_volume:
        if layout == ttnn.TILE_LAYOUT and dim is None:
            pytest.skip("ttnn.argmax does not support dim=None with TILE layout")
        if rank > 1 and dim is not None:
            normalized_dim = dim if dim >= 0 else dim + rank
            if normalized_dim != rank - 1:
                pytest.skip("ttnn.argmax only supports reduction on the last dimension")

    torch_tensor = torch.randn(tensor_shape, dtype=dtype)
    ttnn_tensor = ttnn.from_torch(torch_tensor, layout=layout, device=device)

    torch_errored = False
    try:
        torch_result = torch.argmax(torch_tensor, dim=dim, keepdim=keepdim)
    except (IndexError, TypeError, RuntimeError):
        torch_errored = True

    ttnn_errored = False
    try:
        ttnn_result = ttnn.argmax(ttnn_tensor, dim=dim, keepdim=keepdim)
    except (IndexError, TypeError, RuntimeError):
        ttnn_errored = True

    assert torch_errored == ttnn_errored, f"torch_errored: {torch_errored}, ttnn_errored: {ttnn_errored}"

    # Skip the rest of the test if an exception was raised in both
    if torch_errored:
        return

    ttnn_result = ttnn.to_torch(ttnn.from_device(ttnn_result))

    # For 0-volume results, verify shapes match
    if torch_result.numel() == 0 and ttnn_result.numel() == 0:
        assert (
            torch_result.shape == ttnn_result.shape
        ), f"Shape mismatch on 0-volume result: torch: {torch_result.shape}, ttnn: {ttnn_result.shape}"
        # Other checks are not meaningful for empty tensors.
        return

    # Secondary check: PCC on raw indices (ties are rare with random bfloat16)
    atol = rtol = 0.1
    pcc = 0.999
    passing, output_pcc = comp_allclose_and_pcc(torch_result, ttnn_result, pcc=pcc, rtol=rtol, atol=atol)
    assert passing, f"Indices PCC: {output_pcc}, torch: {torch_result}, ttnn: {ttnn_result_i64}"

    ttnn_result_i64 = ttnn_result.to(torch.int64)

    # Primary check: semantic validation - verify the values at ttnn's indices
    # match the values at torch's indices (robust against tie-breaking differences)
    torch_result_i64 = torch_result.to(torch.int64)
    if dim is not None:
        # torch.gather requires that index has the same number of dimensions as input.
        # When keepdim=False, the reduced dimension is removed, making the index tensor one
        # rank lower than the input; unsqueeze(dim) adds that dimension back.
        ttnn_gather_indices = ttnn_result_i64 if keepdim else ttnn_result_i64.unsqueeze(dim)
        torch_gather_indices = torch_result_i64 if keepdim else torch_result_i64.unsqueeze(dim)
        # Comparing indices directly may not be a good measure because when there are ties
        # (duplicate values), both implementations may return different but equally valid
        # index positions. Instead, we gather the values at the indices each implementation
        # thinks are the max, and compare them.
        ttnn_gathered = torch.gather(torch_tensor, dim, ttnn_gather_indices)
        torch_gathered = torch.gather(torch_tensor, dim, torch_gather_indices)
        if not keepdim:
            # Undo the unsqueeze that was added to make torch.gather work.
            # This helps make errors more readable in case of test failure.
            ttnn_gathered = ttnn_gathered.squeeze(dim)
            torch_gathered = torch_gathered.squeeze(dim)
        assert torch.allclose(
            ttnn_gathered.float(), torch_gathered.float(), atol=atol, rtol=rtol
        ), f"Values at ttnn indices don't match values at torch indices: ttnn={ttnn_gathered}, torch={torch_gathered}"
    else:
        # When dim is None, torch.argmax flattens the entire tensor and returns a single scalar
        # index into the flattened view. Therefore, we can't use torch.gather with a dim argument
        # on the original multi-dimensional tensor. Instead, we flatten the tensor and use the index
        # that ttnn returned to gather the value from the flattened tensor.
        ttnn_value = torch_tensor.flatten()[ttnn_result_i64.item()]
        # Do the same for torch then compare them.
        torch_value = torch_tensor.flatten()[torch_result_i64.item()]
        assert torch.allclose(
            ttnn_value.float(), torch_value.float(), atol=atol, rtol=rtol
        ), f"Value at ttnn index {ttnn_result_i64.item()} is {ttnn_value}, expected {torch_value} (at torch index {torch_result_i64.item()})"


@pytest.mark.parametrize("tensor_shape", [(), (2,), (3, 6, 40, 63, 20), (6, 0, 32)])
@pytest.mark.parametrize("dim", [0])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("op", ["cumsum", "cumprod"])
def test_accumulation(device, tensor_shape, dim, dtype, layout, op):
    """
    Test the compatibility of the torch and ttnn output for cumsum/cumprod and different
    tensor shapes and dim values.
    Unlike standard reductions, cumsum/cumprod produce same-shape outputs (accumulations).
    Checks that resulting tensors are within a certain tolerance of PyTorch outputs.
    Some operations raise exceptions in torch, we check if the same behavior is observed in ttnn.
    Note: We do not enforce the same exception type or message.
    """
    torch.manual_seed(0)
    rank = len(tensor_shape)

    torch_tensor = torch.randn(tensor_shape, dtype=dtype)
    pad_value = 1.0 if op == "cumprod" else None
    ttnn_tensor = ttnn.from_torch(torch_tensor, layout=layout, device=device, pad_value=pad_value)

    torch_op, ttnn_op = getattr(torch, op), getattr(ttnn, op)

    # Run on both and flag exceptions
    torch_errored = False
    try:
        torch_result = torch_op(torch_tensor, dim)
    except (IndexError, TypeError, RuntimeError):
        torch_errored = True

    ttnn_errored = False
    try:
        ttnn_result = ttnn_op(ttnn_tensor, dim)
    except (IndexError, TypeError, RuntimeError):
        ttnn_errored = True

    assert torch_errored == ttnn_errored, f"torch_errored: {torch_errored}, ttnn_errored: {ttnn_errored}"

    # Skip the rest of the test if an exception was raised in both
    if torch_errored:
        return

    ttnn_result = ttnn.to_torch(ttnn.from_device(ttnn_result))

    # For 0-volume results, verify shapes match
    if torch_result.numel() == 0 and ttnn_result.numel() == 0:
        assert (
            torch_result.shape == ttnn_result.shape
        ), f"Shape mismatch on 0-volume result: torch: {torch_result.shape}, ttnn: {ttnn_result.shape}"
        # Other checks are not meaningful for empty tensors.
        return

    atol = rtol = 0.1
    pcc = 0.999
    passing, output_pcc = comp_allclose_and_pcc(torch_result, ttnn_result, pcc=pcc, rtol=rtol, atol=atol)
    assert passing, f"{output_pcc}, torch: {torch_result}, ttnn: {ttnn_result}"
