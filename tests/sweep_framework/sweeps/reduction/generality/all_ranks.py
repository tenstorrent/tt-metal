# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import itertools
import pytest
import random
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

random.seed(0)

# Test configurations
ranks = range(0, 5)
keepdims = [True, False]
zero_dim_sizes = [0, 32]
dims_to_test = lambda rank: list(range(-rank, rank)) if rank > 0 else [None]  # Rank 0 has no dimensions

# Reduction operations to test
reduction_ops = [
    "sum",
    "mean",
    "max",
    "min",
    "std",
    "var",
]


def get_params() -> dict:
    """Generate test parameters for all combinations of ranks, tensor shapes, reduction operations, and dimensions.

    Returns a dictionary with keys "argvalues" and "ids". The "argvalues" key contains a list of tuples, each tuple
    containing the parameters for a test case. The "ids" key contains a list of strings, each string being a unique
    identifier for a test case.
    """
    argvalues = []
    ids = []
    for rank in ranks:
        shape = [2 if i not in range(rank) else 0 for i in range(rank)]
        for zero_dim_combination in itertools.product(zero_dim_sizes, repeat=rank):
            tensor_shape = [zero_dim_combination[i] if shape[i] == 0 else shape[i] for i in range(rank)]
            tensor_shape_str = str(tensor_shape).replace(",", "")

            for keepdim, dim, op in itertools.product(keepdims, dims_to_test(rank), reduction_ops):
                argvalues.append((tensor_shape, keepdim, dim, op))
                ids.append(f"rank={rank}, tensor_shape={tensor_shape_str}, keepdim={keepdim}, dim={dim}, op={op}")
    return {"argvalues": argvalues, "ids": ids}


@pytest.mark.parametrize(argnames="tensor_shape, keepdim, dim, op", **get_params())
def test_reduction(device, tensor_shape, keepdim, dim, op):
    """
    Test the compatibility of the torch and ttnn output for the given operation and different
    tensor shapes, keepdim, and dim values.
    Checks for the exactness of shape, values, and dtype of the output tensors.
    Some operations raise exceptions in torch, we check if the same behavior is observed in ttnn.
    Note: We do not enforce the same exception type or message.
    """
    rank = len(tensor_shape)

    torch_tensor = torch.randn(*tensor_shape) if rank > 0 else torch.randn(())
    ttnn_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    torch_op, ttnn_op = getattr(torch, op), getattr(ttnn, op)

    # Run on both and flag exceptions
    torch_errored = False
    try:
        torch_result = torch_op(torch_tensor, dim=dim, keepdim=keepdim) if dim is not None else torch_op(torch_tensor)
    except IndexError:
        torch_errored = True

    ttnn_errored = False
    try:
        ttnn_result = ttnn_op(ttnn_tensor, dim=dim, keepdim=keepdim) if dim is not None else ttnn_op(ttnn_tensor)
    except RuntimeError:
        ttnn_errored = True

    assert torch_errored == ttnn_errored, f"torch: {torch_errored}, ttnn: {ttnn_errored}"

    # Skip the rest of the test if an exception was raised in both
    if torch_errored:
        return

    # torch's min/max double as argmin/argmax, so we need to extract the values only
    torch_result = (
        torch_result.values
        if isinstance(torch_result, (torch.return_types.min, torch.return_types.max))
        else torch_result
    )

    ttnn_result = ttnn.to_torch(ttnn.from_device(ttnn_result))

    assert_with_pcc(torch_result, ttnn_result, 0.99)

    if op not in ["std", "var"]:
        # There is a scale factor difference between torch and ttnn for std and var
        # But for other operations, it should be close. Issue #19478
        assert torch.allclose(
            torch_result, ttnn_result, atol=0.2, rtol=0.05, equal_nan=True
        ), f"torch: {torch_result}, ttnn: {ttnn_result}"
