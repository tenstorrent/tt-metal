# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import itertools
import pytest
import random
import torch
import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

random.seed(0)

# Test configurations
ranks = range(0, 5)  # Rank from 1 to 4
keepdims = [True, False]
zero_dim_sizes = [0, 2]  # Exclude zero dimensions
dims_to_test = lambda rank: list(range(-rank, rank)) if rank > 0 else [None]

# Reduction operations to test
reduction_ops = [
    "sum",
    "mean",
    # "prod",
    "min",
    "max",
    "std",
    "var",
]


def get_params() -> dict:
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
    rank = len(tensor_shape)

    tensor_torch = torch.randn(*tensor_shape) if rank > 0 else torch.randn(())
    tensor_ttnn = ttnn.from_torch(tensor_torch, layout=ttnn.TILE_LAYOUT, device=device)

    op_torch = getattr(torch, op)
    result_torch = op_torch(tensor_torch, dim=dim, keepdim=keepdim) if dim is not None else op_torch(tensor_torch)

    op_ttnn = getattr(ttnn, op)
    result_ttnn = op_ttnn(tensor_ttnn, dim=dim, keepdim=keepdim) if dim is not None else op_ttnn(tensor_ttnn)
    result_ttnn = ttnn.to_torch(ttnn.from_device(result_ttnn))

    result_torch = (
        result_torch.values
        if isinstance(result_torch, (torch.return_types.min, torch.return_types.max))
        else result_torch
    )

    assert result_torch.shape == result_ttnn.shape
    assert result_torch.dim() == result_ttnn.dim()
    assert result_torch.numel() == result_ttnn.numel()

    # There is a scale factor difference between torch and ttnn for std and var
    if op in ["std", "var"]:
        val, msg = check_with_pcc(result_torch, result_ttnn, 0.99)
        assert val, msg
    else:
        assert torch.allclose(
            result_torch, result_ttnn, atol=0.2, rtol=0.05
        ), f"torch: {result_torch}, ttnn: {result_ttnn}"
    assert result_torch.dtype == result_ttnn.dtype
