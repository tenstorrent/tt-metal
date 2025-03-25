# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import itertools
import math
import pytest
import random
import sys
import torch
import ttnn

from tests.sweep_framework.sweep_utils.utils import gen_pytest_parametrize_args
from tests.ttnn.utils_for_testing import check_with_pcc

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

random.seed(0)

DIM_SIZES = [0, 32]
"""Possible tensor dimensions are picked from this list"""


parameters = {
    f"rank_{rank}": {
        "tensor_shape": list(itertools.product(DIM_SIZES, repeat=rank)),
        "dim": list(range(-rank, rank)) if rank > 0 else [None],  # Rank 0 has no dimensions
        "keepdim": [True, False],
        # Reduction operations to test
        "op": [
            "sum",
            "mean",
            "max",
            "min",
            "std",
            "var",
        ],
    }
    for rank in range(5)
}


def run_reduction(device, tensor_shape, dim, keepdim, op) -> list:
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

    # Skip the rest of the test if an exception was raised in both
    if torch_errored:
        return [True, f"mismatch in errors raised: torch: {torch_errored}, ttnn: {ttnn_errored}"]

    # torch's min/max double as argmin/argmax, so we need to extract the values only
    torch_result = (
        torch_result.values
        if isinstance(torch_result, (torch.return_types.min, torch.return_types.max))
        else torch_result
    )

    ttnn_result = ttnn.to_torch(ttnn.from_device(ttnn_result))

    pcc_result, msg = check_with_pcc(torch_result, ttnn_result, 0.99)

    if not pcc_result:
        return [False, msg]

    atol = rtol = 0.1
    # There is a scale factor difference between torch and ttnn for std and var
    # But for other operations, it should be close. Issue #19478
    if op == "std":
        atol, rtol = sys.maxsize, 0.1 + math.sqrt(2)
    elif op == "var":
        atol, rtol = sys.maxsize, 0.1 + 2

    return [
        torch.allclose(torch_result, ttnn_result, atol=atol, rtol=rtol, equal_nan=True),
        f"mismatch in allclose: torch: {torch_result}, ttnn: {ttnn_result}",
    ]


@pytest.mark.parametrize(**gen_pytest_parametrize_args(parameters))
def test_reduction(
    device,
    tensor_shape,
    dim,
    keepdim,
    op,
):
    run_reduction(
        device,
        tensor_shape,
        dim,
        keepdim,
        op,
    )


def run(
    tensor_shape,
    dim,
    keepdim,
    op,
    *,
    device,
) -> list:
    return run_reduction(
        device,
        tensor_shape,
        dim,
        keepdim,
        op,
    )
