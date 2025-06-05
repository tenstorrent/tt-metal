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
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from tests.sweep_framework.sweep_utils.roofline_utils import get_run_return

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
        "dtype": [torch.bfloat16, torch.float32],
    }
    for rank in range(5)
}


def run_reduction(device, tensor_shape, dim, keepdim, op, dtype) -> list:
    """
    Test the compatibility of the torch and ttnn output for the given operation and different
    tensor shapes, keepdim, and dim values.
    Checks for the exactness of shape, values, and dtype of the output tensors.
    Some operations raise exceptions in torch, we check if the same behavior is observed in ttnn.
    Note: We do not enforce the same exception type or message.
    """
    rank = len(tensor_shape)

    torch_tensor = torch.randn(*tensor_shape, dtype=dtype) if rank > 0 else torch.randn((), dtype=dtype)
    ttnn_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    torch_op, ttnn_op = getattr(torch, op), getattr(ttnn, op)

    # Run on both and flag exceptions
    torch_errored = False
    try:
        torch_result = torch_op(torch_tensor, dim=dim, keepdim=keepdim) if dim is not None else torch_op(torch_tensor)
    except IndexError:
        torch_errored = True

    ttnn_errored = False
    start_time = start_measuring_time()
    try:
        op_output_tensor = ttnn_op(ttnn_tensor, dim=dim, keepdim=keepdim) if dim is not None else ttnn_op(ttnn_tensor)
        output_tensor = ttnn.to_torch(ttnn.from_device(op_output_tensor))
    except RuntimeError:
        ttnn_errored = True
    e2e_perf = stop_measuring_time(start_time)

    # Skip the rest of the test if an exception was raised in both
    if torch_errored:
        return [(True, f"mismatch in errors raised: torch: {torch_errored}, ttnn: {ttnn_errored}"), e2e_perf]

    # torch's min/max double as argmin/argmax, so we need to extract the values only
    torch_result = (
        torch_result.values
        if isinstance(torch_result, (torch.return_types.min, torch.return_types.max))
        else torch_result
    )

    atol = rtol = 0.1
    # There is a scale factor difference between torch and ttnn for std and var
    # But for other operations, it should be close. Issue #19478
    if op == "std":
        atol, rtol = sys.maxsize, 0.1 + math.sqrt(2)
    elif op == "var":
        atol, rtol = sys.maxsize, 0.1 + 2

    allclose = torch.allclose(torch_result, output_tensor, atol=atol, rtol=rtol, equal_nan=True)
    if not allclose:
        return [(False, f"mismatch in allclose: torch: {torch_result}, ttnn: {output_tensor}"), e2e_perf]
    expected_pcc = 0.99
    tensors = [ttnn_tensor, op_output_tensor]
    return get_run_return(torch_result, output_tensor, expected_pcc, tensors, e2e_perf)


@pytest.mark.parametrize(**gen_pytest_parametrize_args(parameters))
def test_reduction(
    device,
    tensor_shape,
    dim,
    keepdim,
    op,
    dtype,
):
    result, msg = run_reduction(
        device,
        tensor_shape,
        dim,
        keepdim,
        op,
        dtype,
    )
    assert result, msg


def run(
    tensor_shape,
    dim,
    keepdim,
    op,
    dtype,
    *,
    device,
) -> list:
    return run_reduction(
        device,
        tensor_shape,
        dim,
        keepdim,
        op,
        dtype,
    )
