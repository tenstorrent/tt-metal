# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import pytest
import random
import torch
import ttnn
from loguru import logger

from tests.sweep_framework.sweep_utils.utils import gen_pytest_parametrize_args
from tests.ttnn.utils_for_testing import start_measuring_time, stop_measuring_time
from tests.sweep_framework.sweep_utils.roofline_utils import get_run_return

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

random.seed(0)

DIM_SIZES = [0, 32]
"""Possible tensor dimensions are picked from this list"""

parameters = {
    f"rank_{rank}": {
        "tensor_shape": list(itertools.product(DIM_SIZES, repeat=rank)),
        "dim": list(range(-rank, rank)) if rank > 0 else [0, -1],  # Rank 0 has no dimensions
        # normalization operations to test
        "op": [
            "softmax",
        ],
    }
    for rank in range(5)
}


def run_softmax(device, tensor_shape, dim, op) -> tuple:
    """
    Test the compatibility of the torch and ttnn output for the given operation and different
    tensor shapes, and dim values.
    Checks for the exactness of shape, values, and dtype of the output tensors.
    Some operations raise exceptions in torch, we check if the same behavior is observed in ttnn.
    Note: We do not enforce the same exception type or message.
    """
    rank = len(tensor_shape)

    torch_tensor = (
        torch.randn(*tensor_shape, dtype=torch.bfloat16) if rank > 0 else torch.randn((), dtype=torch.bfloat16)
    )
    ttnn_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    scale = random.uniform(0.1, 10.0)
    is_scale_op = op in ("scale_mask_softmax", "scale_mask_softmax_in_place")
    torch_op, ttnn_op = torch.softmax, getattr(ttnn, op)

    # Run on both and flag exceptions
    torch_errored = False
    torch_error_msg = ""
    try:
        torch_result = torch_op(torch_tensor, dim=dim)  # if dim is not None else torch_op(torch_tensor)
    except IndexError as e:
        torch_errored = True
        torch_error_msg = str(e)

    ttnn_errored = False
    ttnn_error_msg = ""
    start_time = start_measuring_time()
    try:
        op_ttnn_result = ttnn_op(ttnn_tensor, scale=scale, dim=dim) if is_scale_op else ttnn_op(ttnn_tensor, dim=dim)
    except RuntimeError as e:
        ttnn_errored = True
        ttnn_error_msg = str(e)

    if torch_errored != ttnn_errored:
        return (
            False,
            f"mismatch in errors raised: torch: {torch_errored} ({torch_error_msg}), ttnn: {ttnn_errored} ({ttnn_error_msg})",
        )

    # Skip the rest of the test if an exception was raised in both
    if torch_errored:
        logger.warning(f"both torch and ttnn raised errors: torch: {torch_error_msg}, ttnn: {ttnn_error_msg}")
        return (True, "")

    ttnn_result = ttnn.to_torch(ttnn.from_device(op_ttnn_result))
    e2e_perf = stop_measuring_time(start_time)

    # Add scaling to the torch result to match the ttnn result
    if is_scale_op:
        torch_result *= scale

    atol = rtol = 0.1

    allclose = (torch.allclose(torch_result, ttnn_result, atol=atol, rtol=rtol, equal_nan=True),)
    if not allclose:
        return [(False, f"mismatch in allclose: torch: {torch_result}, ttnn: {ttnn_result}"), e2e_perf]
    # TODO: Verify that the sum of the output tensor is equal to 1.0 over the specified dimension
    expected_pcc = 0.99
    tensors = [ttnn_tensor, op_ttnn_result]
    return get_run_return(torch_result, ttnn_result, expected_pcc, tensors, e2e_perf)


@pytest.mark.parametrize(**gen_pytest_parametrize_args(parameters))
def test_normalization(
    device,
    tensor_shape,
    dim,
    op,
):
    result, error_msg = run_softmax(
        device,
        tensor_shape,
        dim,
        op,
    )
    assert result, error_msg


def run(
    tensor_shape,
    dim,
    op,
    *,
    device,
) -> tuple:
    return run_softmax(
        device,
        tensor_shape,
        dim,
        op,
    )
