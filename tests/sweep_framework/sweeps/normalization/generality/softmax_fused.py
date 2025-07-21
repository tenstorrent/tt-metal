# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

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
        # normalization operations to test
        "op": [
            "scale_mask_softmax",
            "softmax_in_place",
            "scale_mask_softmax_in_place",
            "scale_causal_mask_hw_dims_softmax_in_place",
        ],
    }
    for rank in range(5)
}


def run_softmax(device, tensor_shape, op) -> list:
    """
    Test the compatibility of the torch and ttnn output for the given operation and different
    tensor and shapes values.
    Checks for the exactness of shape, values, and dtype of the output tensors.
    Some operations raise exceptions in torch, we check if the same behavior is observed in ttnn.
    Note: We do not enforce the same exception type or message.
    """
    rank = len(tensor_shape)

    torch_tensor = (
        torch.randn(*tensor_shape, dtype=torch.bfloat16) if rank > 0 else torch.randn((), dtype=torch.bfloat16)
    )

    ttnn_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    torch_mask = torch_tensor.clone() < 0
    ttnn_mask = ttnn.from_torch(torch_mask.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)

    scale = random.uniform(0.1, 10.0)
    has_scale_mask = ("scale" in op) and ("mask" in op)
    torch_op, ttnn_op = torch.softmax, getattr(ttnn, op)

    # Log tensor and scale
    logger.debug(
        f"torch: {torch_tensor.shape}, {torch_tensor.dtype}, {torch_tensor.device}, {torch_tensor} \n"
        f"scale: {scale} \n"
        f"torch_mask: {torch_mask.shape}, {torch_mask.dtype}, {torch_mask.device}, {torch_mask} \n"
    )

    # Run on both and flag exceptions
    torch_errored = False
    torch_error_msg = ""
    try:
        if has_scale_mask:
            torch_result = torch_op(torch_tensor.masked_fill(torch_mask, float("-inf")), dim=-1) * scale
        else:
            torch_result = torch_op(torch_tensor, dim=-1)
    except IndexError as e:
        torch_errored = True
        torch_error_msg = str(e)

    ttnn_errored = False
    ttnn_error_msg = ""
    start_time = start_measuring_time()
    try:
        op_ttnn_result = ttnn_op(ttnn_tensor, scale, ttnn_mask) if has_scale_mask else ttnn_op(ttnn_tensor)
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
        logger.info(f"both torch and ttnn raised errors: torch: {torch_error_msg}, ttnn: {ttnn_error_msg}")
        return (True, "")

    ttnn_result = ttnn.to_torch(ttnn.from_device(op_ttnn_result))
    e2e_perf = stop_measuring_time(start_time)

    atol = rtol = 0.1

    allclose = (torch.allclose(torch_result, ttnn_result, atol=atol, rtol=rtol, equal_nan=True),)
    if not allclose:
        return [(False, f"mismatch in allclose: torch: {torch_result}, ttnn: {ttnn_result}"), e2e_perf]
    # TODO: Verify that the sum of the output tensor is equal to 1.0 over the specified dimension
    expected_pcc = 0.99
    tensors = [ttnn_tensor, op_ttnn_result]
    if has_scale_mask:
        tensors.append(ttnn_mask)
    return get_run_return(torch_result, ttnn_result, expected_pcc, tensors, e2e_perf)


@pytest.mark.parametrize(**gen_pytest_parametrize_args(parameters))
def test_normalization(
    device,
    tensor_shape,
    op,
):
    result, error_msg = run_softmax(
        device,
        tensor_shape,
        op,
    )
    assert result, error_msg


def run(
    tensor_shape,
    op,
    *,
    device,
) -> list:
    return run_softmax(
        device,
        tensor_shape,
        op,
    )
