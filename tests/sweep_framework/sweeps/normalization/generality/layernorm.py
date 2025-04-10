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
from tests.ttnn.utils_for_testing import check_with_pcc

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
            "rms_norm",
            "layer_norm",
        ],
    }
    for rank in range(5)
}


def run_normalization(device, tensor_shape, op) -> tuple:
    """
    Test the compatibility of the torch and ttnn output for the given operation and different
    tensor shapes. Checks for the exactness of shape, values, and dtype of the output tensors.
    Some operations raise exceptions in torch, we check if the same behavior is observed in ttnn.
    Note: We do not enforce the same exception type or message.
    """
    rank = len(tensor_shape)

    torch_tensor = (
        torch.randn(*tensor_shape, dtype=torch.bfloat16) if rank > 0 else torch.randn((), dtype=torch.bfloat16)
    )
    ttnn_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    # Generate additional inputs
    epsilon = random.uniform(1e-5, 1e-2)  # Small constant for numerical stability
    weight = torch.randn(tensor_shape[-1:], dtype=torch.bfloat16) if rank > 0 else None
    bias = torch.randn(tensor_shape[-1:], dtype=torch.bfloat16) if rank > 0 else None

    ttnn_weight = ttnn.from_torch(weight, layout=ttnn.TILE_LAYOUT, device=device) if weight is not None else None
    ttnn_bias = ttnn.from_torch(bias, layout=ttnn.TILE_LAYOUT, device=device) if bias is not None else None

    # Run on both and flag exceptions
    torch_errored = False
    torch_error_msg = ""
    try:
        if op == "rms_norm":
            # Note: torch's RMSNorm is not available in the current version, so we use a manual implementation
            torch_result = torch_tensor / torch.sqrt(torch.mean(torch_tensor**2, dim=-1, keepdim=True) + epsilon)
            if weight is not None:
                torch_result *= weight
            if bias is not None:
                torch_result += bias
        elif op == "layer_norm":
            torch_result = torch.nn.functional.layer_norm(
                torch_tensor, tensor_shape[-1:], weight=weight, bias=bias, eps=epsilon
            )
        else:
            raise ValueError(f"Unsupported operation: {op}")
    except RuntimeError as e:
        torch_errored = True
        torch_error_msg = str(e)

    ttnn_errored = False
    ttnn_error_msg = ""
    try:
        if op == "rms_norm":
            ttnn_result = ttnn.rms_norm(ttnn_tensor, epsilon=epsilon, weight=ttnn_weight, bias=ttnn_bias)
        elif op == "layer_norm":
            ttnn_result = ttnn.layer_norm(ttnn_tensor, epsilon=epsilon, weight=ttnn_weight, bias=ttnn_bias)
        else:
            raise ValueError(f"Unsupported operation: {op}")

    # Note: The exception type may differ from torch, so we only check if an exception was raised
    # and not the specific type or message.
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

    ttnn_result = ttnn.to_torch(ttnn.from_device(ttnn_result))

    pcc_result, msg = check_with_pcc(torch_result, ttnn_result, 0.99)

    if not pcc_result:
        return (False, msg)

    atol = rtol = 0.1

    return (
        torch.allclose(torch_result, ttnn_result, atol=atol, rtol=rtol, equal_nan=True),
        f"mismatch in allclose: torch: {torch_result}, ttnn: {ttnn_result}",
    )


@pytest.mark.parametrize(**gen_pytest_parametrize_args(parameters))
def test_normalization(
    device,
    tensor_shape,
    op,
):
    result, error_msg = run_normalization(
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
) -> tuple:
    return run_normalization(
        device,
        tensor_shape,
        op,
    )
