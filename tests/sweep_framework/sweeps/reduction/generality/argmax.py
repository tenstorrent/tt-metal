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
        "dim": [None] + [rank, -1] if rank > 0 else [],
        "keepdim": [True, False],
        "use_multicore": [True, False],
    }
    for rank in range(5)
}


def run_argmax(device, tensor_shape, dim, keepdim, use_multicore) -> list:
    """
    Test the compatibility of the torch and ttnn output for the given operation and different
    tensor shapes and dim values.
    Checks for the exactness of shape, values, and dtype of the output tensors.
    Some operations raise exceptions in torch, we check if the same behavior is observed in ttnn.
    Note: We do not enforce the same exception type or message.
    """
    rank = len(tensor_shape)

    torch_tensor = (
        torch.randn(*tensor_shape, dtype=torch.bfloat16) if rank > 0 else torch.randn((), dtype=torch.bfloat16)
    )
    ttnn_tensor = ttnn.from_torch(torch_tensor, device=device)

    torch_op, ttnn_op = getattr(torch, "argmax"), getattr(ttnn, "argmax")

    # Run on both and flag exceptions
    torch_errored = False
    torch_error_msg = ""
    try:
        torch_result = torch_op(torch_tensor, dim=dim, keepdim=keepdim) if dim is not None else torch_op(torch_tensor)
    except (IndexError, RuntimeError) as e:
        torch_errored = True
        torch_error_msg = str(e)

    ttnn_errored = False
    ttnn_error_msg = ""
    try:
        ttnn_result = (
            ttnn_op(ttnn_tensor, dim=dim, keepdim=keepdim, use_multicore=use_multicore)
            if dim is not None
            else ttnn_op(ttnn_tensor, use_multicore=use_multicore)
        )
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
        return (False, msg + f"mismatch in pcc: torch: {torch_result}, ttnn: {ttnn_result}")

    # Convert torch dtype from uint64 to int32
    # Note: torch does not have uint32
    torch_result = torch_result.to(torch.int32)

    atol = rtol = 0.1
    return (
        torch.allclose(torch_result, ttnn_result, atol=atol, rtol=rtol, equal_nan=True),
        f"mismatch in allclose: torch: {torch_result}, ttnn: {ttnn_result}",
    )


@pytest.mark.parametrize(**gen_pytest_parametrize_args(parameters))
def test_argmax(
    device,
    tensor_shape,
    dim,
    keepdim,
    use_multicore,
):
    result, error_msg = run_argmax(
        device,
        tensor_shape,
        dim,
        keepdim,
        use_multicore,
    )
    assert result, error_msg


def run(
    tensor_shape,
    dim,
    keepdim,
    use_multicore,
    *,
    device,
) -> list:
    return run_argmax(
        device,
        tensor_shape,
        dim,
        keepdim,
        use_multicore,
    )
