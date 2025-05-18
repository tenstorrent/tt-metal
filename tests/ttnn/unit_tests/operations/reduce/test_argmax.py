# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from loguru import logger
from tests.ttnn.utils_for_testing import check_with_pcc


@pytest.mark.parametrize(
    argnames="tensor_shape, dim, keepdim, use_multicore",
    argvalues=[
        ([], None, True, True),
        ([32], -1, False, False),
        ([32, 0], 1, True, True),
        ([64], -1, True, False),
        ([1, 512], -1, True, True),
        ([1, 1024], -1, True, True),
        ([1, 65], -1, True, True),
        ([8, 10, 129], 2, True, False),
        ([1, 8, 160], -1, False, True),
        ([1, 256, 1024 * 8], -1, False, True),
        ([32, 32, 32, 1], -1, True, True),
    ],
)
def test_argmax(device, tensor_shape, dim, keepdim, use_multicore):
    """
    Test the compatibility of the torch and ttnn output for argmax of different
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
        if dim is not None:
            ttnn_result = ttnn_op(ttnn_tensor, dim=dim, keepdim=keepdim, use_multicore=use_multicore)
        else:
            ttnn_result = ttnn_op(ttnn_tensor, use_multicore=use_multicore)
    except RuntimeError as e:
        ttnn_errored = True
        ttnn_error_msg = str(e)

    assert (
        torch_errored == ttnn_errored
    ), f"mismatch in errors raised: torch: {torch_errored} ({torch_error_msg}), ttnn: {ttnn_errored} ({ttnn_error_msg})"

    # Skip the rest of the test if an exception was raised in both
    if torch_errored:
        logger.warning(f"both torch and ttnn raised errors: torch: {torch_error_msg}, ttnn: {ttnn_error_msg}")
        return

    ttnn_result = ttnn.to_torch(ttnn.from_device(ttnn_result))

    pcc_result, msg = check_with_pcc(torch_result, ttnn_result, 0.99)

    assert pcc_result, msg + f"mismatch in pcc: torch: {torch_result}, ttnn: {ttnn_result}"

    # Convert torch dtype from uint64 to int32
    # Note: torch does not have uint32
    torch_result = torch_result.to(torch.int32)

    atol = rtol = 0.1
    assert torch.allclose(
        torch_result, ttnn_result, atol=atol, rtol=rtol, equal_nan=True
    ), f"mismatch in allclose: torch: {torch_result}, ttnn: {ttnn_result}"
