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
    start_time = start_measuring_time()
    try:
        op_output_tensor = (
            ttnn_op(ttnn_tensor, dim=dim, keepdim=keepdim, use_multicore=use_multicore)
            if dim is not None
            else ttnn_op(ttnn_tensor, use_multicore=use_multicore)
        )
        output_tensor = ttnn.to_torch(ttnn.from_device(op_output_tensor))
    except RuntimeError as e:
        ttnn_errored = True
        ttnn_error_msg = str(e)
    e2e_perf = stop_measuring_time(start_time)

    if torch_errored != ttnn_errored:
        return [
            (
                False,
                f"mismatch in errors raised: torch: {torch_errored} ({torch_error_msg}), ttnn: {ttnn_errored} ({ttnn_error_msg})",
            ),
            e2e_perf,
        ]

    # Skip the rest of the test if an exception was raised in both
    if torch_errored:
        logger.warning(f"both torch and ttnn raised errors: torch: {torch_error_msg}, ttnn: {ttnn_error_msg}")
        return [(True, ""), e2e_perf]

    # Convert torch dtype from uint64 to int32
    # Note: torch does not have uint32
    int32_torch_result = torch_result.to(torch.int32)
    atol = rtol = 0.1
    allclose = (torch.allclose(int32_torch_result, output_tensor, atol=atol, rtol=rtol, equal_nan=True),)
    if not allclose:
        return [(False, f"mismatch in allclose: torch: {int32_torch_result}, ttnn: {output_tensor}"), e2e_perf]

    expected_pcc = 0.99
    tensors = [ttnn_tensor, op_output_tensor]
    return get_run_return(torch_result, output_tensor, expected_pcc, tensors, e2e_perf)


@pytest.mark.parametrize(**gen_pytest_parametrize_args(parameters))
def test_argmax(
    device,
    tensor_shape,
    dim,
    keepdim,
    use_multicore,
):
    (result, error_msg), e2e_perf = run_argmax(
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
