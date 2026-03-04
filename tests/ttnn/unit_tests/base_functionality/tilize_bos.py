# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc_without_tensor_printout
from models.common.utility_functions import is_blackhole, torch_random


def offset_increment_tensor(shape, offset=0, dtype=torch.int32, step=1):
    """
    Create a tensor of given shape where values start from `offset`
    and increment by `step` in row-major order.

    Args:
        shape  : tuple of ints for the tensor dimensions
        offset : starting value
        dtype  : torch dtype
        step   : increment between consecutive elements
    """
    numel = 1
    for s in shape:
        numel *= s
    return torch.arange(
        offset,
        offset + numel * step,
        step=step,
        dtype=dtype,
    ).reshape(shape)


dtypes = [ttnn.uint8, ttnn.uint16]


@pytest.mark.parametrize("in_dtype", dtypes)
@pytest.mark.parametrize("use_multicore", [True])
@pytest.mark.parametrize("H", [32])
@pytest.mark.parametrize("W", [32])
def test_tilize_2D(device, in_dtype, use_multicore, H, W):
    torch_input_shape = [H, W]

    torch_input = offset_increment_tensor(torch_input_shape, dtype=torch.int8)

    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=in_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

    output_tt = ttnn.tilize(ttnn_input, use_multicore=use_multicore)
    output_torch = ttnn.to_torch(output_tt)

    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_input, output_torch)
    logger.info(pcc_msg)
    assert passing, "PCC check failed for tilize output"
