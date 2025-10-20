# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc_without_tensor_printout
from models.common.utility_functions import is_grayskull, is_blackhole, torch_random

# seed torch random for reproducibility
torch.manual_seed(0)


def random_torch_tensor(dtype, shape):
    if dtype == ttnn.uint8:
        return torch.randint(0, 100, shape).to(torch.uint8)
    if dtype == ttnn.uint16:
        return torch.randint(0, 100, shape).to(torch.int16)
    if dtype == ttnn.int32:
        return torch.randint(-(2**31), 2**31, shape, dtype=torch.int32)
    if dtype == ttnn.uint32:
        return torch.randint(0, 2**31, shape, dtype=torch.int32)
    return torch.rand(shape).bfloat16().float()


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


# dtypes = [ttnn.bfloat16, ttnn.float32, ttnn.int32]
# dtypes = [ttnn.uint16, ttnn.uint8]
dtypes = [ttnn.uint8]
# dtypes = [ttnn.uint16]


@pytest.mark.parametrize("in_dtype", dtypes)
@pytest.mark.parametrize("use_multicore", [True])
@pytest.mark.parametrize("H", [32])
@pytest.mark.parametrize("W", [32])
def test_tilize_2D(device, in_dtype, use_multicore, H, W):
    torch_input_shape = [H, W]

    # torch_input = random_torch_tensor(in_dtype, torch_input_shape)
    torch_input = offset_increment_tensor(torch_input_shape, dtype=torch.uint8)

    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=in_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
    print(ttnn_input)

    output_tt = ttnn.tilize(ttnn_input, use_multicore=use_multicore)
    print(output_tt)
    output_torch = ttnn.to_torch(output_tt)

    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_input, output_torch)
    logger.info(pcc_msg)
    print(torch_input)
    assert passing
