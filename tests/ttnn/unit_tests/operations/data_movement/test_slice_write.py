# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from models.utility_functions import is_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.unit_tests.operations.test_utils import round_up
from models.utility_functions import skip_for_blackhole
import math
import random


def random_torch_tensor(dtype, shape):
    if dtype == ttnn.uint16:
        return torch.randint(0, 100, shape).to(torch.int16)
    if dtype == ttnn.int32:
        return torch.randint(-(2**31), 2**31, shape, dtype=torch.int32)
    return torch.rand(shape).bfloat16().float()


def _rand_shape(dim_size, choices=(8, 16, 32, 64, 128, 256)):
    # Choose a reasonable size per dim (keeps DRAM/L1 happy)
    ret = [random.choice(choices) for _ in range(dim_size)]
    if dim_size > 5:
        ret[: dim_size - 5] = [1 for _ in range(dim_size - 5)]  # rank 6+ tensors are too big
    return ret


def _rand_slice_params(shape):
    begins, ends, strides = [], [], []
    ndim = len(shape)
    for dim, size in enumerate(shape):
        s = 1  # stride always 1
        # pick a random start < size
        b = random.randint(0, size - 1)
        # choose a random end > b, ≤ size
        e = random.randint(b + 1, size)
        begins.append(b)
        ends.append(e)
        strides.append(s)
    return begins, ends, strides


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


@skip_for_blackhole("Fails on Blackhole. Issue #28021")
@pytest.mark.parametrize("rank", range(1, 9))  # 1D … 8D
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT])
def test_slice_write_nd(rank, layout, device):
    base_seed = 2005
    random.seed(base_seed)
    torch.manual_seed(base_seed)

    shape = _rand_shape(rank)
    begins, ends, strides = _rand_slice_params(shape)
    # Build PyTorch reference slice
    slices = tuple(slice(b, e, s) for b, e, s in zip(begins, ends, strides))

    # Destination and source (match slice shape)
    torch_out_ref = torch.zeros(shape, dtype=torch.bfloat16)
    torch_src = offset_increment_tensor(torch_out_ref[slices].shape, dtype=torch.bfloat16)

    # PyTorch ground truth
    torch_out_ref[slices] = torch_src

    # TTNN copies
    tt_out = ttnn.from_torch(torch_out_ref * 0, device=device, layout=layout, dtype=ttnn.bfloat16)
    tt_out = ttnn.to_memory_config(tt_out, ttnn.DRAM_MEMORY_CONFIG)

    tt_in = ttnn.from_torch(torch_src, device=device, layout=layout, dtype=ttnn.bfloat16)
    tt_in = ttnn.to_memory_config(tt_in, ttnn.L1_MEMORY_CONFIG)

    # Perform the slice write
    ttnn.slice_write(tt_in, tt_out, begins, ends, strides)

    # Compare full tensors and the written region explicitly
    out_host = ttnn.to_torch(tt_out)
    written_region = out_host[slices]

    assert_with_pcc(written_region, torch_src, 0.9999)
    assert_with_pcc(out_host, torch_out_ref, 0.9999)
