# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import tt_lib as ttl
from models.utility_functions import print_diff_argmax
from models.utility_functions import is_wormhole_b0, skip_for_wormhole_b0


def transpose(input_shape, device, dim0, dim1, expected_program_cache_size=None):
    output_shape = list(input_shape)
    output_shape[dim0], output_shape[dim1] = input_shape[dim1], input_shape[dim0]

    x = torch.randn(input_shape).bfloat16().float()

    xt = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )
    xtt = ttl.tensor.transpose(xt, dim0, dim1)
    assert xtt.shape() == output_shape
    transposed_ref = x.transpose(dim0, dim1)

    tt_got_back = xtt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    eq = torch.equal(tt_got_back, transposed_ref)
    assert eq

    if expected_program_cache_size != None:
        assert ttl.program_cache.num_entries() == expected_program_cache_size


@skip_for_wormhole_b0()
def test_transpose_hc(device):
    N = 3
    C = 32 * 2
    H = 32 * 4
    W = 32 * 3
    input_shape = (N, C, H, W)
    transpose(input_shape, device, dim0=1, dim1=-2)


def test_transpose_hc_program_cache(device, use_program_cache):
    N = 3
    C = 32 * 2
    H = 32 * 4
    W = 32 * 3
    input_shape = (N, C, H, W)
    transpose(input_shape, device, dim0=1, dim1=-2, expected_program_cache_size=1)

    # changing shape
    N = 1
    C = C * 2
    H = H * 3
    W = W
    input_shape = (N, C, H, W)
    transpose(input_shape, device, dim0=1, dim1=-2, expected_program_cache_size=1)

    # changing shape, single core
    N = 1
    C = 1
    H = 32
    W = 32
    input_shape = (N, C, H, W)
    # CACHE MISS since its single core
    # Cache size 2 more because of pad op in single core impl + transpose
    transpose(input_shape, device, dim0=1, dim1=-2, expected_program_cache_size=3)


def test_transpose_cn_program_cache(device, use_program_cache):
    N = 3
    C = 32 * 2
    H = 32 * 4
    W = 32 * 3
    input_shape = (N, C, H, W)
    transpose(input_shape, device, dim0=0, dim1=1, expected_program_cache_size=1)

    N = 1
    C = 32
    H = 32 * 4
    W = 32 * 3
    input_shape = (N, C, H, W)
    transpose(input_shape, device, dim0=0, dim1=1, expected_program_cache_size=1)


def test_transpose_wh_program_cache(device, use_program_cache):
    N = 3
    C = 32 * 2
    H = 32 * 4
    W = 32 * 3
    input_shape = (N, C, H, W)
    transpose(input_shape, device, dim0=-2, dim1=-1, expected_program_cache_size=1)

    # changing shape
    N = 1
    C = C * 2
    H = H * 3
    W = W
    input_shape = (N, C, H, W)
    transpose(input_shape, device, dim0=-2, dim1=-1, expected_program_cache_size=1)

    # changing shape, single core
    N = 1
    C = 1
    H = 32
    W = 32
    input_shape = (N, C, H, W)
    # CACHE MISS since its single core
    transpose(input_shape, device, dim0=-2, dim1=-1, expected_program_cache_size=2)
