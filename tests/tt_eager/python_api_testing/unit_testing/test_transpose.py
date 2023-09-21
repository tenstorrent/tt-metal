# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import torch

import tt_lib as ttl
from models.utility_functions import print_diff_argmax
from tests.tt_eager.python_api_testing.sweep_tests.common import is_wormhole_b0, skip_for_wormhole_b0


def transpose(input_shape, device, dim='hc', expected_program_cache_size=None):
    N = input_shape[0]
    C = input_shape[1]
    H = input_shape[2]
    W = input_shape[3]

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
    assert(dim =='hc' or dim =='cn' or dim=='wh')
    if(dim == 'hc'):
        xtt = ttl.tensor.transpose_hc(xt)
        assert xtt.shape() == [N, H, C, W]
        transposed_ref = x.permute(0, 2, 1, 3)
    elif (dim == 'cn'):
        xtt = ttl.tensor.transpose_cn(xt)
        assert xtt.shape() == [C, N, H, W]
        transposed_ref = x.permute(1, 0, 2, 3)
    elif (dim == 'wh'):
        xtt = ttl.tensor.transpose(xt)
        assert xtt.shape() == [N, C, W, H]
        transposed_ref = x.permute(0,1,3,2)


    tt_got_back = xtt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    assert torch.equal(tt_got_back, transposed_ref)

    if(expected_program_cache_size != None):
        assert ttl.program_cache.num_entries() == expected_program_cache_size

@skip_for_wormhole_b0
def test_transpose_hc(device):
    N = 3
    C = 32 * 2
    H = 32 * 4
    W = 32 * 3
    input_shape = (N, C, H, W)
    transpose(input_shape, device, dim='hc')


def test_transpose_hc_program_cache(device, use_program_cache):
    N = 3
    C = 32 * 2
    H = 32 * 4
    W = 32 * 3
    input_shape = (N, C, H, W)
    transpose(input_shape, device, dim='hc', expected_program_cache_size=1)

    #changing shape
    N = 1
    C = C * 2
    H =  H * 3
    W = W
    input_shape = (N, C, H, W)
    transpose(input_shape, device, dim='hc', expected_program_cache_size=1)


    #changing shape, single core
    N = 1
    C = 1
    H =  32
    W = 32
    input_shape = (N, C, H, W)
    # CACHE MISS since its single core
    # Cache size 2 more because of pad op in single core impl + transpose
    transpose(input_shape, device, dim='hc', expected_program_cache_size=3)

def test_transpose_cn_program_cache(device, use_program_cache):
    N = 3
    C = 32 * 2
    H = 32 * 4
    W = 32 * 3
    input_shape = (N, C, H, W)
    transpose(input_shape, device, dim='cn', expected_program_cache_size=1)

    N = 1
    C = 32
    H = 32 * 4
    W = 32 * 3
    input_shape = (N, C, H, W)
    transpose(input_shape, device, dim='cn', expected_program_cache_size=1)


def test_transpose_wh_program_cache(device, use_program_cache):
    N = 3
    C = 32 * 2
    H = 32 * 4
    W = 32 * 3
    input_shape = (N, C, H, W)
    transpose(input_shape, device, dim='wh', expected_program_cache_size=1)

    #changing shape
    N = 1
    C = C * 2
    H =  H * 3
    W = W
    input_shape = (N, C, H, W)
    transpose(input_shape, device, dim='wh', expected_program_cache_size=1)

    #changing shape, single core
    N = 1
    C = 1
    H =  32
    W = 32
    input_shape = (N, C, H, W)
    # CACHE MISS since its single core
    transpose(input_shape, device, dim='wh', expected_program_cache_size=2)
