import sys
import pytest
import itertools

from pathlib import Path

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import numpy as np
import torch

from libs import tt_lib as ttl
from python_api_testing.models.utility_functions import (
    tilize_to_list,
    tilize,
    untilize,
    comp_pcc,
)

TILE_HEIGHT = TILE_WIDTH = 32

## parameters
# matrix sizes as number of blocks along h and w:
a_height_nblocks = [1, 7]
a_width_nblocks = [1, 7]
b_width_nblocks = [1, 7]
# block sizes as number of tiles along h and w:
a_block_height_ntiles = [4]
a_block_width_ntiles = [4]
b_block_width_ntiles = [16]
# output sublobcking per block:
out_subblock_height_ntiles = [4] ## == a_block_height_ntiles, <= 8
out_subblock_width_ntiles = [2]  ## == b_block_width_ntiles, <= 8
tilize_a = [True, False]
untilize_out = [True, False]


@pytest.mark.parametrize(
    'a_height_nblocks, a_width_nblocks, b_width_nblocks,\
     a_block_height_ntiles, a_block_width_ntiles, b_block_width_ntiles,\
     out_subblock_height_ntiles, out_subblock_width_ntiles,\
     tilize_a, untilize_out',
    itertools.product(a_height_nblocks, a_width_nblocks, b_width_nblocks,
                      a_block_height_ntiles, a_block_width_ntiles, b_block_width_ntiles,
                      out_subblock_height_ntiles, out_subblock_width_ntiles,
                      tilize_a, untilize_out)
)
def test_run_bmm_single_core_tilize_untilize(a_height_nblocks,
                                             a_width_nblocks,
                                             b_width_nblocks,
                                             a_block_height_ntiles,
                                             a_block_width_ntiles,
                                             b_block_width_ntiles,
                                             out_subblock_height_ntiles,
                                             out_subblock_width_ntiles,
                                             tilize_a,
                                             untilize_out):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    a_batch = b_batch = 1
    a_channel = b_channel = 1
    a_height = a_height_nblocks * a_block_height_ntiles * TILE_HEIGHT
    a_width = a_width_nblocks * a_block_width_ntiles * TILE_WIDTH   # == b_height
    b_width = b_width_nblocks * b_block_width_ntiles * TILE_WIDTH
    a_shape = [a_batch, a_channel, a_height, a_width]
    b_shape = [b_batch, b_channel, a_width, b_width]
    out_shape = [a_batch, a_channel, a_height, b_width]

    torch.manual_seed(0)
    a = torch.randn(a_shape, dtype=torch.bfloat16).float()
    b = torch.randn(b_shape, dtype=torch.bfloat16).float()

    if tilize_a:
        ## a in row-major
        a_layout = ttl.tensor.Layout.ROW_MAJOR
        a_list = a.flatten().tolist()
    else:
        ## a in tile
        a_layout = ttl.tensor.Layout.TILE
        a_list = tilize_to_list(a)
    tta = ttl.tensor.Tensor(
        a_list,
        a_shape,
        ttl.tensor.DataType.BFLOAT16,
        a_layout,
        device)
    ## b in tile
    b_list = tilize_to_list(b)
    ttb = ttl.tensor.Tensor(
        b_list,
        b_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        device)

    torch.set_printoptions(
       precision=2, threshold=10000,
       sci_mode=False, edgeitems=80, linewidth=400)

    # tta_pytorch = untilize(torch.tensor(tta.to(host).data()).reshape(a_shape))
    # print("a slice:\n", tta_pytorch[0, 0, 0:32*a_block_height_ntiles*a_height_nblocks:32*a_block_height_ntiles, 0:32*a_width_nblocks*a_block_width_ntiles:1])

    # ttb_pytorch = untilize(torch.tensor(ttb.to(host).data()).reshape(b_shape))
    # print("b slice:\n", ttb_pytorch[0, 0, 0:32*a_block_width_ntiles*a_width_nblocks:32, 0:32*b_width_nblocks*b_block_width_ntiles:1])

    ## compute out
    out = ttl.tensor.bmm_tilize_untilize(tta, ttb,
                                         a_height_nblocks, a_width_nblocks, b_width_nblocks,
                                         a_block_height_ntiles, a_block_width_ntiles, b_block_width_ntiles,
                                         out_subblock_height_ntiles, out_subblock_width_ntiles,
                                         tilize_a, untilize_out)
    out = out.to(host)
    if not untilize_out:
        ## output is in tiled format
        out_pytorch = untilize(torch.tensor(out.data()).reshape(out_shape))
    else:
        out_pytorch = torch.tensor(out.data()).reshape(out_shape)

    # print("out slice:\n", out_pytorch)

    ttl.device.CloseDevice(device)

    ## reference
    golden_pytorch = torch.matmul(a, b)

    # print("golden out slice:\n", golden_pytorch)

    ## test for equivalance
    assert(out_pytorch.shape == golden_pytorch.shape)
    passing_pcc, output_pcc = comp_pcc(golden_pytorch, out_pytorch)
    print(f'Passing PCC = {passing_pcc}')
    print(f'Output PCC = {output_pcc}')

    assert(passing_pcc)
