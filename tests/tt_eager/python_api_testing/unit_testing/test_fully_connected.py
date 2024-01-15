# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import tt_lib as ttl

from tt_lib.utils import _nearest_32
from models.utility_functions import comp_pcc
from models.utility_functions import is_wormhole_b0, skip_for_wormhole_b0

TILE_HEIGHT = TILE_WIDTH = 32


def shape_padded(shape):
    return [shape[0], shape[1], _nearest_32(shape[2]), _nearest_32(shape[3])]


@skip_for_wormhole_b0()
@pytest.mark.parametrize(
    "shapes",
    (
        ([1, 1, 1, 2048], [1, 1, 2048, 1024], [1, 1, 1, 1024], [1, 1, 1, 1024]),
        ([1, 1, 32, 64], [1, 1, 64, 1024], [1, 1, 1, 1024], [1, 1, 32, 1024]),
    ),
    ids=["resnet50_unpadded", "tile_divisible"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT16,),
    ids=[
        "BFLOAT16",
    ],
)
@pytest.mark.parametrize("has_bias", [False, True])
def test_run_fully_connected(shapes, dtype, has_bias, device):
    act_shape, weight_shape, bias_shape, out_shape = shapes

    torch.manual_seed(0)
    act = torch.randn(act_shape, dtype=torch.bfloat16).float()
    weights = torch.randn(weight_shape, dtype=torch.bfloat16).float()

    ttact = ttl.tensor.Tensor(torch.flatten(act).tolist(), act_shape, dtype, ttl.tensor.Layout.ROW_MAJOR)
    ttweight = ttl.tensor.Tensor(
        torch.flatten(weights).tolist(),
        weight_shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    act_shape_padded = shape_padded(act_shape)
    weight_shape_padded = shape_padded(weight_shape)
    out_shape_padded = shape_padded(out_shape)

    if act_shape != act_shape_padded:
        ttact = ttact.pad_to_tile(0.0)
    if weight_shape != weight_shape_padded:
        ttweight = ttweight.pad_to_tile(0.0)

    ttact = ttact.to(ttl.tensor.Layout.TILE).to(device)
    ttweight = ttweight.to(ttl.tensor.Layout.TILE).to(device)

    if has_bias:
        ## with bias
        ## NOTE: bias is always unpadded, 1d row major
        bias = torch.randn(bias_shape, dtype=torch.bfloat16).float()
        ttbias = ttl.tensor.Tensor(torch.flatten(bias).tolist(), bias_shape, dtype, ttl.tensor.Layout.ROW_MAJOR).to(
            device
        )
        out = ttl.tensor.fully_connected(ttact, ttweight, ttbias)
    else:
        ## without bias
        bias = None
        out = ttl.tensor.fully_connected(ttact, ttweight)

    out = out.cpu().to(ttl.tensor.Layout.ROW_MAJOR)
    if out_shape != out_shape_padded:
        out = out.unpad_from_tile(out_shape)
    out_pytorch = out.to_torch()

    ## reference
    golden_pytorch = torch.nn.functional.linear(act, torch.transpose(weights, 2, 3)[0][0], bias)

    ## test for equivalance
    passing_pcc, output_pcc = comp_pcc(golden_pytorch, out_pytorch)
    print(f"Passing PCC = {passing_pcc}")
    print(f"Output PCC = {output_pcc}")

    assert passing_pcc
