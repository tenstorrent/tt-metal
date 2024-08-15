# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest


import torch


from tt_lib.utils import _nearest_32
from models.utility_functions import comp_pcc

import ttnn

TILE_HEIGHT = TILE_WIDTH = 32


def shape_padded(shape):
    return [shape[0], shape[1], _nearest_32(shape[2]), _nearest_32(shape[3])]


@pytest.mark.parametrize(
    "act_shape",
    (([1, 7, 7, 2048], ([1, 1, 32, 64]))),
    ids=["resnet50_unpadded", "tile_divisible"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16,),
    ids=[
        "BFLOAT16",
    ],
)
def test_run_average_pool(act_shape, dtype, device):
    batch_size, _, _, channels = act_shape

    torch.manual_seed(0)

    act = torch.randn(act_shape, dtype=torch.bfloat16).float()
    ttact = ttnn.Tensor(act, ttnn.bfloat16)
    act_shape_padded = shape_padded(act_shape)
    if act_shape != act_shape_padded:
        ttact = ttact.pad_to_tile(0.0)
    ttact = ttact.to(device)

    out = ttnn.avg_pool2d(ttact)

    out = out.cpu().to(ttnn.ROW_MAJOR_LAYOUT)
    out_shape = [batch_size, 1, 1, channels]
    out_shape_padded = shape_padded(out_shape)
    if out_shape != out_shape_padded:
        out = out.unpad_from_tile(out_shape)

    out_pytorch = out.to_torch()

    ## reference
    act_channels_first = torch.permute(act, (0, 3, 1, 2))  # Torch operates on channels-first tensors
    golden_pytorch = torch.nn.AdaptiveAvgPool2d((1, 1))(act_channels_first)

    ## test for equivalance
    passing_pcc, output_pcc = comp_pcc(golden_pytorch, out_pytorch)
    print(f"Passing PCC = {passing_pcc}")
    print(f"Output PCC = {output_pcc}")

    assert passing_pcc
