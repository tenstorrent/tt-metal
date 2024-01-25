# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
import math
from models.utility_functions import skip_for_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc
import ttnn


@skip_for_wormhole_b0()
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
def test_run_average_pool(
    act_shape,
    dtype,
    device,
):
    torch.manual_seed(0)

    act = torch.randn(act_shape, dtype=torch.bfloat16)
    ttact = ttnn.from_torch(act, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    out = ttnn.average_pool2d(ttact)

    out_pytorch = ttnn.to_torch(ttnn.from_device(out))

    ## reference
    act_channels_first = torch.permute(act, (0, 3, 1, 2))  # Torch operates on channels-first tensors
    golden_pytorch = torch.nn.AdaptiveAvgPool2d((1, 1))(act_channels_first)
    golden_pytorch = torch.permute(golden_pytorch, (0, 2, 3, 1))

    ## test for equivalance
    assert_with_pcc(golden_pytorch, out_pytorch)
