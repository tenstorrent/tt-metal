# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from models.experimental.oft.reference.resnet import BasicBlock
from models.experimental.oft.tt.tt_resnet import TTBasicBlock
from tests.ttnn.utils_for_testing import assert_with_pcc

# from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.oft.tt.model_preprocessing import create_OFT_model_parameters_resnet


@pytest.mark.parametrize(
    "n, in_ch, out_ch, h, w, stride, sharding",
    [
        # (1, 128, 128, 80, 64, 1),
        # (1, 128, 128, 48, 160, 1),
        # (1, 128, 256, 64, 80, 2),
        # (1, 128, 256, 48, 160, 2),
        # (1, 64, 64, 96, 320, 1, "HS"),
        (1, 512, 256, 12, 40, 1, "BS"),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8 * 1024}], indirect=True)
def test_tt_basicblock(device, n, in_ch, out_ch, h, w, stride, sharding):
    torch.manual_seed(42)
    input_tensor = torch.randn(n, in_ch, h, w)
    torch_model = BasicBlock(inplanes=in_ch, planes=out_ch, stride=stride)
    out = torch_model.forward(input_tensor)

    params = create_OFT_model_parameters_resnet(torch_model, input_tensor, device)

    print("-----------------------------------------")
    print(f"{params=}")
    print("-----------------------------------------")

    block = TTBasicBlock(
        device,
        params,
        params.conv_args,
        inplanes=in_ch,
        planes=out_ch,
        channels=out_ch,
        cell_size=h,
        grid_height=w,
        stride=stride,
    )

    n, c, h, w = input_tensor.shape
    x_for_ttnn = input_tensor.permute(0, 2, 3, 1).view(1, 1, n * h * w, c)
    ttnn_x = ttnn.from_torch(x_for_ttnn, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_out = block.forward(device, ttnn_x, gn_shard=sharding)

    print(f"Output shape: {ttnn_out.shape}, torch out {out.shape}")
    B, C, H, W = out.shape
    ttnn_out = ttnn.to_torch(ttnn_out)

    out = out.permute(0, 2, 3, 1).reshape(1, 1, B * H * W, C)
    pcc, message = assert_with_pcc(ttnn_out, out, 0.99)
    print(f"PCC: {pcc}, Message: {message}")
