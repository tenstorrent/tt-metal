# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from models.experimental.oft.reference.resnet import BasicBlock, ResNetFeatures
from models.experimental.oft.tt.tt_resnet import TTBasicBlock, TTResNetFeatures
from tests.ttnn.utils_for_testing import assert_with_pcc

# from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.oft.tt.model_preprocessing import create_OFT_model_parameters_resnet


@pytest.mark.parametrize(
    "input_shape, layers",
    [
        ((1, 3, 384, 1280), [2, 2, 2, 2]),  # ResNet-18
        # ((2, 3, 128, 128), [2, 2, 2, 2]),  # batch size 2
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 10 * 1024}], indirect=True)
def test_resnetfeatures_forward(device, input_shape, layers):
    torch.manual_seed(0)
    model = ResNetFeatures(BasicBlock, layers)
    torch_tensor = torch.randn(*input_shape)
    # feats8, feats16, feats32 = model.forward(torch_tensor)
    feats8, feats16, feats32 = model.forward(torch_tensor)

    params = create_OFT_model_parameters_resnet(model, torch_tensor, device)

    print("-----------------------------------------")
    # print(f"{params=}")
    print(
        f"Input shape: {torch_tensor.shape}, feats8 shape: {feats8.shape}, feats16 shape: {feats16.shape}, feats32 shape: {feats32.shape}"
    )
    print("-----------------------------------------")
    n, c, h, w = feats16.shape
    feats16 = feats16.permute(0, 2, 3, 1)
    feats16 = feats16.reshape(1, 1, n * h * w, c)

    n, c, h, w = feats32.shape
    feats32 = feats32.permute(0, 2, 3, 1)
    feats32 = feats32.reshape(1, 1, n * h * w, c)

    ttnn_input = torch_tensor.permute(0, 2, 3, 1)
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    tt_module = TTResNetFeatures(
        device,
        params,
        params.conv_args,
        TTBasicBlock,
        layers,
    )
    # ttnn_feats8, ttnn_feats16, ttnn_feats32 = tt_module.forward(device, ttnn_input)
    # ttnn_feats8, ttnn_feats16 = tt_module.forward(device, ttnn_input)
    ttnn_feat16 = tt_module.forward(device, ttnn_input)
    # ttnn_feat8 = ttnn.to_torch(ttnn_feat8)
    ttnn_feat16 = ttnn.to_torch(ttnn_feat16)
    print(f"TTNN feats16 shape: {ttnn_feat16.shape}")
    # message, pcc = assert_with_pcc(ttnn_feat8, feats8, 0.99)
    # print(f"Passing: {message}, PCC: {pcc}")
    message, pcc = assert_with_pcc(ttnn_feat16, feats32, 0.99)
    print(f"Passing: {message}, PCC: {pcc}")
