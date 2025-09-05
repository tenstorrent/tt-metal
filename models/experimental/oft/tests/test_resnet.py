# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from models.experimental.oft.reference.resnet import resnet18
from models.experimental.oft.tt.tt_resnet import TTBasicBlock, TTResNetFeatures
from tests.ttnn.utils_for_testing import assert_with_pcc

# from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.oft.tt.model_preprocessing import create_OFT_model_parameters_resnet


@pytest.mark.parametrize(
    "input_shape, layers",
    [
        ((1, 3, 384, 1280), [2, 2, 2, 2]),  # ResNet-18
        # ((2, 3, 128, 128), [2, 2, 2, 2]),  # batch size2
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 10 * 1024}], indirect=True)
def test_resnetfeatures_forward(device, input_shape, layers):
    torch.manual_seed(0)
    model = resnet18(pretrained=False)
    torch_tensor = torch.randn(*input_shape)

    params = create_OFT_model_parameters_resnet(model, torch_tensor, device)
    feats8, feats16, feats32 = model.forward(torch_tensor)

    n, c, h, w = feats8.shape
    feats8 = feats8.permute(0, 2, 3, 1)
    feats8 = feats8.reshape(1, 1, n * h * w, c)
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
    ttnn_feats8, ttnn_feats16, ttnn_feats32 = tt_module.forward(device, ttnn_input)
    print("-----------------------------------------")
    print(
        f"TTNN feats8 shape: {ttnn_feats8.shape} dtype: {ttnn_feats8.dtype}, layout: {ttnn_feats8.layout}, memory_config: {ttnn_feats8.memory_config()}"
    )
    print(
        f"TTNN feats16 shape: {ttnn_feats16.shape} dtype: {ttnn_feats16.dtype},layout: {ttnn_feats16.layout}, memory_config: {ttnn_feats16.memory_config()}"
    )
    print(
        f"TTNN feats32 shape: {ttnn_feats32.shape} dtype: {ttnn_feats32.dtype},layout: {ttnn_feats32.layout}, memory_config: {ttnn_feats32.memory_config()}"
    )
    print("------------------------------------------")

    ttnn_feats8 = ttnn.to_torch(ttnn_feats8)
    ttnn_feats16 = ttnn.to_torch(ttnn_feats16)
    print(f"TTNN feats16 shape: {ttnn_feats16.shape}")
    ttnn_feats32 = ttnn.to_torch(ttnn_feats32)

    message, pcc = assert_with_pcc(ttnn_feats8, feats8, 0.99)
    print(f"Passing: {message}, PCC: {pcc}")
    message, pcc = assert_with_pcc(ttnn_feats16, feats16, 0.99)
    print(f"Passing: {message}, PCC: {pcc}")
    message, pcc = assert_with_pcc(ttnn_feats32, feats32, 0.99)
    print(f"Passing: {message}, PCC: {pcc}")
