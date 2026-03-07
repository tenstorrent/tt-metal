#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.rvc.torch_impl.vc.hubert import ConvFeatureExtractionModel as TorchConvFeatureExtractionModel
from models.demos.rvc.tt_impl.vc.hubert import ConvFeatureExtractionModel as TTConvFeatureExtractionModel
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("mode", ["layer_norm", "default"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_hubert_conv_feature_extraction_model(device, mode):
    torch.manual_seed(0)

    conv_layers = [(32, 5, 2), (32, 3, 2), (32, 3, 2)]
    conv_bias = True
    batch_size = 1
    input_length = 128

    torch_model = TorchConvFeatureExtractionModel(
        conv_layers=conv_layers,
        mode=mode,
        conv_bias=conv_bias,
    ).eval()
    tt_model = TTConvFeatureExtractionModel(
        device=device,
        conv_layers=conv_layers,
        mode=mode,
        conv_bias=conv_bias,
    )

    parameters = {f"fe.{k}": v for k, v in torch_model.state_dict().items()}
    tt_model.load_parameters(parameters=parameters, prefix="fe.")

    torch_x = torch.randn(batch_size, input_length, dtype=torch.float32)
    torch_output = torch_model(torch_x)

    tt_x = ttnn.from_torch(
        torch_x.to(torch.bfloat16).reshape(batch_size, input_length, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    tt_output = tt_model(tt_x)
    print(f"Output shape from TT model: {tt_output.shape}")
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32).permute(0, 2, 1)
    print(f"Output shape after converting to Torch: {tt_output_torch.shape}")

    assert tuple(tt_output_torch.shape) == tuple(torch_output.shape)
    assert_with_pcc(torch_output, tt_output_torch, pcc=0.95)
