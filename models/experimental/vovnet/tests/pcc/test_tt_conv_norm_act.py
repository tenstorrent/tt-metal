# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from models.experimental.vovnet.tt.conv_norm_act import TtConvNormAct
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.vovnet.common import load_torch_model, VOVNET_L1_SMALL_SIZE
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.vovnet.tt.model_preprocessing import create_custom_mesh_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": VOVNET_L1_SMALL_SIZE}], indirect=True)
def test_vovnet_conv_norm_act_inference(device, reset_seeds, model_location_generator):
    base_address = f"stem.0"  # f"stages.0.blocks.0.conv_reduction"
    model = load_torch_model(model_location_generator)
    model.eval()
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=create_custom_mesh_preprocessor(mesh_mapper=None),
        device=device,
    )

    torch_model = model.stem[0]  # stages[0].blocks[0].conv_reduction
    tt_model = TtConvNormAct(
        stride=2,  # 1
        padding=1,  # 0
        base_address=base_address,
        device=device,
        parameters=parameters,
    )

    input = torch.rand(1, 3, 224, 224)  # (1, 64, 56, 56)
    model_output = torch_model(input)

    input = input.permute(0, 2, 3, 1)
    input = input.reshape(1, 1, input.shape[0] * input.shape[1] * input.shape[2], input.shape[-1])
    tt_input = ttnn.from_torch(input, device=device, dtype=ttnn.bfloat16)
    tt_output = tt_model.forward(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output[0])

    tt_output_torch = tt_output_torch.reshape(model_output.permute(0, 2, 3, 1).shape)
    tt_output_torch = tt_output_torch.permute(0, 3, 1, 2)
    assert_with_pcc(model_output, tt_output_torch, 0.99)
