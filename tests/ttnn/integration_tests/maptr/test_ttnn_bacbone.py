# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.maptr.reference import resnet
from models.experimental.maptr.ttnn import ttnn_resnet
from models.experimental.maptr.ttnn.model_preprocessing import (
    create_maptr_model_parameters,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_maptr_backbone(
    device,
    reset_seeds,
    use_program_cache,
):
    weights_path = "models/experimental/maptr/maptr_weights_sd.pth"
    torch_model = resnet.ResNet(
        depth=18, num_stages=4, out_indices=(3,), frozen_stages=-1, norm_eval=False, style="pytorch"
    )
    torch_dict = torch.load(weights_path)
    new_state_dict = dict(zip(torch_model.state_dict().keys(), torch_dict.values()))
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input = torch.randn((6, 3, 192, 320), dtype=torch.bfloat16)
    torch_input = torch_input.float()

    torch_output = torch_model(torch_input)

    ttnn_input_tensor = torch.permute(torch_input, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn_input_tensor.reshape(
        1,
        1,
        (ttnn_input_tensor.shape[0] * ttnn_input_tensor.shape[1] * ttnn_input_tensor.shape[2]),
        ttnn_input_tensor.shape[3],
    )

    ttnn_input_tensor = ttnn.from_torch(ttnn_input_tensor, device=device)

    parameter = create_maptr_model_parameters(torch_model, torch_input)

    ttnn_model = ttnn_resnet.TtnnResnet18(parameter.conv_args, parameter.res_model, device)

    ttnn_output = ttnn_model(ttnn_input_tensor, batch_size=6)

    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.reshape(
        torch_output[0].shape[0], torch_output[0].shape[2], torch_output[0].shape[3], torch_output[0].shape[1]
    )
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)

    pcc_passed, pcc_message = assert_with_pcc(ttnn_output, torch_output[0], 0.997)
    logger.info(pcc_message)
