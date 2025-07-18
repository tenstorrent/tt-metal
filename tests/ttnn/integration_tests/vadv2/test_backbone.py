# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.vadv2.reference import resnet as backbone
from models.experimental.vadv2.tt import ttnn_backbone
from models.experimental.vadv2.tt.model_preprocessing import (
    create_vadv2_model_parameters,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_vadv2_backbone(
    device,
    reset_seeds,
):
    weights_path = "models/experimental/vadv2/tt/vadv2_weights_1.pth"
    torch_model = backbone.ResNet(
        layers=[3, 4, 6, 3],
        out_indices=(1, 2, 3),
        block=backbone.Bottleneck,
    )
    torch_dict = torch.load(weights_path)

    state_dict = {k: v for k, v in torch_dict.items() if (k.startswith("img_backbone"))}
    new_state_dict = dict(zip(torch_model.state_dict().keys(), state_dict.values()))

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input = torch.randn((6, 3, 384, 640), dtype=torch.bfloat16)
    torch_input = torch_input.float()

    torch_output = torch_model(torch_input)[0]

    ttnn_input_tensor = torch.permute(torch_input, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn_input_tensor.reshape(
        1,
        1,
        (ttnn_input_tensor.shape[0] * ttnn_input_tensor.shape[1] * ttnn_input_tensor.shape[2]),
        ttnn_input_tensor.shape[3],
    )

    ttnn_input_tensor = ttnn.from_torch(ttnn_input_tensor, device=device, dtype=ttnn.bfloat16)

    parameter = create_vadv2_model_parameters(torch_model, torch_input)

    ttnn_model = ttnn_backbone.TtnnResnet50(parameter.conv_args, parameter.res_model, device)

    ttnn_output = ttnn_model(ttnn_input_tensor, batch_size=6)[0]

    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.reshape(
        torch_output.shape[0], torch_output.shape[2], torch_output.shape[3], torch_output.shape[1]
    ).to(torch.float32)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)

    pcc_passed, pcc_message = assert_with_pcc(ttnn_output, torch_output, 0.997)
    logger.info(pcc_message)
