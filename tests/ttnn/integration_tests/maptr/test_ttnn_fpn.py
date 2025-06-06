# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.maptr.reference import fpn
from models.experimental.maptr.ttnn import ttnn_fpn
from models.experimental.maptr.ttnn.model_preprocessing import (
    create_maptr_model_parameters,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_maptr_fpn(
    device,
    reset_seeds,
    use_program_cache,
):
    weights_path = "models/experimental/maptr/maptr_weights_sd.pth"
    torch_model = fpn.FPN(
        in_channels=[512],
        out_channels=256,
        start_level=0,
        add_extra_convs="on_output",
        num_outs=1,
        relu_before_extra_convs=True,
    )
    torch_dict = torch.load(weights_path)
    state_dict = {k: v for k, v in torch_dict.items() if (k.startswith("img_neck."))}
    new_state_dict = dict(zip(torch_model.state_dict().keys(), state_dict.values()))
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    input = torch.randn((6, 512, 6, 10), dtype=torch.bfloat16)
    input = input.float()

    torch_input = []
    torch_input.append(input)

    torch_output = torch_model(torch_input)

    ttnn_tensor = torch.permute(input, (0, 2, 3, 1))
    ttnn_tensor = ttnn_tensor.reshape(
        1,
        1,
        (ttnn_tensor.shape[0] * ttnn_tensor.shape[1] * ttnn_tensor.shape[2]),
        ttnn_tensor.shape[3],
    )

    ttnn_tensor = ttnn.from_torch(ttnn_tensor, device=device)

    ttnn_input_tensor = []
    ttnn_input_tensor.append(ttnn_tensor)

    parameter = create_maptr_model_parameters(torch_model, torch_input)

    ttnn_model = ttnn_fpn.TtnnFPN(parameter.conv_args, parameter.fpn, device)

    ttnn_output = ttnn_model(ttnn_input_tensor)

    ttnn_output = ttnn.to_torch(ttnn_output[0])
    ttnn_output = ttnn_output.reshape(
        torch_output[0].shape[0], torch_output[0].shape[2], torch_output[0].shape[3], torch_output[0].shape[1]
    )
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)

    pcc_passed, pcc_message = assert_with_pcc(ttnn_output, torch_output[0], 0.999)
    logger.info(pcc_message)
