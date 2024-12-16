# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from diffusers import AutoPipelineForText2Image
from models.demos.stable_diffusion.tt.resnetblock2d import ResnetBlock2D
from models.demos.stable_diffusion.tt.utils import custom_preprocessor
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.demos.stable_diffusion.tt.resnetblock2d_utils import update_params


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, in_channels, input_height, input_width, index1, index2, block_name",
    [
        (1, 320, 128, 128, 0, 0, "down"),
        (1, 320, 128, 128, 0, 1, "down"),
        (1, 320, 64, 64, 1, 0, "down"),  #  0.9790
        (1, 640, 64, 64, 1, 1, "down"),  #  0.982
        (1, 640, 32, 32, 2, 0, "down"),  #  0.95
        (1, 1280, 32, 32, 2, 1, "down"),
        (1, 1280, 32, 32, 0, 0, "mid"),  # 0.9858
        (1, 1280, 32, 32, 1, 0, "mid"),  # 0.9746
        (1, 2560, 32, 32, 0, 0, "up"),  #  0.9509
        (1, 2560, 32, 32, 0, 1, "up"),  #  0.9595
        (1, 1920, 32, 32, 0, 2, "up"),  #  0.7976363194085021
        (1, 1920, 64, 64, 1, 0, "up"),  #  0.9596
        (1, 1280, 64, 64, 1, 1, "up"),  #  0.9555
        (1, 960, 64, 64, 1, 2, "up"),  # 0.9764
        (1, 960, 128, 128, 2, 0, "up"),  #  OOM
        (1, 640, 128, 128, 2, 1, "up"),  #  0.93
        (1, 640, 128, 128, 2, 2, "up"),  # 0.9760
    ],
)
def test_resnet_block_2d_1024x1024(
    device,
    batch_size,
    in_channels,
    input_height,
    input_width,
    index1,
    index2,
    block_name,
    reset_seeds,
):
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo")
    model = pipe.unet
    model.eval()
    config = model.config

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters = update_params(parameters)
    if block_name == "up":
        parameters = parameters.up_blocks[index1].resnets[index2]
        resnet = model.up_blocks[index1].resnets[index2]
    elif block_name == "down":
        parameters = parameters.down_blocks[index1].resnets[index2]
        resnet = model.down_blocks[index1].resnets[index2]
    else:
        parameters = parameters.mid_block.resnets[index1]
        resnet = model.mid_block.resnets[index1]

    temb_channels = 1280

    hidden_states_shape = [batch_size, in_channels, input_height, input_width]
    temb_shape = [1, temb_channels]

    input = torch.randn(hidden_states_shape)
    temb = torch.randn(temb_shape)
    torch_output = resnet(input, temb)

    input = ttnn.from_torch(
        input,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    temb = ttnn.from_torch(temb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    resnet_block = ResnetBlock2D(
        config,
        input,
        temb,
        parameters,
        device,
    )
    resnet_block = ttnn.to_torch(resnet_block)
    assert_with_pcc(torch_output, resnet_block, 0.99)
