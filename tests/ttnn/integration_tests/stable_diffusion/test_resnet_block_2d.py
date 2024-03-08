# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from diffusers import StableDiffusionPipeline
import pytest

from tests.ttnn.utils_for_testing import assert_with_pcc

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_resnetblock2d import resnetBlock2D
from models.experimental.functional_stable_diffusion.custom_preprocessing import custom_preprocessor

from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_utility_functions import (
    pre_process_input,
    post_process_output,
)


def ttnn_to_torch(input):
    input = ttnn.to_layout(input, ttnn.ROW_MAJOR_LAYOUT)
    input = ttnn.from_device(input)
    input = ttnn.to_torch(input)
    return input


@pytest.mark.parametrize(
    "batch_size, in_channels, input_height, input_width, index1,index2,block_name,out_channels",
    [
        (2, 320, 32, 32, 0, 0, "down", None),
        (2, 320, 16, 16, 0, 0, "down", None),
        (2, 640, 16, 16, 1, 1, "down", None),
        (2, 640, 8, 8, 1, 1, "down", None),
        (2, 1280, 8, 8, 2, 1, "down", None),
        (2, 1280, 4, 4, 2, 1, "down", None),
        (2, 2560, 4, 4, 0, 0, "up", 1280),
        (2, 2560, 8, 8, 0, 0, "up", 1280),
        (2, 1920, 8, 8, 2, 0, "up", 640),
        (2, 1920, 16, 16, 2, 0, "up", 640),
        (2, 1280, 16, 16, 3, 0, "down", None),
        (2, 960, 16, 16, 3, 0, "up", 320),
        (2, 960, 32, 32, 3, 0, "up", 320),
        (2, 640, 32, 32, 3, 1, "up", 320),
    ],
)
def test_resnet_block_2d_256x256(
    device, batch_size, in_channels, input_height, input_width, index1, index2, block_name, out_channels
):
    # setup pytorch model
    model_name = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)

    model = pipe.unet
    model.eval()

    parameters = preprocess_model_parameters(
        model_name=model_name, initialize_model=lambda: model, custom_preprocessor=custom_preprocessor, device=device
    )

    if block_name == "up":
        parameters = parameters.up_blocks[index1].resnets[index2]
        resnet = pipe.unet.up_blocks[index1].resnets[index2]
    elif block_name == "down":
        parameters = parameters.down_blocks[index1].resnets[index2]
        resnet = pipe.unet.down_blocks[index1].resnets[index2]
    else:
        parameters = parameters.mid_block.resnets[index2]
        resnet = pipe.unet.mid_block.resnets[index2]

    ############ start of residual block #############
    temb_channels = 1280
    groups = 32
    time_embedding_norm = "default"
    output_scale_factor = 1
    use_in_shortcut = None
    ########## end of residual block #############
    hidden_states_shape = [batch_size, in_channels, input_height, input_width]
    temb_shape = [1, 1, 2, 1280]

    input = torch.randn(hidden_states_shape)
    temb = torch.randn(temb_shape)

    torch_output = resnet(input, temb.squeeze(0).squeeze(0))

    input = ttnn.from_torch(input, ttnn.bfloat16)
    input = ttnn.to_layout(input, ttnn.TILE_LAYOUT)
    input = ttnn.to_device(input, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    temb = ttnn.from_torch(temb, ttnn.bfloat16)
    temb = ttnn.to_layout(temb, ttnn.TILE_LAYOUT)
    temb = ttnn.to_device(temb, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn_output = resnetBlock2D(
        input,
        temb=temb,
        temb_channels=temb_channels,
        time_embedding_norm=time_embedding_norm,
        in_channels=in_channels,
        out_channels=out_channels,
        use_in_shortcut=use_in_shortcut,
        groups=groups,
        output_scale_factor=output_scale_factor,
        parameters=parameters,
        device=device,
    )
    ttnn_output = ttnn_to_torch(ttnn_output)
    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)


@pytest.mark.parametrize(
    "batch_size, in_channels, input_height, input_width, index1,index2,block_name,out_channels",
    [
        (2, 320, 64, 64, 0, 0, "down", None),
        (2, 320, 32, 32, 0, 0, "down", None),
        (2, 640, 32, 32, 1, 1, "down", None),
        (2, 640, 16, 16, 1, 1, "down", None),
        (2, 1280, 16, 16, 2, 1, "down", None),
        (2, 1280, 8, 8, 2, 1, "down", None),
        (2, 2560, 8, 8, 0, 0, "up", 1280),
        (2, 2560, 16, 16, 0, 0, "up", 1280),
        (2, 1920, 16, 16, 2, 0, "up", 1280),
        (2, 1920, 32, 32, 2, 0, "up", 640),
        (2, 1280, 32, 32, 3, 0, "down", None),
        (2, 960, 32, 32, 3, 0, "up", 640),
        (2, 960, 64, 64, 3, 0, "up", 320),
        (2, 640, 64, 64, 3, 1, "up", 320),
    ],
)
def test_resnet_block_2d_512x512(
    device, batch_size, in_channels, input_height, input_width, index1, index2, block_name, out_channels
):
    load_from_disk = False
    if not load_from_disk:
        # setup pytorch model
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)

        model = pipe.unet
        model.eval()
        config = model.config

        parameters = preprocess_model_parameters(
            initialize_model=lambda: model, custom_preprocessor=custom_preprocessor, device=device
        )

        if block_name == "up":
            parameters = parameters.up_blocks[index1].resnets[index2]
            resnet = pipe.unet.up_blocks[index1].resnets[index2]
        elif block_name == "down":
            parameters = parameters.down_blocks[index1].resnets[index2]
            resnet = pipe.unet.down_blocks[index1].resnets[index2]
        else:
            parameters = parameters.mid_block.resnets[index2]
            resnet = pipe.unet.mid_block.resnets[index2]
        torch.save(resnet, "resnet.pt")
        torch.save(config, "config.pt")

    else:
        resnet = torch.load("resnet.pt")
        config = torch.load("config.pt")
        parameters = preprocess_model_parameters(
            initialize_model=lambda: resnet, custom_preprocessor=custom_preprocessor, device=device
        )

    ############ start of residual block #############
    temb_channels = 1280
    groups = 32
    time_embedding_norm = "default"
    output_scale_factor = 1
    use_in_shortcut = None
    ########## end of residual block #############
    hidden_states_shape = [batch_size, in_channels, input_height, input_width]
    temb_shape = [1, 1, 2, 1280]

    input = torch.randn(hidden_states_shape)
    temb = torch.randn(temb_shape)

    torch_output = resnet(input, temb.squeeze(0).squeeze(0))
    reader_patterns_cache = {}
    resnet_block = resnetBlock2D(
        device, parameters, reader_patterns_cache, batch_size, input_height, input_width, group_norm_on_device=True
    )

    input = torch.permute(input, (0, 2, 3, 1))
    input = ttnn.from_torch(input, ttnn.bfloat16)
    input = ttnn.reshape(input, (1, 1, batch_size * input_height * input_width, in_channels))
    input = ttnn.to_device(input, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    input = ttnn.to_layout(input, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
    # input = ttnn.to_memory_config(input, resnet_block.conv1s[0].conv.input_sharded_memory_config)

    temb = ttnn.from_torch(temb, ttnn.bfloat16)
    temb = ttnn.to_layout(temb, ttnn.TILE_LAYOUT)
    temb = ttnn.to_device(temb, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    ttnn_output = resnet_block(
        input,
        temb=temb,
        in_channels=in_channels,
        out_channels=out_channels,
        temb_channels=temb_channels,
        use_in_shortcut=use_in_shortcut,
        eps=1e-6,
        groups=groups,
        time_embedding_norm=time_embedding_norm,
        non_linearity="silu",
        output_scale_factor=output_scale_factor,
        # pre_norm=resnet_pre_norm,
        # group_norm_sharded_config={
        #     "shard_strategy": resnet_block.conv1s[0].conv.input_shard_scheme,
        #     "shard_orientation": resnet_block.conv1s[0].conv.input_shard_orientation,
        #     "grid_size": resnet_block.conv1s[0].conv.grid_size,
        # },
    )
    ttnn_output = ttnn.to_memory_config(ttnn_output, ttnn.L1_MEMORY_CONFIG)
    ttnn_output = ttnn_to_torch(ttnn_output)
    ttnn_output = torch.reshape(
        ttnn_output,
        (
            batch_size,
            input_height,
            input_width,
            out_channels if out_channels is not None else in_channels,
        ),
    )
    ttnn_output = torch.permute(ttnn_output, (0, 3, 1, 2))

    assert_with_pcc(torch_output, ttnn_output, pcc=0.98)
