# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn
from diffusers import StableDiffusionPipeline
import ttnn

from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_cross_attention_down_block_2d import (
    cross_attention_down_block_2d,
)
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import (
    skip_for_grayskull,
)
from models.demos.wormhole.stable_diffusion.custom_preprocessing import custom_preprocessor
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_utility_functions import (
    run_ttnn_conv_with_pre_and_post_tensor_formatting,
    pre_process_input,
    post_process_output,
)


@skip_for_grayskull()
@pytest.mark.parametrize("model_name", ["CompVis/stable-diffusion-v1-4"])
@pytest.mark.parametrize(
    "N, C, H, W, index, in_channels",
    [
        (
            2,
            320,
            32,
            32,
            1,
            640,
        ),
        (
            2,
            320,
            16,
            16,
            1,
            640,
        ),
        (
            2,
            640,
            8,
            8,
            2,
            1280,
        ),
    ],
)
def test_cross_attn_down_block_2d_256x256(device, model_name, N, C, H, W, index, in_channels):
    pytest.skip()
    torch.manual_seed(0)

    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
    down_block = pipe.unet.down_blocks[index]
    down_block.eval()
    state_dict = pipe.unet.state_dict()
    config = pipe.unet.config

    parameters = preprocess_model_parameters(
        initialize_model=lambda: down_block,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    model = cross_attention_down_block_2d(device, parameters, None, N, H, W)

    hidden_states_shape = torch.Size([N, C, H, W])
    hidden_states = torch.randn(hidden_states_shape)

    encoder_hidden_states_shape = torch.Size([1, 2, 77, 768])
    encoder_hidden_states = torch.randn(encoder_hidden_states_shape)

    temb_shape = torch.Size([1, 1, 2, 1280])
    temb = torch.randn(temb_shape)

    attention_mask = None
    cross_attention_kwargs = None

    torch_output, torch_list_out = down_block(
        hidden_states,
        temb.squeeze(0).squeeze(0),
        encoder_hidden_states=encoder_hidden_states.squeeze(0),
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
    )

    hidden_states = ttnn.from_torch(hidden_states, ttnn.bfloat16)
    hidden_states = ttnn.to_device(hidden_states, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, ttnn.bfloat8_b)

    encoder_hidden_states = torch.nn.functional.pad(encoder_hidden_states, (0, 0, 0, 19))
    encoder_hidden_states = ttnn.from_torch(encoder_hidden_states, ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    encoder_hidden_states = ttnn.to_device(encoder_hidden_states, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    temb = ttnn.from_torch(temb, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    temb = ttnn.to_device(temb, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    hidden_states = pre_process_input(device, hidden_states)
    ttnn_output, _ = model(
        hidden_states,
        encoder_hidden_states,
        temb,
        in_channels=in_channels,
        out_channels=in_channels,
        attention_mask=None,
        add_downsample=True,
        cross_attention_kwargs={},
        config=config,
    )
    ttnn_output = post_process_output(device, ttnn_output, N, H // 2, W // 2, in_channels)
    ttnn_output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_output, ttnn_output, 0.98)


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("model_name", ["CompVis/stable-diffusion-v1-4"])
@pytest.mark.parametrize(
    "N, C, H, W, index, in_channels",
    [
        (
            2,
            320,
            64,
            64,
            0,
            320,
        ),
        (
            2,
            320,
            32,
            32,
            1,
            640,
        ),
        (
            2,
            640,
            16,
            16,
            2,
            1280,
        ),
    ],
)
def test_cross_attn_down_block_2d_512x512(device, model_name, N, C, H, W, index, in_channels):
    # TODO
    pytest.skip()
    torch.manual_seed(0)

    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
    down_block = pipe.unet.down_blocks[index]
    down_block.eval()
    state_dict = pipe.unet.state_dict()
    config = pipe.unet.config
    reader_patterns_cache = {}

    parameters = preprocess_model_parameters(
        initialize_model=lambda: down_block,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    model = cross_attention_down_block_2d(device, parameters, reader_patterns_cache, N, H, W, compute_kernel_config)

    hidden_states_shape = torch.Size([N, C, H, W])
    hidden_states = torch.randn(hidden_states_shape)
    encoder_hidden_states_shape = torch.Size([1, 2, 77, 768])
    encoder_hidden_states = torch.randn(encoder_hidden_states_shape)

    temb_shape = torch.Size([1, 1, 2, 1280])
    temb = torch.randn(temb_shape)

    attention_mask = None
    cross_attention_kwargs = None

    torch_output, torch_list_out = down_block(
        hidden_states,
        temb.squeeze(0).squeeze(0),
        encoder_hidden_states=encoder_hidden_states.squeeze(0),
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
    )

    hidden_states = ttnn.from_torch(hidden_states, ttnn.bfloat16)
    hidden_states = ttnn.to_device(hidden_states, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, ttnn.bfloat8_b)

    encoder_hidden_states = torch.nn.functional.pad(encoder_hidden_states, (0, 0, 0, 19))
    encoder_hidden_states = ttnn.from_torch(encoder_hidden_states, ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    encoder_hidden_states = ttnn.to_device(encoder_hidden_states, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    temb = temb.permute(2, 0, 1, 3)  # pre-permute temb
    temb = ttnn.from_torch(temb, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    temb = ttnn.to_device(temb, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    hidden_states = pre_process_input(device, hidden_states)
    ttnn_output, _ = model(
        hidden_states,
        encoder_hidden_states,
        temb,
        in_channels=C,
        out_channels=in_channels,
        attention_mask=None,
        add_downsample=True,
        cross_attention_kwargs={},
        config=config,
    )
    ttnn_output = post_process_output(device, ttnn_output, N, H // 2, W // 2, in_channels)
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.96)
