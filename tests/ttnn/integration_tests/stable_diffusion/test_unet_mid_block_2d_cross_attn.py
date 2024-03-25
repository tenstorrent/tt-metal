# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from torch import nn
from diffusers import StableDiffusionPipeline

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import (
    skip_for_grayskull,
)
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.functional_stable_diffusion.custom_preprocessing import custom_preprocessor
from models.experimental.functional_stable_diffusion.tt.ttnn_functional_unet_mid_block_2d_cross_attn import (
    unet_mid_block_2d_cross_attn as ttnn_unet_mid_block_2d_cross_attn,
)
from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_unet_mid_block_2d_cross_attn import (
    unet_mid_block_2d_cross_attn as tt2_ttnn_unet_mid_block_2d_cross_attn,
)
from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_utility_functions import (
    pre_process_input,
    post_process_output,
)


@skip_for_grayskull()
@pytest.mark.parametrize(
    "hidden_state_shapes,",
    [
        (
            2,
            1280,
            4,
            4,
        ),
    ],
)
@pytest.mark.parametrize("model_name", ["CompVis/stable-diffusion-v1-4"])
def test_unet_mid_block_2d_cross_attn_256x256(device, model_name, hidden_state_shapes, reset_seeds):
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    config = unet.config
    mid_block = pipe.unet.mid_block

    num_layers = 1
    resnet_eps = 1e-05
    resnet_time_scale_shift = "default"
    resnet_act_fn = "silu"
    resnet_groups = 32
    resnet_pre_norm = True
    attn_num_head_channels = 8
    output_scale_factor = 1
    cross_attention_dim = 768
    dual_cross_attention = False
    use_linear_projection = False
    upcast_attention = False

    temb_shape = (1, 1, 2, 1280)
    encoder_hidden_states_shape = (1, 2, 77, 768)
    attention_mask = None
    cross_attention_kwargs = None

    _, in_channels, _, _ = hidden_state_shapes
    _, _, _, temb_channels = temb_shape

    hidden_states = torch.randn(hidden_state_shapes)
    temb = torch.randn(temb_shape)
    encoder_hidden_states = torch.randn(encoder_hidden_states_shape)

    torch_output = mid_block(
        hidden_states,
        temb.squeeze(0).squeeze(0),
        encoder_hidden_states=encoder_hidden_states.squeeze(0),
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: unet, custom_preprocessor=custom_preprocessor, device=device
    )
    parameters = parameters.mid_block

    ttnn_temb = ttnn.from_torch(temb, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    ttnn_hidden_state = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    ttnn_encoder_hidden_states = ttnn.from_torch(
        encoder_hidden_states, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
    )

    ttnn_mid_block = ttnn_unet_mid_block_2d_cross_attn(
        hidden_states=ttnn_hidden_state,
        temb=ttnn_temb,
        encoder_hidden_states=ttnn_encoder_hidden_states,
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
        parameters=parameters,
        config=config,
        in_channels=in_channels,
        temb_channels=temb_channels,
        device=device,
        resnet_eps=resnet_eps,
        resnet_time_scale_shift=resnet_time_scale_shift,
        resnet_act_fn=resnet_act_fn,
        resnet_groups=resnet_groups,
        resnet_pre_norm=resnet_pre_norm,
        attn_num_head_channels=attn_num_head_channels,
        output_scale_factor=output_scale_factor,
        dual_cross_attention=dual_cross_attention,
        use_linear_projection=use_linear_projection,
        upcast_attention=upcast_attention,
        cross_attention_dim=cross_attention_dim,
    )

    ttnn_output_torch = ttnn.to_torch(ttnn_mid_block)
    assert_with_pcc(torch_output, ttnn_output_torch, 0.99)


@skip_for_grayskull()
@pytest.mark.parametrize(
    "hidden_state_shapes,",
    [
        (
            2,
            1280,
            8,
            8,
        ),
    ],
)
@pytest.mark.parametrize("model_name", ["CompVis/stable-diffusion-v1-4"])
def test_unet_mid_block_2d_cross_attn_512x512(device, model_name, hidden_state_shapes, reset_seeds):
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    config = unet.config
    mid_block = pipe.unet.mid_block

    num_layers = 1
    resnet_eps = 1e-05
    resnet_time_scale_shift = "default"
    resnet_act_fn = "silu"
    resnet_groups = 32
    resnet_pre_norm = True
    attn_num_head_channels = 8
    output_scale_factor = 1
    cross_attention_dim = 768
    dual_cross_attention = False
    use_linear_projection = False
    upcast_attention = False
    reader_patterns_cache = {}

    temb_shape = (1, 1, 2, 1280)
    encoder_hidden_states_shape = (1, 2, 77, 768)
    attention_mask = None
    cross_attention_kwargs = None

    N, in_channels, H, W = hidden_state_shapes
    _, _, _, temb_channels = temb_shape

    hidden_states = torch.randn(hidden_state_shapes)
    temb = torch.randn(temb_shape)
    encoder_hidden_states = torch.rand(encoder_hidden_states_shape)

    torch_output = mid_block(
        hidden_states,
        temb.squeeze(0).squeeze(0),
        encoder_hidden_states=encoder_hidden_states.squeeze(0),
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: unet, custom_preprocessor=custom_preprocessor, device=device
    )
    parameters = parameters.mid_block
    model = tt2_ttnn_unet_mid_block_2d_cross_attn(device, parameters, reader_patterns_cache, N, H, W)

    # ttnn_temb = ttnn.from_torch(temb, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    # ttnn_hidden_state = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    # ttnn_encoder_hidden_states = ttnn.from_torch(
    #    encoder_hidden_states, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
    # )

    ttnn_hidden_state = ttnn.from_torch(hidden_states, ttnn.bfloat16)
    ttnn_hidden_state = ttnn.to_device(ttnn_hidden_state, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn_hidden_state = ttnn.to_layout(ttnn_hidden_state, ttnn.TILE_LAYOUT, ttnn.bfloat8_b)

    ttnn_encoder_hidden_states = torch.nn.functional.pad(encoder_hidden_states, (0, 0, 0, 19))
    ttnn_encoder_hidden_states = ttnn.from_torch(ttnn_encoder_hidden_states, ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    ttnn_encoder_hidden_states = ttnn.to_device(ttnn_encoder_hidden_states, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    ttnn_temb = ttnn.from_torch(temb, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_temb = ttnn.to_device(ttnn_temb, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    ttnn_hidden_state = pre_process_input(device, ttnn_hidden_state)
    ttnn_mid_block = model(
        hidden_states=ttnn_hidden_state,
        temb=ttnn_temb,
        encoder_hidden_states=ttnn_encoder_hidden_states,
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
        config=config,
        in_channels=in_channels,
        temb_channels=temb_channels,
        resnet_eps=resnet_eps,
        resnet_time_scale_shift=resnet_time_scale_shift,
        resnet_act_fn=resnet_act_fn,
        resnet_groups=resnet_groups,
        resnet_pre_norm=resnet_pre_norm,
        attn_num_head_channels=attn_num_head_channels,
        output_scale_factor=output_scale_factor,
        dual_cross_attention=dual_cross_attention,
        use_linear_projection=use_linear_projection,
        upcast_attention=upcast_attention,
        cross_attention_dim=cross_attention_dim,
    )

    ttnn_output = post_process_output(device, ttnn_mid_block, N, H, W, in_channels)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_output, ttnn_output_torch, 0.97)
