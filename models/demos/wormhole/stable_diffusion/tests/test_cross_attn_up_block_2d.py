# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from diffusers import StableDiffusionPipeline
from loguru import logger
import ttnn
import pytest
from torch import nn

from models.utility_functions import tt_to_torch_tensor, torch_random
from tests.ttnn.utils_for_testing import assert_with_pcc

from models.utility_functions import (
    skip_for_grayskull,
)
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_cross_attn_upblock_new_conv import (
    cross_attention_upblock2d,
)

from models.demos.wormhole.stable_diffusion.custom_preprocessing import custom_preprocessor

from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_utility_functions import (
    pre_process_input,
    weight_to_bfp8,
    post_process_output,
)


def ttnn_to_torch(input):
    input = ttnn.to_layout(input, ttnn.ROW_MAJOR_LAYOUT)
    input = ttnn.from_device(input)
    input = ttnn.to_torch(input)
    return input


@skip_for_grayskull()
@pytest.mark.skip(reason="#9599: Tests are failing.")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "hidden_states, res_hidden_states_tuple, index, prev_output_channel, in_channels ,out_channels",
    [
        ((2, 1280, 16, 16), ([2, 640, 16, 16], [2, 1280, 16, 16], [2, 1280, 16, 16]), 1, 1280, 640, 1280),
        ((2, 1280, 32, 32), ([2, 320, 32, 32], [2, 640, 32, 32], [2, 640, 32, 32]), 2, 1280, 320, 640),
        ((2, 640, 64, 64), ([2, 320, 64, 64], [2, 320, 64, 64], [2, 320, 64, 64]), 3, 640, 320, 320),
    ],
)
@pytest.mark.parametrize("temb", [[1, 1, 2, 1280]])
@pytest.mark.parametrize("encoder_hidden_states", [[1, 2, 77, 768]])
@pytest.mark.parametrize("cross_attention_kwargs", [None])
@pytest.mark.parametrize("upsample_size", [None])
@pytest.mark.parametrize("attention_mask", [None])
def test_cross_attn_up_block_2d_512x512(
    reset_seeds,
    device,
    res_hidden_states_tuple,
    hidden_states,
    index,
    temb,
    encoder_hidden_states,
    cross_attention_kwargs,
    upsample_size,
    attention_mask,
    prev_output_channel,
    in_channels,
    out_channels,
):
    # TODO
    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    config = unet.config
    state_dict = unet.state_dict()
    unet_upblock = pipe.unet.up_blocks[index]

    parameters = preprocess_model_parameters(
        initialize_model=lambda: unet, custom_preprocessor=custom_preprocessor, device=device
    )
    parameters = parameters.up_blocks[index]
    _, Cout, Hout, Wout = res_hidden_states_tuple[2]

    # synthesize the input
    temb_channels = 1280
    input_shape = hidden_states
    hidden_state = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    temb = torch_random(temb, -0.1, 0.1, dtype=torch.float32)
    res0 = torch_random(res_hidden_states_tuple[0], -0.1, 0.1, dtype=torch.float32)
    res1 = torch_random(res_hidden_states_tuple[1], -0.1, 0.1, dtype=torch.float32)
    res2 = torch_random(res_hidden_states_tuple[2], -0.1, 0.1, dtype=torch.float32)
    res_hidden_states_tuple = (res0, res1, res2)
    encoder_hidden_states = torch_random(encoder_hidden_states, -0.1, 0.1, dtype=torch.float32)
    cross_attention_kwargs = None
    upsample_size = None
    attention_mask = None
    reader_patterns_cache = {}

    # execute pytorch
    torch_output = unet_upblock(
        hidden_states=hidden_state,
        temb=temb.squeeze(0).squeeze(0),
        res_hidden_states_tuple=res_hidden_states_tuple,
        encoder_hidden_states=encoder_hidden_states.squeeze(0),
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
        upsample_size=upsample_size,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    N, _, H, W = input_shape
    model = cross_attention_upblock2d(device, parameters, reader_patterns_cache, N, H, W, compute_kernel_config)

    timestep = (None,)
    class_labels = (None,)
    cross_attention_kwargs = (None,)
    return_dict = True
    num_layers_transformer = 1
    norm_num_groups = 32
    cross_attention_dim = 768
    attention_bias = False
    sample_size = None
    num_vector_embeds = None
    patch_size = None
    activation_fn = "geglu"
    num_embeds_ada_norm = None
    use_linear_projection = False
    only_cross_attention = False
    upcast_attention = False
    norm_type = "layer_norm"
    norm_elementwise_affine = True
    attn_num_head_channels = 8

    hidden_state = ttnn.from_torch(hidden_state, ttnn.bfloat16)
    hidden_state = ttnn.to_layout(hidden_state, ttnn.TILE_LAYOUT)
    hidden_state = ttnn.to_device(hidden_state, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    res0 = ttnn.from_torch(res0, ttnn.bfloat16)
    res0 = ttnn.to_layout(res0, ttnn.TILE_LAYOUT)
    res0 = ttnn.to_device(res0, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    res1 = ttnn.from_torch(res1, ttnn.bfloat16)
    res1 = ttnn.to_layout(res1, ttnn.TILE_LAYOUT)
    res1 = ttnn.to_device(res1, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    res2 = ttnn.from_torch(res2, ttnn.bfloat16)
    res2 = ttnn.to_layout(res2, ttnn.TILE_LAYOUT)
    res2 = ttnn.to_device(res2, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    temb = temb.permute(2, 0, 1, 3)  # pre-permute temb
    temb = ttnn.from_torch(temb, ttnn.bfloat16)
    temb = ttnn.to_layout(temb, ttnn.TILE_LAYOUT)
    temb = ttnn.to_device(temb, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    encoder_hidden_states = torch.nn.functional.pad(encoder_hidden_states, (0, 0, 0, 19))
    ttnn_encoder_hidden_states = ttnn.to_layout(
        ttnn.to_device(ttnn.from_torch(encoder_hidden_states, dtype=ttnn.bfloat16), device), layout=ttnn.TILE_LAYOUT
    )

    add_upsample = True
    if index == 3:
        add_upsample = False
    hidden_state = weight_to_bfp8(pre_process_input(device, hidden_state))
    res_hidden_states_tuple = (
        weight_to_bfp8(pre_process_input(device, res0)),
        weight_to_bfp8(pre_process_input(device, res1)),
        weight_to_bfp8(pre_process_input(device, res2)),
    )
    op = model(
        hidden_state,
        res_hidden_states_tuple,
        in_channels,
        prev_output_channel,
        out_channels,
        temb_channels,
        num_layers=3,
        resnet_eps=1e-6,
        resnet_time_scale_shift="default",
        resnet_act_fn="silu",
        resnet_groups=32,
        resnet_pre_norm=True,
        output_scale_factor=1.0,
        add_upsample=add_upsample,
        temb=temb,
        upsample_size=upsample_size,
        config=config,
        encoder_hidden_states=ttnn_encoder_hidden_states,
        timestep=timestep,
        class_labels=class_labels,
        cross_attention_kwargs=cross_attention_kwargs,
        return_dict=return_dict,
        num_layers_transformer=num_layers_transformer,
        patch_size=patch_size,
        num_embeds_ada_norm=num_embeds_ada_norm,
        use_linear_projection=use_linear_projection,
        norm_type=norm_type,
        only_cross_attention=only_cross_attention,
        upcast_attention=upcast_attention,
        attn_num_head_channels=attn_num_head_channels,
        attention_mask=attention_mask,
        cross_attention_dim=cross_attention_dim,
    )

    op = ttnn_to_torch(op)
    if in_channels == out_channels:
        op = torch.reshape(op, (N, H, W, Cout))
    else:
        op = torch.reshape(op, (N, H * 2, W * 2, Cout))
    op = op.permute(0, 3, 1, 2)

    assert_with_pcc(torch_output, op, 0.92)
