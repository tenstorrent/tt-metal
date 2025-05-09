# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from diffusers import StableDiffusionPipeline
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.wormhole.stable_diffusion.custom_preprocessing import custom_preprocessor
from models.demos.wormhole.stable_diffusion.tests.parameterizations import DOWN_MID_UP_BLOCKS_HIDDEN_STATES_INFO
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_unet_mid_block_2d_cross_attn_new_conv import (
    unet_mid_block_2d_cross_attn,
)
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_utility_functions import (
    get_default_compute_config,
    post_process_output_and_move_to_host,
    preprocess_and_push_input_to_device,
)
from models.utility_functions import skip_for_grayskull, torch_random
from tests.ttnn.utils_for_testing import assert_with_pcc


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "hidden_states, shard_layout, shard_end_core, shard_shape", (DOWN_MID_UP_BLOCKS_HIDDEN_STATES_INFO,)
)
@pytest.mark.parametrize("temb", [[1, 1, 2, 1280]])
def test_cross_attention_midblock_512x512(
    reset_seeds, device, hidden_states, shard_layout, shard_end_core, shard_shape, temb, use_program_cache
):
    # Initialize PyTorch component
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    torch_midblock = unet.mid_block

    # Initialize ttnn component
    parameters = preprocess_model_parameters(
        initialize_model=lambda: unet, custom_preprocessor=custom_preprocessor, device=device
    )
    parameters = parameters.mid_block
    N, _, H, W = hidden_states
    compute_kernel_config = get_default_compute_config(device)

    ttnn_midblock = unet_mid_block_2d_cross_attn(device, parameters, N, H, W, compute_kernel_config)

    # Prepare inputs
    in_channels = hidden_states[1]
    out_channels = in_channels
    temb_channels = 1280
    input_shape = hidden_states
    hidden_states = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    temb = torch_random(temb, -0.1, 0.1, dtype=torch.float32)

    encoder_hidden_states_shape = [1, 2, 77, 768]
    encoder_hidden_states = torch.randn(encoder_hidden_states_shape)

    # Run PyTorch component
    torch_output = torch_midblock(hidden_states, temb.squeeze(0).squeeze(0), encoder_hidden_states.squeeze(0))

    # Prepare inputs for ttnn component
    hidden_states = preprocess_and_push_input_to_device(
        device,
        hidden_states,
        memory_config=ttnn.MemoryConfig(
            shard_layout,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(shard_end_core[0], shard_end_core[1]),
                        ),
                    }
                ),
                shard_shape,
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )

    temb = temb.permute(2, 0, 1, 3)
    temb = ttnn.from_torch(temb, ttnn.bfloat16)
    temb = ttnn.to_layout(temb, ttnn.TILE_LAYOUT, ttnn.bfloat8_b)
    temb = ttnn.to_device(temb, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    encoder_hidden_states = torch.nn.functional.pad(encoder_hidden_states, (0, 0, 0, 19))
    encoder_hidden_states = ttnn.from_torch(
        encoder_hidden_states, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )
    encoder_hidden_states = ttnn.to_device(encoder_hidden_states, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Run ttnn component
    output = ttnn_midblock(
        hidden_states=hidden_states,
        temb=temb,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=None,
        cross_attention_kwargs=None,
        in_channels=in_channels,
        temb_channels=temb_channels,
        resnet_eps=1e-5,
        resnet_act_fn="silu",
        attn_num_head_channels=8,
        config=unet.config,
    )

    # Compare outputs
    output = post_process_output_and_move_to_host(output, N, H, W, out_channels)
    assert_with_pcc(torch_output, output, 0.97)
