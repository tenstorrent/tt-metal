# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from diffusers import StableDiffusionPipeline
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.wormhole.stable_diffusion.custom_preprocessing import custom_preprocessor
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_cross_attention_down_block_2d_new_conv import (
    cross_attention_down_block_2d,
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
    "block_index, hidden_states, shard_layout, shard_end_core, shard_shape, out_channels",
    [
        (0, [2, 320, 64, 64], ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (7, 7), (128, 320), 320),
        (1, [2, 320, 32, 32], ttnn.TensorMemoryLayout.BLOCK_SHARDED, (4, 7), (256, 64), 640),
        (2, [2, 640, 16, 16], ttnn.TensorMemoryLayout.BLOCK_SHARDED, (4, 7), (64, 128), 1280),
    ],
)
@pytest.mark.parametrize("temb", [[1, 1, 2, 1280]])
def test_cross_attention_downblock_512x512(
    reset_seeds,
    device,
    block_index,
    hidden_states,
    shard_layout,
    shard_end_core,
    shard_shape,
    out_channels,
    temb,
    use_program_cache,
):
    # Initialize PyTorch component
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    torch_down_block = unet.down_blocks[block_index]

    # Initialize ttnn component
    parameters = preprocess_model_parameters(
        initialize_model=lambda: unet, custom_preprocessor=custom_preprocessor, device=device
    )
    parameters = parameters.down_blocks[block_index]
    N, _, H, W = hidden_states
    compute_kernel_config = get_default_compute_config(device)

    ttnn_down_block = cross_attention_down_block_2d(device, parameters, N, H, W, compute_kernel_config)

    # Prepare inputs
    in_channels = hidden_states[1]
    temb_channels = 1280
    input_shape = hidden_states
    hidden_states = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    temb = torch_random(temb, -0.1, 0.1, dtype=torch.float32)

    encoder_hidden_states_shape = [1, 2, 77, 768]
    encoder_hidden_states = torch.randn(encoder_hidden_states_shape)

    # Run PyTorch component
    torch_output, torch_residuals = torch_down_block(
        hidden_states, temb.squeeze(0).squeeze(0), encoder_hidden_states.squeeze(0)
    )

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
    output, residuals = ttnn_down_block(
        hidden_states=hidden_states,
        temb=temb,
        encoder_hidden_states=encoder_hidden_states,
        config=unet.config,
        in_channels=in_channels,
        out_channels=out_channels,
        temb_channels=temb_channels,
        add_downsample=True,
        resnet_eps=1e-5,
        resnet_act_fn="silu",
    )

    # Compare outputs
    output = post_process_output_and_move_to_host(output, N, H // 2, W // 2, out_channels)
    assert_with_pcc(torch_output, output, 0.98)

    for residual_index, (torch_residual, residual) in enumerate(zip(torch_residuals, residuals)):
        if residual_index < 2:
            out_height = H
            out_width = W
        else:
            out_height = H // 2
            out_width = W // 2

        residual = post_process_output_and_move_to_host(residual, N, out_height, out_width, out_channels)

        assert_with_pcc(torch_residual, residual, 0.98)
