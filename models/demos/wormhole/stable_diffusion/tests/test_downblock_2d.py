# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from diffusers import StableDiffusionPipeline
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.wormhole.stable_diffusion.custom_preprocessing import custom_preprocessor
from models.demos.wormhole.stable_diffusion.tests.parameterizations import DOWN_MID_UP_BLOCKS_HIDDEN_STATES_INFO
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_downblock_2d_new_conv import downblock2d
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
def test_downblock_512x512(
    reset_seeds, device, hidden_states, shard_layout, shard_end_core, shard_shape, temb, use_program_cache
):
    # Initialize PyTorch component
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    torch_down_block = pipe.unet.down_blocks[3]

    # Initialize ttnn component
    parameters = preprocess_model_parameters(
        initialize_model=lambda: unet, custom_preprocessor=custom_preprocessor, device=device
    )
    parameters = parameters.down_blocks[3]
    N, _, H, W = hidden_states
    compute_kernel_config = get_default_compute_config(device)

    ttnn_down_block = downblock2d(device, parameters, N, H, W, compute_kernel_config)

    # Prepare inputs
    in_channels = hidden_states[1]
    out_channels = in_channels
    temb_channels = 1280
    input_shape = hidden_states
    hidden_states = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    temb = torch_random(temb, -0.1, 0.1, dtype=torch.float32)

    # Run PyTorch component
    torch_output, torch_residuals = torch_down_block(hidden_states, temb.squeeze(0).squeeze(0))

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

    # Run ttnn component
    output, residuals = ttnn_down_block(
        temb,
        hidden_states,
        in_channels,
        out_channels,
        temb_channels,
        num_layers=2,
        resnet_eps=1e-5,
        resnet_act_fn="silu",
    )

    # Compare outputs
    output = post_process_output_and_move_to_host(output, N, H, W, out_channels)
    assert_with_pcc(torch_output, output, 0.97)

    for torch_residual, residual in zip(torch_residuals, residuals):
        residual = post_process_output_and_move_to_host(residual, N, H, W, out_channels)
        assert_with_pcc(torch_residual, residual, 0.97)
