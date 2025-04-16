# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from diffusers import StableDiffusionPipeline
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.wormhole.stable_diffusion.custom_preprocessing import custom_preprocessor
from models.demos.wormhole.stable_diffusion.tests.parameterizations import DOWN_MID_UP_BLOCKS_HIDDEN_STATES_INFO
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_upblock_2d_new_conv import upblock_2d
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_utility_functions import (
    post_process_output_and_move_to_host,
    preprocess_and_push_input_to_device,
    weight_to_bfp8,
)
from models.utility_functions import skip_for_grayskull, torch_random
from tests.ttnn.utils_for_testing import assert_with_pcc


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("res_hidden_states_tuple", [([2, 1280, 8, 8], [2, 1280, 8, 8], [2, 1280, 8, 8])])
@pytest.mark.parametrize(
    "hidden_states, shard_layout, shard_end_core, shard_shape", [DOWN_MID_UP_BLOCKS_HIDDEN_STATES_INFO]
)
@pytest.mark.parametrize("temb", [[1, 1, 2, 1280]])
def test_upblock_512x512(
    reset_seeds,
    device,
    res_hidden_states_tuple,
    hidden_states,
    shard_layout,
    shard_end_core,
    shard_shape,
    temb,
    use_program_cache,
):
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()
    unet_upblock = pipe.unet.up_blocks[0]

    parameters = preprocess_model_parameters(
        initialize_model=lambda: unet, custom_preprocessor=custom_preprocessor, device=device
    )
    parameters = parameters.up_blocks[0]
    N, _, H, W = hidden_states

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    model = upblock_2d(device, parameters, N, H, W, compute_kernel_config)

    # synthesize the input
    in_channels = hidden_states[1]
    out_channels = in_channels
    prev_output_channel = in_channels
    temb_channels = None
    input_shape = hidden_states
    hidden_state = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    res_hidden_states_tuple = (hidden_state, hidden_state, hidden_state)
    temb = torch_random(temb, -0.1, 0.1, dtype=torch.float32)

    # execute pytorch
    torch_output = unet_upblock(hidden_state, res_hidden_states_tuple, temb.squeeze(), None)

    hidden_state = preprocess_and_push_input_to_device(
        device,
        hidden_state,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
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

    temb = temb.permute(2, 0, 1, 3)  # pre-permute temb
    temb = ttnn.from_torch(temb, ttnn.bfloat16)
    temb = ttnn.to_device(temb, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    temb = ttnn.to_layout(temb, ttnn.TILE_LAYOUT, ttnn.bfloat8_b)

    # hidden_state = pre_process_input(device, hidden_state)
    res_hidden_states_tuple = (weight_to_bfp8(hidden_state), weight_to_bfp8(hidden_state), weight_to_bfp8(hidden_state))
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
        add_upsample=True,
        temb=temb,
        upsample_size=None,
    )

    op = post_process_output_and_move_to_host(op, N, H * 2, W * 2, in_channels)

    assert_with_pcc(torch_output, op, 0.94)
