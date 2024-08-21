# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from diffusers import StableDiffusionPipeline
from loguru import logger
import ttnn
import pytest

from models.utility_functions import tt_to_torch_tensor, torch_random
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import (
    skip_for_grayskull,
)
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_upblock_2d import upblock_2d
from models.demos.wormhole.stable_diffusion.custom_preprocessing import custom_preprocessor
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_utility_functions import (
    pre_process_input,
    post_process_output,
    weight_to_bfp8,
)


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("res_hidden_states_tuple", [([2, 1280, 8, 8], [2, 1280, 8, 8], [2, 1280, 8, 8])])
@pytest.mark.parametrize("hidden_states", [[2, 1280, 8, 8]])
@pytest.mark.parametrize("temb", [[1, 1, 2, 1280]])
def test_upblock_512x512(reset_seeds, device, res_hidden_states_tuple, hidden_states, temb):
    # TODO
    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()
    unet_upblock = pipe.unet.up_blocks[0]
    reader_patterns_cache = {}

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
    model = upblock_2d(device, parameters, reader_patterns_cache, N, H, W, compute_kernel_config)

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
    torch_output = unet_upblock(hidden_state, res_hidden_states_tuple, None, None)

    hidden_state = ttnn.from_torch(hidden_state, ttnn.bfloat16)
    hidden_state = ttnn.to_device(hidden_state, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    hidden_state = ttnn.permute(hidden_state, (0, 2, 3, 1))

    hidden_state = ttnn.reshape(hidden_state, (1, 1, N * H * W, in_channels))

    hidden_state = ttnn.to_layout(hidden_state, ttnn.TILE_LAYOUT, ttnn.bfloat8_b)

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

    op = post_process_output(device, op, N, H * 2, W * 2, in_channels)
    op = ttnn.to_torch(op)
    assert_with_pcc(torch_output, op, 0.95)
