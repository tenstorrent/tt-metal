# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from diffusers import StableDiffusionPipeline
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.wormhole.stable_diffusion.custom_preprocessing import custom_preprocessor
from models.demos.wormhole.stable_diffusion.tests.parameterizations import (
    CROSS_UP_BLOCKS_HIDDEN_STATES_INFO,
    DOWN_MID_UP_BLOCKS_HIDDEN_STATES_INFO,
)
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_upsample_2d_new_conv import upsample2d
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_utility_functions import (
    post_process_output_and_move_to_host,
    preprocess_and_push_input_to_device,
)
from models.utility_functions import skip_for_grayskull, torch_random
from tests.ttnn.utils_for_testing import assert_with_pcc


def torch_to_ttnn(input, device, layout=ttnn.TILE_LAYOUT):
    input = ttnn.from_torch(input, ttnn.bfloat16)
    input = ttnn.to_layout(input, layout)
    input = ttnn.to_device(input, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return input


@skip_for_grayskull()
@pytest.mark.parametrize(
    "input_shape, shard_layout, shard_end_core, shard_shape, index",
    [
        DOWN_MID_UP_BLOCKS_HIDDEN_STATES_INFO + (0,),
        CROSS_UP_BLOCKS_HIDDEN_STATES_INFO[0] + (1,),
        CROSS_UP_BLOCKS_HIDDEN_STATES_INFO[1] + (2,),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_upsample2d_512x512(device, input_shape, shard_layout, shard_end_core, shard_shape, index):
    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)

    unet = pipe.unet
    unet.eval()
    unet_upblock = pipe.unet.up_blocks[index]
    resnet_upsampler = unet_upblock.upsamplers[0]

    parameters = preprocess_model_parameters(
        initialize_model=lambda: unet, custom_preprocessor=custom_preprocessor, device=device
    )
    parameters = parameters.up_blocks[index].upsamplers[0]

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    batch_size, in_channels, input_height, input_width = input_shape
    model = upsample2d(device, parameters, batch_size, input_height, input_width, compute_kernel_config)

    out_channels = in_channels
    input = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_output = resnet_upsampler(input)

    ttnn_input = preprocess_and_push_input_to_device(
        device,
        input,
        dtype=ttnn.bfloat16,
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

    tt_up = model(
        ttnn_input,
        in_channels,
        out_channels,
    )
    torch_up = post_process_output_and_move_to_host(tt_up, batch_size, input_height * 2, input_width * 2, in_channels)

    assert_with_pcc(torch_output, torch_up, 0.99)
