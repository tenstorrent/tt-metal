# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from diffusers import StableDiffusionPipeline
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.wormhole.stable_diffusion.custom_preprocessing import custom_preprocessor
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_resnetblock2d_new_conv import resnetBlock2D
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_utility_functions import (
    preprocess_and_push_input_to_device,
)
from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc


def ttnn_to_torch(input):
    input = ttnn.to_layout(input, ttnn.ROW_MAJOR_LAYOUT)
    input = ttnn.from_device(input)
    input = ttnn.to_torch(input)
    return input


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, in_channels, input_height, input_width, memory_layout, buffer_type, shard_end_core, shard_shape, out_channels, use_in_shortcut, block_name, block_index, resnet_index",
    [
        # fmt: off
        # down block 0
        (2, 320, 64, 64, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, (7, 7), (128, 320), 320, False, "down", 0, 0),
        (2, 320, 64, 64, ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, (4, 7), (1024, 64), 320, False, "down", 0, 1),
        # down block 1
        (2, 320, 32, 32, ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, (4, 7), (256, 64), 640, True, "down", 1, 0),
        (2, 640, 32, 32, ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, (4, 7), (256, 128), 640, False, "down", 1, 1),
        # down block 2
        (2, 640, 16, 16, ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, (4, 7), (64, 128), 1280, True, "down", 2, 0),
        (2, 1280, 16, 16, ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, (7, 7), (64, 160), 1280, False, "down", 2, 1),
        # down block 3
        (2, 1280, 8, 8, ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, (7, 3), (32, 160), 1280, None, "down", 3, 0),
        (2, 1280, 8, 8, ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, (7, 3), (32, 160), 1280, None, "down", 3, 1),
        # mid
        (2, 1280, 8, 8, ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, (7, 3), (32, 160), 1280, None, "mid", 0, 0),
        (2, 1280, 8, 8, ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, (7, 3), (32, 160), 1280, None, "mid", 0, 1),
        # up block 0
        (2, 2560, 8, 8, ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None, None, 1280, None, "up", 0, 0,),
        (2, 2560, 8, 8, ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None, None, 1280, None, "up", 0, 1),
        (2, 2560, 8, 8, ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None, None, 1280, None, "up", 0, 2),
        # up block 1
        (2, 2560, 16, 16, ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None, None, 1280, None, "up", 1, 0),
        (2, 2560, 16, 16, ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None, None, 1280, None, "up", 1, 1),
        (2, 1920, 16, 16, ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None, None, 1280, None, "up", 1, 2),
        # up block 2
        (2, 1920, 32, 32, ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None, None, 640, None, "up", 2, 0),
        (2, 1280, 32, 32, ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None, None, 640, None, "up", 2, 1),
        (2, 960, 32, 32, ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None, None, 640, None, "up", 2, 2),
        # up block 3
        (2, 960, 64, 64, ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None, None, 320, None, "up", 3, 0),
        (2, 640, 64, 64, ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None, None, 320, None, "up", 3, 1),
        (2, 640, 64, 64, ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None, None, 320, None, "up", 3, 2),
        # fmt: on
    ],
)
def test_resnet_block_2d_512x512(
    device,
    batch_size,
    in_channels,
    input_height,
    input_width,
    memory_layout,
    buffer_type,
    shard_end_core,
    shard_shape,
    out_channels,
    use_in_shortcut,
    block_name,
    block_index,
    resnet_index,
    use_program_cache,
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
            parameters = parameters.up_blocks[block_index].resnets[resnet_index]
            resnet = pipe.unet.up_blocks[block_index].resnets[resnet_index]
        elif block_name == "down":
            parameters = parameters.down_blocks[block_index].resnets[resnet_index]
            resnet = pipe.unet.down_blocks[block_index].resnets[resnet_index]
        else:
            parameters = parameters.mid_block.resnets[resnet_index]
            resnet = pipe.unet.mid_block.resnets[resnet_index]
        torch.save(resnet, "resnet.pt")
        torch.save(config, "config.pt")

    else:
        resnet = torch.load("resnet.pt")
        config = torch.load("config.pt")
        parameters = preprocess_model_parameters(
            initialize_model=lambda: resnet, custom_preprocessor=custom_preprocessor, device=device
        )

    ttnn.dump_device_memory_state(device, prefix="GN_resnet_1_")

    ############ start of residual block #############
    temb_channels = 1280
    groups = 32
    time_embedding_norm = "default"
    output_scale_factor = 1
    ########## end of residual block #############
    hidden_states_shape = [batch_size, in_channels, input_height, input_width]
    temb_shape = [1, 1, 2, 1280]

    input = torch.randn(hidden_states_shape)
    temb = torch.randn(temb_shape)

    torch_output = resnet(input, temb.squeeze(0).squeeze(0))
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    resnet_block = resnetBlock2D(
        device,
        parameters,
        batch_size,
        input_height,
        input_width,
        group_norm_on_device=True,
        compute_kernel_config=compute_kernel_config,
    )

    memory_config = None
    if memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED:
        memory_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
            buffer_type=buffer_type,
        )
    else:
        memory_config = ttnn.MemoryConfig(
            memory_layout,
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
        )

    input = preprocess_and_push_input_to_device(device, input, memory_config=memory_config)

    temb = temb.permute(2, 0, 1, 3)  # pre-permute temb
    temb = ttnn.from_torch(temb, ttnn.bfloat16)
    temb = ttnn.to_layout(temb, ttnn.TILE_LAYOUT)
    temb = ttnn.to_device(temb, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    ttnn_output_ = resnet_block(
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
    )
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
    )
    ttnn_output = ttnn.to_memory_config(ttnn_output, ttnn.DRAM_MEMORY_CONFIG)
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
