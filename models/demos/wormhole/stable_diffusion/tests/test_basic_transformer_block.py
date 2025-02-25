# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from diffusers import StableDiffusionPipeline
import ttnn

from models.demos.wormhole.stable_diffusion.custom_preprocessing import custom_preprocessor
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_basic_transformer_block import basic_transformer_block
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import (
    skip_for_grayskull,
)
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_utility_functions import (
    preprocess_and_push_input_to_device,
)


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("model_name", ["CompVis/stable-diffusion-v1-4"])
@skip_for_grayskull()
@pytest.mark.parametrize(
    "N, C, H, W, shard_layout, shard_end_core, shard_shape, attention_head_dim, block, index1, index2",
    [
        (
            2,
            320,
            64,
            64,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (4, 7),
            (1024, 64),
            40,
            "down",
            0,
            0,
        ),
        (
            2,
            320,
            64,
            64,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (4, 7),
            (1024, 64),
            40,
            "down",
            0,
            1,
        ),
        (
            2,
            640,
            32,
            32,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (4, 7),
            (256, 128),
            80,
            "down",
            1,
            0,
        ),
        (
            2,
            640,
            32,
            32,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (4, 7),
            (256, 128),
            80,
            "down",
            1,
            1,
        ),
        (
            2,
            1280,
            16,
            16,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (7, 7),
            (64, 160),
            160,
            "down",
            2,
            0,
        ),
        (
            2,
            1280,
            16,
            16,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (7, 7),
            (64, 160),
            160,
            "down",
            2,
            1,
        ),
    ],
)
def test_basic_transformer_block_512x512(
    device, model_name, N, C, H, W, shard_layout, shard_end_core, shard_shape, attention_head_dim, block, index1, index2
):
    torch.manual_seed(0)

    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
    model = pipe.unet
    model.eval()
    config = model.config

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model, custom_preprocessor=custom_preprocessor, device=device
    )

    if block == "up":
        parameters = parameters.up_blocks[index1].attentions[index2].transformer_blocks[0]
        basic_transformer = pipe.unet.up_blocks[index1].attentions[index2].transformer_blocks[0]
    elif block == "down":
        breakpoint()
        parameters = parameters.down_blocks[index1].attentions[index2].transformer_blocks[0]
        basic_transformer = pipe.unet.down_blocks[index1].attentions[index2].transformer_blocks[0]
    elif block == "mid":
        parameters = parameters.mid_block.attentions[0].transformer_blocks[0]
        basic_transformer = pipe.unet.mid_block.attentions[0].transformer_blocks[0]

    hidden_states_shape = torch.Size([N, C, H, W])
    hidden_states = torch.rand(hidden_states_shape) * 0.01
    encoder_hidden_states_shape = [1, 2, 77, 768]
    encoder_hidden_states = torch.rand(encoder_hidden_states_shape)

    timestep = None
    attention_mask = None
    cross_attention_kwargs = None
    class_labels = None

    torch_hidden_states = torch.permute(hidden_states, [0, 2, 3, 1])
    torch_hidden_states = torch.reshape(torch_hidden_states, [N, H * W, C])
    torch_output = basic_transformer(torch_hidden_states, encoder_hidden_states.squeeze(0))

    model = basic_transformer_block(device, parameters, seq_len=H * W)

    encoder_hidden_states = torch.nn.functional.pad(encoder_hidden_states, (0, 0, 0, 19))
    encoder_hidden_states = ttnn.from_torch(encoder_hidden_states, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    encoder_hidden_states = ttnn.to_device(encoder_hidden_states, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

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

    ttnn_output = model(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
        class_labels=class_labels,
        config=config,
        attention_head_dim=attention_head_dim,
    )

    ttnn_output = ttnn.reshape(ttnn_output, [1, 2, ttnn_output.shape[-2] // 2, ttnn_output.shape[-1]])
    ttnn_output = ttnn.from_device(ttnn_output)
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output.unsqueeze(0), ttnn_output, pcc=0.98)
