# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from diffusers import StableDiffusionPipeline
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.wormhole.stable_diffusion.custom_preprocessing import custom_preprocessor
from models.demos.wormhole.stable_diffusion.tests.parameterizations import TRANSFORMER_PARAMETERIZATIONS
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_cross_attention import cross_attention
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_utility_functions import (
    preprocess_and_push_input_to_device,
)
from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import comp_pcc


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("model_name", ["CompVis/stable-diffusion-v1-4"])
@pytest.mark.parametrize("has_encoder_hidden_states", (True, False))
@pytest.mark.parametrize(
    "input_shape, shard_layout, shard_end_core, shard_shape, attention_head_dim, block, block_index, attention_index",
    TRANSFORMER_PARAMETERIZATIONS,
)
def test_cross_attention_512x512(
    device,
    model_name,
    input_shape,
    shard_layout,
    shard_end_core,
    shard_shape,
    attention_head_dim,
    has_encoder_hidden_states,
    block,
    block_index,
    attention_index,
    use_program_cache,
):
    torch.manual_seed(0)
    device.enable_program_cache()

    N, C, H, W = input_shape

    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
    model = pipe.unet
    model.eval()

    hidden_states_shape = torch.Size(input_shape)
    hidden_states = torch.rand(hidden_states_shape)

    if block == "up":
        basic_transformer = pipe.unet.up_blocks[block_index].attentions[attention_index].transformer_blocks[0]
    elif block == "down":
        basic_transformer = pipe.unet.down_blocks[block_index].attentions[attention_index].transformer_blocks[0]
    elif block == "mid":
        basic_transformer = pipe.unet.mid_block.attentions[0].transformer_blocks[0]

    if has_encoder_hidden_states:
        cross_attn = basic_transformer.attn2

        encoder_hidden_states_shape = torch.Size([1, 2, 77, 768])
        encoder_hidden_states = torch.rand(encoder_hidden_states_shape)
        encoder_hidden_states = encoder_hidden_states.squeeze(0)

        ttnn_encoder_hidden_states = ttnn.from_torch(encoder_hidden_states, dtype=ttnn.bfloat16)
        ttnn_encoder_hidden_states = ttnn.to_device(ttnn_encoder_hidden_states, device)
    else:
        cross_attn = basic_transformer.attn1
        encoder_hidden_states = None
        ttnn_encoder_hidden_states = None

    torch_hidden_states = torch.permute(hidden_states, [0, 2, 3, 1])
    torch_hidden_states = torch.reshape(torch_hidden_states, [N, H * W, C])
    encoder_hidden_states = encoder_hidden_states.squeeze(0) if encoder_hidden_states is not None else None
    torch_output = cross_attn(torch_hidden_states.squeeze(0), encoder_hidden_states).unsqueeze(0)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: cross_attn, custom_preprocessor=custom_preprocessor, device=device
    )

    if encoder_hidden_states is not None:
        encoder_hidden_states = torch.nn.functional.pad(encoder_hidden_states, (0, 0, 0, 19))
        ttnn_encoder_hidden_states = ttnn.from_torch(
            encoder_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
    else:
        ttnn_encoder_hidden_states = None

    ttnn_hidden_states = preprocess_and_push_input_to_device(
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

    model = cross_attention(device, parameters, seq_len=H * W)
    ttnn_output = model(
        ttnn_hidden_states,
        ttnn_encoder_hidden_states,
        attention_mask=None,
        dim_head=attention_head_dim,
    )
    ttnn_output = ttnn.to_torch(ttnn_output)

    passing, output = comp_pcc(torch_output, ttnn_output, pcc=0.99)
    print(output)
    assert passing
