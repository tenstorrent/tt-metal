# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from diffusers import UNet2DConditionModel
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.wormhole.stable_diffusion.custom_preprocessing import custom_preprocessor
from models.demos.wormhole.stable_diffusion.tests.parameterizations import TRANSFORMER_PARAMETERIZATIONS
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_geglu import geglu
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_utility_functions import (
    preprocess_and_push_input_to_device,
)
from models.utility_functions import skip_for_grayskull, torch_random
from tests.ttnn.utils_for_testing import assert_with_pcc


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("model_name", ["CompVis/stable-diffusion-v1-4"])
@pytest.mark.parametrize(
    "input_shape, shard_layout, shard_end_core, shard_shape, attention_head_dim, block, block_index, attention_index",
    TRANSFORMER_PARAMETERIZATIONS,
)
def test_geglu_512x512(
    device,
    model_name,
    input_shape,
    shard_layout,
    shard_end_core,
    shard_shape,
    attention_head_dim,
    block,
    block_index,
    attention_index,
    use_program_cache,
):
    model = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet").eval()

    if block == "up":
        basic_transformer = model.up_blocks[block_index].attentions[attention_index].transformer_blocks[0]
    elif block == "down":
        basic_transformer = model.down_blocks[block_index].attentions[attention_index].transformer_blocks[0]
    elif block == "mid":
        basic_transformer = model.mid_block.attentions[0].transformer_blocks[0]

    ref_model = basic_transformer.ff.net[0]
    config = model.config

    N, C, H, W = input_shape

    hidden_states = torch_random(input_shape, -1, 1, dtype=torch.float32)
    torch_hidden_states = torch.permute(hidden_states, [0, 2, 3, 1])
    torch_hidden_states = torch.reshape(torch_hidden_states, [N, H * W, C])

    torch_output = ref_model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_model,
        device=device,
        custom_preprocessor=custom_preprocessor,
    )
    model = geglu(device, parameters=parameters)

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
        dtype=ttnn.bfloat16,
    )

    output = model(config, ttnn_hidden_states)
    output = ttnn.from_device(output)
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.to_torch(output)
    output = output.reshape(1, 2, output.shape[-2] // 2, output.shape[-1])

    assert_with_pcc(torch_output, output.to(torch_output.dtype).squeeze(0), 0.99)
