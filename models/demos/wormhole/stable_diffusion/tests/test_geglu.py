# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
from diffusers import UNet2DConditionModel
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters

from models.demos.wormhole.stable_diffusion.custom_preprocessing import custom_preprocessor

from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_geglu import geglu
from models.utility_functions import torch_random, skip_for_grayskull

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_utility_functions import (
    preprocess_and_push_input_to_device,
)


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("model_name", ["CompVis/stable-diffusion-v1-4"])
@pytest.mark.parametrize(
    "N, C, H, W, shard_layout, shard_end_core, shard_shape, block, index1, index2",
    [
        (
            2,
            320,
            64,
            64,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (4, 7),
            (1024, 64),
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
            "down",
            2,
            1,
        ),
    ],
)
def test_geglu_512x512(
    device, model_name, N, C, H, W, shard_layout, shard_end_core, shard_shape, block, index1, index2
):
    model = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet").eval()
    ref_model = model.down_blocks[index1].attentions[index2].transformer_blocks[0].ff.net[0]
    config = model.config
    hidden_states = torch_random([N, C, H, W], -1, 1, dtype=torch.float32)
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
