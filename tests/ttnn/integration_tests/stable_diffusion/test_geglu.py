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


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("model_name", ["CompVis/stable-diffusion-v1-4"])
@pytest.mark.parametrize(
    "N, C, H, W, index",
    [
        (
            1,
            2,
            4096,
            320,
            3,
        ),
        (
            1,
            2,
            1024,
            640,
            2,
        ),
        (
            1,
            2,
            256,
            1280,
            1,
        ),
        (
            1,
            2,
            64,
            1280,
            1,
        ),
    ],
)
def test_geglu_512x512(device, model_name, N, C, H, W, index, reset_seeds):
    input_shapes = (N, C, H, W)
    model = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet").eval()
    ref_model = model.up_blocks[index].attentions[0].transformer_blocks[0].ff.net[0]
    config = model.config
    torch_hidden_states = torch_random(input_shapes, -1, 1, dtype=torch.float32)
    torch_output = ref_model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_model,
        device=device,
        custom_preprocessor=custom_preprocessor,
    )
    model = geglu(device, parameters=parameters)

    torch_hidden_states = torch_hidden_states.reshape(
        1, 1, torch_hidden_states.shape[-3] * torch_hidden_states.shape[-2], torch_hidden_states.shape[-1]
    )
    ttnn_hidden_state = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_hidden_state = ttnn.to_device(ttnn_hidden_state, device)

    output = model(config, ttnn_hidden_state)
    output = ttnn.from_device(output)
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.to_torch(output)
    output = output.reshape(1, 2, output.shape[-2] // 2, output.shape[-1])

    assert_with_pcc(torch_output, output.to(torch_output.dtype), 0.99)
