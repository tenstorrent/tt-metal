# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
from diffusers import UNet2DConditionModel
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters

from models.experimental.functional_stable_diffusion.tt.ttnn_functional_geglu import geglu as ttnn_geglu
from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_geglu import geglu as tt2_ttnn_geglu
from models.utility_functions import torch_random, skip_for_grayskull

from tests.ttnn.utils_for_testing import assert_with_pcc


@skip_for_grayskull()
@pytest.mark.parametrize("model_name", ["CompVis/stable-diffusion-v1-4"])
@pytest.mark.parametrize(
    "N, C, H, W, index",
    [
        (
            1,
            2,
            1024,
            320,
            0,
        ),
        (
            1,
            2,
            256,
            640,
            1,
        ),
        (
            1,
            2,
            64,
            1280,
            2,
        ),
        (
            1,
            2,
            16,
            1280,
            2,
        ),
    ],
)
def test_geglu_256x256(device, model_name, N, C, H, W, index, reset_seeds):
    input_shapes = (N, C, H, W)
    model = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet").eval()
    ref_model = model.down_blocks[index].attentions[0].transformer_blocks[0].ff.net[0]
    config = model.config
    torch_hidden_states = torch_random(input_shapes, -0.1, 0.1, dtype=torch.float32)
    torch_output = ref_model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_model,
        device=device,
    )

    ttnn_hidden_state = ttnn.to_layout(
        ttnn.to_device(ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat16), device), layout=ttnn.TILE_LAYOUT
    )

    output = ttnn_geglu(
        config,
        ttnn_hidden_state,
        parameters=parameters,
    )
    output = ttnn.from_device(output)
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output.to(torch_output.dtype), 0.96)


@skip_for_grayskull()
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
    torch_hidden_states = torch_random(input_shapes, -0.1, 0.1, dtype=torch.float32)
    torch_output = ref_model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_model,
        device=device,
    )
    model = tt2_ttnn_geglu(device, parameters=parameters)

    ttnn_hidden_state = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_hidden_state = ttnn.to_device(ttnn_hidden_state, device)

    output = model(config, ttnn_hidden_state)
    output = ttnn.from_device(output)
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output.to(torch_output.dtype), 0.99)
