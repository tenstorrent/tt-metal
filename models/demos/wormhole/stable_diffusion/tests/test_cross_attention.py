# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from diffusers import StableDiffusionPipeline
import ttnn

from models.demos.wormhole.stable_diffusion.custom_preprocessing import custom_preprocessor
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_cross_attention import (
    cross_attention,
)
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc, comp_pcc
from models.utility_functions import (
    skip_for_grayskull,
)


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("model_name", ["CompVis/stable-diffusion-v1-4"])
@pytest.mark.parametrize(
    "N, C, H, W, index, has_encoder_hidden_states",
    [
        (
            1,
            2,
            4096,
            320,
            3,
            True,
        ),
        (
            1,
            2,
            4096,
            320,
            3,
            False,
        ),
        (
            1,
            2,
            1024,
            640,
            2,
            True,
        ),
        (
            1,
            2,
            1024,
            640,
            2,
            False,
        ),
        (
            1,
            2,
            256,
            1280,
            1,
            True,
        ),
        (
            1,
            2,
            256,
            1280,
            1,
            False,
        ),
        (
            1,
            2,
            64,
            1280,
            1,
            True,
        ),
        (
            1,
            2,
            64,
            1280,
            1,
            False,
        ),
    ],
)
def test_cross_attention_512x512(device, model_name, N, C, H, W, index, has_encoder_hidden_states):
    torch.manual_seed(0)
    device.enable_program_cache()

    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
    model = pipe.unet
    model.eval()

    hidden_states_shape = torch.Size([N, C, H, W])
    hidden_states = torch.rand(hidden_states_shape)
    if has_encoder_hidden_states:
        cross_attn = pipe.unet.up_blocks[index].attentions[0].transformer_blocks[0].attn2

        encoder_hidden_states_shape = torch.Size([1, 2, 77, 768])
        encoder_hidden_states = torch.rand(encoder_hidden_states_shape)
        encoder_hidden_states = encoder_hidden_states.squeeze(0)

        ttnn_encoder_hidden_states = ttnn.from_torch(encoder_hidden_states, dtype=ttnn.bfloat16)
        ttnn_encoder_hidden_states = ttnn.to_device(ttnn_encoder_hidden_states, device)
    else:
        cross_attn = pipe.unet.up_blocks[index].attentions[0].transformer_blocks[0].attn1
        encoder_hidden_states = None
        ttnn_encoder_hidden_states = None

    encoder_hidden_states = encoder_hidden_states.squeeze(0) if encoder_hidden_states is not None else None
    torch_output = cross_attn(hidden_states.squeeze(0), encoder_hidden_states).unsqueeze(0)

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

    hidden_states = hidden_states.reshape(1, 1, N * C * H, W)
    ttnn_hidden_states = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_hidden_states = ttnn.to_device(ttnn_hidden_states, device)

    model = cross_attention(device, parameters, seq_len=H)
    ttnn_output = model(
        ttnn_hidden_states,
        ttnn_encoder_hidden_states,
        attention_mask=None,
        dim_head=W // 8,
    )
    ttnn_output = ttnn.to_torch(ttnn_output)

    passing, output = comp_pcc(torch_output, ttnn_output, pcc=0.99)
    print(output)
    assert passing
