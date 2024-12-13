# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from diffusers import AutoPipelineForText2Image
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.stable_diffusion_xl_turbo.tt import tt_feed_forward


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "N, H, C, index",
    [
        (1, 4096, 640, 1),
        (1, 1024, 1280, 2),
    ],
)
def test_geglu_down_blocks(device, N, C, H, index, reset_seeds):
    input_shapes = (N, H, C)
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
    )
    model = pipe.unet
    model.eval()
    config = model.config
    model = model.down_blocks[index].attentions[0].transformer_blocks[0].ff.net[0]
    torch_hidden_states = torch_random(input_shapes, -1, 1, dtype=torch.float16)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
    )
    print(parameters)
    torch_output = model(torch_hidden_states)
    ttnn_hidden_state = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_hidden_state = ttnn.to_device(ttnn_hidden_state, device)

    output = tt_feed_forward.sd_geglu(ttnn_hidden_state, parameters, device)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "N, H, C, index",
    [
        (1, 1024, 1280, 0),
        (1, 4096, 640, 1),
    ],
)
def test_geglu_up_blocks(device, N, C, H, index, reset_seeds):
    input_shapes = (N, H, C)
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
    )
    model = pipe.unet
    model.eval()
    config = model.config
    model = model.up_blocks[index].attentions[0].transformer_blocks[0].ff.net[0]
    torch_hidden_states = torch_random(input_shapes, -1, 1, dtype=torch.float16)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
    )
    print(parameters)
    torch_output = model(torch_hidden_states)
    ttnn_hidden_state = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_hidden_state = ttnn.to_device(ttnn_hidden_state, device)

    output = tt_feed_forward.sd_geglu(ttnn_hidden_state, parameters, device)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "N, H, C",
    [
        (1, 1024, 1280),
    ],
)
def test_geglu_mid_block(device, N, C, H, reset_seeds):
    input_shapes = (N, H, C)
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
    )
    model = pipe.unet
    model.eval()
    config = model.config
    model = model.mid_block.attentions[0].transformer_blocks[0].ff.net[0]
    torch_hidden_states = torch_random(input_shapes, -1, 1, dtype=torch.float16)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
    )
    print(parameters)
    torch_output = model(torch_hidden_states)
    ttnn_hidden_state = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_hidden_state = ttnn.to_device(ttnn_hidden_state, device)

    output = tt_feed_forward.sd_geglu(ttnn_hidden_state, parameters, device)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "N, H, C, index",
    [
        (1, 4096, 640, 1),
        (1, 1024, 1280, 2),
    ],
)
def test_feed_forward_down_blocks(device, N, C, H, index, reset_seeds):
    input_shapes = (N, H, C)
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
    )
    model = pipe.unet
    model.eval()
    config = model.config
    model = model.down_blocks[index].attentions[0].transformer_blocks[0].ff
    torch_hidden_states = torch_random(input_shapes, -1, 1, dtype=torch.float16)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
    )
    torch_output = model(torch_hidden_states)
    ttnn_hidden_state = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_hidden_state = ttnn.to_device(ttnn_hidden_state, device)

    output = tt_feed_forward.sd_feed_forward(ttnn_hidden_state, parameters, device)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "N, H, C, index",
    [
        (1, 1024, 1280, 0),
        (1, 4096, 640, 1),
    ],
)
def test_feed_forward_up_blocks(device, N, C, H, index, reset_seeds):
    input_shapes = (N, H, C)
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
    )
    model = pipe.unet
    model.eval()
    config = model.config
    model = model.up_blocks[index].attentions[0].transformer_blocks[0].ff
    torch_hidden_states = torch_random(input_shapes, -1, 1, dtype=torch.float16)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
    )
    torch_output = model(torch_hidden_states)
    ttnn_hidden_state = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_hidden_state = ttnn.to_device(ttnn_hidden_state, device)

    output = tt_feed_forward.sd_feed_forward(ttnn_hidden_state, parameters, device)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "N, H, C",
    [
        (1, 1024, 1280),
    ],
)
def test_geglu_mid_block(device, N, C, H, reset_seeds):
    input_shapes = (N, H, C)
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
    )
    model = pipe.unet
    model.eval()
    config = model.config
    model = model.mid_block.attentions[0].transformer_blocks[0].ff.net[0]
    torch_hidden_states = torch_random(input_shapes, -1, 1, dtype=torch.float16)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
    )
    print(parameters)
    torch_output = model(torch_hidden_states)
    ttnn_hidden_state = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_hidden_state = ttnn.to_device(ttnn_hidden_state, device)

    output = tt_feed_forward.sd_geglu(ttnn_hidden_state, parameters, device)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, 0.99)
