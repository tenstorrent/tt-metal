# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from diffusers import AutoPipelineForText2Image
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.stable_diffusion_xl_turbo.tt import tt_downsample_2d
from models.demos.stable_diffusion_xl_turbo import custom_preprocessing


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "N, C, H, W",
    [
        (1, 320, 128, 128),
    ],
)
def test_downsample_1(device, N, C, H, W, reset_seeds):
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float32, variant="fp16"
    )
    model = pipe.unet
    model.eval()
    model = model.down_blocks[0].downsamplers[0]
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model, custom_preprocessor=custom_preprocessing.custom_preprocessor
    )
    input_tensor = torch.randn((N, C, H, W), dtype=torch.float32)
    torch_output = model(input_tensor)
    input_tensor = input_tensor.permute(0, 2, 3, 1)

    ttnn_hidden_state = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
    )
    output = tt_downsample_2d.downsample_1(ttnn_hidden_state, parameters, device)
    output = ttnn.to_torch(output).permute(0, 3, 1, 2)

    assert_with_pcc(torch_output, output, 0.9996278)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "N, C, H, W",
    [
        (1, 640, 64, 64),
    ],
)
def test_downsample_2(device, N, C, H, W, reset_seeds):
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float32, variant="fp16"
    )
    model = pipe.unet
    model.eval()
    model = model.down_blocks[1].downsamplers[0]
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model, custom_preprocessor=custom_preprocessing.custom_preprocessor
    )
    input_tensor = torch.randn((N, C, H, W), dtype=torch.float32)
    torch_output = model(input_tensor)

    ttnn_hidden_state = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
    )
    output = tt_downsample_2d.downsample_2(ttnn_hidden_state, parameters, device)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, 0.999572)
