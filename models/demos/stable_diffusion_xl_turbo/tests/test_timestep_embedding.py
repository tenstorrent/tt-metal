# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from diffusers import AutoPipelineForText2Image
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.stable_diffusion_xl_turbo.tt import tt_timestep_embedding


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_time_embeddings(device, reset_seeds):
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
    )

    model = pipe.unet
    model.eval()
    time_embedding = model.time_embedding

    parameters = preprocess_model_parameters(initialize_model=lambda: time_embedding, device=device)
    input_tensor = torch.randn((1, 1, 2, 320), dtype=torch.float16)
    torch_output = time_embedding(input_tensor.squeeze(0).squeeze(0))

    input_tensor = ttnn.from_torch(
        input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    ttnn_output = tt_timestep_embedding.timestep_embedding(input_tensor, parameters, device)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.squeeze(0).squeeze(0)
    print(ttnn_output.shape)

    assert_with_pcc(torch_output, ttnn_output, pcc=0.9996)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_add_embeddings(device, reset_seeds):
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
    )

    model = pipe.unet
    model.eval()
    time_embedding = model.add_embedding

    parameters = preprocess_model_parameters(initialize_model=lambda: time_embedding, device=device)
    input_tensor = torch.randn((1, 1, 2, 2816), dtype=torch.float16)
    torch_output = time_embedding(input_tensor.squeeze(0).squeeze(0))

    input_tensor = ttnn.from_torch(
        input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    ttnn_output = tt_timestep_embedding.timestep_embedding(input_tensor, parameters, device)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.squeeze(0).squeeze(0)
    print(ttnn_output.shape)

    assert_with_pcc(torch_output, ttnn_output, pcc=0.99969)
