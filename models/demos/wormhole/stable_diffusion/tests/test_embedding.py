# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from diffusers import StableDiffusionPipeline
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_embeddings import TtTimestepEmbedding
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_embeddings(device, use_program_cache):
    torch.manual_seed(0)
    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)

    model = pipe.unet
    model.eval()
    time_embedding = model.time_embedding

    parameters = preprocess_model_parameters(initialize_model=lambda: time_embedding, device=device)
    model = TtTimestepEmbedding(parameters=parameters)

    input = torch.randn([1, 1, 2, 320])
    torch_output = time_embedding(input.squeeze(0).squeeze(0))

    input = ttnn.from_torch(input, ttnn.bfloat16)
    input = ttnn.to_layout(input, ttnn.TILE_LAYOUT)
    input = ttnn.to_device(input, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    ttnn_output = model(input)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.squeeze(0).squeeze(0)

    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)
