# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from diffusers import StableDiffusionPipeline
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.wormhole.stable_diffusion.common import SD_L1_SMALL_SIZE
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_embeddings import TtTimestepEmbedding
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": SD_L1_SMALL_SIZE}], indirect=True)
def test_embeddings(device, is_ci_env, is_ci_v2_env, model_location_generator):
    torch.manual_seed(0)
    model_location = model_location_generator("stable-diffusion-v1-4", download_if_ci_v2=True, ci_v2_timeout_in_s=1800)
    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4" if not is_ci_v2_env else model_location,
        torch_dtype=torch.float32,
        local_files_only=is_ci_env or is_ci_v2_env,
    )

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
