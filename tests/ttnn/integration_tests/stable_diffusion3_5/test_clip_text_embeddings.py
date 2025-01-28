# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)
from diffusers import StableDiffusion3Pipeline
from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_stable_diffusion3_5.reference.clip_text_embeddings import CLIPTextEmbeddings
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_clip_text_embeddings import ttnn_CLIPTextEmbeddings


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, CLIPTextEmbeddings):
            parameters["token_embedding"] = {}
            parameters["token_embedding"]["weight"] = ttnn.from_torch(
                model.token_embedding.weight, layout=ttnn.TILE_LAYOUT
            )
            parameters["position_embedding"] = {}
            parameters["position_embedding"]["weight"] = ttnn.from_torch(
                model.position_embedding.weight, layout=ttnn.TILE_LAYOUT
            )

            parameters["position_ids"] = {}
            parameters["position_ids"] = ttnn.from_torch(model.position_ids, layout=ttnn.ROW_MAJOR_LAYOUT)

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@skip_for_grayskull()
def test_clip_text_embeddings(device, reset_seeds):
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16
    )
    clip_mlp_config = pipe.text_encoder.config

    reference_model = CLIPTextEmbeddings(clip_mlp_config).to(dtype=torch.bfloat16)

    reference_model.eval()
    hidden_states = torch.normal(0.0, 30.0, size=(1, 77))
    hidden_states = hidden_states.abs()
    hidden_states = hidden_states.to(torch.int64)

    ttnn_hidden_states = ttnn.from_torch(
        hidden_states,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    torch_output = reference_model(hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(device), device=device
    )
    ttnn_model = ttnn_CLIPTextEmbeddings(clip_mlp_config, parameters)

    ttnn_output = ttnn_model(ttnn_hidden_states, parameters=parameters)

    assert_with_pcc(torch_output, ttnn.to_torch(ttnn_output), pcc=0.99)
