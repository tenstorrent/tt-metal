# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_layernorm_parameter,
)
from tests.ttnn.utils_for_testing import assert_with_pcc
from transformers import SegformerModel
import pytest
from models.experimental.functional_segformer.tt.ttnn_segformer_overlap_patch_embeddings import (
    TtSegformerOverlapPatchEmbeddings,
)

from models.experimental.functional_segformer.reference.segformer_overlap_patch_embeddings import (
    SegformerOverlapPatchEmbeddings,
)
from models.utility_functions import skip_for_grayskull


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, SegformerOverlapPatchEmbeddings):
            parameters["proj"] = {}

            parameters["proj"]["weight"] = ttnn.from_torch(model.proj.weight, dtype=ttnn.bfloat16)
            parameters["proj"]["bias"] = ttnn.from_torch(
                torch.reshape(model.proj.bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
            )

            parameters["layer_norm"] = {}
            parameters["layer_norm"]["weight"] = preprocess_layernorm_parameter(
                model.layer_norm.weight, dtype=ttnn.bfloat16
            )
            parameters["layer_norm"]["bias"] = preprocess_layernorm_parameter(
                model.layer_norm.bias, dtype=ttnn.bfloat16
            )

        return parameters

    return custom_preprocessor


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "patch_size, stride, num_channels, hidden_size, batch_size, height, width, patch_emb_i",
    [
        (7, 4, 3, 32, 1, 512, 512, 0),
        (3, 2, 32, 64, 1, 128, 128, 1),
        (3, 2, 64, 160, 1, 64, 64, 2),
        (3, 2, 160, 256, 1, 32, 32, 3),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_segformer_overlap_patch_embeddings(
    patch_size,
    stride,
    num_channels,
    hidden_size,
    batch_size,
    height,
    width,
    patch_emb_i,
    device,
    reset_seeds,
    is_ci_env,
):
    if is_ci_env:
        pytest.skip("Skip in CI, model is WIP, issue# 13357")

    torch_input_tensor = torch.randn(batch_size, num_channels, height, width)
    if width == 512:
        ttnn_input_tensor = ttnn.from_torch(
            torch_input_tensor,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
    else:
        ttnn_input_tensor = ttnn.from_torch(
            torch_input_tensor,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )

    torch_model = SegformerModel.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

    torch_model = torch_model.encoder.patch_embeddings[patch_emb_i]
    reference_model = SegformerOverlapPatchEmbeddings(
        patch_size=patch_size, stride=stride, num_channels=num_channels, hidden_size=hidden_size
    )
    sd = torch_model.state_dict()
    reference_model.load_state_dict(sd)
    reference_model.eval()
    torch_output = reference_model(torch_input_tensor)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(device), device=None
    )

    parameters.layer_norm.weight = ttnn.to_device(parameters.layer_norm.weight, device=device)
    parameters.layer_norm.bias = ttnn.to_device(parameters.layer_norm.bias, device=device)

    ttnn_model = TtSegformerOverlapPatchEmbeddings(
        parameters,
        patch_size=patch_size,
        stride=stride,
    )

    ttnn_output, height, width = ttnn_model(
        ttnn_input_tensor,
        parameters=parameters,
    )
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output[0], ttnn_output[0], pcc=0.99)
