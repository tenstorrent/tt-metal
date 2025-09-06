# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.segformer.common import load_torch_model
from models.demos.segformer.reference.segformer_overlap_patch_embeddings import SegformerOverlapPatchEmbeddings
from models.demos.segformer.tt.common import preprocess_layernorm_parameter
from models.demos.segformer.tt.ttnn_segformer_overlap_patch_embeddings import TtSegformerOverlapPatchEmbeddings
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc


def create_custom_mesh_preprocessor(mesh_mapper=None):
    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        return custom_preprocessor(model, name, mesh_mapper)

    def custom_preprocessor(model, name, mesh_mapper=None):
        parameters = {}
        if isinstance(model, SegformerOverlapPatchEmbeddings):
            parameters["proj"] = {}

            parameters["proj"]["weight"] = ttnn.from_torch(
                model.proj.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            parameters["proj"]["bias"] = ttnn.from_torch(
                torch.reshape(model.proj.bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            parameters["layer_norm"] = {}
            parameters["layer_norm"]["weight"] = preprocess_layernorm_parameter(
                model.layer_norm.weight, dtype=ttnn.bfloat8_b, mesh_mapper=mesh_mapper
            )
            parameters["layer_norm"]["bias"] = preprocess_layernorm_parameter(
                model.layer_norm.bias, dtype=ttnn.bfloat8_b, mesh_mapper=mesh_mapper
            )
        return parameters

    return custom_mesh_preprocessor


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
    model_location_generator,
):
    torch_input_tensor = torch.randn(batch_size, num_channels, height, width)

    reference_model = SegformerOverlapPatchEmbeddings(
        patch_size=patch_size, stride=stride, num_channels=num_channels, hidden_size=hidden_size
    )
    target_prefix = f"encoder.patch_embeddings.{patch_emb_i}"
    reference_model = load_torch_model(
        reference_model, target_prefix, module="semantic_sub", model_location_generator=model_location_generator
    )

    torch_output = reference_model(torch_input_tensor)
    _, weights_mesh_mapper, _ = get_mesh_mappers(device)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=None,
    )

    parameters.layer_norm.weight = ttnn.to_device(parameters.layer_norm.weight, device=device)
    parameters.layer_norm.bias = ttnn.to_device(parameters.layer_norm.bias, device=device)

    ttnn_model = TtSegformerOverlapPatchEmbeddings(
        parameters,
        patch_size=patch_size,
        stride=stride,
    )

    torch_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    CONV2D_MIN_CHANNEL_SIZE = 8
    # adjust padding if necessary
    if num_channels < CONV2D_MIN_CHANNEL_SIZE:
        ttnn_input_tensor = ttnn.pad(
            ttnn_input_tensor, [batch_size, height, width, CONV2D_MIN_CHANNEL_SIZE], [0, 0, 0, 0], 0
        )
    elif num_channels > CONV2D_MIN_CHANNEL_SIZE and num_channels % 32 != 0:
        ttnn_input_tensor = ttnn.pad(
            ttnn_input_tensor, [batch_size, height, width, (num_channels + 31) // 32 * 32], [0, 0, 0, 0], 0
        )

    ttnn_input_tensor = ttnn.to_device(ttnn_input_tensor, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

    ttnn_output, height, width = ttnn_model(
        device,
        ttnn_input_tensor,
        parameters=parameters,
    )
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output[0], ttnn_output[0], pcc=0.99)
