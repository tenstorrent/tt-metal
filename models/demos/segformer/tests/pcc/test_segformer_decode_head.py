# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.common.utility_functions import skip_for_grayskull
from models.demos.segformer.common import load_config, load_torch_model
from models.demos.segformer.reference.segformer_decode_head import SegformerDecodeHead
from models.demos.segformer.tests.pcc.test_segformer_mlp import (
    create_custom_mesh_preprocessor as create_custom_preprocessor_mlp,
)
from models.demos.segformer.tt.common import fold_batch_norm2d_into_conv2d
from models.demos.segformer.tt.ttnn_segformer_decode_head import TtSegformerDecodeHead
from models.demos.utils.common_demo_utils import get_mesh_mappers
from tests.ttnn.utils_for_testing import assert_with_pcc


def create_custom_mesh_preprocessor(mesh_mapper=None):
    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        return custom_preprocessor(model, name, mesh_mapper)

    def custom_preprocessor(model, name, mesh_mapper=None):
        parameters = {}
        if isinstance(model, SegformerDecodeHead):
            parameters["linear_c"] = {}
            for i in range(4):
                parameters["linear_c"][i] = {}
                mlp_preprocess = create_custom_preprocessor_mlp(mesh_mapper=mesh_mapper)
                parameters["linear_c"][i] = mlp_preprocess(model.linear_c[i], None, None, None)

            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.linear_fuse, model.batch_norm)

            parameters["linear_fuse"] = {}
            parameters["linear_fuse"]["weight"] = ttnn.from_torch(
                conv_weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            parameters["linear_fuse"]["bias"] = ttnn.from_torch(
                torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

            parameters["classifier"] = {}
            parameters["classifier"]["weight"] = ttnn.from_torch(
                model.classifier.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            parameters["classifier"]["bias"] = ttnn.from_torch(
                torch.reshape(model.classifier.bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )

        return parameters

    return custom_mesh_preprocessor


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_segformer_decode_head(device, model_location_generator):
    batch_size = 1

    torch_input_tensor_0 = torch.randn(1, 32, 128, 128)
    torch_input_tensor_1 = torch.randn(1, 64, 64, 64)
    torch_input_tensor_2 = torch.randn(1, 160, 32, 32)
    torch_input_tensor_3 = torch.randn(1, 256, 16, 16)

    torch_input_tensor_0_folded = torch.reshape(torch_input_tensor_0, (batch_size, 32, 128 * 128))
    torch_input_tensor_0_folded = torch.permute(torch_input_tensor_0_folded, (0, 2, 1))
    torch_input_tensor_1_folded = torch.reshape(torch_input_tensor_1, (batch_size, 64, 64 * 64))
    torch_input_tensor_1_folded = torch.permute(torch_input_tensor_1_folded, (0, 2, 1))
    torch_input_tensor_2_folded = torch.reshape(torch_input_tensor_2, (batch_size, 160, 32 * 32))
    torch_input_tensor_2_folded = torch.permute(torch_input_tensor_2_folded, (0, 2, 1))
    torch_input_tensor_3_folded = torch.reshape(torch_input_tensor_3, (batch_size, 256, 16 * 16))
    torch_input_tensor_3_folded = torch.permute(torch_input_tensor_3_folded, (0, 2, 1))

    ttnn_input_tensor_0 = ttnn.from_torch(
        torch_input_tensor_0_folded,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn_input_tensor_1 = ttnn.from_torch(
        torch_input_tensor_1_folded,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn_input_tensor_2 = ttnn.from_torch(
        torch_input_tensor_2_folded,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn_input_tensor_3 = ttnn.from_torch(
        torch_input_tensor_3_folded,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    torch_input_tensor = (torch_input_tensor_0, torch_input_tensor_1, torch_input_tensor_2, torch_input_tensor_3)
    ttnn_input_tensor = (ttnn_input_tensor_0, ttnn_input_tensor_1, ttnn_input_tensor_2, ttnn_input_tensor_3)

    config = load_config("configs/segformer_semantic_config.json")
    reference_model = SegformerDecodeHead(config)
    target_prefix = f"decode_head."
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

    for i in range(4):
        parameters["linear_c"][i]["proj"]["weight"] = ttnn.to_device(
            parameters["linear_c"][i]["proj"]["weight"], device=device
        )
        parameters["linear_c"][i]["proj"]["bias"] = ttnn.to_device(
            parameters["linear_c"][i]["proj"]["bias"], device=device
        )

    ttnn_model = TtSegformerDecodeHead(config, parameters)
    ttnn_output = ttnn_model(device, ttnn_input_tensor, parameters)
    ttnn.deallocate(ttnn_input_tensor_0)

    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = torch.permute(ttnn_output, (0, 3, 1, 2))
    h = w = int(math.sqrt(ttnn_output.shape[-1]))
    ttnn_output = torch.reshape(ttnn_output, (ttnn_output.shape[0], ttnn_output.shape[1], h, w))

    print(torch_output.shape, ttnn_output.shape)
    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)
