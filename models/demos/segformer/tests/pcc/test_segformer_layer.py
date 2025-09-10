# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from ttnn.model_preprocessing import ParameterDict, ParameterList, preprocess_model_parameters

import ttnn
from models.demos.segformer.common import load_config, load_torch_model
from models.demos.segformer.reference.segformer_layer import SegformerLayer
from models.demos.segformer.tests.pcc.test_segformer_attention import (
    create_custom_mesh_preprocessor as create_custom_preprocessor_attention,
)
from models.demos.segformer.tests.pcc.test_segformer_mix_ffn import (
    create_custom_mesh_preprocessor as create_custom_preprocessor_mix_ffn,
)
from models.demos.segformer.tt.common import preprocess_layernorm_parameter
from models.demos.segformer.tt.ttnn_segformer_layer import TtSegformerLayer
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc


def create_custom_mesh_preprocessor(mesh_mapper=None):
    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        return custom_preprocessor(model, name, mesh_mapper)

    def custom_preprocessor(model, name, mesh_mapper=None):
        parameters = {}
        if isinstance(model, SegformerLayer):
            parameters["layer_norm_1"] = {}
            parameters["layer_norm_1"]["weight"] = preprocess_layernorm_parameter(
                model.layer_norm_1.weight, dtype=ttnn.bfloat8_b, mesh_mapper=mesh_mapper
            )
            parameters["layer_norm_1"]["bias"] = preprocess_layernorm_parameter(
                model.layer_norm_1.bias, dtype=ttnn.bfloat8_b, mesh_mapper=mesh_mapper
            )

            attention_preprocess = create_custom_preprocessor_attention(mesh_mapper)
            parameters["attention"] = {}
            parameters["attention"] = attention_preprocess(model.attention, None, None, None)

            mix_ffn_preprocess = create_custom_preprocessor_mix_ffn(mesh_mapper)
            parameters["mlp"] = {}
            parameters["mlp"] = mix_ffn_preprocess(model.mlp, None, None, None)

            parameters["layer_norm_2"] = {}
            parameters["layer_norm_2"]["weight"] = preprocess_layernorm_parameter(
                model.layer_norm_2.weight, dtype=ttnn.bfloat8_b, mesh_mapper=mesh_mapper
            )
            parameters["layer_norm_2"]["bias"] = preprocess_layernorm_parameter(
                model.layer_norm_2.bias, dtype=ttnn.bfloat8_b, mesh_mapper=mesh_mapper
            )
        return parameters

    return custom_mesh_preprocessor


def move_to_device(object, device):
    if isinstance(object, ParameterDict):
        for name, value in list(object.items()):
            if name in ["sr", "proj", "dwconv"]:
                continue
            object[name] = move_to_device(value, device)
        return object
    elif isinstance(object, ParameterList):
        for index, element in enumerate(object):
            object[index] = move_to_device(element, device)
        return object
    elif isinstance(object, ttnn.Tensor):
        return ttnn.to_device(object, device)
    else:
        return object


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size, height, width, num_attention_heads, drop_path, sequence_reduction_ratio, mlp_ratio, block_i, segformer_i",
    [
        (1, 16384, 32, 128, 128, 1, 0, 8, 4, 0, 0),
        (1, 16384, 32, 128, 128, 1, 0, 8, 4, 0, 1),
        (1, 4096, 64, 64, 64, 2, 0, 4, 4, 1, 0),
        (1, 4096, 64, 64, 64, 2, 0, 4, 4, 1, 1),
        (1, 1024, 160, 32, 32, 5, 0, 2, 4, 2, 0),
        (1, 1024, 160, 32, 32, 5, 0, 2, 4, 2, 1),
        (1, 256, 256, 16, 16, 8, 0, 1, 4, 3, 0),
        (1, 256, 256, 16, 16, 8, 0, 1, 4, 3, 1),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_segformer_layer(
    batch_size,
    seq_len,
    hidden_size,
    height,
    width,
    num_attention_heads,
    drop_path,
    sequence_reduction_ratio,
    mlp_ratio,
    block_i,
    segformer_i,
    device,
    model_location_generator,
):
    torch_input_tensor = torch.randn(batch_size, 1, seq_len, hidden_size)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    config = load_config("configs/segformer_semantic_config.json")
    reference_model = SegformerLayer(
        config=config,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        sequence_reduction_ratio=sequence_reduction_ratio,
        mlp_ratio=mlp_ratio,
    )
    target_prefix = f"encoder.block.{block_i}.{segformer_i}."
    reference_model = load_torch_model(
        reference_model, target_prefix, module="semantic_sub", model_location_generator=model_location_generator
    )

    torch_input_tensor = torch.reshape(torch_input_tensor, (batch_size, seq_len, hidden_size))
    torch_output = reference_model(torch_input_tensor, height=height, width=width)
    _, weights_mesh_mapper, _ = get_mesh_mappers(device)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=None,
    )

    parameters = move_to_device(parameters, device)

    ttnn_model = TtSegformerLayer(hidden_size, num_attention_heads, sequence_reduction_ratio, parameters, mlp_ratio)

    ttnn_output = ttnn_model(
        device,
        ttnn_input_tensor,
        height,
        width,
        parameters=parameters,
    )
    ttnn_final_output = ttnn.to_torch(ttnn_output[0])
    if len(ttnn_final_output.shape) == 4:
        ttnn_final_output = ttnn_final_output[0]

    assert_with_pcc(torch_output[0], ttnn_final_output, pcc=0.99)
