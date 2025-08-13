# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.segformer.common import load_config, load_torch_model
from models.demos.segformer.reference.segformer_attention import SegformerAttention
from models.demos.segformer.tests.pcc.test_segformer_efficient_selfattention import (
    create_custom_mesh_preprocessor as create_customer_preprocessor_selfattention,
)
from models.demos.segformer.tests.pcc.test_segformer_selfoutput import (
    create_custom_mesh_preprocessor as create_customer_preprocessor_selfoutput,
)
from models.demos.segformer.tt.ttnn_segformer_attention import TtSegformerAttention
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc


def create_custom_mesh_preprocessor(mesh_mapper=None):
    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        return custom_preprocessor(model, name, mesh_mapper)

    def custom_preprocessor(model, name, mesh_mapper):
        parameters = {}
        if isinstance(model, SegformerAttention):
            parameters["self"] = {}
            self_attention_prepocessor = create_customer_preprocessor_selfattention(mesh_mapper)
            parameters["self"] = self_attention_prepocessor(model.self, None, None, None)

            parameters["output"] = {}
            self_output_prepocessor = create_customer_preprocessor_selfoutput(mesh_mapper)
            parameters["output"] = self_output_prepocessor(model.output, None, None, None)

        return parameters

    return custom_mesh_preprocessor


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "hidden_size, num_attention_heads, sequence_reduction_ratio, batch_size, seq_len, height, width, block_i, attention_i",
    [
        (32, 1, 8, 1, 16384, 128, 128, 0, 0),
        (32, 1, 8, 1, 16384, 128, 128, 0, 1),
        (64, 2, 4, 1, 4096, 64, 64, 1, 0),
        (64, 2, 4, 1, 4096, 64, 64, 1, 1),
        (160, 5, 2, 1, 1024, 32, 32, 2, 0),
        (160, 5, 2, 1, 1024, 32, 32, 2, 1),
        (256, 8, 1, 1, 256, 16, 16, 3, 0),
        (256, 8, 1, 1, 256, 16, 16, 3, 1),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_segformer_attention(
    device,
    hidden_size,
    num_attention_heads,
    sequence_reduction_ratio,
    batch_size,
    seq_len,
    height,
    width,
    block_i,
    attention_i,
    model_location_generator,
):
    torch_input_tensor = torch.randn(batch_size, 1, seq_len, hidden_size)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    target_prefix = f"segformer.encoder.block.{block_i}.{attention_i}.attention."

    config = load_config("configs/segformer_semantic_config.json")
    reference_model = SegformerAttention(
        config,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        sequence_reduction_ratio=sequence_reduction_ratio,
    )

    reference_model = load_torch_model(
        reference_model, target_prefix, module="semantic_sub", model_location_generator=model_location_generator
    )

    torch_input_tensor = torch.reshape(torch_input_tensor, (batch_size, seq_len, hidden_size))
    output = reference_model(torch_input_tensor, height, width)
    _, weights_mesh_mapper, _ = get_mesh_mappers(device)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=device,
    )
    if "sr" in parameters["self"]:
        parameters["self"]["sr"]["weight"] = ttnn.from_device(parameters["self"]["sr"]["weight"])
        parameters["self"]["sr"]["bias"] = ttnn.from_device(parameters["self"]["sr"]["bias"])

    ttnn_model = TtSegformerAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        parameters=parameters,
        sequence_reduction_ratio=sequence_reduction_ratio,
    )

    ttnn_output = ttnn_model(device, ttnn_input_tensor, height, width, parameters=parameters)
    ttnn_final_output = ttnn.to_torch(ttnn_output[0])
    if len(ttnn_final_output.shape) == 4:
        ttnn_final_output = ttnn_final_output[0]

    assert_with_pcc(output[0], ttnn_final_output, pcc=0.986)
