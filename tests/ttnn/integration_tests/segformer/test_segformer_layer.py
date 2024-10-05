# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_layernorm_parameter,
    ParameterDict,
    ParameterList,
)
from tests.ttnn.utils_for_testing import assert_with_pcc

from transformers import SegformerModel
import pytest
from models.experimental.functional_segformer.tt.ttnn_segformer_layer import (
    TtSegformerLayer,
)

from models.experimental.functional_segformer.reference.segformer_layer import SegformerLayer
from tests.ttnn.integration_tests.segformer.test_segformer_mix_ffn import (
    create_custom_preprocessor as create_custom_preprocessor_mix_ffn,
)
from tests.ttnn.integration_tests.segformer.test_segformer_attention import (
    create_custom_preprocessor as create_custom_preprocessor_attention,
)
from models.utility_functions import skip_for_grayskull


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, SegformerLayer):
            parameters["layer_norm_1"] = {}
            parameters["layer_norm_1"]["weight"] = preprocess_layernorm_parameter(
                model.layer_norm_1.weight, dtype=ttnn.bfloat16
            )
            parameters["layer_norm_1"]["bias"] = preprocess_layernorm_parameter(
                model.layer_norm_1.bias, dtype=ttnn.bfloat16
            )

            attention_preprocess = create_custom_preprocessor_attention(device)
            parameters["attention"] = {}
            parameters["attention"] = attention_preprocess(model.attention, None, None)

            mix_ffn_preprocess = create_custom_preprocessor_mix_ffn(device)
            parameters["mlp"] = {}
            parameters["mlp"] = mix_ffn_preprocess(model.mlp, None, None)

            parameters["layer_norm_2"] = {}
            parameters["layer_norm_2"]["weight"] = preprocess_layernorm_parameter(
                model.layer_norm_2.weight, dtype=ttnn.bfloat16
            )
            parameters["layer_norm_2"]["bias"] = preprocess_layernorm_parameter(
                model.layer_norm_2.bias, dtype=ttnn.bfloat16
            )
        return parameters

    return custom_preprocessor


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
    reset_seeds,
    is_ci_env,
):
    if is_ci_env:
        pytest.skip("Skip in CI, model is WIP, issue# 13357")

    torch_input_tensor = torch.randn(batch_size, seq_len, hidden_size)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    torch_model = SegformerModel.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    config = torch_model.config

    torch_model = torch_model.encoder.block[block_i][segformer_i]

    reference_model = SegformerLayer(
        config=config,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        sequence_reduction_ratio=sequence_reduction_ratio,
        mlp_ratio=mlp_ratio,
    )

    sd = torch_model.state_dict()
    reference_model.load_state_dict(sd)
    reference_model.eval()

    torch_output = reference_model(torch_input_tensor, height=height, width=width)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(device), device=None
    )

    parameters = move_to_device(parameters, device)

    ttnn_model = TtSegformerLayer(hidden_size, num_attention_heads, sequence_reduction_ratio, parameters, mlp_ratio)

    ttnn_output = ttnn_model(
        ttnn_input_tensor,
        height,
        width,
        parameters=parameters,
        device=device,
    )
    ttnn_final_output = ttnn.to_torch(ttnn_output[0])
    if len(ttnn_final_output.shape) == 4:
        ttnn_final_output = ttnn_final_output[0]

    assert_with_pcc(torch_output[0], ttnn_final_output, pcc=0.94)
