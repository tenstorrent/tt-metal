# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_bias,
    preprocess_linear_weight,
)
from tests.ttnn.utils_for_testing import assert_with_pcc
from transformers import SegformerModel

import pytest
from models.experimental.functional_segformer.tt.ttnn_segformer_mix_ffn import TtSegformerMixFFN
from models.experimental.functional_segformer.reference.segformer_mixffn import SegformerMixFFN
from tests.ttnn.integration_tests.segformer.test_segformer_dwconv import (
    create_custom_preprocessor as create_custom_preprocessor_dwconv,
)
from models.utility_functions import skip_for_grayskull


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, SegformerMixFFN):
            parameters["dense1"] = {}
            parameters["dense1"]["weight"] = preprocess_linear_weight(model.dense1.weight, dtype=ttnn.bfloat8_b)
            parameters["dense1"]["bias"] = preprocess_linear_bias(model.dense1.bias, dtype=ttnn.bfloat8_b)

            parameters["dwconv"] = {}
            dwconv_preprocessor = create_custom_preprocessor_dwconv(device)
            parameters["dwconv"] = dwconv_preprocessor(model.dwconv, None, None)

            parameters["dense2"] = {}
            parameters["dense2"]["weight"] = preprocess_linear_weight(model.dense2.weight, dtype=ttnn.bfloat8_b)
            parameters["dense2"]["bias"] = preprocess_linear_bias(model.dense2.bias, dtype=ttnn.bfloat8_b)

        return parameters

    return custom_preprocessor


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "in_features, hidden_features, out_features, batch_size, seq_len, height, width, block_i, mixffn_i",
    [
        (32, 128, 32, 1, 16384, 128, 128, 0, 0),
        (32, 128, 32, 1, 16384, 128, 128, 0, 1),
        (64, 256, 64, 1, 4096, 64, 64, 1, 0),
        (64, 256, 64, 1, 4096, 64, 64, 1, 1),
        (160, 640, 160, 1, 1024, 32, 32, 2, 0),
        (160, 640, 160, 1, 1024, 32, 32, 2, 1),
        (256, 1024, 256, 1, 256, 16, 16, 3, 0),
        (256, 1024, 256, 1, 256, 16, 16, 3, 1),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_segformer_mix_ffn(
    device,
    in_features,
    hidden_features,
    out_features,
    batch_size,
    seq_len,
    height,
    width,
    block_i,
    mixffn_i,
    reset_seeds,
    is_ci_env,
):
    if is_ci_env:
        pytest.skip("Skip in CI, model is WIP, issue# 13357")

    torch_input_tensor = torch.randn(batch_size, seq_len, in_features)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    torch_model = SegformerModel.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    config = torch_model.config
    torch_model = torch_model.encoder.block[block_i][mixffn_i].mlp

    reference_model = SegformerMixFFN(
        config=config, in_features=in_features, hidden_features=hidden_features, out_features=out_features
    )

    sd = torch_model.state_dict()
    reference_model.load_state_dict(sd)

    reference_model.eval()

    torch_output = reference_model(torch_input_tensor, height, width)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(device), device=None
    )
    parameters["dense1"]["weight"] = ttnn.to_device(parameters["dense1"]["weight"], device=device)
    parameters["dense1"]["bias"] = ttnn.to_device(parameters["dense1"]["bias"], device=device)
    parameters["dense2"]["weight"] = ttnn.to_device(parameters["dense2"]["weight"], device=device)
    parameters["dense2"]["bias"] = ttnn.to_device(parameters["dense2"]["bias"], device=device)

    ttnn_model = TtSegformerMixFFN(parameters, hidden_features)

    ttnn_output = ttnn_model(ttnn_input_tensor, height=height, width=width, parameters=parameters, device=device)
    ttnn_output = ttnn.from_device(ttnn_output)
    ttnn_output = ttnn.to_torch(ttnn_output)[0]

    assert_with_pcc(torch_output, ttnn_output, pcc=0.96)
