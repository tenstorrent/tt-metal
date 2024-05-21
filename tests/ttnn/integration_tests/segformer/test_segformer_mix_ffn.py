# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn.model_preprocessing import preprocess_model
from tests.ttnn.utils_for_testing import assert_with_pcc
from transformers import SegformerModel

import pytest
from models.experimental.functional_segformer.tt.ttnn_segformer_mix_ffn import TtSegformerMixFFN
from models.experimental.functional_segformer.reference.segformer_dwconv import SegformerDWConv
from models.experimental.functional_segformer.reference.segformer_mixffn import SegformerMixFFN


def update_ttnn_module_args(ttnn_module_args):
    ttnn_module_args["use_1d_systolic_array"] = ttnn_module_args.in_channels < 256


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, SegformerDWConv):
            parameters["dwconv"] = {}
            parameters["dwconv"]["weight"] = model.dwconv.weight
            parameters["dwconv"]["bias"] = model.dwconv.bias

        return parameters

    return custom_preprocessor


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
):
    torch_input_tensor = torch.randn(batch_size, seq_len, in_features)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat8_b,
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

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: reference_model,
        run_model=lambda model: model(torch_input_tensor, height, width),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtSegformerMixFFN(parameters, reference_model)

    ttnn_output = ttnn_model(ttnn_input_tensor, height=height, width=width, parameters=parameters, device=device)
    ttnn_output = ttnn.from_device(ttnn_output)
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)
