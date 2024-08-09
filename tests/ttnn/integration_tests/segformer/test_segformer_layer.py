# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn.model_preprocessing import preprocess_model
from tests.ttnn.utils_for_testing import assert_with_pcc

from transformers import SegformerModel
import pytest
from models.experimental.functional_segformer.tt.ttnn_segformer_layer import (
    TtSegformerLayer,
)
from models.experimental.functional_segformer.reference.segformer_dwconv import SegformerDWConv
from models.experimental.functional_segformer.reference.segformer_layer import SegformerLayer


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
):
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

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: reference_model,
        run_model=lambda model: model(torch_input_tensor, height=height, width=width),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtSegformerLayer(
        hidden_size, num_attention_heads, sequence_reduction_ratio, parameters, reference_model
    )

    ttnn_output = ttnn_model(
        ttnn_input_tensor,
        height,
        width,
        parameters=parameters,
        device=device,
    )
    ttnn_final_output = ttnn.to_torch(ttnn_output[0])

    assert_with_pcc(torch_output[0], ttnn_final_output, pcc=0.99)
