# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torchvision import models
from models.experimental.functional_swin_s.reference.swin_transformer_block import SwinTransformerBlock
from models.experimental.functional_swin_s.tt.tt_swin_transformer_block import TtSwinTransformerBlock
from tests.ttnn.utils_for_testing import assert_with_pcc
import ttnn
from models.utility_functions import skip_for_grayskull
from ttnn.model_preprocessing import preprocess_model_parameters, preprocess_layernorm_parameter
from tests.ttnn.integration_tests.swin_s.test_ttnn_shifted_window_attention import (
    create_custom_preprocessor as create_custom_preprocessor_shifted_window_attention,
)
from tests.ttnn.integration_tests.swin_s.test_ttnn_mlp import (
    create_custom_preprocessor as create_custom_preprocessor_mlp,
)
import pytest


def create_custom_preprocessor(device):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}
        if isinstance(torch_model, SwinTransformerBlock):
            parameters["norm1"] = {}
            parameters["norm1"]["weight"] = preprocess_layernorm_parameter(
                torch_model.norm1.weight, dtype=ttnn.bfloat8_b
            )
            parameters["norm1"]["bias"] = preprocess_layernorm_parameter(torch_model.norm1.bias, dtype=ttnn.bfloat8_b)
            parameters["attn"] = {}
            shifted_window_attention_preprocessor = create_custom_preprocessor_shifted_window_attention(device)
            parameters["attn"] = shifted_window_attention_preprocessor(torch_model.attn, None, None)
            parameters["norm2"] = {}
            parameters["norm2"]["weight"] = preprocess_layernorm_parameter(
                torch_model.norm2.weight, dtype=ttnn.bfloat8_b
            )
            parameters["norm2"]["bias"] = preprocess_layernorm_parameter(torch_model.norm2.bias, dtype=ttnn.bfloat8_b)
            parameters["mlp"] = {}
            mlp_preprocessor = create_custom_preprocessor_mlp(device)
            parameters["mlp"] = mlp_preprocessor(torch_model.mlp, None, None)

        return parameters

    return custom_preprocessor


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size",
    [
        1,
        # 8, #MM shard is given for bs=1 alone in the pipeline
    ],
)
@pytest.mark.parametrize(
    "dim,window_size,shift_size,num_heads,seq_len,i,j",
    [
        (96, [7, 7], [0, 0], 3, 128, 1, 0),
        (96, [7, 7], [3, 3], 3, 128, 1, 1),
        (192, [7, 7], [0, 0], 6, 64, 3, 0),
        (192, [7, 7], [3, 3], 6, 64, 3, 1),
        (384, [7, 7], [0, 0], 12, 32, 5, 0),
        (384, [7, 7], [3, 3], 12, 32, 5, 1),
        (768, [7, 7], [0, 0], 24, 16, 7, 0),
        (768, [7, 7], [3, 3], 24, 16, 7, 1),
    ],
)
def test_swin_transformer_block(
    device, batch_size, dim, window_size, shift_size, num_heads, seq_len, i, j, reset_seeds
):
    model = models.swin_s(weights="IMAGENET1K_V1")
    state_dict = model.state_dict()
    ds_state_dict = {k: v for k, v in state_dict.items() if (k.startswith(f"features.{i}.{j}."))}

    torch_model = SwinTransformerBlock(dim, num_heads, window_size, shift_size)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input_tensor = torch.randn(batch_size, seq_len, seq_len, dim)
    torch_output = torch_model(torch_input_tensor)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(device), device=device
    )

    ttnn_model = TtSwinTransformerBlock(device, parameters, dim, num_heads, window_size, shift_size)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_output = ttnn_model(input_tensor)
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 0.99)
