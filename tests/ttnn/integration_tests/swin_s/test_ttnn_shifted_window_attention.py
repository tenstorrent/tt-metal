# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torchvision import models
import pytest
from models.utility_functions import skip_for_grayskull
from models.experimental.functional_swin_s.reference.shifted_window_attention import ShiftedWindowAttention
from models.experimental.functional_swin_s.tt.tt_shifted_window_attention import TtShiftedWindowAttention
from tests.ttnn.utils_for_testing import assert_with_pcc
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters, preprocess_linear_weight, preprocess_linear_bias


def create_custom_preprocessor(device):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}
        if isinstance(torch_model, ShiftedWindowAttention):
            parameters["qkv"] = {}
            parameters["proj"] = {}
            parameters["qkv"]["weight"] = preprocess_linear_weight(torch_model.qkv.weight, dtype=ttnn.bfloat8_b)
            parameters["qkv"]["bias"] = preprocess_linear_bias(torch_model.qkv.bias, dtype=ttnn.bfloat8_b)
            parameters["relative_position_bias"] = ttnn.from_torch(
                torch_model.get_relative_position_bias(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
            )
            parameters["proj"]["weight"] = preprocess_linear_weight(torch_model.proj.weight, dtype=ttnn.bfloat8_b)
            parameters["proj"]["bias"] = preprocess_linear_bias(torch_model.proj.bias, dtype=ttnn.bfloat8_b)

        return parameters

    return custom_preprocessor


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size",
    [
        1,
        8,
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
def test_shifted_window_attention(
    device, batch_size, dim, window_size, shift_size, num_heads, seq_len, i, j, reset_seeds
):
    model = models.swin_s(weights="IMAGENET1K_V1")
    state_dict = model.state_dict()
    ds_state_dict = {k: v for k, v in state_dict.items() if (k.startswith(f"features.{i}.{j}.attn."))}

    torch_model = ShiftedWindowAttention(dim, window_size, shift_size, num_heads)
    print(torch_model)

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

    ttnn_model = TtShiftedWindowAttention(parameters, device, dim, window_size, shift_size, num_heads)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_output = ttnn_model(input_tensor)

    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 0.99)
