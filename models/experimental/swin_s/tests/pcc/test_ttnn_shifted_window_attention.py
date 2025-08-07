# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import ttnn
from models.utility_functions import skip_for_grayskull
from models.experimental.swin_s.reference.shifted_window_attention import ShiftedWindowAttention
from models.experimental.swin_s.tt.tt_shifted_window_attention import TtShiftedWindowAttention
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters, preprocess_linear_weight, preprocess_linear_bias
from models.experimental.swin_s.common import load_torch_model, SWIN_S_L1_SMALL_SIZE


def preprocess_attn_mask(input_shape, patch_size, window_size, shift_size, device):
    h, w = input_shape[2], input_shape[3]
    attention_h = ((h - (patch_size[0] - 1) - 1) // patch_size[0]) + 1
    attention_w = ((w - (patch_size[0] - 1) - 1) // patch_size[0]) + 1
    attn_mask_tuple = ()
    for i in range(4):
        pad_r = (window_size[1] - attention_w % window_size[1]) % window_size[1]
        pad_b = (window_size[0] - attention_h % window_size[0]) % window_size[0]
        pad_H = attention_h + pad_b
        pad_W = attention_w + pad_r
        num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])

        attn_mask = torch.zeros((pad_H, pad_W))
        h_slices = (
            (0, -window_size[0]),
            (-window_size[0], -shift_size[0]),
            (-shift_size[0], None),
        )
        w_slices = (
            (0, -window_size[1]),
            (-window_size[1], -shift_size[1]),
            (-shift_size[1], None),
        )
        count = 0
        for h in h_slices:
            for w in w_slices:
                attn_mask[h[0] : h[1], w[0] : w[1]] = count
                count += 1
        attn_mask = attn_mask.view(pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1])
        attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, window_size[0] * window_size[1])
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(0)
        attn_mask = ttnn.from_torch(attn_mask, device=device, layout=ttnn.TILE_LAYOUT)
        attention_h = attention_h // 2
        attention_w = attention_w // 2

        attn_mask_tuple += (attn_mask,)

    return attn_mask_tuple


def create_custom_preprocessor(device):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}
        if isinstance(torch_model, ShiftedWindowAttention):
            parameters["qkv"] = {}
            parameters["proj"] = {}
            parameters["qkv"]["weight"] = preprocess_linear_weight(torch_model.qkv.weight, dtype=ttnn.bfloat16)
            parameters["qkv"]["bias"] = preprocess_linear_bias(torch_model.qkv.bias, dtype=ttnn.bfloat16)
            parameters["relative_position_bias"] = ttnn.from_torch(
                torch_model.get_relative_position_bias(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )
            parameters["proj"]["weight"] = preprocess_linear_weight(torch_model.proj.weight, dtype=ttnn.bfloat16)
            parameters["proj"]["bias"] = preprocess_linear_bias(torch_model.proj.bias, dtype=ttnn.bfloat16)

        return parameters

    return custom_preprocessor


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": SWIN_S_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize(
    "batch_size",
    [
        1,
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
    device, batch_size, dim, window_size, shift_size, num_heads, seq_len, i, j, reset_seeds, model_location_generator
):
    torch_model = ShiftedWindowAttention(dim, window_size, shift_size, num_heads)

    attn_mask_tuple = preprocess_attn_mask([1, 3, 512, 512], [4, 4], [7, 7], [3, 3], device)

    if seq_len == 128:
        attn_mask = attn_mask_tuple[0]
    elif seq_len == 64:
        attn_mask = attn_mask_tuple[1]
    elif seq_len == 32:
        attn_mask = attn_mask_tuple[2]
    elif seq_len == 16:
        attn_mask = attn_mask_tuple[3]

    torch_model = load_torch_model(
        torch_model=torch_model, i=i, j=j, module="attention", model_location_generator=model_location_generator
    )

    torch_input_tensor = torch.randn(batch_size, seq_len, seq_len, dim)
    torch_output = torch_model(torch_input_tensor)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(device), device=device
    )

    ttnn_model = TtShiftedWindowAttention(
        parameters, device, dim, window_size, shift_size, num_heads, attn_mask=attn_mask
    )

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_output = ttnn_model(input_tensor)

    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 0.99)
