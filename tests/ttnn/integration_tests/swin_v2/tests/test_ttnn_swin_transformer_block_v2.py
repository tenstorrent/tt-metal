# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from torchvision import models
from models.experimental.swin_v2.reference.swin_transformer_block_v2 import SwinTransformerBlockV2
from models.experimental.swin_v2.reference.shifted_window_attention_v2 import ShiftedWindowAttentionV2
from models.experimental.swin_v2.tt.tt_swin_transformer_block_v2 import TtSwinTransformerBlockV2
from tests.ttnn.utils_for_testing import assert_with_pcc
import ttnn
from models.utility_functions import skip_for_grayskull
from ttnn.model_preprocessing import preprocess_model_parameters, preprocess_layernorm_parameter
from ttnn.model_preprocessing import preprocess_model_parameters, preprocess_linear_weight, preprocess_linear_bias
from tests.ttnn.integration_tests.swin_v2.tests.test_ttnn_mlp import (
    create_custom_preprocessor as create_custom_preprocessor_mlp,
)
import pytest


def create_custom_preprocessor_shifted_window_attention_v2(device):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}
        if isinstance(torch_model, ShiftedWindowAttentionV2):
            parameters["qkv"] = {}
            parameters["proj"] = {}
            parameters["qkv"]["weight"] = preprocess_linear_weight(torch_model.qkv.weight, dtype=ttnn.bfloat16)
            parameters["qkv"]["bias"] = preprocess_linear_bias(torch_model.qkv.bias, dtype=ttnn.bfloat16)
            parameters["relative_position_bias"] = ttnn.from_torch(
                torch_model.get_relative_position_bias(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )
            parameters["logit_scale"] = ttnn.from_torch(
                torch_model.logit_scale, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )
            parameters["proj"]["weight"] = preprocess_linear_weight(torch_model.proj.weight, dtype=ttnn.bfloat16)
            parameters["proj"]["bias"] = preprocess_linear_bias(torch_model.proj.bias, dtype=ttnn.bfloat16)

        return parameters

    return custom_preprocessor


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
        if isinstance(torch_model, SwinTransformerBlockV2):
            parameters["norm1"] = {}
            parameters["norm1"]["weight"] = preprocess_layernorm_parameter(
                torch_model.norm1.weight, dtype=ttnn.bfloat16
            )
            parameters["norm1"]["bias"] = preprocess_layernorm_parameter(torch_model.norm1.bias, dtype=ttnn.bfloat16)
            parameters["attn"] = {}
            shifted_window_attention_preprocessor = create_custom_preprocessor_shifted_window_attention_v2(device)
            parameters["attn"] = shifted_window_attention_preprocessor(torch_model.attn, None, None)
            parameters["norm2"] = {}
            parameters["norm2"]["weight"] = preprocess_layernorm_parameter(
                torch_model.norm2.weight, dtype=ttnn.bfloat16
            )
            parameters["norm2"]["bias"] = preprocess_layernorm_parameter(torch_model.norm2.bias, dtype=ttnn.bfloat16)
            parameters["mlp"] = {}
            mlp_preprocessor = create_custom_preprocessor_mlp(device)
            parameters["mlp"] = mlp_preprocessor(torch_model.mlp, None, None)

        return parameters

    return custom_preprocessor


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "dim,shift_size,num_heads,i,j, attn_mask, input_shape",
    [
        (96, [0, 0], 3, 1, 0, [1, 256, 1, 64, 64], [1, 128, 128, 96]),
        (96, [4, 4], 3, 1, 1, [1, 256, 1, 64, 64], [1, 128, 128, 96]),
        (192, [0, 0], 6, 3, 0, [1, 64, 1, 64, 64], [1, 64, 64, 192]),
        (192, [4, 4], 6, 3, 1, [1, 64, 1, 64, 64], [1, 64, 64, 192]),
        (384, [0, 0], 12, 5, 0, [1, 16, 1, 64, 64], [1, 32, 32, 384]),
        (384, [4, 4], 12, 5, 1, [1, 16, 1, 64, 64], [1, 32, 32, 384]),
    ],
)
def test_swin_transformer_block_v2(device, dim, shift_size, num_heads, i, j, attn_mask, input_shape, reset_seeds):
    model = models.swin_v2_s(weights="IMAGENET1K_V1")
    state_dict = model.state_dict()
    ds_state_dict = {k: v for k, v in state_dict.items() if (k.startswith(f"features.{i}.{j}."))}

    attn_mask = torch.rand(attn_mask)
    torch_input_tensor = torch.rand(input_shape)
    torch_model = SwinTransformerBlockV2(dim, num_heads, [8, 8], shift_size)

    new_state_dict = {}
    new_torch_state_dic = {}
    for k, v in ds_state_dict.items():
        if "cbp_mlp" not in k:
            new_state_dict[k] = ds_state_dict[k]
        new_torch_state_dic[k.replace(f"features.{i}.{j}.", "")] = ds_state_dict[k]

    torch_model.load_state_dict(new_torch_state_dic)
    torch_model.eval()

    torch_output = torch_model(torch_input_tensor)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(device), device=device
    )
    attn_mask = ttnn.from_torch(attn_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_model = TtSwinTransformerBlockV2(
        device=device,
        parameters=parameters,
        dim=dim,
        num_heads=num_heads,
        window_size=[8, 8],
        shift_size=shift_size,
        attn_mask=attn_mask,
    )

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_output = ttnn_model(input_tensor)
    tt_output = ttnn.to_torch(tt_output)
    pcc = 0.99
    if i == 1 and j == 0:
        pcc = 0.98
    assert_with_pcc(torch_output, tt_output, pcc)
