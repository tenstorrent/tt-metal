# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torchvision import models
from models.experimental.swin_s.reference.swin_transformer_block import SwinTransformerBlock
from models.experimental.swin_s.tt.tt_swin_transformer_block import TtSwinTransformerBlock
from tests.ttnn.utils_for_testing import assert_with_pcc
import ttnn
from models.utility_functions import skip_for_grayskull
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.integration_tests.swin_s.test_ttnn_shifted_window_attention import (
    create_custom_mesh_preprocessor as create_custom_mesh_preprocessor_shifted_window_attention,
)
from tests.ttnn.integration_tests.swin_s.test_ttnn_mlp import (
    create_custom_mesh_preprocessor as create_custom_mesh_preprocessor_mlp,
)
from models.experimental.swin_s.tt.common import get_mesh_mappers
import pytest


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


def preprocess_layernorm_parameter(parameter, *, dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=None):
    parameter = parameter.reshape((1, -1))
    parameter = ttnn.from_torch(parameter, dtype=dtype, layout=layout, mesh_mapper=mesh_mapper)
    return parameter


def custom_preprocessor(torch_model, name, mesh_mapper=None):
    parameters = {}
    if isinstance(torch_model, SwinTransformerBlock):
        parameters["norm1"] = {}
        parameters["norm1"]["weight"] = preprocess_layernorm_parameter(
            torch_model.norm1.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
        )
        parameters["norm1"]["bias"] = preprocess_layernorm_parameter(
            torch_model.norm1.bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
        )
        parameters["attn"] = {}
        shifted_window_attention_preprocessor = create_custom_mesh_preprocessor_shifted_window_attention(mesh_mapper)
        parameters["attn"] = shifted_window_attention_preprocessor(torch_model.attn, None)
        parameters["norm2"] = {}
        parameters["norm2"]["weight"] = preprocess_layernorm_parameter(
            torch_model.norm2.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
        )
        parameters["norm2"]["bias"] = preprocess_layernorm_parameter(
            torch_model.norm2.bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
        )
        parameters["mlp"] = {}
        mlp_preprocessor = create_custom_mesh_preprocessor_mlp(mesh_mapper)
        parameters["mlp"] = mlp_preprocessor(torch_model.mlp, None)

    return parameters


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
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
        (384, [7, 7], [3, 3], 12, 32, 5, 1),  # Low pcc after precomputing attn_mask outside,  check it
        (768, [7, 7], [0, 0], 24, 16, 7, 0),
        (768, [7, 7], [3, 3], 24, 16, 7, 1),  # Low pcc after precomputing attn_mask outside, check it
    ],
)
def test_swin_transformer_block(
    device, batch_size, dim, window_size, shift_size, num_heads, seq_len, i, j, reset_seeds
):
    model = models.swin_s(weights="IMAGENET1K_V1")
    state_dict = model.state_dict()
    ds_state_dict = {k: v for k, v in state_dict.items() if (k.startswith(f"features.{i}.{j}."))}

    attn_mask_tuple = preprocess_attn_mask([1, 3, 512, 512], [4, 4], [7, 7], [3, 3], device)

    if seq_len == 128:
        attn_mask = attn_mask_tuple[0]
    elif seq_len == 64:
        attn_mask = attn_mask_tuple[1]
    elif seq_len == 32:
        attn_mask = attn_mask_tuple[2]
    elif seq_len == 16:
        attn_mask = attn_mask_tuple[3]

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

    _, weights_mesh_mapper, _ = get_mesh_mappers(device)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=device,
    )

    ttnn_model = TtSwinTransformerBlock(
        device, parameters, dim, num_heads, window_size, shift_size, attn_mask=attn_mask
    )

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_output = ttnn_model(input_tensor)
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 0.99)


def create_custom_mesh_preprocessor(mesh_mapper=None):
    def custom_mesh_preprocessor(model, name):
        return custom_preprocessor(model, name, mesh_mapper)

    return custom_mesh_preprocessor
