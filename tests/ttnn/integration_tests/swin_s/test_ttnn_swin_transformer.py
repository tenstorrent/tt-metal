# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torchvision import models
import pytest
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_grayskull
from models.experimental.swin_s.reference.swin_transformer import SwinTransformer
from models.experimental.swin_s.tt.tt_swin_transformer import TtSwinTransformer
from tests.ttnn.integration_tests.swin_s.test_ttnn_swin_transformer_block import (
    create_custom_mesh_preprocessor as create_custom_mesh_preprocessor_transformer_block,
)
from tests.ttnn.integration_tests.swin_s.test_ttnn_patchmerging import (
    create_custom_mesh_preprocessor as create_custom_mesh_preprocessor_patch_merging,
)
from models.experimental.swin_s.tt.common import get_mesh_mappers


def preprocess_attn_mask(input_shape, patch_size, window_size, shift_size, device, weights_mesh_mapper=None):
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
        attn_mask = ttnn.from_torch(attn_mask, device=device, layout=ttnn.TILE_LAYOUT, mesh_mapper=weights_mesh_mapper)
        attention_h = attention_h // 2
        attention_w = attention_w // 2

        attn_mask_tuple += (attn_mask,)

    return attn_mask_tuple


def preprocess_linear_weight(weight, *, dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=None):
    weight = weight.T.contiguous()
    weight = ttnn.from_torch(weight, dtype=dtype, layout=layout, mesh_mapper=mesh_mapper)
    return weight


def preprocess_linear_bias(bias, *, dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=None):
    bias = bias.reshape((1, -1))
    bias = ttnn.from_torch(bias, dtype=dtype, layout=layout, mesh_mapper=mesh_mapper)
    return bias


def preprocess_layernorm_parameter(parameter, *, dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=None):
    parameter = parameter.reshape((1, -1))
    parameter = ttnn.from_torch(parameter, dtype=dtype, layout=layout, mesh_mapper=mesh_mapper)
    return parameter


def custom_preprocessor(torch_model, name, mesh_mapper=None):
    parameters = {}
    if isinstance(torch_model, SwinTransformer):
        parameters["features"] = {}
        parameters["features"][0] = {}
        parameters["features"][0][0] = {}
        parameters["features"][0][2] = {}
        parameters["features"][0][0]["weight"] = ttnn.from_torch(
            torch_model.features[0][0].weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
        )
        parameters["features"][0][0]["bias"] = ttnn.from_torch(
            torch.reshape(torch_model.features[0][0].bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
        )
        parameters["features"][0][2]["weight"] = preprocess_layernorm_parameter(
            torch_model.features[0][2].weight, dtype=ttnn.bfloat8_b, mesh_mapper=mesh_mapper
        )
        parameters["features"][0][2]["bias"] = preprocess_layernorm_parameter(
            torch_model.features[0][2].bias, dtype=ttnn.bfloat8_b, mesh_mapper=mesh_mapper
        )

        tranformer_block_preprocessor = create_custom_mesh_preprocessor_transformer_block(mesh_mapper)
        patch_merging_preprocessor = create_custom_mesh_preprocessor_patch_merging(mesh_mapper)
        depths_list = [2, 2, 18, 2]
        for i_stage in range(len(depths_list)):
            index_list = [1, 3, 5, 7]
            parameters["features"][index_list[i_stage]] = {}
            for i_layer in range(depths_list[i_stage]):
                parameters["features"][index_list[i_stage]][i_layer] = {}
                parameters["features"][index_list[i_stage]][i_layer] = tranformer_block_preprocessor(
                    torch_model.features[index_list[i_stage]][i_layer], None
                )
        for i_patch_merging in range(2, 7, 2):
            parameters["features"][i_patch_merging] = {}
            parameters["features"][i_patch_merging] = patch_merging_preprocessor(
                torch_model.features[i_patch_merging], None
            )

        parameters["norm"] = {}
        parameters["head"] = {}
        parameters["norm"]["weight"] = preprocess_layernorm_parameter(
            torch_model.norm.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
        )
        parameters["norm"]["bias"] = preprocess_layernorm_parameter(
            torch_model.norm.bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
        )
        parameters["head"]["weight"] = preprocess_linear_weight(
            torch_model.head.weight, dtype=ttnn.bfloat8_b, mesh_mapper=mesh_mapper
        )
        parameters["head"]["bias"] = preprocess_linear_bias(
            torch_model.head.bias, dtype=ttnn.bfloat8_b, mesh_mapper=mesh_mapper
        )

    return parameters


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True, ids=["0"])
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        False,
        True,
    ],
    ids=[
        "pretrained_weight_false",
        "pretrained_weight_true",
    ],
)
def test_swin_s_transformer(device, use_pretrained_weight, reset_seeds):
    torch_model = SwinTransformer(
        patch_size=[4, 4], embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24], window_size=[7, 7]
    )

    if use_pretrained_weight:
        model = models.swin_s(weights="IMAGENET1K_V1")
        state_dict = model.state_dict()
        torch_model.load_state_dict(state_dict)

    torch_model.eval()

    # Input tensor for testing
    torch_input_tensor = torch.randn(1, 3, 512, 512)  # Sample input tensor
    torch_output_tensor = torch_model(torch_input_tensor)

    _, weights_mesh_mapper, _ = get_mesh_mappers(device)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=device,
    )

    attn_mask_tuple = preprocess_attn_mask(
        [1, 3, 512, 512], [4, 4], [7, 7], [3, 3], device, weights_mesh_mapper=weights_mesh_mapper
    )

    # Convert the model to TTNN
    ttnn_model = TtSwinTransformer(
        device,
        parameters,
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        attn_mask_tuple=attn_mask_tuple,
    )

    # Convert input tensor to TTNN format
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Apply TTNN model
    output_tensor = ttnn_model(input_tensor)

    # Convert output tensor back to Torch format
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(
        torch_output_tensor, output_tensor, pcc=0.96 if use_pretrained_weight else 0.99  # pcc=0.9611514804078299
    )  # The drop starts as we use shard MM in patch_mergig & mlp sub_module sub_module


def create_custom_mesh_preprocessor(mesh_mapper=None):
    def custom_mesh_preprocessor(model, name):
        return custom_preprocessor(model, name, mesh_mapper)

    return custom_mesh_preprocessor
