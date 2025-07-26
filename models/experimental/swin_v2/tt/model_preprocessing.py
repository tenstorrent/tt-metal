# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.integration_tests.swin_v2.tests.test_ttnn_swin_v2_s import create_custom_preprocessor
from models.experimental.swin_v2.reference.patchmerging_v2 import PatchMergingV2
from models.experimental.swin_v2.reference.swin_transformer import SwinTransformer
from tests.ttnn.integration_tests.swin_v2.tests.test_ttnn_swin_transformer_block_v2 import (
    create_custom_preprocessor as create_custom_preprocessor_transformer_block_v2,
)
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_layernorm_parameter,
    preprocess_linear_bias,
)


def create_custom_preprocessor_patch_merging_v2(device):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}
        if isinstance(torch_model, PatchMergingV2):
            parameters["reduction"] = {}
            parameters["reduction"]["weight"] = preprocess_linear_weight(
                torch_model.reduction.weight, dtype=ttnn.bfloat16
            )
            parameters["norm"] = {}
            parameters["norm"]["weight"] = preprocess_layernorm_parameter(torch_model.norm.weight, dtype=ttnn.bfloat16)
            parameters["norm"]["bias"] = preprocess_layernorm_parameter(torch_model.norm.bias, dtype=ttnn.bfloat16)
        return parameters

    return custom_preprocessor


def create_custom_preprocessor(device):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}

        if isinstance(torch_model, SwinTransformer):
            parameters["features"] = {}
            parameters["features"][0] = {}
            parameters["features"][0][0] = {}
            parameters["features"][0][2] = {}
            parameters["features"][0][0]["weight"] = ttnn.from_torch(
                torch_model.features[0][0].weight, dtype=ttnn.bfloat16
            )
            parameters["features"][0][0]["bias"] = ttnn.from_torch(
                torch.reshape(torch_model.features[0][0].bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
            )
            parameters["features"][0][2]["weight"] = preprocess_layernorm_parameter(
                torch_model.features[0][2].weight, dtype=ttnn.bfloat8_b
            )
            parameters["features"][0][2]["bias"] = preprocess_layernorm_parameter(
                torch_model.features[0][2].bias, dtype=ttnn.bfloat8_b
            )

            tranformer_block_preprocessor = create_custom_preprocessor_transformer_block_v2(device)

            patch_merging_preprocessor = create_custom_preprocessor_patch_merging_v2(device)
            depths_list = [2, 2, 18, 2]
            for i_stage in range(len(depths_list)):
                index_list = [1, 3, 5, 7]
                parameters["features"][index_list[i_stage]] = {}
                for i_layer in range(depths_list[i_stage]):
                    parameters["features"][index_list[i_stage]][i_layer] = {}
                    parameters["features"][index_list[i_stage]][i_layer] = tranformer_block_preprocessor(
                        torch_model.features[index_list[i_stage]][i_layer], None, None
                    )
            for i_patch_merging in range(2, 7, 2):
                parameters["features"][i_patch_merging] = {}
                parameters["features"][i_patch_merging] = patch_merging_preprocessor(
                    torch_model.features[i_patch_merging], None, None
                )

            parameters["norm"] = {}
            parameters["head"] = {}
            parameters["norm"]["weight"] = preprocess_layernorm_parameter(torch_model.norm.weight, dtype=ttnn.bfloat16)
            parameters["norm"]["bias"] = preprocess_layernorm_parameter(torch_model.norm.bias, dtype=ttnn.bfloat16)
            parameters["head"]["weight"] = preprocess_linear_weight(torch_model.head.weight, dtype=ttnn.bfloat8_b)
            parameters["head"]["bias"] = preprocess_linear_bias(torch_model.head.bias, dtype=ttnn.bfloat8_b)

        return parameters

    return custom_preprocessor


def create_swinv2_model_parameters(torch_model, device):
    # model = models.swin_v2_s(weights="IMAGENET1K_V1")
    # state_dict = model.state_dict()

    # torch_model.load_state_dict(state_dict)
    # torch_model.eval()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(device), device=device
    )
    return parameters


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
