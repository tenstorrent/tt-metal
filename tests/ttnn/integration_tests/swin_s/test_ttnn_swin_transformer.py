# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torchvision import models
import pytest
import ttnn
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_layernorm_parameter,
    preprocess_linear_bias,
)
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_grayskull
from models.experimental.swin_s.reference.swin_transformer import SwinTransformer
from models.experimental.swin_s.tt.tt_swin_transformer import TtSwinTransformer
from tests.ttnn.integration_tests.swin_s.test_ttnn_swin_transformer_block import (
    create_custom_preprocessor as create_custom_preprocessor_transformer_block,
)
from tests.ttnn.integration_tests.swin_s.test_ttnn_patchmerging import (
    create_custom_preprocessor as create_custom_preprocessor_patch_merging,
)


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

            tranformer_block_preprocessor = create_custom_preprocessor_transformer_block(device)
            patch_merging_preprocessor = create_custom_preprocessor_patch_merging(device)
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


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
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

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(device), device=device
    )

    attn_mask_tuple = preprocess_attn_mask([1, 3, 512, 512], [4, 4], [7, 7], [3, 3], device)

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
        torch_output_tensor, output_tensor, pcc=0.96 if use_pretrained_weight else 0.99  # pcc=0.9614840639526093
    )  # The drop starts as we use shard MM in patch_mergig & mlp sub_module sub_module
