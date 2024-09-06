# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

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
from models.experimental.functional_swin_s.reference.swin_transformer import SwinTransformer
from models.experimental.functional_swin_s.tt.tt_swin_transformer import TtSwinTransformer
from tests.ttnn.integration_tests.swin_s.test_ttnn_swin_transformer_block import (
    create_custom_preprocessor as create_custom_preprocessor_transformer_block,
)
from tests.ttnn.integration_tests.swin_s.test_ttnn_patchmerging import (
    create_custom_preprocessor as create_custom_preprocessor_patch_merging,
)


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
            parameters["norm"]["weight"] = preprocess_layernorm_parameter(torch_model.norm.weight, dtype=ttnn.bfloat8_b)
            parameters["norm"]["bias"] = preprocess_layernorm_parameter(torch_model.norm.bias, dtype=ttnn.bfloat8_b)
            parameters["head"]["weight"] = preprocess_linear_weight(torch_model.head.weight, dtype=ttnn.bfloat8_b)
            parameters["head"]["bias"] = preprocess_linear_bias(torch_model.head.bias, dtype=ttnn.bfloat8_b)

        return parameters

    return custom_preprocessor


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_swin_s_transformer(device, reset_seeds):
    model = models.swin_s(weights="IMAGENET1K_V1")
    state_dict = model.state_dict()

    torch_model = SwinTransformer(
        patch_size=[4, 4], embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24], window_size=[7, 7]
    )

    torch_model.load_state_dict(state_dict)
    torch_model.eval()

    # Input tensor for testing
    torch_input_tensor = torch.randn(1, 3, 512, 512)  # Sample input tensor
    torch_output_tensor = torch_model(torch_input_tensor)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(device), device=device
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
    )

    # Convert input tensor to TTNN format
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Apply TTNN model
    output_tensor = ttnn_model(input_tensor)

    # Convert output tensor back to Torch format
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(
        torch_output_tensor, output_tensor, pcc=0.82
    )  # The drop starts as we use shard MM in patch_mergig & mlp sub_module sub_module
