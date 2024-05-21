# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torchvision import models
import pytest
import ttnn
from ttnn.model_preprocessing import (
    preprocess_model,
    preprocess_linear_weight,
    preprocess_layernorm_parameter,
    preprocess_linear_bias,
    preprocess_conv2d,
)
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_swin_s.reference.swin_transformer import SwinTransformer
from models.experimental.functional_swin_s.tt.tt_swin_transformer import TtSwinTransformer
from calflops import calculate_flops


def update_ttnn_module_args(ttnn_module_args):
    # ttnn_module_args["dtype"] = ttnn.bfloat8_b
    ttnn_module_args["math_fidelity"] = ttnn.MathFidelity.LoFi
    # ttnn_module_args["weights_dtype"] = ttnn.bfloat8_b
    ttnn_module_args["deallocate_activation"] = True


def create_custom_preprocessor(device):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        # print("Arguments:", ttnn_module_args)
        parameters = {}
        if isinstance(torch_model, torch.nn.Conv2d):
            print("TT Arguments:", ttnn_module_args)
            print("torch model: ", torch_model)
            # ttnn_module_args["conv2d"] = ttnn_module_args#.features["0"]["0"]
            # update_ttnn_module_args(ttnn_module_args)
            # ttnn_module_args["use_1d_systolic_array"] = True
            # ttnn_module_args["conv_blocking_and_parallelization_config_override"] = {
            #     "act_block_h": 16 * 32
            # }
            # ttnn_module_args["use_shallow_conv_variant"] = True
            # print("Arguments:", ttnn_module_args)
            parameters["conv2d"] = preprocess_conv2d(torch_model.weight, torch_model.bias, ttnn_module_args)

        # if isinstance(torch_model,torch.nn.Linear):
        #     print("torch model: ", torch_model)
        #     print("TT Arguments:", ttnn_module_args)
        #     parameters["weight"] = preprocess_linear_weight(torch_model.weight, dtype=ttnn.bfloat16)
        #     parameters["bias"] = preprocess_linear_bias(torch_model.bias, dtype=ttnn.bfloat16)
        # if isinstance(torch_model, torch.nn.LayerNorm):
        #     parameters["norm_weight"] = preprocess_layernorm_parameter(torch_model.weight, dtype=ttnn.bfloat16)
        #     parameters["norm_bias"] = preprocess_layernorm_parameter(torch_model.bias, dtype=ttnn.bfloat16)
        return parameters

    return custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_patchmerging(device, reset_seeds):
    model = models.swin_s(weights="IMAGENET1K_V1")
    state_dict = model.state_dict()

    torch_model = SwinTransformer(
        patch_size=[4, 4], embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24], window_size=[7, 7]
    )

    torch_model.load_state_dict(state_dict)
    torch_model.eval()

    # Input tensor for testing
    torch_input_tensor = torch.randn(8, 3, 512, 512)  # Sample input tensor
    torch_output_tensor = torch_model(torch_input_tensor)

    input_shape = (8, 3, 512, 512)
    flops, macs, params = calculate_flops(
        model=torch_model, input_shape=input_shape, output_as_string=True, output_precision=4
    )
    print("Swin Transformer FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    # Convert the model to TTNN
    ttnn_model = TtSwinTransformer(
        ttnn.device,
        parameters,
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
    )

    # Convert input tensor to TTNN format
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # Apply TTNN model
    output_tensor = ttnn_model(input_tensor)

    # Convert output tensor back to Torch format
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)
