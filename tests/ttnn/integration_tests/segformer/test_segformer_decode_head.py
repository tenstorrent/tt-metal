# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from transformers import SegformerForSemanticSegmentation
from ttnn.model_preprocessing import preprocess_model, fold_batch_norm2d_into_conv2d, preprocess_conv2d
from models.experimental.functional_segformer.reference.segformer_decode_head import SegformerDecodeHead
from models.experimental.functional_segformer.tt.ttnn_segformer_decode_head import TtSegformerDecodeHead


def update_ttnn_module_args(ttnn_module_args):
    ttnn_module_args["use_1d_systolic_array"] = ttnn_module_args.in_channels < 256
    ttnn_module_args["dtype"] = ttnn.bfloat8_b
    ttnn_module_args["math_fidelity"] = ttnn.MathFidelity.LoFi
    ttnn_module_args["weights_dtype"] = ttnn.bfloat8_b
    ttnn_module_args["activation"] = None


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, SegformerDecodeHead):
            for i in range(4):
                parameters[f"mlp_{i}"] = {}
                parameters[f"mlp_{i}"]["proj"] = {}
                parameters[f"mlp_{i}"]["proj"]["weight"] = ttnn.from_torch(
                    getattr(model, f"mlp_{i}").proj.weight.T,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )
                parameters[f"mlp_{i}"]["proj"]["bias"] = ttnn.from_torch(
                    getattr(model, f"mlp_{i}").proj.bias,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )

            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.linear_fuse, model.batch_norm)
            update_ttnn_module_args(ttnn_module_args["linear_fuse"])
            ttnn_module_args["linear_fuse"]["activation"] = "relu"
            ttnn_module_args["linear_fuse"]["use_shallow_conv_variant"] = True
            ttnn_module_args["linear_fuse"]["use_1d_systolic_array"] = True
            ttnn_module_args["linear_fuse"]["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 64}
            parameters["linear_fuse"], _ = preprocess_conv2d(
                conv_weight, conv_bias, ttnn_module_args["linear_fuse"], return_parallel_config=True
            )

            parameters["classifier"], _ = preprocess_conv2d(
                model.classifier.weight,
                model.classifier.bias,
                ttnn_module_args["classifier"],
                return_parallel_config=True,
            )

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_segformer_decode_head(device, batch_size=1):
    torch_input_tensor_0 = torch.randn(1, 32, 128, 128)
    torch_input_tensor_1 = torch.randn(1, 64, 64, 64)
    torch_input_tensor_2 = torch.randn(1, 160, 32, 32)
    torch_input_tensor_3 = torch.randn(1, 256, 16, 16)

    ttnn_input_tensor_0 = ttnn.from_torch(
        torch_input_tensor_0,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn_input_tensor_1 = ttnn.from_torch(
        torch_input_tensor_1,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn_input_tensor_2 = ttnn.from_torch(
        torch_input_tensor_2,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn_input_tensor_3 = ttnn.from_torch(
        torch_input_tensor_3,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    torch_input_tensor = (torch_input_tensor_0, torch_input_tensor_1, torch_input_tensor_2, torch_input_tensor_3)
    ttnn_input_tensor = (ttnn_input_tensor_0, ttnn_input_tensor_1, ttnn_input_tensor_2, ttnn_input_tensor_3)

    torch_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    torch_model = torch_model.decode_head
    config = torch_model.config

    state_dict = torch_model.state_dict()

    reference_model = SegformerDecodeHead(config)

    new_state_dict = {}
    keys = [name for name, parameter in reference_model.state_dict().items()]
    values = [parameter for name, parameter in state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    reference_model.load_state_dict(new_state_dict)
    reference_model.eval()

    torch_output = reference_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: reference_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtSegformerDecodeHead(config, parameters)
    ttnn_output = ttnn_model(ttnn_input_tensor, parameters)

    ttnn_output = ttnn.from_device(ttnn_output)
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)
