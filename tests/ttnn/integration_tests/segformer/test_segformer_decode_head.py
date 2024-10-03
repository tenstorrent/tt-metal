# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from transformers import SegformerForSemanticSegmentation
from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d, preprocess_model_parameters
from models.experimental.functional_segformer.reference.segformer_decode_head import SegformerDecodeHead
from models.experimental.functional_segformer.tt.ttnn_segformer_decode_head import TtSegformerDecodeHead
from tests.ttnn.integration_tests.segformer.test_segformer_mlp import (
    create_custom_preprocessor as create_custom_preprocessor_mlp,
)
from models.utility_functions import skip_for_grayskull


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, SegformerDecodeHead):
            parameters["linear_c"] = {}
            for i in range(4):
                parameters["linear_c"][i] = {}
                mlp_preprocess = create_custom_preprocessor_mlp(device)
                parameters["linear_c"][i] = mlp_preprocess(model.linear_c[i], None, None)

            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.linear_fuse, model.batch_norm)

            parameters["linear_fuse"] = {}
            parameters["linear_fuse"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["linear_fuse"]["bias"] = ttnn.from_torch(
                torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
            )

            parameters["classifier"] = {}
            parameters["classifier"]["weight"] = ttnn.from_torch(model.classifier.weight, dtype=ttnn.bfloat16)
            parameters["classifier"]["bias"] = ttnn.from_torch(
                torch.reshape(model.classifier.bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
            )

        return parameters

    return custom_preprocessor


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_segformer_decode_head(device, batch_size=1):
    torch_input_tensor_0 = torch.randn(1, 32, 128, 128)
    torch_input_tensor_1 = torch.randn(1, 64, 64, 64)
    torch_input_tensor_2 = torch.randn(1, 160, 32, 32)
    torch_input_tensor_3 = torch.randn(1, 256, 16, 16)

    if 0:
        torch_input_tensor_0_folded = torch_input_tensor_0
        torch_input_tensor_1_folded = torch_input_tensor_1
        torch_input_tensor_2_folded = torch_input_tensor_2
        torch_input_tensor_3_folded = torch_input_tensor_3
    else:
        torch_input_tensor_0_folded = torch.reshape(torch_input_tensor_0, (batch_size, 32, 128 * 128))
        torch_input_tensor_0_folded = torch.permute(torch_input_tensor_0_folded, (0, 2, 1))
        torch_input_tensor_1_folded = torch.reshape(torch_input_tensor_1, (batch_size, 64, 64 * 64))
        torch_input_tensor_1_folded = torch.permute(torch_input_tensor_1_folded, (0, 2, 1))
        torch_input_tensor_2_folded = torch.reshape(torch_input_tensor_2, (batch_size, 160, 32 * 32))
        torch_input_tensor_2_folded = torch.permute(torch_input_tensor_2_folded, (0, 2, 1))
        torch_input_tensor_3_folded = torch.reshape(torch_input_tensor_3, (batch_size, 256, 16 * 16))
        torch_input_tensor_3_folded = torch.permute(torch_input_tensor_3_folded, (0, 2, 1))

    ttnn_input_tensor_0 = ttnn.from_torch(
        torch_input_tensor_0_folded,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn_input_tensor_1 = ttnn.from_torch(
        torch_input_tensor_1_folded,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn_input_tensor_2 = ttnn.from_torch(
        torch_input_tensor_2_folded,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn_input_tensor_3 = ttnn.from_torch(
        torch_input_tensor_3_folded,
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

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(device), device=None
    )

    for i in range(4):
        parameters["linear_c"][i]["proj"]["weight"] = ttnn.to_device(
            parameters["linear_c"][i]["proj"]["weight"], device=device
        )
        parameters["linear_c"][i]["proj"]["bias"] = ttnn.to_device(
            parameters["linear_c"][i]["proj"]["bias"], device=device
        )

    ttnn_model = TtSegformerDecodeHead(config, parameters)
    ttnn_output = ttnn_model(ttnn_input_tensor, parameters)

    ttnn_output = ttnn.from_device(ttnn_output)
    ttnn_output = ttnn.to_torch(ttnn_output)

    # torch_output = torch.permute(torch_output,(0,3,2,1))
    # torch_output = torch.reshape(torch_output,(batch_size, 1, 16384, 256))

    # print("ddd", torch_output.shape, ttnn_output.shape)

    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)
