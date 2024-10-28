# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from models.experimental.functional_petr.reference.vovnetcp import (
    VoVNetCP,
    Hsigmoid,
    eSEModule,
    _OSA_module,
    _OSA_stage,
)
from models.experimental.functional_petr.tt.tt_vovnetcp import (
    ttnn_hsigmoid,
    ttnn_esemodule,
    ttnn_osa_module,
    ttnn_osa_stage,
    ttnn_VoVNetCP,
)
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters, fold_batch_norm2d_into_conv2d


@pytest.mark.parametrize(
    "n, c, h, w",
    (
        (6, 256, 1, 1),
        (6, 768, 1, 1),
        (6, 512, 1, 1),
        (6, 1024, 1, 1),
    ),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_vovnetcp_hsigmoid(device, reset_seeds, n, c, h, w):
    input_tensor = torch.randn((n, c, h, w))
    torch_model = Hsigmoid()
    torch_output = torch_model(input_tensor)
    input_tensor = torch.permute(input_tensor, (0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_model = ttnn_hsigmoid(device)
    ttnn_output = ttnn_model(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = torch.permute(ttnn_output, (0, 3, 1, 2))
    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)


def stem_parameters_preprocess(model):
    parameters = {}
    if isinstance(model, VoVNetCP):
        if hasattr(model, "stem"):
            layers = list(model.stem.named_children())
        for i, (name, layer) in enumerate(layers):
            if "conv" in name:
                conv_name, conv_layer = layers[i]
                norm_name, norm_layer = layers[i + 1]

                # Extract prefix (part before '/')
                prefix = conv_name.split("/")[0]

                # Initialize dictionary for each prefix
                if prefix not in parameters:
                    parameters[prefix] = {}

                conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(conv_layer, norm_layer)

                parameters[prefix]["weight"] = conv_weight
                parameters[prefix]["bias"] = conv_bias
    return parameters


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, eSEModule):
            parameters["fc"] = {}
            parameters["fc"]["weight"] = ttnn.from_torch(model.fc.weight, dtype=ttnn.bfloat16)
            parameters["fc"]["bias"] = ttnn.from_torch(torch.reshape(model.fc.bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)
        if isinstance(model, _OSA_module):
            if hasattr(model, "conv_reduction"):
                first_layer_name, _ = list(model.conv_reduction.named_children())[0]
                base_name = first_layer_name.split("/")[0]
                parameters[base_name] = {}
                conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.conv_reduction[0], model.conv_reduction[1])
                parameters[base_name]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
                parameters[base_name]["bias"] = ttnn.from_torch(
                    torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                )

            for i, layers in enumerate(model.layers):
                first_layer_name = list(layers.named_children())[0][0]
                prefix = first_layer_name.split("/")[0]
                parameters[prefix] = {}
                conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(layers[0], layers[1])
                if "OSA2_1" in prefix:
                    parameters[prefix]["weight"] = conv_weight
                    parameters[prefix]["bias"] = conv_bias
                else:
                    parameters[prefix]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
                    parameters[prefix]["bias"] = ttnn.from_torch(
                        torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                    )

            first_layer_name, _ = list(model.concat.named_children())[0]
            base_name = first_layer_name.split("/")[0]
            parameters[base_name] = {}
            if "OSA2_1" in base_name:
                parameters[base_name]["weight"] = model.concat[0].weight
                parameters[base_name]["bias"] = model.concat[0].bias
            else:
                concat_weight, concat_bias = fold_batch_norm2d_into_conv2d(model.concat[0], model.concat[1])
                parameters[base_name]["weight"] = ttnn.from_torch(concat_weight, dtype=ttnn.bfloat16)
                parameters[base_name]["bias"] = ttnn.from_torch(
                    torch.reshape(concat_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                )

            parameters["fc"] = {}
            parameters["fc"]["weight"] = ttnn.from_torch(model.ese.fc.weight, dtype=ttnn.bfloat16)
            parameters["fc"]["bias"] = ttnn.from_torch(
                torch.reshape(model.ese.fc.bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
            )
        if isinstance(model, _OSA_stage):
            if isinstance(model, _OSA_module):
                if hasattr(model, "conv_reduction"):
                    first_layer_name, _ = list(model.conv_reduction.named_children())[0]
                    base_name = first_layer_name.split("/")[0]
                    parameters[base_name] = {}
                    conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(
                        model.conv_reduction[0], model.conv_reduction[1]
                    )
                    parameters[base_name]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
                    parameters[base_name]["bias"] = ttnn.from_torch(
                        torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                    )

                for i, layers in enumerate(model.layers):
                    first_layer_name = list(layers.named_children())[0][0]
                    prefix = first_layer_name.split("/")[0]
                    parameters[prefix] = {}
                    conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(layers[0], layers[1])
                    parameters[prefix]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
                    parameters[prefix]["bias"] = ttnn.from_torch(
                        torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                    )

                first_layer_name, _ = list(model.concat.named_children())[0]
                base_name = first_layer_name.split("/")[0]
                parameters[base_name] = {}
                parameters[base_name]["weight"] = model.concat[0].weight
                parameters[base_name]["bias"] = model.concat[0].bias

                parameters["fc"] = {}
                parameters["fc"]["weight"] = ttnn.from_torch(model.ese.fc.weight, dtype=ttnn.bfloat16)
                parameters["fc"]["bias"] = ttnn.from_torch(
                    torch.reshape(model.ese.fc.bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                )

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize(
    "n, c, h, w",
    (
        # (6, 256, 80, 200),
        # (6, 768, 20, 50),
        (6, 256, 40, 100),
        # (6, 1024, 10, 25),
    ),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_vovnetcp_esemodule(device, n, c, h, w):
    torch_input_tensor = torch.randn(n, c, h, w)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor.permute(0, 2, 3, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    torch_model = eSEModule(c)
    torch_model.eval()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(None), device=None
    )

    torch_output = torch_model(torch_input_tensor)
    ttnn_model = ttnn_esemodule(parameters)

    ttnn_output = ttnn_model(device=device, x=ttnn_input_tensor)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)

    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)


@pytest.mark.parametrize(
    "n, c, h, w",
    (
        (6, 128, 80, 200),
        # (6, 256, 40, 100), 0.928
        # (6, 768, 10, 25),
        # (6, 1024, 10, 25),
        # (6, 768, 20, 50),
        # (6, 512, 40, 100), 0.87
        # 2_1 = 0.99
        # 3_1 = 0.928
        # 3_2, 3_3 = 0.87
        # 4_1 = 0.82
        # 4_2, to 4_9 = 0.87
        # 5_1 = 0.95
        # 5_2, 5_3 = 0.95
    ),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_vovnetcp_osa_module(device, reset_seeds, n, c, h, w):
    torch_input_tensor = torch.randn(6, 256, 40, 100)
    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor.permute(0, 2, 3, 1), dtype=ttnn.bfloat16, device=device)

    torch_model = _OSA_module(256, 160, 256, 5, "OSA3_1", SE=True)
    torch_model.eval()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(None), device=None
    )
    torch_output = torch_model(torch_input_tensor)

    ttnn_model = ttnn_osa_module(parameters, 256, 160, 256, 5, "OSA3_1", SE=True)
    ttnn_output = ttnn_model(device=device, x=ttnn_input_tensor)

    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)

    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)


@pytest.mark.parametrize(
    "in_ch, stage_ch, concat_ch, block_per_stage, layer_per_block, stage_num,input_shape",
    [
        # (128, 128, 256, 1, 5, 2, [6, 128, 80, 200]),
        (256, 160, 512, 3, 5, 3, [6, 256, 80, 200]),  # Maxpool issue 15093
        # (512,192,768,9,5,4,[6,512,40,100]), #Maxpool issue 15093
        # (768,224,1024,3,5,5,[6,768,20,50]), #Maxpool issue 15093,14292
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_vovnetcp_osa_stage(
    device, reset_seeds, in_ch, stage_ch, concat_ch, block_per_stage, layer_per_block, stage_num, input_shape
):
    torch_input_tensor = torch.randn(input_shape)
    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor.permute(0, 2, 3, 1), dtype=ttnn.bfloat16, device=device)
    torch_model = _OSA_stage(
        in_ch, stage_ch, concat_ch, block_per_stage, layer_per_block, stage_num, SE=True, depthwise=False
    )
    torch_model.eval()
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(None), device=None
    )

    torch_output = torch_model(torch_input_tensor)
    ttnn_model = ttnn_osa_stage(
        parameters, in_ch, stage_ch, concat_ch, block_per_stage, layer_per_block, stage_num, SE=True, depthwise=False
    )
    ttnn_output = ttnn_model(device=device, x=ttnn_input_tensor)

    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)

    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_vovnetcp(
    device,
):
    torch_input_tensor = torch.randn(6, 3, 320, 800)
    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor.permute(0, 2, 3, 1), dtype=ttnn.bfloat16, device=device)
    torch_model = VoVNetCP("V-99-eSE")
    torch_model.eval()
    stem_parameters = stem_parameters_preprocess(torch_model)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(None), device=None
    )
    output = torch_model(torch_input_tensor)

    ttnn_model = ttnn_VoVNetCP(parameters, stem_parameters, device)

    ttnn_output = ttnn_model(device, ttnn_input_tensor)

    #
    # Tensor Postprocessing
    #

    ttnn_output[1] = ttnn.to_torch(ttnn_output[1])
    ttnn_output[1] = ttnn_output[1].permute(0, 3, 1, 2)
    ttnn_output[1] = ttnn_output[1].reshape(output[1].shape)
    ttnn_output[1] = ttnn_output[1].to(torch_input_tensor.dtype)
    assert_with_pcc(output[1], ttnn_output[1], pcc=0.95)
