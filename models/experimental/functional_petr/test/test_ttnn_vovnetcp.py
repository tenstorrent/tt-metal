# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

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
from models.experimental.functional_petr.tt.ttnn_vovnetcp import (
    ttnn_hsigmoid,
    ttnn_esemodule,
    ttnn_osa_stage,
    ttnn_VoVNetCP,
)
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters, fold_batch_norm2d_into_conv2d
from loguru import logger


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_vovnetcp(device, reset_seeds):
    input_tensor = torch.randn((6, 3, 320, 800))
    torch_model = VoVNetCP("V-99-eSE", out_features=["stage4", "stage5"])
    print(torch_model)

    output = torch_model(input_tensor)


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
    passed, msg = check_with_pcc(torch_output, ttnn_output, pcc=0.99)

    logger.info(
        f"vovnetcp_hsigmoid test passed: "
        # f"batch_size={batch_size}, "
        # # f"act_dtype={self.model_config['ACTIVATIONS_DTYPE']}, "
        # f"weight_dtype={self.model_config['WEIGHTS_DTYPE']}, "
        # f"math_fidelity={self.model_config['MATH_FIDELITY']}, "
        f"PCC={msg}"
    )


def stem_parameters_preprocess(model):
    parameters = {}
    if isinstance(model, VoVNetCP):
        if hasattr(model, "stem"):
            layers = list(model.stem.named_children())

        for i, (name, layer) in enumerate(layers):
            if "conv" in name:
                conv_name, conv_layer = layers[i]
                norm_name, norm_layer = layers[i + 1]
                prefix = conv_name.split("/")[0]

                if prefix not in parameters:
                    parameters[prefix] = {}

                conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(conv_layer, norm_layer)

                logger.info(
                    f"[PREPROCESS] {prefix}: weight shape={conv_weight.shape}, mean={conv_weight.mean():.6f}, std={conv_weight.std():.6f}"
                )

                # Convert to ttnn format (same as other Conv layers)
                parameters[prefix]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)  # Use float32 for stem
                parameters[prefix]["bias"] = ttnn.from_torch(
                    torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                )

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
                # if "OSA2_1" in prefix:
                #     parameters[prefix]["weight"] = conv_weight
                #     parameters[prefix]["bias"] = conv_bias
                # else:
                parameters[prefix]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
                parameters[prefix]["bias"] = ttnn.from_torch(
                    torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                )

            first_layer_name, _ = list(model.concat.named_children())[0]
            base_name = first_layer_name.split("/")[0]
            parameters[base_name] = {}
            # if "OSA2_1" in base_name:
            #     parameters[base_name]["weight"] = model.concat[0].weight
            #     parameters[base_name]["bias"] = model.concat[0].bias
            # else:
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
        (6, 256, 80, 200),
        (6, 768, 20, 50),
        (6, 256, 40, 100),
        (6, 1024, 10, 25),
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
    passed, msg = check_with_pcc(torch_output, ttnn_output, pcc=0.99)

    logger.info(
        f"vovnetcp_esemodule test passed: "
        # f"batch_size={batch_size}, "
        # # f"act_dtype={self.model_config['ACTIVATIONS_DTYPE']}, "
        # f"weight_dtype={self.model_config['WEIGHTS_DTYPE']}, "
        # f"math_fidelity={self.model_config['MATH_FIDELITY']}, "
        f"PCC={msg}"
    )


@pytest.mark.parametrize(
    "in_ch, stage_ch, concat_ch, block_per_stage, layer_per_block, stage_num,input_shape",
    [
        (128, 128, 256, 1, 5, 2, [1, 128, 80, 200]),
        (256, 160, 512, 3, 5, 3, [1, 256, 80, 200]),
        (512, 192, 768, 9, 5, 4, [1, 512, 40, 100]),
        (768, 224, 1024, 3, 5, 5, [1, 768, 20, 50]),
    ],
)
# @pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
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
    if len(ttnn_output.shape) == 4 and ttnn_output.shape[2] == 1:
        # Calculate original H and W from the torch output shape
        target_h = torch_output.shape[2]
        target_w = torch_output.shape[3]
        ttnn_output = ttnn_output.reshape(ttnn_output.shape[0], ttnn_output.shape[1], target_h, target_w)

    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)
    passed, msg = check_with_pcc(torch_output, ttnn_output, pcc=0.99)

    logger.info(
        f"vovnetcp_osa_stage_{stage_num} test passed: "
        # f"batch_size={batch_size}, "
        # # f"act_dtype={self.model_config['ACTIVATIONS_DTYPE']}, "
        # f"weight_dtype={self.model_config['WEIGHTS_DTYPE']}, "
        # f"math_fidelity={self.model_config['MATH_FIDELITY']}, "
        f"PCC={msg}"
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_vovnetcp(
    device,
):
    torch_input_tensor = torch.randn(1, 3, 320, 800)
    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor.permute(0, 2, 3, 1), dtype=ttnn.bfloat16, device=device)
    torch_model = VoVNetCP("V-99-eSE")
    torch_model.eval()
    stem_parameters = stem_parameters_preprocess(torch_model)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(None), device=None
    )

    with torch.no_grad():
        output = torch_model(torch_input_tensor)

    logger.info(f"Torch output[0] (stage4): {output[0].shape}")
    logger.info(f"Torch output[1] (stage5): {output[1].shape}")

    ttnn_model = ttnn_VoVNetCP(parameters, stem_parameters, device)

    ttnn_output = ttnn_model(device, ttnn_input_tensor)
    logger.info(f"TTNN output[0]: {ttnn_output[0].shape}")
    logger.info(f"TTNN output[1]: {ttnn_output[1].shape}")

    #
    # Tensor Postprocessing
    #

    # ttnn_output[1] = ttnn.to_torch(ttnn_output[1])
    # ttnn_output[1] = ttnn_output[1].permute(0, 3, 1, 2)
    # ttnn_output[1] = ttnn_output[1].reshape(output[1].shape)
    # ttnn_output[1] = ttnn_output[1].to(torch_input_tensor.dtype)
    # assert_with_pcc(output[1], ttnn_output[1], pcc=0.99)
    # passed, msg = check_with_pcc(output[1], ttnn_output[1], pcc=0.99)

    # logger.info(
    #     f"vovnetcp test passed: "
    #     # f"batch_size={batch_size}, "
    #     # # f"act_dtype={self.model_config['ACTIVATIONS_DTYPE']}, "
    #     # f"weight_dtype={self.model_config['WEIGHTS_DTYPE']}, "
    #     # f"math_fidelity={self.model_config['MATH_FIDELITY']}, "
    #     f"PCC={msg}"
    # )
    #  Convert TTNN outputs to torch for comparison
    ttnn_out0_torch = ttnn.to_torch(ttnn_output[0]).permute(0, 3, 1, 2)
    ttnn_out1_torch = ttnn.to_torch(ttnn_output[1]).permute(0, 3, 1, 2)
    logger.info(f"After permute - TTNN out0: {ttnn_out0_torch.shape}")
    logger.info(f"After permute - TTNN out1: {ttnn_out1_torch.shape}")
    logger.info(f"Torch out0: {output[0].shape}")
    logger.info(f"Torch out1: {output[1].shape}")

    # Reshape if needed
    if ttnn_out0_torch.shape != output[0].shape:
        ttnn_out0_torch = ttnn_out0_torch.reshape(output[0].shape)
    if ttnn_out1_torch.shape != output[1].shape:
        ttnn_out1_torch = ttnn_out1_torch.reshape(output[1].shape)

    # Compare
    passed0, msg0 = check_with_pcc(output[0], ttnn_out0_torch, pcc=0.99)
    passed1, msg1 = check_with_pcc(output[1], ttnn_out1_torch, pcc=0.99)

    logger.info("=" * 60)
    logger.info("FINAL BACKBONE RESULTS:")
    logger.info("=" * 60)
    logger.info(f"Stage 4 output PCC: {msg0}")
    logger.info(f"Stage 5 output PCC: {msg1}")
    assert_with_pcc(output[0], ttnn_out0_torch, pcc=0.99)
    assert_with_pcc(output[1], ttnn_out1_torch, pcc=0.99)
    assert passed0, f"Stage 4 PCC failed: {msg0}"
    assert passed1, f"Stage 5 PCC failed: {msg1}"
