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
    weights_state_dict = torch.load(
        "models/experimental/functional_petr/resources/petr_vovnet_gridmask_p4_800x320-e2191752.pth", weights_only=False
    )["state_dict"]
    torch_model = VoVNetCP("V-99-eSE")
    torch_model.load_state_dict(
        {k.replace("img_backbone.", ""): v for k, v in weights_state_dict.items() if "img_backbone" in k}
    )
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


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_vovnetcp_layer_by_layer_debug(device, reset_seeds):
    """Debug backbone layer by layer to find where PCC drops"""

    from models.experimental.functional_petr.reference.vovnetcp import VoVNetCP
    from models.experimental.functional_petr.tt.ttnn_vovnetcp import ttnn_VoVNetCP

    # Load trained model
    torch_model = VoVNetCP("V-99-eSE")
    weights_state_dict = torch.load(
        "models/experimental/functional_petr/resources/petr_vovnet_gridmask_p4_800x320-e2191752.pth", weights_only=False
    )["state_dict"]
    backbone_weights = {k.replace("img_backbone.", ""): v for k, v in weights_state_dict.items() if "img_backbone" in k}
    torch_model.load_state_dict(backbone_weights)
    torch_model.eval()

    # Preprocess
    stem_parameters = stem_parameters_preprocess(torch_model)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_preprocessor(None),
        device=None,
    )

    # Create TTNN model
    ttnn_model = ttnn_VoVNetCP(parameters, stem_parameters, device)

    # Test input
    test_input = torch.randn(1, 3, 320, 800)

    print("\n" + "=" * 80)
    print("LAYER-BY-LAYER PCC ANALYSIS WITH TRAINED WEIGHTS")
    print("=" * 80)

    # ============ TEST STEM ============
    print("\n[STEM LAYER BY LAYER]")

    with torch.no_grad():
        # Torch stem
        x_torch = test_input

        # Stem conv 1
        x_torch_s1 = torch_model.stem[0](x_torch)  # conv
        x_torch_s1 = torch_model.stem[1](x_torch_s1)  # norm
        x_torch_s1 = torch_model.stem[2](x_torch_s1)  # relu

        # Stem conv 2
        x_torch_s2 = torch_model.stem[3](x_torch_s1)  # conv
        x_torch_s2 = torch_model.stem[4](x_torch_s2)  # norm
        x_torch_s2 = torch_model.stem[5](x_torch_s2)  # relu

        # Stem conv 3
        x_torch_s3 = torch_model.stem[6](x_torch_s2)  # conv
        x_torch_s3 = torch_model.stem[7](x_torch_s3)  # norm
        x_torch_s3 = torch_model.stem[8](x_torch_s3)  # relu

    # TTNN stem
    x_ttnn = ttnn.from_torch(test_input.permute(0, 2, 3, 1), dtype=ttnn.bfloat16, device=device)

    x_ttnn_s1 = ttnn_model.stem_conv1(device, x_ttnn)
    x_ttnn_s1_torch = ttnn.to_torch(ttnn.permute(x_ttnn_s1, (0, 3, 1, 2)))
    passed, pcc1 = check_with_pcc(x_torch_s1, x_ttnn_s1_torch, pcc=0.99)
    print(f"  Stem Conv1: PCC = {float(pcc1):.6f}")
    print(f"    Torch: mean={x_torch_s1.mean():.6f}, std={x_torch_s1.std():.6f}")
    print(f"    TTNN:  mean={x_ttnn_s1_torch.mean():.6f}, std={x_ttnn_s1_torch.std():.6f}")

    x_ttnn_s2 = ttnn_model.stem_conv2(device, x_ttnn_s1)
    x_ttnn_s2_torch = ttnn.to_torch(ttnn.permute(x_ttnn_s2, (0, 3, 1, 2)))
    passed, pcc2 = check_with_pcc(x_torch_s2, x_ttnn_s2_torch, pcc=0.99)
    print(f"  Stem Conv2: PCC = {float(pcc2):.6f}")
    print(f"    Torch: mean={x_torch_s2.mean():.6f}, std={x_torch_s2.std():.6f}")
    print(f"    TTNN:  mean={x_ttnn_s2_torch.mean():.6f}, std={x_ttnn_s2_torch.std():.6f}")

    x_ttnn_s3 = ttnn_model.stem_conv3(device, x_ttnn_s2)
    x_ttnn_s3_torch = ttnn.to_torch(ttnn.permute(x_ttnn_s3, (0, 3, 1, 2)))
    passed, pcc3 = check_with_pcc(x_torch_s3, x_ttnn_s3_torch, pcc=0.99)
    print(f"  Stem Conv3: PCC = {float(pcc3):.6f}")
    print(f"    Torch: mean={x_torch_s3.mean():.6f}, std={x_torch_s3.std():.6f}")
    print(f"    TTNN:  mean={x_ttnn_s3_torch.mean():.6f}, std={x_ttnn_s3_torch.std():.6f}")

    if float(pcc3) < 0.99:
        print(f"  STEM ALREADY FAILING! PCC = {float(pcc3):.6f}")
        print(f"  Max diff: {(x_torch_s3 - x_ttnn_s3_torch).abs().max():.6f}")

    # ============ TEST EACH STAGE ============
    stages = ["stage2", "stage3", "stage4", "stage5"]

    x_torch = x_torch_s3
    x_ttnn = x_ttnn_s3

    for stage_name in stages:
        print(f"\n[{stage_name.upper()}]")

        # Torch stage
        with torch.no_grad():
            torch_stage = getattr(torch_model, stage_name)
            x_torch_stage = torch_stage(x_torch)

        # TTNN stage
        ttnn_stage = getattr(ttnn_model, stage_name)
        x_ttnn_stage = ttnn_stage(device, x_ttnn)

        x_ttnn_stage_torch = ttnn.to_torch(ttnn.permute(x_ttnn_stage, (0, 3, 1, 2)))

        passed, pcc = check_with_pcc(x_torch_stage, x_ttnn_stage_torch, pcc=0.99)
        print(f"  Full {stage_name}: PCC = {float(pcc):.6f}")
        print(
            f"    Torch: mean={x_torch_stage.mean():.6f}, std={x_torch_stage.std():.6f}, "
            f"min={x_torch_stage.min():.6f}, max={x_torch_stage.max():.6f}"
        )
        print(
            f"    TTNN:  mean={x_ttnn_stage_torch.mean():.6f}, std={x_ttnn_stage_torch.std():.6f}, "
            f"min={x_ttnn_stage_torch.min():.6f}, max={x_ttnn_stage_torch.max():.6f}"
        )
        print(
            f"    Diff: mean={abs(x_torch_stage - x_ttnn_stage_torch).mean():.6f}, "
            f"max={abs(x_torch_stage - x_ttnn_stage_torch).max():.6f}"
        )

        if float(pcc) < 0.99:
            print(f"  PCC DROPPED SIGNIFICANTLY IN {stage_name}!")

            # Test individual blocks within this stage
            print(f"  Testing individual blocks in {stage_name}:")

            x_torch_block = x_torch
            x_ttnn_block = x_ttnn

            # Get blocks from torch stage
            for block_name, block_module in torch_stage.named_children():
                if "OSA" in block_name or "Pooling" in block_name:
                    with torch.no_grad():
                        x_torch_block = block_module(x_torch_block)

                    # Find corresponding TTNN block
                    if "Pooling" in block_name:
                        # Handle pooling - convert to torch, pool, convert back
                        x_ttnn_block_torch = ttnn.to_torch(x_ttnn_block)
                        x_ttnn_block_torch = torch.nn.functional.max_pool2d(
                            x_ttnn_block_torch.permute(0, 3, 1, 2), kernel_size=3, stride=2, ceil_mode=True
                        ).permute(0, 2, 3, 1)
                        x_ttnn_block = ttnn.from_torch(x_ttnn_block_torch, dtype=ttnn.bfloat16, device=device)
                        x_ttnn_block = ttnn.to_layout(x_ttnn_block, ttnn.TILE_LAYOUT)
                    else:
                        ttnn_block = getattr(ttnn_stage, block_name)
                        x_ttnn_block = ttnn_block(device, x_ttnn_block)

                    x_ttnn_block_torch = ttnn.to_torch(ttnn.permute(x_ttnn_block, (0, 3, 1, 2)))
                    passed, block_pcc = check_with_pcc(x_torch_block, x_ttnn_block_torch, pcc=0.99)
                    print(f"    {block_name}: PCC = {float(block_pcc):.6f}")

                    if float(block_pcc) < 0.99:
                        print(f"      FOUND PROBLEM BLOCK: {block_name}")
                        break

        # Update for next stage
        x_torch = x_torch_stage
        x_ttnn = x_ttnn_stage

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Stem Conv1: {float(pcc1):.6f}")
    print(f"Stem Conv2: {float(pcc2):.6f}")
    print(f"Stem Conv3: {float(pcc3):.6f}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_osa_block_internals(device, reset_seeds):
    """Debug inside a single OSA block to find the exact failing layer"""

    from models.experimental.functional_petr.reference.vovnetcp import VoVNetCP
    from models.experimental.functional_petr.tt.ttnn_vovnetcp import ttnn_osa_module

    # Load trained model
    torch_model = VoVNetCP("V-99-eSE")
    weights_state_dict = torch.load(
        "models/experimental/functional_petr/resources/petr_vovnet_gridmask_p4_800x320-e2191752.pth", weights_only=False
    )["state_dict"]
    backbone_weights = {k.replace("img_backbone.", ""): v for k, v in weights_state_dict.items() if "img_backbone" in k}
    torch_model.load_state_dict(backbone_weights)
    torch_model.eval()

    # Get the failing OSA3_1 block
    torch_osa_block = torch_model.stage4.OSA4_3

    # Preprocess just this block
    from ttnn.model_preprocessing import preprocess_model_parameters

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_osa_block,
        custom_preprocessor=create_custom_preprocessor(None),
        device=None,
    )

    # Create TTNN OSA block
    ttnn_osa_block = ttnn_osa_module(
        parameters=parameters,
        in_ch=768,  # 512, #256,
        stage_ch=192,  # 160,
        concat_ch=768,  # 512,
        layer_per_block=5,
        module_name="OSA4_2",
        SE=False,
        identity=False,
        depthwise=False,
    )

    # Create appropriate input (output from Stage2 + Pooling)
    # Approximate shape after stage2 pooling
    test_input = torch.randn(1, 768, 20, 50)
    # test_input = torch.randn(1, 512, 20, 50)
    # test_input = torch.randn(1, 256, 40, 100)

    print("\n" + "=" * 80)
    print("OSA4_2 BLOCK INTERNAL LAYER ANALYSIS")
    print("=" * 80)

    # ===== TORCH FORWARD (step by step) =====
    with torch.no_grad():
        x_torch = test_input
        output_torch = []

        print("\n[TORCH Forward]")
        output_torch.append(x_torch)
        print(f"Initial input: mean={x_torch.mean():.6f}, std={x_torch.std():.6f}")

        # Check if has conv_reduction
        if hasattr(torch_osa_block, "conv_reduction") and torch_osa_block.isReduced:
            x_torch = torch_osa_block.conv_reduction(x_torch)
            print(f"After reduction: mean={x_torch.mean():.6f}, std={x_torch.std():.6f}")

        # Process through layers
        for i, layer in enumerate(torch_osa_block.layers):
            x_torch = layer(x_torch)
            output_torch.append(x_torch)
            print(f"After layer {i}: mean={x_torch.mean():.6f}, std={x_torch.std():.6f}")

        # Concatenate
        x_torch_cat = torch.cat(output_torch, dim=1)
        print(f"After concat: shape={x_torch_cat.shape}, mean={x_torch_cat.mean():.6f}")

        # Concat conv
        x_torch_concat = torch_osa_block.concat(x_torch_cat)
        print(f"After concat conv: mean={x_torch_concat.mean():.6f}, std={x_torch_concat.std():.6f}")

        # eSE
        x_torch_final = torch_osa_block.ese(x_torch_concat)
        print(f"After eSE: mean={x_torch_final.mean():.6f}, std={x_torch_final.std():.6f}")

    # ===== TTNN FORWARD (step by step) =====
    print("\n[TTNN Forward]")
    x_ttnn = ttnn.from_torch(test_input.permute(0, 2, 3, 1), dtype=ttnn.bfloat16, device=device)

    output_ttnn = []
    x_ttnn = ttnn.to_layout(x_ttnn, ttnn.ROW_MAJOR_LAYOUT) if x_ttnn.get_layout() != ttnn.ROW_MAJOR_LAYOUT else x_ttnn
    output_ttnn.append(x_ttnn)

    x_ttnn_torch = ttnn.to_torch(x_ttnn).permute(0, 3, 1, 2)
    print(f"Initial input: mean={x_ttnn_torch.mean():.6f}, std={x_ttnn_torch.std():.6f}")
    passed, pcc = check_with_pcc(test_input, x_ttnn_torch, pcc=0.99)
    print(f"  vs Torch input: PCC = {float(pcc):.6f}")

    # Reduction if exists
    if hasattr(ttnn_osa_block, "conv_reduction") and ttnn_osa_block.isReduced:
        x_ttnn = ttnn_osa_block.conv_reduction(device, x_ttnn)
        x_ttnn_torch = ttnn.to_torch(x_ttnn).permute(0, 3, 1, 2)
        print(f"After reduction: mean={x_ttnn_torch.mean():.6f}, std={x_ttnn_torch.std():.6f}")

    # Layers
    for i, layer in enumerate(ttnn_osa_block.layers):
        x_ttnn = layer(device, x_ttnn)
        if x_ttnn.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            x_ttnn = ttnn.to_layout(x_ttnn, ttnn.ROW_MAJOR_LAYOUT)
        if hasattr(x_ttnn, "memory_config") and x_ttnn.memory_config().is_sharded():
            x_ttnn = ttnn.to_memory_config(x_ttnn, ttnn.DRAM_MEMORY_CONFIG)
        output_ttnn.append(x_ttnn)

        x_ttnn_torch = ttnn.to_torch(x_ttnn).permute(0, 3, 1, 2)
        print(f"After layer {i}: mean={x_ttnn_torch.mean():.6f}, std={x_ttnn_torch.std():.6f}")

        # Compare with torch
        passed, pcc = check_with_pcc(output_torch[i + 1], x_ttnn_torch, pcc=0.99)
        print(f"  vs Torch layer {i}: PCC = {float(pcc):.6f}")

        if float(pcc) < 0.99:
            print(f"  LAYER {i} FAILED! PCC = {float(pcc):.6f}")
            print(f"     Torch: mean={output_torch[i+1].mean():.6f}, std={output_torch[i+1].std():.6f}")
            print(f"     TTNN:  mean={x_ttnn_torch.mean():.6f}, std={x_ttnn_torch.std():.6f}")
            print(f"     Diff: max={abs(output_torch[i+1] - x_ttnn_torch).max():.6f}")
            break

    # Concatenate
    print(f"\nConcatenating {len(output_ttnn)} tensors...")
    for idx in range(len(output_ttnn)):
        if output_ttnn[idx].get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            output_ttnn[idx] = ttnn.to_layout(output_ttnn[idx], ttnn.ROW_MAJOR_LAYOUT)
        if hasattr(output_ttnn[idx], "memory_config") and output_ttnn[idx].memory_config().is_sharded():
            output_ttnn[idx] = ttnn.to_memory_config(output_ttnn[idx], ttnn.L1_MEMORY_CONFIG)

    x_ttnn_cat = ttnn.concat(output_ttnn, dim=3)
    x_ttnn_cat_torch = ttnn.to_torch(x_ttnn_cat).permute(0, 3, 1, 2)
    print(f"After concat: shape={x_ttnn_cat_torch.shape}, mean={x_ttnn_cat_torch.mean():.6f}")

    passed, pcc = check_with_pcc(x_torch_cat, x_ttnn_cat_torch, pcc=0.99)
    print(f"  vs Torch concat: PCC = {float(pcc):.6f}")

    if float(pcc) < 0.99:
        print(f"  CONCATENATION FAILED! PCC = {float(pcc):.6f}")
        print(f"     Checking individual tensors before concat:")
        for idx in range(len(output_ttnn)):
            out_torch = output_torch[idx]
            out_ttnn_torch = ttnn.to_torch(output_ttnn[idx]).permute(0, 3, 1, 2)
            passed, pcc_i = check_with_pcc(out_torch, out_ttnn_torch, pcc=0.99)
            print(f"     Tensor {idx}: PCC = {float(pcc_i):.6f}")

    # Concat conv
    x_ttnn_concat = ttnn_osa_block.conv_concat(device, x_ttnn_cat)
    x_ttnn_concat_torch = (
        ttnn.to_torch(x_ttnn_concat).permute(0, 3, 1, 2)
        if x_ttnn_concat.get_layout() == ttnn.ROW_MAJOR_LAYOUT
        else ttnn.to_torch(ttnn.to_layout(x_ttnn_concat, ttnn.ROW_MAJOR_LAYOUT)).permute(0, 3, 1, 2)
    )

    passed, pcc = check_with_pcc(x_torch_concat, x_ttnn_concat_torch, pcc=0.99)
    print(f"After concat conv: PCC = {float(pcc):.6f}")

    # eSE
    x_ttnn_final = ttnn_osa_block.ese(device, x_ttnn_concat)
    x_ttnn_final_torch = (
        ttnn.to_torch(x_ttnn_final).permute(0, 3, 1, 2)
        if x_ttnn_final.get_layout() == ttnn.ROW_MAJOR_LAYOUT
        else ttnn.to_torch(ttnn.to_layout(x_ttnn_final, ttnn.ROW_MAJOR_LAYOUT)).permute(0, 3, 1, 2)
    )

    passed, pcc = check_with_pcc(x_torch_final, x_ttnn_final_torch, pcc=0.99)
    print(f"After eSE (FINAL): PCC = {float(pcc):.6f}")

    print("\n" + "=" * 80)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_weight_loading_verification(device, reset_seeds):
    """Verify that trained weights are actually loaded into the model"""

    from models.experimental.functional_petr.reference.vovnetcp import VoVNetCP

    # Create model with random weights
    model_random = VoVNetCP("V-99-eSE")

    # Create model with trained weights
    model_trained = VoVNetCP("V-99-eSE")
    weights_state_dict = torch.load(
        "models/experimental/functional_petr/resources/petr_vovnet_gridmask_p4_800x320-e2191752.pth", weights_only=False
    )["state_dict"]

    # Extract backbone weights
    backbone_weights = {k.replace("img_backbone.", ""): v for k, v in weights_state_dict.items() if "img_backbone" in k}

    print("\n" + "=" * 80)
    print("AVAILABLE KEYS IN CHECKPOINT")
    print("=" * 80)
    print("First 20 backbone keys in checkpoint:")
    for i, key in enumerate(list(backbone_weights.keys())[:]):
        print(f"  {key}: {backbone_weights[key].shape}")

    print("\n" + "=" * 80)
    print("AVAILABLE KEYS IN MODEL")
    print("=" * 80)
    print("First 20 keys in VoVNetCP model:")
    model_keys = list(model_trained.state_dict().keys())[:]
    for key in model_keys:
        print(f"  {key}: {model_trained.state_dict()[key].shape}")

    # Try to load
    print("\n" + "=" * 80)
    print("LOADING WEIGHTS")
    print("=" * 80)

    try:
        missing_keys, unexpected_keys = model_trained.load_state_dict(backbone_weights, strict=False)
        print(f"Missing keys: {len(missing_keys)}")
        if missing_keys:
            print("First 10 missing keys:")
            for key in missing_keys[:]:
                print(f"  - {key}")

        print(f"\nUnexpected keys: {len(unexpected_keys)}")
        if unexpected_keys:
            print("First 10 unexpected keys:")
            for key in unexpected_keys[:]:
                print(f"  - {key}")
    except Exception as e:
        print(f"ERROR loading weights: {e}")

    model_trained.eval()

    # Compare specific layer weights
    print("\n" + "=" * 80)
    print("COMPARING LAYER WEIGHTS (Random vs Trained)")
    print("=" * 80)

    # Check stem weights
    for name, param_random in model_random.named_parameters():
        if "stem" in name:
            param_trained = dict(model_trained.named_parameters())[name]
            diff = (param_random - param_trained).abs().max().item()
            same = diff < 1e-6
            print(f"{name}:")
            print(f"  Random: mean={param_random.mean():.6f}, std={param_random.std():.6f}")
            print(f"  Trained: mean={param_trained.mean():.6f}, std={param_trained.std():.6f}")
            print(f"  Difference: {diff:.6f} {'✅ DIFFERENT' if not same else '❌ SAME (NOT LOADED!)'}")

    # Check first stage weights
    print("\nChecking stage2 first conv:")
    for name, param_random in model_random.named_parameters():
        if "stage2" in name and "conv" in name and "weight" in name:
            param_trained = dict(model_trained.named_parameters())[name]
            diff = (param_random - param_trained).abs().max().item()
            same = diff < 1e-6
            print(f"{name}:")
            print(f"  Random: mean={param_random.mean():.6f}, std={param_random.std():.6f}")
            print(f"  Trained: mean={param_trained.mean():.6f}, std={param_trained.std():.6f}")
            print(f"  Difference: {diff:.6f} {'✅ DIFFERENT' if not same else '❌ SAME (NOT LOADED!)'}")
            break  # Just check first one

    # Run inference to see if outputs differ
    print("\n" + "=" * 80)
    print("INFERENCE COMPARISON")
    print("=" * 80)

    test_input = torch.randn(1, 3, 320, 800)

    with torch.no_grad():
        output_random = model_random(test_input)
        output_trained = model_trained(test_input)

    for stage_idx in range(2):
        diff = (output_random[stage_idx] - output_trained[stage_idx]).abs().max().item()
        print(f"Stage {stage_idx+4} output difference: {diff:.6f}")
        if diff < 1e-3:
            print(f"  ⚠️ WARNING: Outputs are nearly identical! Weights may not be loaded correctly!")
