# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import math

import torch
import torchvision

import ttnn
from ttnn.model_preprocessing import (
    preprocess_model,
    preprocess_conv2d,
    fold_batch_norm2d_into_conv2d,
    fold_conv7s2_into_conv4s1,
    preprocess_remaining_children_and_parameters,
    convert_torch_model_to_ttnn_model,
)

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import (
    skip_for_wormhole_b0,
    pad_and_fold_conv_activation_for_unity_stride,
    is_wormhole_b0,
    is_grayskull,
)

from models.experimental.functional_resnet.tt.ttnn_functional_resnet import resnet_basic_block, resnet_bottleneck_block


def update_ttnn_module_args(ttnn_module_args):
    ttnn_module_args["use_1d_systolic_array"] = ttnn_module_args.in_channels < 256
    ttnn_module_args["enable_auto_formatting"] = ttnn_module_args.kernel_size < (7, 7)
    ttnn_module_args["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 32}
    ttnn_module_args["deallocate_activation"] = True if ttnn_module_args.kernel_size == (3, 3) else False


def custom_preprocessor(model, name, ttnn_module_args):
    parameters = {}
    if isinstance(model, torchvision.models.resnet.BasicBlock):
        ttnn_module_args.conv1["activation"] = "relu"  # Fuse relu with conv1

        conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
        conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.conv2, model.bn2)

        update_ttnn_module_args(ttnn_module_args.conv1)
        update_ttnn_module_args(ttnn_module_args.conv2)

        parameters["conv1"] = preprocess_conv2d(conv1_weight, conv1_bias, ttnn_module_args.conv1)
        parameters["conv2"] = preprocess_conv2d(conv2_weight, conv2_bias, ttnn_module_args.conv2)

        if model.downsample is not None:
            update_ttnn_module_args(ttnn_module_args.downsample)
            parameters["downsample"] = preprocess_conv2d(
                model.downsample.weight, model.downsample.bias, ttnn_module_args.downsample
            )

    elif isinstance(model, torchvision.models.resnet.Bottleneck):
        ttnn_module_args.conv2["activation"] = "relu"  # Fuse relu with conv1

        conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
        conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.conv2, model.bn2)
        conv3_weight, conv3_bias = fold_batch_norm2d_into_conv2d(model.conv3, model.bn3)

        update_ttnn_module_args(ttnn_module_args.conv1)
        update_ttnn_module_args(ttnn_module_args.conv2)
        update_ttnn_module_args(ttnn_module_args.conv3)

        parameters["conv1"] = preprocess_conv2d(conv1_weight, conv1_bias, ttnn_module_args.conv1)
        parameters["conv2"] = preprocess_conv2d(conv2_weight, conv2_bias, ttnn_module_args.conv2)
        parameters["conv3"] = preprocess_conv2d(conv3_weight, conv3_bias, ttnn_module_args.conv3)

        if model.downsample is not None:
            update_ttnn_module_args(ttnn_module_args.downsample)
            parameters["downsample"] = preprocess_conv2d(
                model.downsample.weight, model.downsample.bias, ttnn_module_args.downsample
            )

    elif isinstance(model, torch.nn.Conv2d) and model.kernel_size == (7, 7) and model.stride == (2, 2):
        return fold_conv7s2_into_conv4s1(model.weight, model.bias, ttnn_module_args)

    return parameters


@skip_for_wormhole_b0()
def test_basic_block(device):
    torch.manual_seed(0)

    torch_model = torchvision.models.resnet.BasicBlock(inplanes=64, planes=64, stride=1).eval()

    torch_input_tensor = torch.rand((8, 64, 56, 56), dtype=torch.float32)
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=custom_preprocessor,
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    input_tensor = torch.reshape(input_tensor, (1, 1, -1, input_tensor.shape[-1]))

    padded_input_channels = math.ceil(input_tensor.shape[3] / 16) * 16
    input_tensor = torch.nn.functional.pad(input_tensor, (0, padded_input_channels - input_tensor.shape[3], 0, 0, 0, 0))
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = resnet_basic_block(input_tensor, parameters=parameters)

    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.reshape(
        output_tensor,
        [
            torch_input_tensor.shape[0],
            torch_input_tensor.shape[2],
            torch_input_tensor.shape[3],
            torch_input_tensor.shape[1],
        ],
    )
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.9997)


@skip_for_wormhole_b0()
def test_basic_block_with_downsample(device):
    torch.manual_seed(0)

    def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> torch.nn.Conv2d:
        """1x1 convolution"""
        return torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    torch_model = torchvision.models.resnet.BasicBlock(
        inplanes=64, planes=64, stride=1, downsample=conv1x1(64, 64, 1)
    ).eval()

    torch_input_tensor = torch.rand((8, 64, 56, 56), dtype=torch.float32)
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=custom_preprocessor,
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    input_tensor = torch.reshape(input_tensor, (1, 1, -1, input_tensor.shape[-1]))

    padded_input_channels = math.ceil(input_tensor.shape[3] / 16) * 16
    input_tensor = torch.nn.functional.pad(input_tensor, (0, padded_input_channels - input_tensor.shape[3], 0, 0, 0, 0))
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = resnet_basic_block(input_tensor, parameters=parameters)

    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.reshape(
        output_tensor,
        [
            torch_input_tensor.shape[0],
            torch_input_tensor.shape[2],
            torch_input_tensor.shape[3],
            torch_input_tensor.shape[1],
        ],
    )
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99966)


@skip_for_wormhole_b0()
def test_resnet_conv7s2(device):
    in_planes = 64

    torch_model = torch.nn.Conv2d(3, in_planes, kernel_size=7, stride=2, padding=[3, 3], bias=False)

    torch_input_tensor = torch.rand((8, 3, 224, 224), dtype=torch.float32)
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=custom_preprocessor,
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    input_tensor = pad_and_fold_conv_activation_for_unity_stride(
        torch_input_tensor, *torch_model.padding, *torch_model.stride
    )

    input_tensor = torch.permute(input_tensor, (0, 2, 3, 1))
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16)

    output_tensor = parameters.copy_input_to_device(input_tensor)
    output_tensor = parameters(output_tensor)
    output_tensor = parameters.copy_output_from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.9998)


@skip_for_wormhole_b0()
def test_resnet(device):
    torch.manual_seed(0)

    torch_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).eval()

    torch_input_tensor = torch.rand((8, 3, 224, 224), dtype=torch.float32)
    torch_output_tensor = torch_model(torch_input_tensor)

    def custom_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        parameters = {}
        if isinstance(model, torchvision.models.resnet.BasicBlock):
            ttnn_module_args.conv1["activation"] = "relu"  # Fuse relu with conv1

            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
            conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.conv2, model.bn2)

            update_ttnn_module_args(ttnn_module_args.conv1)
            update_ttnn_module_args(ttnn_module_args.conv2)

            parameters["conv1"] = preprocess_conv2d(conv1_weight, conv1_bias, ttnn_module_args.conv1)
            parameters["conv2"] = preprocess_conv2d(conv2_weight, conv2_bias, ttnn_module_args.conv2)

            if model.downsample is not None:
                downsample_weight, downsample_bias = fold_batch_norm2d_into_conv2d(
                    model.downsample[0], model.downsample[1]
                )
                update_ttnn_module_args(ttnn_module_args.downsample[0])
                parameters["downsample"] = preprocess_conv2d(
                    downsample_weight, downsample_bias, ttnn_module_args.downsample[0]
                )
                ttnn_module_args["downsample"] = ttnn_module_args.downsample[0]

        elif isinstance(model, torchvision.models.resnet.ResNet):
            ttnn_module_args.conv1["activation"] = "relu"  # Fuse relu with conv1
            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
            parameters["conv1"]: ttnn.Conv2d = fold_conv7s2_into_conv4s1(
                conv1_weight, conv1_bias, ttnn_module_args.conv1
            )

            return preprocess_remaining_children_and_parameters(
                model,
                name=name,
                convert_to_ttnn=convert_to_ttnn,
                custom_preprocessor=custom_preprocessor,
                parameters=parameters,
                ttnn_module_args=ttnn_module_args,
                already_preprocessed_children={"conv1", "bn1", "relu1"},
            )

            for child_name, child in tuple(model.named_children()) + named_parameters:
                if child_name in {"conv1", "bn1"}:
                    continue
                parameters[child_name] = convert_torch_model_to_ttnn_model(
                    child,
                    name=name,
                    convert_to_ttnn=convert_to_ttnn,
                    custom_preprocessor=custom_preprocessor,
                    ttnn_module_args=ttnn_module_args.get(child_name, None),
                )
                if child_name == "maxpool":
                    ttnn_module_args.maxpool["parallel_config_override"] = {
                        "grid_size": parameters["conv1"]["parallel_config"],
                        "ncores_nhw": parameters["conv1"]["num_cores_nhw"],
                    }
                    # update_ttnn_module_args(ttnn_module_args.maxpool)

        return parameters

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        reader_patterns_cache=reader_patterns_cache,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    input_tensor = pad_and_fold_conv_activation_for_unity_stride(
        torch_input_tensor, *torch_model.conv1.padding, *torch_model.conv1.stride
    )

    input_tensor = torch.permute(input_tensor, (0, 2, 3, 1))
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16)

    output_tensor = parameters.conv1.copy_input_to_device(input_tensor)
    output_tensor = parameters.conv1(output_tensor)
    output_tensor = parameters.maxpool(output_tensor)
    output_tensor = ttnn.to_memory_config(output_tensor, ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)

    for basic_block_parameters in parameters.layer1.values():
        output_tensor = resnet_basic_block(output_tensor, parameters=basic_block_parameters)
    for basic_block_parameters in parameters.layer2.values():
        output_tensor = resnet_basic_block(output_tensor, parameters=basic_block_parameters)
    for basic_block_parameters in parameters.layer3.values():
        output_tensor = resnet_basic_block(output_tensor, parameters=basic_block_parameters)
    for basic_block_parameters in parameters.layer4.values():
        output_tensor = resnet_basic_block(output_tensor, parameters=basic_block_parameters)

    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.reshape(output_tensor, (8, 1, 49, 512))
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.global_avg_pool2d(output_tensor)
    output_tensor = output_tensor @ parameters.fc.weight + parameters.fc.bias

    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.reshape(output_tensor, (8, 1000))

    # The check below doesn't work yet
    # assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.999)


@skip_for_wormhole_b0()
def test_bottleneck_block(device):
    torch.manual_seed(0)

    torch_model = torchvision.models.resnet.Bottleneck(inplanes=2048, planes=512, stride=1).eval()
    torch_input_tensor = torch.rand((8, 2048, 7, 7), dtype=torch.float32)
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=custom_preprocessor,
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    """input preparation and invocation of the bottleneck class output reshape"""
    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    input_tensor = torch.reshape(input_tensor, (1, 1, -1, input_tensor.shape[-1]))
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = resnet_bottleneck_block(input_tensor, parameters)

    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.reshape(
        output_tensor,
        [
            torch_output_tensor.shape[0],
            torch_output_tensor.shape[2],
            torch_output_tensor.shape[3],
            torch_output_tensor.shape[1],
        ],
    )
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@skip_for_wormhole_b0()
def test_bottleneck_block_with_downsample(device):
    torch.manual_seed(0)

    def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> torch.nn.Conv2d:
        """1x1 convolution"""
        return torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)

    torch_model = torchvision.models.resnet.Bottleneck(
        inplanes=512, planes=256, stride=2, downsample=conv1x1(512, 512 * 2, stride=2)
    ).eval()
    torch_input_tensor = torch.rand((8, 512, 28, 28), dtype=torch.float32)
    torch_output_tensor = torch_model(torch_input_tensor)
    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=custom_preprocessor,
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )
    # input preparation
    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    input_tensor = torch.reshape(input_tensor, (1, 1, -1, input_tensor.shape[-1]))

    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # intialization of bottleneck class and invocation
    output_tensor = resnet_bottleneck_block(input_tensor, parameters)

    # output tensor reshaping and comparison
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.reshape(
        output_tensor,
        [
            torch_output_tensor.shape[0],
            torch_output_tensor.shape[2],
            torch_output_tensor.shape[3],
            torch_output_tensor.shape[1],
        ],
    )
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)

    # validation of the output
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.98)


@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, math_fidelity",
    (
        (8, ttnn.bfloat16, ttnn.bfloat16, ttnn.MathFidelity.HiFi4),  ## pass
        # (8, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),    ## legacy shape assertion
        # (16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2),
        # (16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),
        # (20, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2),
        # (20, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),
    ),
)
def test_resnet_50(device, batch_size, act_dtype, weight_dtype, math_fidelity):
    torch.manual_seed(0)

    torch_model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1).eval()
    activation = {}

    # compute_kernel_config = None
    # if is_grayskull():
    #     compute_kernel_config = ttnn.types.GrayskullComputeKernelConfig(
    #         math_fidelity=math_fedility,
    #         math_approx_mode=True,
    #         fp32_dest_acc_en=False,
    #         packer_l1_acc=False,
    #     )

    def update_ttnn_module_args_resnet50(ttnn_module_args):
        ttnn_module_args["use_1d_systolic_array"] = True
        ttnn_module_args["enable_auto_formatting"] = False
        ttnn_module_args["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 32}
        ttnn_module_args["deallocate_activation"] = True if ttnn_module_args.kernel_size == (3, 3) else False
        ttnn_module_args["weights_dtype"] = weight_dtype
        ttnn_module_args["math_fidelity"] = math_fidelity

    def custom_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        parameters = {}
        if isinstance(model, torchvision.models.resnet.Bottleneck):
            ttnn_module_args.conv1["activation"] = "relu"  # Fuse relu with conv1
            ttnn_module_args.conv2["activation"] = "relu"  # Fuse relu with conv1
            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
            conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.conv2, model.bn2)
            conv3_weight, conv3_bias = fold_batch_norm2d_into_conv2d(model.conv3, model.bn3)

            update_ttnn_module_args_resnet50(ttnn_module_args.conv1)
            update_ttnn_module_args_resnet50(ttnn_module_args.conv2)
            update_ttnn_module_args_resnet50(ttnn_module_args.conv3)
            if model.downsample is not None:
                update_ttnn_module_args_resnet50(ttnn_module_args.downsample[0])

            ## TODO: Cleanup this atrocity
            if ttnn_module_args.conv1.input_height <= 14 and ttnn_module_args.conv1.input_width <= 14:
                ttnn_module_args.conv1["use_1d_systolic_array"] = False
                if model.downsample is not None:
                    ttnn_module_args.downsample[0]["use_1d_systolic_array"] = False
            else:
                if ttnn_module_args.conv1.input_height == 28 and ttnn_module_args.conv1.input_width == 28:
                    if ttnn_module_args.conv1.stride == (2, 2):
                        ttnn_module_args.conv1["use_1d_systolic_array"] = False
                        if model.downsample is not None:
                            ttnn_module_args.downsample[0]["use_1d_systolic_array"] = False
                    else:
                        ttnn_module_args.conv1["use_1d_systolic_array"] = True
                        if model.downsample is not None:
                            ttnn_module_args.downsample[0]["use_1d_systolic_array"] = True
                else:
                    ttnn_module_args.conv1["use_1d_systolic_array"] = True
                    if model.downsample is not None:
                        ttnn_module_args.downsample[0]["use_1d_systolic_array"] = True

            if ttnn_module_args.conv2.input_height <= 14 and ttnn_module_args.conv2.input_width <= 14:
                ttnn_module_args.conv2["use_1d_systolic_array"] = False
                if model.downsample is not None:
                    ttnn_module_args.downsample[0]["use_1d_systolic_array"] = False
            else:
                if ttnn_module_args.conv2.input_height == 28 and ttnn_module_args.conv2.input_width == 28:
                    if ttnn_module_args.conv2.stride == (2, 2):
                        ttnn_module_args.conv2["use_1d_systolic_array"] = False
                        if model.downsample is not None:
                            ttnn_module_args.downsample[0]["use_1d_systolic_array"] = False
                    else:
                        ttnn_module_args.conv2["use_1d_systolic_array"] = True
                else:
                    ttnn_module_args.conv2["use_1d_systolic_array"] = True

            if ttnn_module_args.conv3.input_height <= 14 and ttnn_module_args.conv3.input_width <= 14:
                ttnn_module_args.conv3["use_1d_systolic_array"] = False
                if model.downsample is not None:
                    ttnn_module_args.downsample[0]["use_1d_systolic_array"] = False
            else:
                if ttnn_module_args.conv3.input_height == 28 and ttnn_module_args.conv3.input_width == 28:
                    if ttnn_module_args.conv3.stride == (2, 2):
                        ttnn_module_args.conv3["use_1d_systolic_array"] = False
                        if model.downsample is not None:
                            ttnn_module_args.downsample[0]["use_1d_systolic_array"] = False
                    else:
                        ttnn_module_args.conv3["use_1d_systolic_array"] = True
                else:
                    ttnn_module_args.conv3["use_1d_systolic_array"] = True

            parameters["conv1"], pconfig1 = preprocess_conv2d(conv1_weight, conv1_bias, ttnn_module_args.conv1, True)
            parameters["conv2"], pconfig2 = preprocess_conv2d(conv2_weight, conv2_bias, ttnn_module_args.conv2, True)
            parameters["conv3"], pconfig3 = preprocess_conv2d(conv3_weight, conv3_bias, ttnn_module_args.conv3, True)

            logger.debug(f"pconfig1: {pconfig1.num_cores_nhw}")
            logger.debug(f"pconfig2: {pconfig2.num_cores_nhw}")
            logger.debug(f"pconfig3: {pconfig3.num_cores_nhw}")

            if model.downsample is not None:
                ttnn_module_args.downsample[0]["use_dram_for_matmul"] = True
                downsample_weight, downsample_bias = fold_batch_norm2d_into_conv2d(
                    model.downsample[0], model.downsample[1]
                )
                parameters["downsample"], pconfig4 = preprocess_conv2d(
                    downsample_weight, downsample_bias, ttnn_module_args.downsample[0], True
                )
                ttnn_module_args["downsample"] = ttnn_module_args.downsample[0]

                logger.debug(f"pconfig4: {pconfig4.num_cores_nhw}")

        elif isinstance(model, torchvision.models.resnet.ResNet):
            ttnn_module_args.conv1["activation"] = "relu"  # Fuse relu with conv1
            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
            parameters["conv1"] = fold_conv7s2_into_conv4s1(conv1_weight, conv1_bias, ttnn_module_args.conv1)

            named_parameters = tuple(
                (name, parameter) for name, parameter in model.named_parameters() if "." not in name
            )
            for child_name, child in tuple(model.named_children()) + named_parameters:
                if child_name in {"conv1", "bn1"}:
                    continue
                parameters[child_name] = convert_torch_model_to_ttnn_model(
                    child,
                    name=name,
                    convert_to_ttnn=convert_to_ttnn,
                    custom_preprocessor=custom_preprocessor,
                    ttnn_module_args=ttnn_module_args.get(child_name, None),
                )
        return parameters

    reader_patterns_cache = {}
    torch_input_tensor = torch.rand((batch_size, 3, 224, 224), dtype=torch.float32)
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        reader_patterns_cache=reader_patterns_cache,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    for module in range(1, 5):
        parameters.layer3[module].conv1.conv.is_1d_systolic = False
        parameters.layer3[module].conv2.conv.is_1d_systolic = False
        parameters.layer3[module].conv3.conv.is_1d_systolic = False
    for module in range(1, 3):
        parameters.layer4[module].conv1.conv.is_1d_systolic = False
        parameters.layer4[module].conv2.conv.is_1d_systolic = False
        parameters.layer4[module].conv3.conv.is_1d_systolic = False

    torch_model.to(torch.bfloat16)
    torch_input_tensor_test = torch_input_tensor.to(torch.bfloat16)
    torch_output_tensor = torch_model(torch_input_tensor_test)

    input_tensor = pad_and_fold_conv_activation_for_unity_stride(
        torch_input_tensor, *torch_model.conv1.padding, *torch_model.conv1.stride
    )
    input_tensor = torch.permute(input_tensor, (0, 2, 3, 1))
    layout = ttnn.TILE_LAYOUT if act_dtype == ttnn.bfloat8_b else ttnn.ROW_MAJOR_LAYOUT
    input_tensor = ttnn.from_torch(input_tensor, dtype=act_dtype, layout=layout)
    output_tensor = parameters.conv1.copy_input_to_device(input_tensor)

    output_tensor = parameters.conv1(output_tensor)
    output_tensor = parameters.maxpool(output_tensor)

    """
    1st bottleneck layer. all the blocks implemented by ttnn
    """
    output_tensor = ttnn.reshape(output_tensor, (1, 1, 56 * 56 * batch_size, 64))
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)

    module = 1
    for bottleneck_block_parameters in list(parameters.layer1.values()):
        logger.debug(f"parameters 1st block {bottleneck_block_parameters}")
        output_tensor = resnet_bottleneck_block(
            output_tensor, bottleneck_block_parameters, layer=1, module=module, device=device
        )
        module += 1

    """
    2nd bottleneck layer. 1st block implemented by torch rest by ttnn
    """
    module = 1
    for bottleneck_block_parameters in list(parameters.layer2.values()):
        logger.debug(f"parameters 2nd block {bottleneck_block_parameters}")
        output_tensor = resnet_bottleneck_block(
            output_tensor, bottleneck_block_parameters, layer=2, device=device, module=module
        )
        module += 1

    """
    3rd bottleneck layer. 1st block implemented by torch rest by ttnn
    """
    module = 1
    for bottleneck_block_parameters in list(parameters.layer3.values()):
        logger.debug(f"parameters 3rd block {bottleneck_block_parameters}")
        output_tensor = resnet_bottleneck_block(
            output_tensor, bottleneck_block_parameters, layer=3, module=module, device=device
        )
        module += 1

    """
    4th bottleneck layer. 1st block implemented by torch rest by ttnn
    """
    module = 1
    for bottleneck_block_parameters in list(parameters.layer4.values()):
        logger.debug(f"parameters 4th block {bottleneck_block_parameters}")
        output_tensor = resnet_bottleneck_block(
            output_tensor, bottleneck_block_parameters, layer=4, module=module, device=device
        )
        module += 1

    # """
    # the last layers of the resnet
    # """
    output_tensor = ttnn.to_memory_config(output_tensor, ttnn.L1_MEMORY_CONFIG)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.reshape(output_tensor, (batch_size, 1, 49, 2048))
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.global_avg_pool2d(output_tensor)
    output_tensor = output_tensor @ parameters.fc.weight + parameters.fc.bias

    """
    output verify
    """
    output_tensor_test = ttnn.to_torch(output_tensor)
    output_tensor_test = torch.reshape(output_tensor_test, (batch_size, 1000))
    assert_with_pcc(torch_output_tensor, output_tensor_test, pcc=0.98)
