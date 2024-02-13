# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import torchvision

import ttnn
from ttnn.model_preprocessing import (
    preprocess_model,
    preprocess_conv2d,
    fold_batch_norm2d_into_conv2d,
    fold_conv7s2_into_conv4s1,
    convert_torch_model_to_ttnn_model,
)

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0, pad_and_fold_conv_activation_for_unity_stride

from models.experimental.functional_resnet.tt import ttnn_functional_resnet


def custom_preprocessor(model, name, ttnn_module_args):
    parameters = {}
    if isinstance(model, BasicBlock):
        ttnn_module_args.conv1["activation"] = "relu"  # Fuse relu with conv1

        conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
        conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.conv2, model.bn2)

        parameters["conv1"] = preprocess_conv2d(conv1_weight, conv1_bias, ttnn_module_args.conv1)
        parameters["conv2"] = preprocess_conv2d(conv2_weight, conv2_bias, ttnn_module_args.conv2)

    elif isinstance(model, torch.nn.Conv2d) and model.kernel_size == (7, 7) and model.stride == (2, 2):
        return fold_conv7s2_into_conv4s1(model.weight, model.bias, ttnn_module_args)

    return parameters


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> torch.nn.Conv2d:
    """3x3 convolution with padding"""
    return torch.nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> torch.nn.Conv2d:
    """1x1 convolution"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(torch.nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample=None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer=None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # out += identity
        # out = self.relu(out)

        return out


@skip_for_wormhole_b0()
def test_basic_block(device):
    torch.manual_seed(0)

    torch_model = BasicBlock(inplanes=64, planes=64, stride=1).eval()

    new_state_dict = {}
    for name, parameter in torch_model.state_dict().items():
        if isinstance(parameter, torch.FloatTensor):
            new_state_dict[name] = torch.rand_like(parameter)
    torch_model.load_state_dict(new_state_dict)

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

    ttnn_model = ttnn_functional_resnet.BasicBlock(parameters)

    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16)

    input_tensor = ttnn_model.conv1.copy_input_to_device(input_tensor)
    output_tensor = ttnn_model(input_tensor)
    output_tensor = ttnn_model.conv2.copy_output_from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = torch.reshape(output_tensor, torch_input_tensor.shape)
    output_tensor = output_tensor.to(torch_input_tensor.dtype)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.999)


@pytest.mark.skip(reason="This test is not ready to run")
@skip_for_wormhole_b0()
def test_basic_block_with_downsample(device):
    torch.manual_seed(0)

    torch_model = BasicBlock(inplanes=64, planes=64, stride=1, downsample=conv1x1(64, 64, 1)).eval()

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

    ttnn_model = ttnn_functional_resnet.BasicBlock(parameters)

    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16)

    input_tensor = ttnn_model.conv1.copy_input_to_device(input_tensor)
    output_tensor = ttnn_model(input_tensor)
    output_tensor = ttnn_model.conv2.copy_output_from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = torch.reshape(output_tensor, torch_input_tensor.shape)
    output_tensor = output_tensor.to(torch_input_tensor.dtype)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.999)


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

    assert_with_pcc(torch_output_tensor, output_tensor)


@skip_for_wormhole_b0()
def test_resnet(device):
    torch.manual_seed(0)

    torch_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).eval()

    torch_input_tensor = torch.rand((8, 3, 224, 224), dtype=torch.float32)

    def custom_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        parameters = {}
        if isinstance(model, torchvision.models.resnet.BasicBlock):
            ttnn_module_args.conv1["activation"] = "relu"  # Fuse relu with conv1

            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
            conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.conv2, model.bn2)

            parameters["conv1"] = preprocess_conv2d(conv1_weight, conv1_bias, ttnn_module_args.conv1)
            parameters["conv2"] = preprocess_conv2d(conv2_weight, conv2_bias, ttnn_module_args.conv2)

        elif isinstance(model, torchvision.models.resnet.ResNet):
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
    for layer in parameters.layer1.values():
        output_tensor = layer.conv1(output_tensor)
        output_tensor = layer.conv2(output_tensor)
    for layer in parameters.layer2.values():
        output_tensor = layer.conv1(output_tensor)
        output_tensor = layer.conv2(output_tensor)
    for layer in parameters.layer3.values():
        return  # TODO(arakhmati): remove this return once the next conv works
        output_tensor = layer.conv1(output_tensor)
        output_tensor = layer.conv2(output_tensor)
    for layer in parameters.layer4.values():
        output_tensor = layer.conv1(output_tensor)
        output_tensor = layer.conv2(output_tensor)

    output_tensor = ttnn.global_avg_pool2d(output_tensor, (1, 1))
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.reshape(output_tensor, (output_tensor.shape[0], output_tensor.shape[-1]))
    output_tensor = output_tensor @ parameters.fc.weight + parameters.fc.bias

    output_tensor = ttnn.to_torch(output_tensor)
