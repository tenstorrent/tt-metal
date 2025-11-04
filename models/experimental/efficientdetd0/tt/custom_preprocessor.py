# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn as nn

from ttnn.dot_access import make_dot_access_dict
from ttnn.model_preprocessing import (
    ModuleArgs,
    MaxPool2dArgs,
    convert_torch_model_to_ttnn_model,
    fold_batch_norm2d_into_conv2d,
)

from models.experimental.efficientdetd0.reference.efficientdet import EfficientDetBackbone
from models.experimental.efficientdetd0.reference.modules import Regressor, Classifier, BiFPN
from models.experimental.efficientdetd0.reference.modules import SeparableConvBlock


def preprocess_conv_parameter(parameter, *, dtype):
    parameter = ttnn.from_torch(parameter, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return parameter


def _extract_seperable_conv(model, bn=None):
    assert isinstance(model, SeparableConvBlock)
    parameters = {}
    parameters["depthwise_conv"] = {}
    parameters["depthwise_conv"]["weight"] = ttnn.from_torch(model.depthwise_conv.weight, dtype=ttnn.float32)
    parameters["depthwise_conv"]["bias"] = None
    if model.depthwise_conv.bias is not None:
        bias = model.depthwise_conv.bias
        bias = bias.reshape((1, 1, 1, -1))
        parameters["depthwise_conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

    parameters["pointwise_conv"] = {}
    if bn is not None:
        weight, bias = fold_batch_norm2d_into_conv2d(model.pointwise_conv, bn)
    elif hasattr(model, "bn"):
        weight, bias = fold_batch_norm2d_into_conv2d(model.pointwise_conv, model.bn)
    else:
        weight, bias = model.pointwise_conv.weight, model.pointwise_conv.bias

    parameters["pointwise_conv"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
    if bias is not None:
        bias = bias.reshape((1, 1, 1, -1))
        parameters["pointwise_conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

    return parameters


def custom_preprocessor(
    model, name, ttnn_module_args, convert_to_ttnn, custom_preprocessor_func=None, mesh_mapper=None
):
    parameters = {}
    weight_dtype = ttnn.bfloat16

    if isinstance(model, torch.nn.BatchNorm2d):
        weight = model.weight
        bias = model.bias
        running_mean = model.running_mean
        running_var = model.running_var
        weight = weight[None, :, None, None]
        bias = bias[None, :, None, None]
        running_mean = running_mean[None, :, None, None]
        running_var = running_var[None, :, None, None]
        parameters["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        parameters["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        parameters["running_mean"] = ttnn.from_torch(running_mean, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        parameters["running_var"] = ttnn.from_torch(running_var, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        parameters["eps"] = model.eps
    if isinstance(model, SeparableConvBlock):
        parameters = _extract_seperable_conv(model)
    if isinstance(
        model,
        (
            Regressor,
            Classifier,
        ),
    ):
        parameters["conv_list"] = {}
        parameters["header_list"] = {}
        # Creating batchnorm folded conv weights; multiple copies of conv, one for each pyramid layer
        for layer_num, pyramid_layer_bn_list in enumerate(model.bn_list):
            parameters["conv_list"][layer_num] = {}
            for id, bn in enumerate(pyramid_layer_bn_list):
                # parameters["conv_list"][layer_num][id] = _extract_seperable_conv(model.conv_list[id])
                parameters["conv_list"][layer_num][id] = _extract_seperable_conv(model.conv_list[id], bn)
            parameters["header_list"][layer_num] = _extract_seperable_conv(model.header)

    if isinstance(model, BiFPN):
        # Process all separable conv blocks for upsampling path
        parameters["conv6_up"] = _extract_seperable_conv(model.conv6_up)
        parameters["conv5_up"] = _extract_seperable_conv(model.conv5_up)
        parameters["conv4_up"] = _extract_seperable_conv(model.conv4_up)
        parameters["conv3_up"] = _extract_seperable_conv(model.conv3_up)

        # Process all separable conv blocks for downsampling path
        parameters["conv4_down"] = _extract_seperable_conv(model.conv4_down)
        parameters["conv5_down"] = _extract_seperable_conv(model.conv5_down)
        parameters["conv6_down"] = _extract_seperable_conv(model.conv6_down)
        parameters["conv7_down"] = _extract_seperable_conv(model.conv7_down)

        if model.use_p8:
            parameters["conv7_up"] = _extract_seperable_conv(model.conv7_up)
            parameters["conv8_down"] = _extract_seperable_conv(model.conv8_down)

        # Process first_time channel reduction layers
        if model.first_time:
            # Extract conv + batchnorm for channel reduction
            parameters["p3_down_channel"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.p3_down_channel[0], model.p3_down_channel[1])
            parameters["p3_down_channel"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.float32)
            parameters["p3_down_channel"]["bias"] = ttnn.from_torch(
                conv_bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32
            )

            parameters["p4_down_channel"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.p4_down_channel[0], model.p4_down_channel[1])
            parameters["p4_down_channel"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.float32)
            parameters["p4_down_channel"]["bias"] = ttnn.from_torch(
                conv_bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32
            )

            parameters["p5_down_channel"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.p5_down_channel[0], model.p5_down_channel[1])
            parameters["p5_down_channel"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.float32)
            parameters["p5_down_channel"]["bias"] = ttnn.from_torch(
                conv_bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32
            )

            # P5 to P6 conversion (conv + bn + maxpool)
            parameters["p5_to_p6_conv"] = {}
            # import pdb; pdb.set_trace()
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.p5_to_p6[0], model.p5_to_p6[1])
            parameters["p5_to_p6_conv"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.float32)
            parameters["p5_to_p6_conv"]["bias"] = ttnn.from_torch(conv_bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32)
            # Note: p5_to_p6[2] is MaxPool2d, handled separately in conv_params

            # Additional channel reduction for bottom-up path
            parameters["p4_down_channel_2"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(
                model.p4_down_channel_2[0], model.p4_down_channel_2[1]
            )
            parameters["p4_down_channel_2"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.float32)
            parameters["p4_down_channel_2"]["bias"] = ttnn.from_torch(
                conv_bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32
            )

            parameters["p5_down_channel_2"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(
                model.p5_down_channel_2[0], model.p5_down_channel_2[1]
            )
            parameters["p5_down_channel_2"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.float32)
            parameters["p5_down_channel_2"]["bias"] = ttnn.from_torch(
                conv_bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32
            )

        # Store attention weights if using fast attention
        if model.attention:
            parameters["p6_w1"] = model.p6_w1.data
            parameters["p5_w1"] = model.p5_w1.data
            parameters["p4_w1"] = model.p4_w1.data
            parameters["p3_w1"] = model.p3_w1.data
            parameters["p4_w2"] = model.p4_w2.data
            parameters["p5_w2"] = model.p5_w2.data
            parameters["p6_w2"] = model.p6_w2.data
            parameters["p7_w2"] = model.p7_w2.data

    elif isinstance(
        model,
        (EfficientDetBackbone,),
    ):
        # Let the sub-modules handle their own preprocessing
        for child_name, child in model.named_children():
            parameters[child_name] = convert_torch_model_to_ttnn_model(
                child,
                name=f"{name}.{child_name}",
                custom_preprocessor=custom_preprocessor_func,
                convert_to_ttnn=convert_to_ttnn,
                ttnn_module_args=ttnn_module_args,
            )

    return parameters


def create_custom_mesh_preprocessor(mesh_mapper=None):
    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        return custom_preprocessor(
            model, name, ttnn_module_args, convert_to_ttnn, custom_mesh_preprocessor, mesh_mapper
        )

    return custom_mesh_preprocessor


class ConvArgs(ModuleArgs):
    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        return super().__repr__()


class Args(ModuleArgs):
    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        return super().__repr__()


def infer_module_args(model):
    if isinstance(
        model,
        (
            torch.nn.Conv1d,
            torch.nn.Conv2d,
        ),
    ):
        return ConvArgs(
            in_channels=model.in_channels,
            out_channels=model.out_channels,
            kernel_size=model.kernel_size,
            stride=model.stride,
            padding=model.padding,
            dilation=model.dilation,
            groups=model.groups,
            padding_mode=model.padding_mode,
        )
    elif isinstance(model, torch.nn.MaxPool2d):
        return MaxPool2dArgs(
            kernel_size=model.kernel_size,
            stride=model.stride,
            padding=model.padding,
            dilation=model.dilation,
        )
    else:
        module_args = {}
        for child_name, child in model.named_children():
            module_args[child_name] = infer_module_args(child)

    return make_dot_access_dict(module_args, ignore_types=(ModuleArgs,))


def register_layer_hooks(model, layer_type):
    """Register hooks on all instances of a given layer type."""
    layer_info = {}

    def hook_fn(module, input, output):
        # input and output are tuples
        input_shape = tuple(input[0].shape) if isinstance(input, (tuple, list)) else tuple(input.shape)
        output_shape = tuple(output.shape) if isinstance(output, torch.Tensor) else tuple(output[0].shape)

        layer_info[len(layer_info)] = Args(
            kernel_size=getattr(module, "kernel_size", None),
            stride=getattr(module, "stride", None),
            padding=getattr(module, "padding", None),
            padding_mode=getattr(module, "padding_mode", None),
            dilation=getattr(module, "dilation", None),
            groups=getattr(module, "groups", None),
            in_channels=getattr(module, "in_channels", None),
            out_channels=getattr(module, "out_channels", None),
            batch_size=input_shape[0],
            input_height=input_shape[-2],
            input_width=input_shape[-1],
        )

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, layer_type):
            hooks.append(module.register_forward_hook(hook_fn))

    return layer_info, hooks


def infer_torch_module_args(model, input, layer_type=(nn.Conv2d, nn.MaxPool2d)):
    """Run forward pass and collect layer information."""
    model.eval()

    layer_info, hooks = register_layer_hooks(model, layer_type)

    with torch.no_grad():
        _ = model(input)

    # Remove hooks to avoid memory leaks
    for h in hooks:
        h.remove()

    return layer_info
