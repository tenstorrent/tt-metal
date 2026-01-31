# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn as nn

from ttnn.dot_access import make_dot_access_dict
from ttnn.model_preprocessing import (
    ModuleArgs,
    Conv2dArgs,
    MaxPool2dArgs,
    fold_batch_norm2d_into_conv2d,
    convert_torch_model_to_ttnn_model,
)
from efficientnet_pytorch import EfficientNet
from models.experimental.efficientdetd0.reference.efficientdet import EfficientDetBackbone
from models.experimental.efficientdetd0.reference.modules import BiFPN, Regressor, Classifier, SeparableConvBlock


def _preprocess_conv_bn_parameter(conv, bn, *, dtype=ttnn.bfloat16, mesh_mapper=None):
    parameters = {}
    conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(conv, bn)
    parameters["weight"] = ttnn.from_torch(conv_weight, dtype=dtype, mesh_mapper=mesh_mapper)
    parameters["bias"] = ttnn.from_torch(conv_bias.reshape((1, 1, 1, -1)), dtype=dtype, mesh_mapper=mesh_mapper)
    return parameters


def _preprocess_conv_params(conv, *, dtype=ttnn.bfloat16, mesh_mapper=None):
    parameters = {}
    weight = conv.weight
    bias = conv.bias
    parameters["weight"] = ttnn.from_torch(weight, dtype=dtype, mesh_mapper=mesh_mapper)
    parameters["bias"] = ttnn.from_torch(bias.reshape((1, 1, 1, -1)), dtype=dtype, mesh_mapper=mesh_mapper)
    return parameters


def _extract_seperable_conv(model, bn=None, dtype=ttnn.bfloat16, mesh_mapper=None):
    assert isinstance(model, SeparableConvBlock)
    parameters = {}
    parameters["depthwise_conv"] = {}
    parameters["depthwise_conv"]["weight"] = ttnn.from_torch(
        model.depthwise_conv.weight, dtype=dtype, mesh_mapper=mesh_mapper
    )
    parameters["depthwise_conv"]["bias"] = None
    if model.depthwise_conv.bias is not None:
        bias = model.depthwise_conv.bias
        bias = bias.reshape((1, 1, 1, -1))
        parameters["depthwise_conv"]["bias"] = ttnn.from_torch(bias, dtype=dtype, mesh_mapper=mesh_mapper)

    parameters["pointwise_conv"] = {}
    if bn is not None:
        weight, bias = fold_batch_norm2d_into_conv2d(model.pointwise_conv, bn)
    elif hasattr(model, "bn"):
        weight, bias = fold_batch_norm2d_into_conv2d(model.pointwise_conv, model.bn)
    else:
        weight, bias = model.pointwise_conv.weight, model.pointwise_conv.bias

    parameters["pointwise_conv"]["weight"] = ttnn.from_torch(weight, dtype=dtype, mesh_mapper=mesh_mapper)
    if bias is not None:
        bias = bias.reshape((1, 1, 1, -1))
        parameters["pointwise_conv"]["bias"] = ttnn.from_torch(bias, dtype=dtype, mesh_mapper=mesh_mapper)

    return parameters


def custom_preprocessor(
    model, name, ttnn_module_args, convert_to_ttnn, custom_preprocessor_func=None, mesh_mapper=None
):
    parameters = {}
    weight_dtype = ttnn.bfloat16

    if isinstance(model, SeparableConvBlock):
        parameters = _extract_seperable_conv(model, dtype=weight_dtype, mesh_mapper=mesh_mapper)
    elif isinstance(
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
                parameters["conv_list"][layer_num][id] = _extract_seperable_conv(
                    model.conv_list[id], bn, dtype=weight_dtype, mesh_mapper=mesh_mapper
                )
            parameters["header_list"][layer_num] = _extract_seperable_conv(
                model.header, dtype=weight_dtype, mesh_mapper=mesh_mapper
            )
    elif isinstance(model, BiFPN):
        # Let the sub-modules handle their own preprocessing
        for child_name, child in model.named_children():
            if isinstance(child, SeparableConvBlock):
                parameters[child_name] = _extract_seperable_conv(child, dtype=weight_dtype, mesh_mapper=mesh_mapper)
            elif isinstance(child, nn.Sequential) and len(child) > 1:
                if isinstance(child[0], nn.Conv2d) and isinstance(child[1], nn.BatchNorm2d):
                    parameters[child_name] = {}
                    parameters[child_name][0] = _preprocess_conv_bn_parameter(
                        child[0], child[1], dtype=weight_dtype, mesh_mapper=mesh_mapper
                    )
                else:
                    continue  # Maxpool case
        # Store attention weights if using fast attention
        if model.attention:
            parameters["p6_w1"] = ttnn.from_torch(
                model.p6_w1.data, dtype=weight_dtype, mesh_mapper=mesh_mapper, layout=ttnn.TILE_LAYOUT
            )
            parameters["p5_w1"] = ttnn.from_torch(
                model.p5_w1.data, dtype=weight_dtype, mesh_mapper=mesh_mapper, layout=ttnn.TILE_LAYOUT
            )
            parameters["p4_w1"] = ttnn.from_torch(
                model.p4_w1.data, dtype=weight_dtype, mesh_mapper=mesh_mapper, layout=ttnn.TILE_LAYOUT
            )
            parameters["p3_w1"] = ttnn.from_torch(
                model.p3_w1.data, dtype=weight_dtype, mesh_mapper=mesh_mapper, layout=ttnn.TILE_LAYOUT
            )
            parameters["p4_w2"] = ttnn.from_torch(
                model.p4_w2.data, dtype=weight_dtype, mesh_mapper=mesh_mapper, layout=ttnn.TILE_LAYOUT
            )
            parameters["p5_w2"] = ttnn.from_torch(
                model.p5_w2.data, dtype=weight_dtype, mesh_mapper=mesh_mapper, layout=ttnn.TILE_LAYOUT
            )
            parameters["p6_w2"] = ttnn.from_torch(
                model.p6_w2.data, dtype=weight_dtype, mesh_mapper=mesh_mapper, layout=ttnn.TILE_LAYOUT
            )
            parameters["p7_w2"] = ttnn.from_torch(
                model.p7_w2.data, dtype=weight_dtype, mesh_mapper=mesh_mapper, layout=ttnn.TILE_LAYOUT
            )
    elif isinstance(model, EfficientNet):
        parameters = {}
        parameters["_conv_stem"] = _preprocess_conv_bn_parameter(
            model._conv_stem, model._bn0, dtype=weight_dtype, mesh_mapper=mesh_mapper
        )
        parameters["_blocks"] = {}
        for idx, block in enumerate(model._blocks):
            block_parameters = {}
            if hasattr(block, "_expand_conv"):
                block_parameters["_expand_conv"] = _preprocess_conv_bn_parameter(
                    block._expand_conv, block._bn0, dtype=weight_dtype, mesh_mapper=mesh_mapper
                )
            block_parameters["_depthwise_conv"] = _preprocess_conv_bn_parameter(
                block._depthwise_conv, block._bn1, dtype=weight_dtype, mesh_mapper=mesh_mapper
            )
            block_parameters["_se_reduce"] = _preprocess_conv_params(
                block._se_reduce, dtype=weight_dtype, mesh_mapper=mesh_mapper
            )
            block_parameters["_se_expand"] = _preprocess_conv_params(
                block._se_expand, dtype=weight_dtype, mesh_mapper=mesh_mapper
            )
            block_parameters["_project_conv"] = _preprocess_conv_bn_parameter(
                block._project_conv, block._bn2, dtype=weight_dtype, mesh_mapper=mesh_mapper
            )
            parameters["_blocks"][idx] = block_parameters
    elif isinstance(model, EfficientDetBackbone):
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


class UpsampleArgs(ModuleArgs):
    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        return super().__repr__()


def register_layer_hooks(model, layer_type):
    """Register hooks on all instances of a given layer type."""
    layer_info = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            # input and output are tuples
            input_shape = tuple(input[0].shape) if isinstance(input, (tuple, list)) else tuple(input.shape)
            output_shape = tuple(output.shape) if isinstance(output, torch.Tensor) else tuple(output[0].shape)
            if name not in layer_info:
                layer_info[name] = {}
            if isinstance(module, torch.nn.Conv2d):
                layer_info[name][len(layer_info[name])] = Conv2dArgs(
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
                    dtype=ttnn.bfloat16,
                )
            elif isinstance(module, torch.nn.MaxPool2d):
                layer_info[name][len(layer_info[name])] = MaxPool2dArgs(
                    kernel_size=getattr(module, "kernel_size", None),
                    stride=getattr(module, "stride", None),
                    padding=getattr(module, "padding", None),
                    padding_mode=getattr(module, "padding_mode", None),
                    dilation=getattr(module, "dilation", None),
                    batch_size=input_shape[0],
                    channels=input_shape[1],
                    input_height=input_shape[-2],
                    input_width=input_shape[-1],
                    dtype=ttnn.bfloat16,
                )
            elif isinstance(module, torch.nn.Upsample):
                layer_info[name][len(layer_info[name])] = UpsampleArgs(
                    scale_factor=getattr(module, "scale_factor", None),
                    mode=getattr(module, "mode", "nearest"),
                    batch_size=input_shape[0],
                    channels=input_shape[1],
                    input_height=input_shape[-2],
                    input_width=input_shape[-1],
                    dtype=ttnn.bfloat16,
                )
            else:
                layer_info[name][len(layer_info[name])] = {}

        return hook_fn

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, layer_type):
            hooks.append(module.register_forward_hook(make_hook(name)))

    return layer_info, hooks


def _expand_dotted_keys(flat_dict):
    """
    Helper function to convert dot separated layer name keys to nested dict format
    {"classifier.conv1.kernel_size": 3} -> {"classifier": {"conv1": {"kernel_size": 3}}}
    """
    result = {}

    for key, value in flat_dict.items():
        parts = key.split(".")
        current = result
        for i, part in enumerate(parts):
            # convert numeric keys to int
            if part.isdigit():
                part = int(part)

            if i == len(parts) - 1:
                # last part — assign the value
                current[part] = value
            else:
                if part not in current or not isinstance(current[part], dict):
                    current[part] = {}
                current = current[part]
    return result


def _fix_layername(layer_info):
    """
    Helper function for updating layer names to represent model init structure instead of the forward flow retured by hook.
    """
    structured_info = {}
    for layer_name, instances in layer_info.items():
        if len(instances) > 1:
            # Cases where same layer is called multiple times in model's forward call
            op_name = layer_name[layer_name.rfind(".") + 1 :]
            layer_tree = layer_name[: layer_name.rfind(".")]
            for idx, instance in instances.items():
                if "conv_list" in layer_tree:
                    # Fix for nested loop in forward call, we need the instance index to be right after "conv_list" in params
                    updated_layer_tree = (
                        layer_tree[: layer_tree.rfind(".") + 1] + str(idx) + layer_tree[layer_tree.rfind(".") :]
                    )
                    updated_layer_name = updated_layer_tree + "." + op_name
                else:
                    updated_layer_name = layer_tree + f".{idx}." + op_name
                structured_info[updated_layer_name] = instance
        else:
            structured_info[layer_name] = instances[0]
    return structured_info


def _make_dot_accessible_args(layer_info):
    structured_info = _fix_layername(layer_info)
    structured_args = _expand_dotted_keys(structured_info)
    return make_dot_access_dict(structured_args, ignore_types=(ModuleArgs,))


def infer_torch_module_args(model, input, layer_type=(nn.Conv2d, nn.MaxPool2d, nn.Upsample)):
    """Run forward pass and collect layer information."""
    model.eval()

    layer_info, hooks = register_layer_hooks(model, layer_type)

    with torch.no_grad():
        _ = model(input)

    # Remove hooks to avoid memory leaks
    for h in hooks:
        h.remove()

    return _make_dot_accessible_args(layer_info)
