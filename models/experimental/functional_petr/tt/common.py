# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch

from ttnn.model_preprocessing import (
    preprocess_linear_weight,
    preprocess_layernorm_parameter,
    preprocess_linear_bias,
    ParameterDict,
    ParameterList,
    fold_batch_norm2d_into_conv2d,
)

from models.experimental.functional_petr.reference.petr_head import PETRHead
from models.experimental.functional_petr.reference.cp_fpn import CPFPN
from models.experimental.functional_petr.reference.vovnetcp import VoVNetCP, eSEModule, _OSA_module, _OSA_stage
from torch.nn import Conv2d, Linear

from torch import nn


class Conv:
    def __init__(
        self,
        conv_params,
        parameters,
        *,
        act_block_h=None,
        reshard=False,
        deallocate=True,
        height_sharding=True,
        width_sharding=None,
        activation="",
        dtype=ttnn.bfloat16,
        groups=1,
        dilation=1,
        use_shallow_conv_variant=False,
    ) -> None:
        self.weights = parameters["weight"]
        self.bias = parameters["bias"]

        self.kernel_size = (self.weights.shape[2], self.weights.shape[3])
        self.conv_params = conv_params
        self.out_channels = self.weights.shape[0]
        self.act_block_h = act_block_h
        self.reshard = reshard
        self.deallocate = deallocate
        self.activation = activation
        self.dtype = dtype
        if width_sharding == True:
            self.shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
        else:
            self.shard_layout = (
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED if height_sharding else ttnn.TensorMemoryLayout.BLOCK_SHARDED
            )
        self.groups = groups
        self.dilation = dilation
        self.use_shallow_conv_variant = use_shallow_conv_variant

    def __call__(self, device, input_tensor):
        batch_size = input_tensor.shape[0]
        input_height = input_tensor.shape[1]
        input_width = input_tensor.shape[2]
        input_channels = input_tensor.shape[3]

        # Ensure input is in DRAM and interleaved
        input_tensor = ttnn.to_memory_config(input_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        if hasattr(input_tensor, "memory_config") and input_tensor.memory_config().is_sharded():
            input_tensor = ttnn.sharded_to_interleaved(input_tensor, ttnn.L1_MEMORY_CONFIG)

        # Ensure weights are in DRAM and interleaved
        if hasattr(self.weights, "memory_config") and self.weights.memory_config().is_sharded():
            self.weights = ttnn.sharded_to_interleaved(self.weights, ttnn.L1_MEMORY_CONFIG)

        # High precision compute config
        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
            math_approx_mode=False,
        )

        # Single unified path for all convolutions
        [output_tensor, [_out_height, _out_width]] = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights,
            bias_tensor=self.bias,
            device=device,
            in_channels=input_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            kernel_size=self.kernel_size,
            stride=(self.conv_params[0], self.conv_params[1]),
            padding=(self.conv_params[2], self.conv_params[3]),
            dilation=(self.dilation, self.dilation),
            groups=self.groups,
            compute_config=compute_config,
            return_output_dim=True,
            return_weights_and_bias=False,
        )

        # Apply activation if specified
        if self.activation == "relu":
            output_tensor = ttnn.relu(output_tensor)
        elif self.activation == "gelu":
            output_tensor = ttnn.gelu(output_tensor)

        # Post-processing
        if hasattr(output_tensor, "memory_config") and output_tensor.memory_config().is_sharded():
            output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)

        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.reshape(output_tensor, (batch_size, _out_height, _out_width, output_tensor.shape[3]))
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)

        return output_tensor


class Conv_with_split:
    def __init__(
        self,
        conv_params,
        parameters,
        *,
        act_block_h=None,
        reshard=False,
        deallocate=False,
        height_sharding=True,
        width_sharding=None,
        activation="",
        dtype=ttnn.bfloat16,
        groups=1,
        dilation=1,
        use_shallow_conv_variant=False,
        split_factor=2,
    ) -> None:
        self.weights = parameters["weight"]
        self.bias = parameters["bias"]
        input_channels = self.weights.shape[1]
        self.output_channels = self.weights.shape[0]
        assert input_channels % split_factor == 0
        self.split_input_channels = input_channels // split_factor
        if width_sharding == True:
            self.shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
        else:
            self.shard_layout = (
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED if height_sharding else ttnn.TensorMemoryLayout.BLOCK_SHARDED
            )
        self.dtype = dtype
        self.split_factor = split_factor
        self.act_block_h = act_block_h
        self.conv_params = conv_params
        self.activation = activation
        self.kernel_size = (self.weights.shape[2], self.weights.shape[3])

    def __call__(self, device, input_tensor):
        batch, height, width, channel = input_tensor.shape

        # Convert input to torch if needed
        if not isinstance(input_tensor, torch.Tensor):
            input_tensor = ttnn.to_torch(input_tensor)

        # Convert weights to torch if needed
        if isinstance(self.weights, torch.Tensor):
            # Already torch tensor, use as-is
            weights_torch = self.weights
        else:
            # Convert from ttnn
            weights_torch = ttnn.to_torch(self.weights)

        # Convert bias to torch if needed
        if isinstance(self.bias, torch.Tensor):
            bias_torch = self.bias
        else:
            bias_torch = ttnn.to_torch(self.bias)

        split_input_tensors = torch.split(input_tensor, self.split_input_channels, 3)
        split_weight_tensors = torch.split(weights_torch, self.split_input_channels, 1)

        weights_dtype = ttnn.bfloat16

        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
            math_approx_mode=False,
        )

        for i in range(self.split_factor):
            tt_weight_tensor = ttnn.from_torch(split_weight_tensors[i], weights_dtype)

            tt_bias_tensor = None
            tt_input_tensor = ttnn.from_torch(
                split_input_tensors[i], ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
            )

            [tt_output_tensor_on_device, [out_height, out_width]] = ttnn.conv2d(
                input_tensor=tt_input_tensor,
                weight_tensor=tt_weight_tensor,
                bias_tensor=tt_bias_tensor,
                device=device,
                in_channels=self.split_input_channels,
                out_channels=self.output_channels,
                batch_size=batch,
                input_height=height,
                input_width=width,
                kernel_size=self.kernel_size,
                stride=(self.conv_params[0], self.conv_params[1]),
                padding=(self.conv_params[2], self.conv_params[3]),
                dilation=(1, 1),
                groups=1,
                compute_config=compute_config,
                return_output_dim=True,
                return_weights_and_bias=False,
            )

            tt_conv_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
            torch_conv_output_tensor = ttnn.to_torch(tt_conv_output_tensor)

            if i == 0:
                torch_output_tensor = torch_conv_output_tensor
            else:
                torch_output_tensor = torch.add(torch_output_tensor, torch_conv_output_tensor)

        if isinstance(bias_torch, torch.Tensor):
            torch_output_tensor = torch_output_tensor + bias_torch.view(1, 1, 1, -1)
        # Shape handling
        if len(torch_output_tensor.shape) == 2:
            torch_output_tensor = torch_output_tensor.reshape(batch, out_height, out_width, self.output_channels)
        elif torch_output_tensor.shape[1] == 1 and torch_output_tensor.shape[2] != out_width:
            torch_output_tensor = torch_output_tensor.reshape(batch, out_height, out_width, self.output_channels)

        output_tensor = ttnn.from_torch(torch_output_tensor, dtype=ttnn.bfloat16, device=device)

        if output_tensor.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)

        if output_tensor.shape[1] != out_height or output_tensor.shape[2] != out_width:
            output_tensor = ttnn.reshape(output_tensor, (batch, out_height, out_width, self.output_channels))

        expected_shape = (batch, out_height, out_width, self.output_channels)
        if output_tensor.shape != expected_shape:
            output_tensor = ttnn.reshape(output_tensor, expected_shape)

        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)

        if self.activation == "relu":
            output_tensor = ttnn.relu(output_tensor)

        del out_height, out_width
        return output_tensor


def move_to_device(object, device):
    if isinstance(object, ParameterDict):
        for name, value in list(object.items()):
            if name in ["input_proj", "adapt_pos3d", "position_encoder"]:
                continue
            object[name] = move_to_device(value, device)
        return object
    elif isinstance(object, ParameterList):
        for index, element in enumerate(object):
            object[index] = move_to_device(element, device)
        return object
    elif isinstance(object, ttnn.Tensor):
        return ttnn.to_device(object, device)
    else:
        return object


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
                # Convert to ttnn format
                parameters[prefix]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
                parameters[prefix]["bias"] = ttnn.from_torch(
                    torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                )

    return parameters


def create_custom_preprocessor_cpfpn(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, CPFPN):
            parameters["lateral_convs"] = {}
            for i, child in enumerate(model.lateral_convs):
                parameters["lateral_convs"][i] = {}
                parameters["lateral_convs"][i]["conv"] = {}
                parameters["lateral_convs"][i]["conv"]["weight"] = ttnn.from_torch(
                    child.conv.weight, dtype=ttnn.bfloat16
                )
                parameters["lateral_convs"][i]["conv"]["bias"] = ttnn.from_torch(
                    torch.reshape(child.conv.bias, (1, 1, 1, -1)),
                    dtype=ttnn.bfloat16,
                )
            parameters["fpn_convs"] = {}
            for i, child in enumerate(model.fpn_convs):
                parameters["fpn_convs"][i] = {}
                parameters["fpn_convs"][i]["conv"] = {}
                parameters["fpn_convs"][i]["conv"]["weight"] = ttnn.from_torch(child.conv.weight, dtype=ttnn.bfloat16)
                parameters["fpn_convs"][i]["conv"]["bias"] = ttnn.from_torch(
                    torch.reshape(child.conv.bias, (1, 1, 1, -1)),
                    dtype=ttnn.bfloat16,
                )
        return parameters

    return custom_preprocessor


def create_custom_preprocessor_vovnetcp(device):
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
                parameters[prefix]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
                parameters[prefix]["bias"] = ttnn.from_torch(
                    torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                )

            first_layer_name, _ = list(model.concat.named_children())[0]
            base_name = first_layer_name.split("/")[0]
            parameters[base_name] = {}
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


def create_custom_preprocessor_petr_head(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, PETRHead):
            parameters["input_proj"] = {}
            parameters["input_proj"]["weight"] = ttnn.from_torch(model.input_proj.weight, dtype=ttnn.bfloat16)
            parameters["input_proj"]["bias"] = ttnn.from_torch(
                torch.reshape(model.input_proj.bias, (1, 1, 1, -1)),
                dtype=ttnn.bfloat16,
            )

            parameters["cls_branches"] = {}
            for index, child in enumerate(model.cls_branches):
                parameters["cls_branches"][index] = {}
                for index1, child1 in enumerate(child):
                    parameters["cls_branches"][index][index1] = {}
                    if isinstance(child1, Linear):
                        parameters["cls_branches"][index][index1]["weight"] = preprocess_linear_weight(
                            child1.weight, dtype=ttnn.bfloat16
                        )
                        parameters["cls_branches"][index][index1]["bias"] = preprocess_linear_bias(
                            child1.bias, dtype=ttnn.bfloat16
                        )
                    elif isinstance(child1, nn.LayerNorm):
                        parameters["cls_branches"][index][index1]["weight"] = preprocess_layernorm_parameter(
                            child1.weight, dtype=ttnn.bfloat16
                        )
                        parameters["cls_branches"][index][index1]["bias"] = preprocess_layernorm_parameter(
                            child1.bias, dtype=ttnn.bfloat16
                        )

            parameters["reg_branches"] = {}
            for index, child in enumerate(model.reg_branches):
                parameters["reg_branches"][index] = {}
                for index1, child1 in enumerate(child):
                    parameters["reg_branches"][index][index1] = {}
                    if isinstance(child1, Linear):
                        parameters["reg_branches"][index][index1]["weight"] = preprocess_linear_weight(
                            child1.weight, dtype=ttnn.bfloat16
                        )
                        parameters["reg_branches"][index][index1]["bias"] = preprocess_linear_bias(
                            child1.bias, dtype=ttnn.bfloat16
                        )

            parameters["adapt_pos3d"] = {}
            for index, child in enumerate(model.adapt_pos3d):
                parameters["adapt_pos3d"][index] = {}
                if isinstance(child, Conv2d):
                    parameters["adapt_pos3d"][index]["weight"] = ttnn.from_torch(child.weight, dtype=ttnn.bfloat16)
                    parameters["adapt_pos3d"][index]["bias"] = ttnn.from_torch(
                        torch.reshape(child.bias, (1, 1, 1, -1)),
                        dtype=ttnn.bfloat16,
                    )

            parameters["position_encoder"] = {}
            for index, child in enumerate(model.position_encoder):
                parameters["position_encoder"][index] = {}
                if isinstance(child, Conv2d):
                    parameters["position_encoder"][index]["weight"] = ttnn.from_torch(child.weight, dtype=ttnn.bfloat16)
                    parameters["position_encoder"][index]["bias"] = ttnn.from_torch(
                        torch.reshape(child.bias, (1, 1, 1, -1)),
                        dtype=ttnn.bfloat16,
                    )

            parameters["query_embedding"] = {}
            for index, child in enumerate(model.query_embedding):
                parameters["query_embedding"][index] = {}
                if isinstance(child, Linear):
                    parameters["query_embedding"][index]["weight"] = preprocess_linear_weight(
                        child.weight, dtype=ttnn.bfloat16
                    )
                    parameters["query_embedding"][index]["bias"] = preprocess_linear_bias(
                        child.bias, dtype=ttnn.bfloat16
                    )
            parameters["reference_points"] = {}
            parameters["reference_points"]["weight"] = ttnn.from_torch(
                model.reference_points.weight, dtype=ttnn.bfloat16, device=device
            )

        return parameters

    return custom_preprocessor
