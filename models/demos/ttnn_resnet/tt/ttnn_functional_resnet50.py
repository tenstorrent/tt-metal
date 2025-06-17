# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List

import torch
from loguru import logger

import ttnn
from models.demos.ttnn_resnet.tt.ttnn_functional_resnet50_model_utils import get_conv_input_memory_config
from models.utility_functions import _nearest_y, is_blackhole, is_grayskull, is_wormhole_b0

hardcoded_matmul_config_linear = {
    8: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 4),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    ),
    16: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 4),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    ),
    20: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    ),
    32: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    ),
}

ops_parallel_config = {
    "layer1_module1_input": None,
}


def ResnetLinear(
    in_features: int,
    out_features: int,
    weight: ttnn.Tensor,
    bias: ttnn.Tensor,
    output_mem_config,
    model_config,
    device,
    batch_size,
    compute_kernel_config,
):
    """
    Returns a function for linear operation in resnet with bias.
    """

    matmul_config = hardcoded_matmul_config_linear[batch_size]
    weight = weight.reshape(weight.shape.to_rank(4))
    bias = bias.reshape(bias.shape.to_rank(4))

    def linear_(act):
        output = ttnn.linear(
            act,
            weight,
            bias=bias,
            program_config=matmul_config,
            memory_config=output_mem_config,
            dtype=model_config["ACTIVATIONS_DTYPE"],
            compute_kernel_config=compute_kernel_config,
        )
        return output

    return linear_


def do_nothing_op(x):
    return x


import math


def _nearest_32(x):
    return math.ceil(x / 32) * 32


# TODO: this function is required because conv is preprocessed before in TTNN model preprocessing flow
# We need to skip conv preprocessing there
def permute_conv_weights(weight, bias):
    weight = ttnn.to_layout(weight, layout=ttnn.ROW_MAJOR_LAYOUT)
    weight = ttnn.to_torch(weight)
    weight = torch.permute(weight, (2, 3, 0, 1))
    bias = ttnn.to_layout(bias, layout=ttnn.ROW_MAJOR_LAYOUT)
    bias = ttnn.to_torch(bias)
    return weight, bias


class resnet50Bottleneck:
    expansion: int = 4

    def __init__(self, parameters, downsample, stride, model_config) -> None:
        # init is just to pre-process pytorch weights and bias tensors
        self.conv1_weight_tensor = parameters.conv1.weight
        self.conv1_bias_tensor = parameters.conv1.bias
        self.conv1_input_channels = self.conv1_weight_tensor.shape[1]
        self.conv1_output_channels = self.conv1_weight_tensor.shape[0]
        assert self.conv1_weight_tensor.shape[2] == 1

        self.conv2_weight_tensor = parameters.conv2.weight
        self.conv2_bias_tensor = parameters.conv2.bias
        self.conv2_input_channels = self.conv2_weight_tensor.shape[1]
        self.conv2_output_channels = self.conv2_weight_tensor.shape[0]
        self.conv2_stride = 2 if downsample else 1
        assert self.conv2_weight_tensor.shape[2] == 3

        self.conv3_weight_tensor = parameters.conv3.weight
        self.conv3_bias_tensor = parameters.conv3.bias
        self.conv3_input_channels = self.conv3_weight_tensor.shape[1]
        self.conv3_output_channels = self.conv3_weight_tensor.shape[0]
        assert self.conv3_weight_tensor.shape[2] == 1

        self.downsample = downsample
        self.stride = stride
        if downsample:
            self.ds_conv_weight_tensor = parameters.downsample.weight
            self.ds_conv_bias_tensor = parameters.downsample.bias
            self.ds_conv_input_channels = self.ds_conv_weight_tensor.shape[1]
            self.ds_conv_output_channels = self.ds_conv_weight_tensor.shape[0]
            assert self.ds_conv_weight_tensor.shape[2] == 1
        self.model_config = model_config
        return

    def run_downsample_if_req(
        self,
        x,
        device,
        batch_size,
        input_height,
        input_width,
        reshard_if_not_optimal=False,
        height_sharding=None,
        packer_l1_accum_enabled=True if not is_grayskull() else False,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        enable_subblock_padding=False,
    ):
        if self.downsample:
            logger.debug(f"Running downsample")
            conv_kwargs = {
                "in_channels": self.ds_conv_input_channels,
                "out_channels": self.ds_conv_output_channels,
                "batch_size": batch_size,
                "input_height": input_height,
                "input_width": input_width,
                "kernel_size": (1, 1),
                "stride": (self.stride, self.stride),
                "padding": (0, 0),
                "dilation": (1, 1),
                "groups": 1,
                "device": device,
                "conv_config": ttnn.Conv2dConfig(
                    weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                    shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                    if height_sharding
                    else ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                    deallocate_activation=True,
                    reallocate_halo_output=True,
                    reshard_if_not_optimal=reshard_if_not_optimal,
                    enable_act_double_buffer=enable_act_double_buffer
                    if height_sharding
                    else True
                    if input_width < 56
                    else False,
                    enable_weights_double_buffer=True if input_width < 56 else False,
                    enable_split_reader=enable_split_reader,
                    enable_subblock_padding=enable_subblock_padding,
                ),
            }

            if not ttnn.is_tensor_storage_on_device(self.ds_conv_weight_tensor):
                self.ds_conv_weight_tensor = ttnn.prepare_conv_weights(
                    weight_tensor=self.ds_conv_weight_tensor,
                    weights_format="OIHW",
                    input_memory_config=x.memory_config(),
                    input_layout=x.get_layout(),
                    has_bias=True,
                    **conv_kwargs,
                    input_dtype=self.model_config["ACTIVATIONS_DTYPE"],
                )

                self.ds_conv_bias_tensor = ttnn.prepare_conv_bias(
                    bias_tensor=self.ds_conv_bias_tensor,
                    input_memory_config=x.memory_config(),
                    input_layout=x.get_layout(),
                    **conv_kwargs,
                    input_dtype=self.model_config["ACTIVATIONS_DTYPE"],
                )
                self.ds_conv_weight_tensor = ttnn.to_device(self.ds_conv_weight_tensor, device)
                self.ds_conv_bias_tensor = ttnn.to_device(self.ds_conv_bias_tensor, device)

            ds_out = ttnn.conv2d(
                input_tensor=x,
                weight_tensor=self.ds_conv_weight_tensor,
                bias_tensor=self.ds_conv_bias_tensor,
                **conv_kwargs,
                compute_config=ttnn.init_device_compute_kernel_config(
                    device.arch(),
                    math_fidelity=self.model_config["MATH_FIDELITY"],
                    packer_l1_acc=packer_l1_accum_enabled,
                ),
                return_output_dim=False,
                return_weights_and_bias=False,
                dtype=self.model_config["ACTIVATIONS_DTYPE"],
            )
            ttnn.deallocate(x)
            ds_out = ttnn.reallocate(ds_out)
        else:
            ds_out = x
        return ds_out

    def __call__(
        self,
        x,
        device,
        batch_size,
        input_height,
        input_width,
        reshard_if_not_optimal=False,
        height_sharding=None,
        eltwise_binary_out_in_place=True,
        packer_l1_acc=True if not is_grayskull() else False,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        enable_subblock_padding=False,
        ops_parallel_config=None,
        layer_module=None,
    ):
        logger.debug(
            f"==== Running {batch_size}, {input_height}, {input_width}, {self.conv1_input_channels}, {self.conv1_output_channels}"
        )

        ds_input_height = input_height
        ds_input_width = input_width

        # conv1 is 1x1 conv
        logger.debug(f"Running conv1")
        module_input_height = input_height
        module_input_width = input_width
        conv_kwargs_1 = {
            "in_channels": self.conv1_input_channels,
            "out_channels": self.conv1_output_channels,
            "batch_size": batch_size,
            "input_height": input_height,
            "input_width": input_width,
            "kernel_size": (1, 1),
            "stride": (1, 1),
            "padding": (0, 0),
            "dilation": (1, 1),
            "groups": 1,
            "device": device,
            "conv_config": ttnn.Conv2dConfig(
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                activation="relu",
                shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                if height_sharding
                else ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                reshard_if_not_optimal=reshard_if_not_optimal,
            ),
        }

        if not ttnn.is_tensor_storage_on_device(self.conv1_weight_tensor):
            self.conv1_weight_tensor = ttnn.prepare_conv_weights(
                weight_tensor=self.conv1_weight_tensor,
                weights_format="OIHW",
                input_memory_config=x.memory_config(),
                input_layout=x.get_layout(),
                has_bias=True,
                **conv_kwargs_1,
                input_dtype=self.model_config["ACTIVATIONS_DTYPE"],
            )
            self.conv1_bias_tensor = ttnn.prepare_conv_bias(
                bias_tensor=self.conv1_bias_tensor,
                input_memory_config=x.memory_config(),
                input_layout=x.get_layout(),
                **conv_kwargs_1,
                input_dtype=self.model_config["ACTIVATIONS_DTYPE"],
            )

            self.conv1_weight_tensor = ttnn.to_device(self.conv1_weight_tensor, device)
            self.conv1_bias_tensor = ttnn.to_device(self.conv1_bias_tensor, device)

        out, [input_height, input_width] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.conv1_weight_tensor,
            bias_tensor=self.conv1_bias_tensor,
            **conv_kwargs_1,
            compute_config=ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                packer_l1_acc=packer_l1_acc,
            ),
            return_output_dim=True,
            return_weights_and_bias=False,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        act_block_h_override = 0
        run_downsample_before_conv2 = True
        ds_out = None

        if is_grayskull():
            if self.conv2_output_channels == 64 and input_height == 56 and batch_size == 20:
                act_block_h_override = 320
        elif is_wormhole_b0():
            run_downsample_before_conv2 = False

        if run_downsample_before_conv2:
            if layer_module and layer_module == "layer4_module1":
                if ops_parallel_config and "layer4_module1_downsample" in ops_parallel_config:
                    x = ttnn.to_memory_config(x, ops_parallel_config["layer4_module1_downsample"])
            if is_grayskull():
                if input_height == 56 and self.conv1_input_channels == 256 and self.downsample:
                    x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
                    ttnn.deallocate(x)
                    x = ttnn.reallocate(x_rm)
            ds_out = self.run_downsample_if_req(
                x,
                device,
                batch_size,
                ds_input_height,
                ds_input_width,
                reshard_if_not_optimal,
                height_sharding,
                packer_l1_accum_enabled=packer_l1_acc,
                enable_act_double_buffer=False,
                enable_split_reader=enable_split_reader,
                enable_subblock_padding=enable_subblock_padding,
            )
            if layer_module and layer_module == "layer4_module1":
                if ops_parallel_config and "layer4_module1_downsample" not in ops_parallel_config:
                    x_memory_config = ttnn.get_memory_config(ds_out)
                    sharded_config = ttnn.create_sharded_memory_config_(
                        ttnn.Shape([batch_size, ds_input_height, ds_input_width, self.conv1_input_channels]),
                        x_memory_config.shard_spec.grid,
                        x_memory_config.memory_layout,
                        x_memory_config.shard_spec.orientation,
                        tile_layout=True,
                    )
                    ops_parallel_config["layer4_module1_downsample"] = sharded_config

        logger.debug(f"Running conv2")

        if layer_module and layer_module == "layer4_module1":
            if ops_parallel_config and "layer4_module1_input" in ops_parallel_config:
                out = ttnn.to_memory_config(out, ops_parallel_config["layer4_module1_input"])

        conv_kwargs_2 = {
            "in_channels": self.conv2_input_channels,
            "out_channels": self.conv2_output_channels,
            "batch_size": batch_size,
            "input_height": input_height,
            "input_width": input_width,
            "kernel_size": (3, 3),
            "stride": (self.stride, self.stride),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "device": device,
            "conv_config": ttnn.Conv2dConfig(
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                activation="relu",
                deallocate_activation=True,
                reallocate_halo_output=not is_wormhole_b0(),
                act_block_h_override=act_block_h_override,
                shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                if height_sharding
                else ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                reshard_if_not_optimal=reshard_if_not_optimal,
                enable_act_double_buffer=enable_act_double_buffer,
                enable_weights_double_buffer=True,
                enable_split_reader=enable_split_reader,
                enable_subblock_padding=enable_subblock_padding,
            ),
        }

        if is_blackhole():
            conv_kwargs_2["conv_config"].act_block_h_override = 2 * 32
            conv_kwargs_2["conv_config"].enable_subblock_padding = False
            if (
                batch_size == 32
                and layer_module
                and (
                    layer_module == "layer1_module2"
                    or layer_module == "layer1_module3"
                    or layer_module == "layer2_module2"
                    or layer_module == "layer2_module3"
                    or layer_module == "layer2_module4"
                )
            ):
                conv_kwargs_2["conv_config"].act_block_h_override = 0
            elif (
                batch_size == 20
                and layer_module
                and (layer_module == "layer4_module2" or layer_module == "layer4_module3")
            ):
                conv_kwargs_2["conv_config"].act_block_h_override = 0
            elif (
                batch_size == 16
                and layer_module
                and (layer_module == "layer1_module2" or layer_module == "layer1_module3")
            ):
                conv_kwargs_2["conv_config"].act_block_h_override = 0

        if not ttnn.is_tensor_storage_on_device(self.conv2_weight_tensor):
            self.conv2_weight_tensor = ttnn.prepare_conv_weights(
                weight_tensor=self.conv2_weight_tensor,
                weights_format="OIHW",
                input_memory_config=x.memory_config(),
                input_layout=out.get_layout(),
                has_bias=True,
                **conv_kwargs_2,
                input_dtype=self.model_config["ACTIVATIONS_DTYPE"],
            )
            self.conv2_bias_tensor = ttnn.prepare_conv_bias(
                bias_tensor=self.conv2_bias_tensor,
                input_memory_config=x.memory_config(),
                input_layout=out.get_layout(),
                **conv_kwargs_2,
                input_dtype=self.model_config["ACTIVATIONS_DTYPE"],
            )
            self.conv2_weight_tensor = ttnn.to_device(self.conv2_weight_tensor, device)
            self.conv2_bias_tensor = ttnn.to_device(self.conv2_bias_tensor, device)

        out, [input_height, input_width] = ttnn.conv2d(
            input_tensor=out,
            weight_tensor=self.conv2_weight_tensor,
            bias_tensor=self.conv2_bias_tensor,
            **conv_kwargs_2,
            compute_config=ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                packer_l1_acc=packer_l1_acc,
            ),
            return_output_dim=True,
            return_weights_and_bias=False,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        if layer_module and layer_module == "layer4_module1":
            if ops_parallel_config and "layer4_module1_input" not in ops_parallel_config:
                x_memory_config = ttnn.get_memory_config(out)
                sharded_config = ttnn.create_sharded_memory_config_(
                    ttnn.Shape([batch_size, module_input_height, module_input_width, self.conv2_input_channels]),
                    x_memory_config.shard_spec.grid,
                    x_memory_config.memory_layout,
                    x_memory_config.shard_spec.orientation,
                    tile_layout=True,
                )
                ops_parallel_config["layer4_module1_input"] = sharded_config

        # conv3 is 1x1 conv
        logger.debug(f"Running conv3")
        conv_kwargs_3 = {
            "in_channels": self.conv3_input_channels,
            "out_channels": self.conv3_output_channels,
            "batch_size": batch_size,
            "input_height": input_height,
            "input_width": input_width,
            "kernel_size": (1, 1),
            "stride": (1, 1),
            "padding": (0, 0),
            "dilation": (1, 1),
            "groups": 1,
            "device": device,
            "conv_config": ttnn.Conv2dConfig(
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                if height_sharding
                else ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                reshard_if_not_optimal=reshard_if_not_optimal,
            ),
        }

        if not ttnn.is_tensor_storage_on_device(self.conv3_weight_tensor):
            self.conv3_weight_tensor = ttnn.prepare_conv_weights(
                weight_tensor=self.conv3_weight_tensor,
                weights_format="OIHW",
                input_memory_config=x.memory_config(),
                input_layout=out.get_layout(),
                has_bias=True,
                **conv_kwargs_3,
                input_dtype=self.model_config["ACTIVATIONS_DTYPE"],
            )
            self.conv3_bias_tensor = ttnn.prepare_conv_bias(
                bias_tensor=self.conv3_bias_tensor,
                input_memory_config=x.memory_config(),
                input_layout=out.get_layout(),
                **conv_kwargs_3,
                input_dtype=self.model_config["ACTIVATIONS_DTYPE"],
            )
            self.conv3_weight_tensor = ttnn.to_device(self.conv3_weight_tensor, device)
            self.conv3_bias_tensor = ttnn.to_device(self.conv3_bias_tensor, device)
        out = ttnn.conv2d(
            input_tensor=out,
            weight_tensor=self.conv3_weight_tensor,
            bias_tensor=self.conv3_bias_tensor,
            **conv_kwargs_3,
            compute_config=ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                packer_l1_acc=packer_l1_acc,
            ),
            return_output_dim=False,
            return_weights_and_bias=False,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        if not run_downsample_before_conv2:
            ds_out = self.run_downsample_if_req(
                x,
                device,
                batch_size,
                ds_input_height,
                ds_input_width,
                reshard_if_not_optimal,
                height_sharding,
                packer_l1_accum_enabled=packer_l1_acc,
                enable_act_double_buffer=enable_act_double_buffer,
                enable_split_reader=enable_split_reader,
                enable_subblock_padding=enable_subblock_padding,
            )

        assert ds_out is not None, "ds_out is None"

        assert ttnn.get_memory_config(out) == ttnn.get_memory_config(
            ds_out
        ), f"{ttnn.get_memory_config(out)} != {ttnn.get_memory_config(ds_out)}"

        if eltwise_binary_out_in_place:
            # underscore version is in_place = True
            out = ttnn.add_(
                out,
                ds_out,
                activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)],
            )
        else:
            out = ttnn.add(
                out,
                ds_out,
                activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        ttnn.deallocate(ds_out)
        return out, input_height, input_width


class resnet50:
    def __init__(
        self,
        device,
        parameters,
        batch_size,
        model_config,
        input_shape,
        kernel_size,
        stride,
        dealloc_input=True,
        final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
    ) -> None:
        super().__init__()
        layers = [3, 4, 6, 3]
        num_classes = 1000
        conv_input_face_shape_hw = [224, 224]
        self.device = device
        self.conv_input_face_shape_hw = conv_input_face_shape_hw
        self.batch_size = batch_size
        self.model_config = model_config
        self.inplanes = 64
        self.final_output_mem_config = final_output_mem_config
        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=model_config["MATH_FIDELITY"],
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.conv1_weight_tensor = parameters.conv1.weight
        self.conv1_bias_tensor = parameters.conv1.bias
        self.conv1_input_channels = self.conv1_weight_tensor.shape[1]
        self.conv1_output_channels = self.conv1_weight_tensor.shape[0]
        assert self.conv1_weight_tensor.shape[2] == 4

        self.layer1 = self._make_layer(
            parameters=parameters.layer1,
            planes=64,
            blocks=layers[0],
            stride=1,
            model_config=model_config,
        )
        self.layer2 = self._make_layer(
            parameters=parameters.layer2,
            planes=128,
            blocks=layers[1],
            stride=2,
            model_config=model_config,
        )
        self.layer3 = self._make_layer(
            parameters=parameters.layer3,
            planes=256,
            blocks=layers[2],
            stride=2,
            model_config=model_config,
        )
        self.layer4 = self._make_layer(
            parameters=parameters.layer4,
            planes=512,
            blocks=layers[3],
            stride=2,
            model_config=model_config,
        )

        # All modules in RN50 are unrolled here. One variable for each module. Only specific number of modules supported - layers MUST equal to [3, 4, 6, 3]
        assert layers == [3, 4, 6, 3]
        self.layer1_module1 = self.layer1[0]
        self.layer1_module2 = self.layer1[1]
        self.layer1_module3 = self.layer1[2]

        self.layer2_module1 = self.layer2[0]
        self.layer2_module2 = self.layer2[1]
        self.layer2_module3 = self.layer2[2]
        self.layer2_module4 = self.layer2[3]

        self.layer3_module1 = self.layer3[0]
        self.layer3_module2 = self.layer3[1]
        self.layer3_module3 = self.layer3[2]
        self.layer3_module4 = self.layer3[3]
        self.layer3_module5 = self.layer3[4]
        self.layer3_module6 = self.layer3[5]

        self.layer4_module1 = self.layer4[0]
        self.layer4_module2 = self.layer4[1]
        self.layer4_module3 = self.layer4[2]

        self.avgpool = ttnn.global_avg_pool2d
        self.fc = ResnetLinear(
            in_features=512 * resnet50Bottleneck.expansion,
            out_features=1024,
            weight=ttnn.to_device(parameters.fc.weight, device),
            bias=ttnn.to_device(parameters.fc.bias, device),
            output_mem_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            model_config=model_config,
            device=self.device,
            batch_size=batch_size,
            compute_kernel_config=compute_kernel_config,
        )  # num_classes = 1000

        act_block_h_override = 0

        if is_wormhole_b0():
            act_block_h_override = 1568

        if is_blackhole() and self.batch_size == 32:
            act_block_h_override = 49 * 32

        self.conv1_config = ttnn.Conv2dConfig(
            weights_dtype=self.model_config["WEIGHTS_DTYPE"],
            activation="relu",
            deallocate_activation=dealloc_input,
            act_block_h_override=act_block_h_override,
            enable_act_double_buffer=is_wormhole_b0() or is_blackhole(),
            enable_split_reader=True,
            enable_subblock_padding=False,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            reshard_if_not_optimal=False,
        )
        self.conv1_compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=self.model_config["MATH_FIDELITY"],
            packer_l1_acc=True,
        )
        if is_wormhole_b0():
            # Issue #13145: Temp workaround for Galaxy to avoid hangs
            if device.get_num_devices() > 8:
                self.conv1_config.act_block_h_override = 64
            else:
                self.conv1_config.act_block_h_override = 49 * 32

        self.conv1_kernel_size = (4, 4)
        self.conv1_stride = (1, 1)
        self.conv1_padding = (0, 0)
        self.conv1_input_height = 115
        self.conv1_input_width = 115
        self.conv1_output_height = (
            (self.conv1_input_height - self.conv1_kernel_size[0] + 2 * self.conv1_padding[0]) // self.conv1_stride[0]
        ) + 1
        self.conv1_output_width = (
            (self.conv1_input_width - self.conv1_kernel_size[1] + 2 * self.conv1_padding[1]) // self.conv1_stride[1]
        ) + 1

        # fold params
        self.fold_stride_h = stride
        self.fold_stride_w = stride
        _, c, h, w = input_shape
        n = batch_size
        h += kernel_size * 2
        w += kernel_size * 2
        C = _nearest_y(c, 4)
        self.fold_pad_c = C - c
        self.fold_pad_h = kernel_size
        self.fold_pad_w = kernel_size
        self.fold_output_shape = (
            n,
            h // self.fold_stride_h,
            w // self.fold_stride_w,
            C * (self.fold_stride_h * self.fold_stride_w),
        )
        num_cores_x = 8
        num_cores_y = 8
        if self.batch_size == 16:
            num_cores_x = 8
            num_cores_y = 8
            self.fold_compute_grid_size = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))}
            )
        elif self.batch_size == 20:
            if is_grayskull():
                num_cores_x = 10
                num_cores_y = 8
            elif is_wormhole_b0():
                num_cores_x = 8
                num_cores_y = 5
            elif is_blackhole():
                num_cores_x = 10
                num_cores_y = 8
            self.fold_compute_grid_size = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))}
            )
        elif self.batch_size == 32:
            core_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(12, 8)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 9), ttnn.CoreCoord(10, 9)),
                }
            )
            self.fold_compute_grid_size = core_grid

        conv_dummy_tensor = torch.rand((self.fold_output_shape), dtype=torch.bfloat16)
        conv_dummy_tensor = ttnn.from_torch(conv_dummy_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)

        self.override_fold_mem_config = get_conv_input_memory_config(
            self.batch_size,
            self.conv1_input_channels,
            self.conv1_input_height,
            self.conv1_input_width,
            self.conv1_output_channels,
            self.conv1_output_height,
            self.conv1_output_width,
            device.compute_with_storage_grid_size(),
            input_channels_alignment=8,
            override_num_cores=is_grayskull() or is_blackhole(),
        )

    def __del__(self):
        # Nothing to do
        pass

    def _make_layer(
        self,
        parameters,
        planes: int,
        blocks: int,
        stride: int,
        model_config=None,
    ) -> List[resnet50Bottleneck]:
        layers = []
        layers.append(
            resnet50Bottleneck(
                parameters=parameters[0],
                downsample=stride != 1 or self.inplanes != planes * resnet50Bottleneck.expansion,
                stride=stride,
                model_config=model_config,
            )
        )
        self.inplanes = planes * resnet50Bottleneck.expansion
        for block_num in range(1, blocks):
            layers.append(
                resnet50Bottleneck(
                    parameters=parameters[block_num],
                    downsample=False,
                    stride=1,
                    model_config=model_config,
                )
            )
        return layers

    def __call__(self, input_tensor, device, ops_parallel_config) -> ttnn.Tensor:
        return self.run(
            input_tensor,
            device,
            ops_parallel_config,
        )

    ## merged runs (first and optimized)
    def run(self, input_tensor, device, ops_parallel_config) -> ttnn.Tensor:
        is_first_run = False
        if not ops_parallel_config:
            is_first_run = True
            logger.debug(f"==== First run")
        else:
            logger.debug(f"==== Optimized run")

        logger.debug(f"==== fold on device")

        # run fold
        fold_output_tensor = ttnn.fold(
            input_tensor,
            self.fold_stride_h,
            self.fold_stride_w,
            use_transpose_as_fold=True,
            pad_c=self.fold_pad_c,
            pad_h=self.fold_pad_h,
            pad_w=self.fold_pad_w,
            grid_size=self.fold_compute_grid_size,
            override_memory_config=self.override_fold_mem_config,
        )
        n, c, h, w = fold_output_tensor.shape
        fold_output_tensor = ttnn.reshape(fold_output_tensor, (1, 1, n * c * h, w))

        ttnn.deallocate(input_tensor)

        logger.debug(f"==== first conv")

        # first conv
        conv_kwargs = {
            "in_channels": self.conv1_input_channels,
            "out_channels": self.conv1_output_channels,
            "batch_size": self.batch_size,
            "input_height": self.conv1_input_height,
            "input_width": self.conv1_input_width,
            "kernel_size": self.conv1_kernel_size,
            "stride": self.conv1_stride,
            "padding": self.conv1_padding,
            "dilation": (1, 1),
            "groups": 1,
            "device": device,
            "conv_config": self.conv1_config,
        }

        if not ttnn.is_tensor_storage_on_device(self.conv1_weight_tensor):
            self.conv1_weight_tensor = ttnn.prepare_conv_weights(
                weight_tensor=self.conv1_weight_tensor,
                weights_format="OIHW",
                input_memory_config=fold_output_tensor.memory_config(),
                input_layout=fold_output_tensor.get_layout(),
                has_bias=True,
                **conv_kwargs,
                input_dtype=self.model_config["ACTIVATIONS_DTYPE"],
            )

            self.conv1_bias_tensor = ttnn.prepare_conv_bias(
                bias_tensor=self.conv1_bias_tensor,
                input_memory_config=fold_output_tensor.memory_config(),
                input_layout=fold_output_tensor.get_layout(),
                **conv_kwargs,
                input_dtype=self.model_config["ACTIVATIONS_DTYPE"],
            )
            self.conv1_weight_tensor = ttnn.to_device(self.conv1_weight_tensor, device)
            self.conv1_bias_tensor = ttnn.to_device(self.conv1_bias_tensor, device)

        x, [x_height, x_width] = ttnn.conv2d(
            input_tensor=fold_output_tensor,
            weight_tensor=self.conv1_weight_tensor,
            bias_tensor=self.conv1_bias_tensor,
            **conv_kwargs,
            compute_config=self.conv1_compute_config,
            return_output_dim=True,
            return_weights_and_bias=False,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        # Relu is fused with conv1
        if self.batch_size == 20:
            x = ttnn.reallocate(x)

        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=self.batch_size,
            input_h=x_height,
            input_w=x_width,
            channels=self.conv1_output_channels,
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
        )

        x_height = 56
        x_width = 56
        x = ttnn.reshape(x, (1, 1, x_height * x_width * self.batch_size, 64))

        if is_blackhole():
            ## 112
            core_range_set = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(12, 7),
                    ),
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 8),
                        ttnn.CoreCoord(7, 8),
                    ),
                }
            )
        elif is_wormhole_b0():
            core_range_set = ttnn.CoreGrid(x=8, y=7)

        if is_blackhole() or is_wormhole_b0():
            mem_config = ttnn.create_sharded_memory_config_(
                ttnn.Shape([self.batch_size * x_height * x_width, 64]),
                core_range_set,
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
                tile_layout=True,
            )
            x = ttnn.to_memory_config(x, mem_config)

        if self.batch_size == 20 and is_grayskull():
            x = ttnn.reallocate(x)

        if not is_blackhole():
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, dtype=self.model_config["ACTIVATIONS_DTYPE"])

        logger.debug(f"==== Running layer 1 module 1")
        layer1_module1_input_shape = ttnn.Shape(x.padded_shape)

        reshard = is_blackhole()
        height_shard = True

        x, x_height, x_width = self.layer1_module1(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            reshard_if_not_optimal=reshard,
            height_sharding=height_shard,
            enable_act_double_buffer=True,
            enable_split_reader=True,
            enable_subblock_padding=not is_grayskull(),
        )

        if is_first_run:
            x_memory_config = ttnn.get_memory_config(x)
            ops_parallel_config["layer1_module1_input"] = ttnn.create_sharded_memory_config_(
                layer1_module1_input_shape,
                x_memory_config.shard_spec.grid,
                x_memory_config.memory_layout,
                x_memory_config.shard_spec.orientation,
                tile_layout=True,
            )

        logger.debug(f"==== Running layer 1 module 2")
        x, x_height, x_width = self.layer1_module2(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            enable_act_double_buffer=False,
            enable_split_reader=True,
            enable_subblock_padding=not is_grayskull(),
            layer_module="layer1_module2",
        )

        logger.debug(f"==== Running layer 1 module 3")
        x, x_height, x_width = self.layer1_module3(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            enable_act_double_buffer=False,
            enable_split_reader=True,
            enable_subblock_padding=not is_grayskull(),
            layer_module="layer1_module3",
        )

        layer2_module1_input_shape = ttnn.Shape(x.padded_shape)

        reshard = is_blackhole() or not (is_wormhole_b0() or is_grayskull())
        height_shard = True

        if is_blackhole() and self.batch_size < 20:
            ## 98
            core_range_set = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(12, 6),
                    ),
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 7),
                        ttnn.CoreCoord(6, 7),
                    ),
                }
            )
            mem_config = ttnn.create_sharded_memory_config_(
                layer2_module1_input_shape,
                core_range_set,
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
                tile_layout=True,
            )
            x = ttnn.to_memory_config(x, mem_config)

        logger.debug(f"==== Running layer 2 module 1")
        x, x_height, x_width = self.layer2_module1(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            reshard_if_not_optimal=reshard,
            height_sharding=height_shard,
            enable_act_double_buffer=True,
            enable_split_reader=True,
            enable_subblock_padding=False,
            layer_module="layer2_module1",
        )

        if is_first_run:
            x_memory_config = ttnn.get_memory_config(x)
            ops_parallel_config["layer2_module1_input"] = ttnn.create_sharded_memory_config_(
                layer2_module1_input_shape,
                x_memory_config.shard_spec.grid,
                x_memory_config.memory_layout,
                x_memory_config.shard_spec.orientation,
                tile_layout=True,
            )

        logger.debug(f"==== Running layer 2 module 2")
        x, x_height, x_width = self.layer2_module2(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            enable_act_double_buffer=True,
            enable_split_reader=True,
            enable_subblock_padding=False,
            layer_module="layer2_module2",
        )

        logger.debug(f"==== Running layer 2 module 3")
        x, x_height, x_width = self.layer2_module3(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            enable_act_double_buffer=True,
            enable_split_reader=True,
            enable_subblock_padding=False,
            layer_module="layer2_module3",
        )

        logger.debug(f"==== Running layer 2 module 4")
        x, x_height, x_width = self.layer2_module4(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            enable_act_double_buffer=True,
            enable_split_reader=True,
            enable_subblock_padding=False,
            layer_module="layer2_module4",
        )

        layer3_module1_input_shape = ttnn.Shape(x.padded_shape)

        reshard = is_wormhole_b0() or is_grayskull()
        height_shard = False

        if is_blackhole():
            ## 104
            core_range_set = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(12, 7),
                    ),
                }
            )
            mem_config = ttnn.create_sharded_memory_config_(
                layer3_module1_input_shape,
                core_range_set,
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.ShardOrientation.COL_MAJOR,
                tile_layout=True,
            )
            x = ttnn.to_memory_config(x, mem_config)

        logger.debug(f"==== Running layer 3 module 1")
        x, x_height, x_width = self.layer3_module1(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            reshard_if_not_optimal=reshard,
            height_sharding=height_shard,
            enable_act_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
        )

        if is_first_run:
            x_memory_config = ttnn.get_memory_config(x)
            ops_parallel_config["layer3_module1_input"] = ttnn.create_sharded_memory_config_(
                layer3_module1_input_shape,
                x_memory_config.shard_spec.grid,
                x_memory_config.memory_layout,
                x_memory_config.shard_spec.orientation,
                tile_layout=True,
            )

        logger.debug(f"==== Running layer 3 module 2")
        x, x_height, x_width = self.layer3_module2(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            enable_act_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
        )

        logger.debug(f"==== Running layer 3 module 3")
        x, x_height, x_width = self.layer3_module3(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            enable_act_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            layer_module="layer3_module3",
        )

        logger.debug(f"==== Running layer 3 module 4")
        x, x_height, x_width = self.layer3_module4(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            enable_act_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            layer_module="layer3_module4",
        )

        logger.debug(f"==== Running layer 3 module 5")
        x, x_height, x_width = self.layer3_module5(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            enable_act_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            layer_module="layer3_module5",
        )

        logger.debug(f"==== Running layer 3 module 6")
        x, x_height, x_width = self.layer3_module6(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            eltwise_binary_out_in_place=True,
            enable_act_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
        )

        reshard = is_grayskull() or (is_blackhole() and self.batch_size == 20)
        height_shard = False

        layer4_module1_input_shape = ttnn.Shape(x.padded_shape)
        if is_blackhole() and self.batch_size != 20:
            # 104
            grid_size = (13, 8)
            core_range_set = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(grid_size[0] - 1, grid_size[1] - 1),
                    ),
                }
            )
            mem_config = ttnn.create_sharded_memory_config_(
                layer4_module1_input_shape,
                core_range_set,
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.ShardOrientation.COL_MAJOR,
                tile_layout=True,
            )
            x = ttnn.to_memory_config(x, mem_config)
        elif is_wormhole_b0():
            core_range_set = ttnn.CoreGrid(x=8, y=7)
            shard_config = ttnn.create_sharded_memory_config_(
                layer4_module1_input_shape,
                core_range_set,
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
                tile_layout=True,
            )
            x = ttnn.to_memory_config(x, shard_config)

        logger.debug(f"==== Running layer 4 module 1")
        x, x_height, x_width = self.layer4_module1(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            reshard_if_not_optimal=reshard,
            height_sharding=height_shard,
            enable_act_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            ops_parallel_config=ops_parallel_config,
            layer_module="layer4_module1",
        )

        logger.debug(f"==== Running layer 4 module 2")
        x, x_height, x_width = self.layer4_module2(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            enable_act_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            layer_module="layer4_module2",
        )

        logger.debug(f"==== Running layer 4 module 3")
        x, x_height, x_width = self.layer4_module3(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            enable_act_double_buffer=True,
            enable_split_reader=False,
            enable_subblock_padding=False,
            layer_module="layer4_module3",
        )

        grid_size = (8, 4)
        if self.batch_size > 16:
            grid_size = (8, 8)
        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(grid_size[0] - 1, grid_size[1] - 1),
                )
            }
        )
        shard_shape = [
            x.volume() // x.padded_shape[-1],
            x.padded_shape[-1] // (grid_size[0] * grid_size[1]),
        ]
        shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        width_sharded_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec
        )
        x = ttnn.to_memory_config(x, width_sharded_mem_config)

        unpadded_shape = x.shape
        x = ttnn.untilize_with_unpadding(
            x,
            output_tensor_end=(
                unpadded_shape[0] - 1,
                unpadded_shape[1] - 1,
                unpadded_shape[2] - 1,
                unpadded_shape[3] - 1,
            ),
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )

        x = ttnn.reshape(
            x,
            (
                self.batch_size,
                x.shape[1],
                x.shape[2] // self.batch_size,
                x.shape[3],
            ),
        )

        unpadded_shape = x.padded_shape
        padded_shape = [
            unpadded_shape[0],
            unpadded_shape[1],
            _nearest_32(unpadded_shape[2]),
            _nearest_32(unpadded_shape[3]),
        ]
        x = ttnn.tilize_with_val_padding(
            x,
            padded_shape,
            0.0,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        x = self.avgpool(x, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)

        unpadded_shape_end = [
            x.padded_shape[0] - 1,
            x.padded_shape[1] - 1,
            1 - 1,
            x.padded_shape[3] - 1,
        ]
        x = ttnn.untilize_with_unpadding(
            x, output_tensor_end=unpadded_shape_end, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
        )

        x = ttnn.reshape(
            x,
            (
                1,
                x.padded_shape[1],
                self.batch_size * x.padded_shape[2],
                x.padded_shape[3],
            ),
        )

        unpadded_shape = x.padded_shape
        padded_shape = [
            unpadded_shape[0],
            unpadded_shape[1],
            _nearest_32(unpadded_shape[2]),
            _nearest_32(unpadded_shape[3]),
        ]

        x = ttnn.tilize_with_val_padding(
            x,
            padded_shape,
            0.0,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        x = self.fc(x)
        desired_shape = list(x.shape)
        desired_shape[-1] = 1000
        x = ttnn.untilize_with_unpadding(
            x,
            output_tensor_end=(desired_shape[0] - 1, desired_shape[1] - 1, desired_shape[2] - 1, desired_shape[3] - 1),
            memory_config=self.final_output_mem_config,
        )
        x = ttnn.reshape(
            x,
            (
                self.batch_size,
                x.shape[1],
                x.shape[2] // self.batch_size,
                x.shape[3],
            ),
        )

        return x
