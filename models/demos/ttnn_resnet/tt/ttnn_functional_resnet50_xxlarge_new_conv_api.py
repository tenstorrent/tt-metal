# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from models.utility_functions import (
    is_grayskull,
    is_wormhole_b0,
    pad_and_fold_conv_activation_for_unity_stride,
)
from typing import List
from loguru import logger

hardcoded_matmul_config_linear = {
    1: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
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
    weight_shape = weight.get_legacy_shape()
    weight = weight.reshape(1, 1, weight_shape[-2], weight_shape[-1])
    bias_shape = bias.get_legacy_shape()
    bias = bias.reshape(1, 1, bias_shape[-2], bias_shape[-1])

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
        conv_op_cache,
        reshard_if_not_optimal=False,
        height_sharding=None,
    ):
        if self.downsample:
            ds_out, _, _, self.ds_conv_weight_tensor, self.ds_conv_bias_tensor = ttnn.conv2d(
                input_tensor=x,
                weight_tensor=self.ds_conv_weight_tensor,
                in_channels=self.ds_conv_input_channels,
                out_channels=self.ds_conv_output_channels,
                device=device,
                bias_tensor=self.ds_conv_bias_tensor,
                kernel_size=(1, 1),
                stride=(self.stride, self.stride),
                padding=(0, 0),
                batch_size=batch_size,
                input_height=input_height,
                input_width=input_width,
                conv_config=ttnn.Conv2dConfig(
                    dtype=self.model_config["ACTIVATIONS_DTYPE"],
                    weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                    math_fidelity=self.model_config["MATH_FIDELITY"],
                    shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                    if height_sharding
                    else ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                    deallocate_activation=True,
                    reallocate_halo_output=True,
                    reshard_if_not_optimal=reshard_if_not_optimal,
                    transpose_shards=height_sharding,
                ),
                conv_op_cache=conv_op_cache,
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
        conv_op_cache,
        reshard_if_not_optimal=False,
        height_sharding=None,
        eltwise_binary_out_in_place=True,
    ):
        logger.info(f"Running resnet50Bottleneck")
        logger.info(
            f"input_height: {input_height}, input_width: {input_width}, batch_size: {batch_size}, conv1_input_channels: {self.conv1_input_channels}, conv1_output_channels: {self.conv1_output_channels}, conv2_input_channels: {self.conv2_input_channels}, conv2_output_channels: {self.conv2_output_channels}, conv3_input_channels: {self.conv3_input_channels}, conv3_output_channels: {self.conv3_output_channels}"
        )

        # conv1 is 1x1 conv
        # print("Running conv1")
        module_input_height = input_height
        out, input_height, input_width, self.conv1_weight_tensor, self.conv1_bias_tensor = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.conv1_weight_tensor,
            in_channels=self.conv1_input_channels,
            out_channels=self.conv1_output_channels,
            device=device,
            bias_tensor=self.conv1_bias_tensor,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            conv_config=ttnn.Conv2dConfig(
                dtype=self.model_config["ACTIVATIONS_DTYPE"],
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                math_fidelity=self.model_config["MATH_FIDELITY"],
                activation="relu",
                shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                if height_sharding
                else ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                reshard_if_not_optimal=reshard_if_not_optimal,
                transpose_shards=height_sharding,
            ),
            conv_op_cache=conv_op_cache,
        )

        if is_wormhole_b0():
            out_rm = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(out)
            out = ttnn.reallocate(out_rm)

        act_block_h_override = 0
        if is_grayskull():
            if self.conv2_output_channels == 64 and input_height == 56 and batch_size == 20:
                act_block_h_override = 320
        elif is_wormhole_b0():
            if (
                self.conv2_input_channels == 128
                and self.conv2_output_channels == 128
                and input_height == 56
                and batch_size == 20
            ):
                act_block_h_override = 160
            elif batch_size == 1:
                act_block_h_override = 512

        self.run_downsample_before_conv2 = False
        if not (input_height == 56 and self.conv1_input_channels == 64):
            self.run_downsample_before_conv2 = True
        if is_wormhole_b0() and (
            (
                input_height == 256
                and batch_size == 1
                and self.conv1_input_channels == 256
                and self.conv1_output_channels == 128
            )
            or (input_height == 128 and self.conv1_input_channels == 512 and self.conv1_output_channels == 128)
        ):
            # self.run_downsample_before_conv2 = False
            act_block_h_override = 256
        if is_wormhole_b0() and (
            (
                input_height == 64
                and batch_size == 1
                and self.conv1_input_channels == 1024
                and self.conv1_output_channels == 512
            )
            or (
                input_height == 32
                and batch_size == 1
                and self.conv1_input_channels == 2048
                and self.conv1_output_channels == 512
            )
        ):
            act_block_h_override = 128

        if self.run_downsample_before_conv2:
            ttnn.dump_device_memory_state(device, "before_reallocate_")
            if (
                (is_wormhole_b0() and batch_size == 1 and input_height == 256)
                or (input_height == 56 and self.conv1_input_channels == 256)
            ) and self.downsample:
                x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
                ttnn.deallocate(x)
                x = ttnn.reallocate(x_rm)
            ttnn.dump_device_memory_state(device, "before_downsample_")
            ds_out = self.run_downsample_if_req(
                x,
                device,
                batch_size,
                input_height,
                input_width,
                conv_op_cache,
                reshard_if_not_optimal,
                height_sharding,
            )

        reallocate_halo_output = (
            batch_size
            == 20  # and
            # input_height == 56 and
            # self.conv1_input_channels == 256 and
            # self.downsample
        )
        out, input_height, input_width, self.conv2_weight_tensor, self.conv2_bias_tensor = ttnn.conv2d(
            input_tensor=out,
            weight_tensor=self.conv2_weight_tensor,
            in_channels=self.conv2_input_channels,
            out_channels=self.conv2_output_channels,
            device=device,
            bias_tensor=self.conv2_bias_tensor,
            kernel_size=(3, 3),
            stride=(self.stride, self.stride),
            padding=(1, 1),
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            conv_config=ttnn.Conv2dConfig(
                dtype=self.model_config["ACTIVATIONS_DTYPE"],
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                math_fidelity=self.model_config["MATH_FIDELITY"],
                activation="relu",
                deallocate_activation=True,
                reallocate_halo_output=reallocate_halo_output,
                act_block_h_override=act_block_h_override,
                shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                if height_sharding
                else ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                reshard_if_not_optimal=reshard_if_not_optimal,
                transpose_shards=height_sharding,
            ),
            conv_op_cache=conv_op_cache,
        )

        # conv3 is 1x1 conv
        # print("Running conv3")
        out, _, _, self.conv3_weight_tensor, self.conv3_bias_tensor = ttnn.conv2d(
            input_tensor=out,
            weight_tensor=self.conv3_weight_tensor,
            in_channels=self.conv3_input_channels,
            out_channels=self.conv3_output_channels,
            device=device,
            bias_tensor=self.conv3_bias_tensor,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            conv_config=ttnn.Conv2dConfig(
                dtype=self.model_config["ACTIVATIONS_DTYPE"],
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                math_fidelity=self.model_config["MATH_FIDELITY"],
                shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                if height_sharding
                else ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                reshard_if_not_optimal=reshard_if_not_optimal,
                transpose_shards=height_sharding,
            ),
            conv_op_cache=conv_op_cache,
        )

        if not self.run_downsample_before_conv2:
            ds_out = self.run_downsample_if_req(
                x,
                device,
                batch_size,
                input_height,
                input_width,
                conv_op_cache,
                reshard_if_not_optimal,
                height_sharding,
            )

        assert ttnn.get_memory_config(out) == ttnn.get_memory_config(ds_out)
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
            )  ## TODO: check why not out mem config???
        ttnn.deallocate(ds_out)
        if batch_size == 20 and module_input_height == 56 and self.conv1_input_channels == 64:
            out = ttnn.reallocate(out)
        return out, input_height, input_width


class resnet50:
    def __init__(
        self,
        device,
        parameters,
        batch_size,
        model_config,
    ) -> None:
        super().__init__()
        layers = [3, 4, 6, 3]
        num_classes = 1000
        conv_input_face_shape_hw = [1024, 1024]
        self.device = device
        self.conv_input_face_shape_hw = conv_input_face_shape_hw
        self.batch_size = batch_size
        self.model_config = model_config
        self.conv_op_cache = {}
        self.inplanes = 64
        if is_grayskull():
            compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
                math_fidelity=model_config["MATH_FIDELITY"],
                math_approx_mode=True,
            )
        else:
            compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=model_config["MATH_FIDELITY"],
                math_approx_mode=True,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )
        self.conv1_weight_tensor = parameters.conv1.weight
        self.conv1_bias_tensor = parameters.conv1.bias
        self.conv1_input_channels = self.conv1_weight_tensor.shape[1]
        self.conv1_output_channels = self.conv1_weight_tensor.shape[0]
        assert self.conv1_weight_tensor.shape[2] == 4

        self.max_pool_reader_patterns_cache = {}
        max_pool_parallel_config_override = {}

        self.max_pool = ttnn.MaxPool2d(
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            dilation=(1, 1),
            dtype=ttnn.bfloat16,
            device=self.device,
            batch_size=self.batch_size,
            input_height=512,
            input_width=512,
            reader_patterns_cache=self.max_pool_reader_patterns_cache,
            deallocate_activation=True,
            parallel_config_override=max_pool_parallel_config_override,
            channels=self.conv1_output_channels,
        )

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

    def __del__(self):
        # Need to clear global configs for each Resnet run
        self.conv_op_cache.clear()
        self.max_pool_reader_patterns_cache.clear()

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

    def preprocessing(self, torch_input_tensor):
        resnet50_first_conv_kernel_size = 3
        resnet50_first_conv_stride = 2
        input_tensor = pad_and_fold_conv_activation_for_unity_stride(
            torch_input_tensor,
            resnet50_first_conv_kernel_size,
            resnet50_first_conv_kernel_size,
            resnet50_first_conv_stride,
            resnet50_first_conv_stride,
        )
        input_tensor = torch.permute(input_tensor, (0, 2, 3, 1))
        input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16)
        return input_tensor

    def __call__(self, input_tensor, device, batch_size, ops_parallel_config) -> ttnn.Tensor:
        if not ops_parallel_config:
            return self.first_run(input_tensor, device, batch_size, ops_parallel_config)
        else:
            return self.optimized_run(input_tensor, device, batch_size, ops_parallel_config, self.conv_op_cache)

    def first_run(self, input_tensor, device, batch_size, ops_parallel_config) -> ttnn.Tensor:
        ## copy input to device sharded directly
        # x = ttnn.to_device(input_tensor, device=self.device, memory_config=self.conv1.conv.input_sharded_memory_config)
        conv_op_cache = {}
        act_block_h_override = 0
        if is_wormhole_b0():
            if batch_size == 16:
                act_block_h_override = 1568
            elif batch_size == 20:
                act_block_h_override = 640
            elif batch_size == 1:
                act_block_h_override = 256
        else:
            act_block_h_override = 0

        x, x_height, x_width, self.conv1_weight_tensor, self.conv1_bias_tensor = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.conv1_weight_tensor,
            in_channels=self.conv1_input_channels,
            out_channels=self.conv1_output_channels,
            device=device,
            bias_tensor=self.conv1_bias_tensor,
            kernel_size=(4, 4),
            stride=(1, 1),
            padding=(0, 0),
            batch_size=self.batch_size,
            input_height=515,
            input_width=515,
            conv_config=ttnn.Conv2dConfig(
                dtype=self.model_config["ACTIVATIONS_DTYPE"],
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                math_fidelity=self.model_config["MATH_FIDELITY"],
                activation="relu",
                deallocate_activation=True,
                reallocate_halo_output=True,
                input_channels_alignment=16 if not is_wormhole_b0() else 32,
                act_block_h_override=act_block_h_override,
            ),
            conv_op_cache=conv_op_cache,
        )
        # Relu is fused with conv1

        if self.batch_size == 20 or self.batch_size == 1:
            x = ttnn.reallocate(x)

        if is_wormhole_b0() and (self.batch_size == 20 or self.batch_size == 1):
            # TODO: fix the need to do the reshard here
            # x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
            x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(x)
            x = ttnn.reallocate(x_rm)
            x = ttnn.to_memory_config(x, self.max_pool.max_pool.input_sharded_memory_config)
        x = self.max_pool(x)

        x_height = 256
        x_width = 256

        x = ttnn.reshape(x, (1, 1, x_height * x_width * self.batch_size, 64))
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, dtype=self.model_config["ACTIVATIONS_DTYPE"])

        if self.batch_size == 20 and not is_wormhole_b0():
            x = ttnn.reallocate(x)

        print(f"=================================== layer: 1, module: 1")
        layer1_module1_input_shape = [
            x.get_legacy_shape()[0],
            x.get_legacy_shape()[1],
            x.get_legacy_shape()[2],
            x.get_legacy_shape()[3],
        ]
        if is_wormhole_b0() and self.batch_size == 20:
            x, x_height, x_width = self.layer1_module1(
                x,
                device,
                batch_size,
                x_height,
                x_width,
                conv_op_cache,
                reshard_if_not_optimal=True,
                height_sharding=True,
            )
        else:
            x, x_height, x_width = self.layer1_module1(x, device, batch_size, x_height, x_width, conv_op_cache)
        x_memory_config = ttnn.get_memory_config(x)
        ops_parallel_config["layer1_module1_input"] = ttnn.create_sharded_memory_config_(
            layer1_module1_input_shape,
            x_memory_config.shard_spec.grid,
            x_memory_config.memory_layout,
            x_memory_config.shard_spec.orientation,
            tile_layout=True,
        )
        print(f"=================================== layer: 1, module: 2")
        x, x_height, x_width = self.layer1_module2(x, device, batch_size, x_height, x_width, conv_op_cache)
        print(f"=================================== layer: 1, module: 3")
        x, x_height, x_width = self.layer1_module3(x, device, batch_size, x_height, x_width, conv_op_cache)
        if self.batch_size == 20 and is_wormhole_b0():
            x = ttnn.reallocate(x)

        layer2_module1_input_shape = [
            x.get_legacy_shape()[0],
            x.get_legacy_shape()[1],
            x.get_legacy_shape()[2],
            x.get_legacy_shape()[3],
        ]
        print(f"=================================== layer: 2, module: 1")
        if is_wormhole_b0() and self.batch_size == 20:
            x, x_height, x_width = self.layer2_module1(
                x,
                device,
                batch_size,
                x_height,
                x_width,
                conv_op_cache,
                reshard_if_not_optimal=True,
                height_sharding=True,
            )
        else:
            x, x_height, x_width = self.layer2_module1(x, device, batch_size, x_height, x_width, conv_op_cache)
        x_memory_config = ttnn.get_memory_config(x)
        ops_parallel_config["layer2_module1_input"] = ttnn.create_sharded_memory_config_(
            layer2_module1_input_shape,
            x_memory_config.shard_spec.grid,
            x_memory_config.memory_layout,
            x_memory_config.shard_spec.orientation,
            tile_layout=True,
        )
        print(f"=================================== layer: 2, module: 2")
        x, x_height, x_width = self.layer2_module2(x, device, batch_size, x_height, x_width, conv_op_cache)
        print(f"=================================== layer: 2, module: 3")
        x, x_height, x_width = self.layer2_module3(x, device, batch_size, x_height, x_width, conv_op_cache)
        print(f"=================================== layer: 2, module: 4")
        x, x_height, x_width = self.layer2_module4(x, device, batch_size, x_height, x_width, conv_op_cache)

        print(f"=================================== layer: 3, module: 1")
        layer3_module1_input_shape = [
            x.get_legacy_shape()[0],
            x.get_legacy_shape()[1],
            x.get_legacy_shape()[2],
            x.get_legacy_shape()[3],
        ]
        x, x_height, x_width = self.layer3_module1(
            x,
            device,
            batch_size,
            x_height,
            x_width,
            conv_op_cache,
            reshard_if_not_optimal=True,
            height_sharding=False,
        )
        x_memory_config = ttnn.get_memory_config(x)
        ops_parallel_config["layer3_module1_input"] = ttnn.create_sharded_memory_config_(
            layer3_module1_input_shape,
            x_memory_config.shard_spec.grid,
            x_memory_config.memory_layout,
            x_memory_config.shard_spec.orientation,
            tile_layout=True,
        )
        print(f"=================================== layer: 3, module: 2")
        x, x_height, x_width = self.layer3_module2(x, device, batch_size, x_height, x_width, conv_op_cache)
        print(f"=================================== layer: 3, module: 3")
        x, x_height, x_width = self.layer3_module3(x, device, batch_size, x_height, x_width, conv_op_cache)
        print(f"=================================== layer: 3, module: 4")
        x, x_height, x_width = self.layer3_module4(x, device, batch_size, x_height, x_width, conv_op_cache)
        print(f"=================================== layer: 3, module: 5")
        x, x_height, x_width = self.layer3_module5(x, device, batch_size, x_height, x_width, conv_op_cache)
        print(f"=================================== layer: 3, module: 6")
        x, x_height, x_width = self.layer3_module6(
            x,
            device,
            batch_size,
            x_height,
            x_width,
            conv_op_cache,
            eltwise_binary_out_in_place=False,
        )

        print(f"=================================== layer: 4, module: 1")
        layer4_module1_input_shape = [
            x.get_legacy_shape()[0],
            x.get_legacy_shape()[1],
            x.get_legacy_shape()[2],
            x.get_legacy_shape()[3],
        ]
        x, x_height, x_width = self.layer4_module1(
            x,
            device,
            batch_size,
            x_height,
            x_width,
            conv_op_cache,
            reshard_if_not_optimal=True,
            height_sharding=False,
        )
        x_memory_config = ttnn.get_memory_config(x)
        ops_parallel_config["layer4_module1_input"] = ttnn.create_sharded_memory_config_(
            layer4_module1_input_shape,
            x_memory_config.shard_spec.grid,
            x_memory_config.memory_layout,
            x_memory_config.shard_spec.orientation,
            tile_layout=True,
        )
        print(f"=================================== layer: 4, module: 2")
        x, x_height, x_width = self.layer4_module2(x, device, batch_size, x_height, x_width, conv_op_cache)
        print(f"=================================== layer: 4, module: 3")
        x, x_height, x_width = self.layer4_module3(x, device, batch_size, x_height, x_width, conv_op_cache)

        unpadded_shape = x.shape_without_padding()
        x = ttnn.untilize_with_unpadding(
            x,
            output_tensor_end=(
                unpadded_shape[0] - 1,
                unpadded_shape[1] - 1,
                unpadded_shape[2] - 1,
                unpadded_shape[3] - 1,
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        x = ttnn.reshape(
            x,
            (
                self.batch_size,
                x.get_legacy_shape()[1],
                (int)(x.get_legacy_shape()[2] / self.batch_size),
                x.get_legacy_shape()[3],
            ),
        )

        grid_size = (8, 4)
        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(grid_size[0] - 1, grid_size[1] - 1),
                )
            }
        )
        shard_shape = [
            x.volume() // x.get_legacy_shape()[-1],
            x.get_legacy_shape()[-1] // (grid_size[0] * grid_size[1]),
        ]
        shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR, False)
        width_sharded_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec
        )
        x = ttnn.to_memory_config(x, width_sharded_mem_config)
        unpadded_shape = x.get_legacy_shape()
        padded_shape = [
            unpadded_shape[0],
            unpadded_shape[1],
            _nearest_32(unpadded_shape[2]),
            _nearest_32(unpadded_shape[3]),
        ]
        x = ttnn.tilize_with_val_padding(
            x,
            padded_shape,
            0,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        x = self.avgpool(x, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)

        unpadded_shape_end = [
            x.get_legacy_shape()[0] - 1,
            x.get_legacy_shape()[1] - 1,
            1 - 1,
            x.get_legacy_shape()[3] - 1,
        ]
        x = ttnn.untilize_with_unpadding(
            x,
            output_tensor_end=unpadded_shape_end,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )

        x = ttnn.reshape(
            x,
            (
                1,
                x.get_legacy_shape()[1],
                self.batch_size * x.get_legacy_shape()[2],
                x.get_legacy_shape()[3],
            ),
        )

        unpadded_shape = x.get_legacy_shape()
        padded_shape = [
            unpadded_shape[0],
            unpadded_shape[1],
            _nearest_32(unpadded_shape[2]),
            _nearest_32(unpadded_shape[3]),
        ]

        x = ttnn.tilize_with_val_padding(
            x,
            padded_shape,
            0,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        x = self.fc(x)
        desired_shape = list(x.shape_without_padding())
        desired_shape[-1] = 1000
        x = ttnn.untilize_with_unpadding(
            x,
            output_tensor_end=(
                desired_shape[0] - 1,
                desired_shape[1] - 1,
                desired_shape[2] - 1,
                desired_shape[3] - 1,
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        x = ttnn.reshape(
            x,
            (
                self.batch_size,
                x.get_legacy_shape()[1],
                (int)(x.get_legacy_shape()[2] / self.batch_size),
                x.get_legacy_shape()[3],
            ),
        )

        return x

    def optimized_run(self, input_tensor, device, batch_size, ops_parallel_config, conv_op_cache) -> ttnn.Tensor:
        ## copy input to device sharded directly
        # x = ttnn.to_device(input_tensor, device=self.device, memory_config=self.conv1.conv.input_sharded_memory_config)
        act_block_h_override = 0
        if is_wormhole_b0():
            if batch_size == 16:
                act_block_h_override = 1568
            elif batch_size == 20:
                act_block_h_override = 640
            elif batch_size == 1:
                act_block_h_override = 256

        x, x_height, x_width, self.conv1_weight_tensor, self.conv1_bias_tensor = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.conv1_weight_tensor,
            in_channels=self.conv1_input_channels,
            out_channels=self.conv1_output_channels,
            device=device,
            bias_tensor=self.conv1_bias_tensor,
            kernel_size=(4, 4),
            stride=(1, 1),
            padding=(0, 0),
            batch_size=self.batch_size,
            input_height=515,
            input_width=515,
            conv_config=ttnn.Conv2dConfig(
                dtype=self.model_config["ACTIVATIONS_DTYPE"],
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                math_fidelity=self.model_config["MATH_FIDELITY"],
                activation="relu",
                deallocate_activation=True,
                input_channels_alignment=16 if not is_wormhole_b0() else 32,
                act_block_h_override=act_block_h_override,
            ),
            conv_op_cache=conv_op_cache,
        )
        # Relu is fused with conv1

        if self.batch_size == 20 or self.batch_size == 1:
            x = ttnn.reallocate(x)

        if is_wormhole_b0() and self.batch_size == 20 or self.batch_size == 1:
            # TODO: fix the need to do the reshard here
            # x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
            x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(x)
            x = ttnn.reallocate(x_rm)
            x = ttnn.to_memory_config(x, self.max_pool.max_pool.input_sharded_memory_config)
        x = self.max_pool(x)

        x_height = 256
        x_width = 256

        x = ttnn.reshape(x, (1, 1, x_height * x_width * self.batch_size, 64))
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, dtype=self.model_config["ACTIVATIONS_DTYPE"])

        if self.batch_size == 20 and not is_wormhole_b0():
            x = ttnn.reallocate(x)

        if is_wormhole_b0() and batch_size == 20:
            x = ttnn.to_memory_config(x, ops_parallel_config["layer1_module1_input"])
        x, x_height, x_width = self.layer1_module1(x, device, batch_size, x_height, x_width, conv_op_cache)
        x, x_height, x_width = self.layer1_module2(x, device, batch_size, x_height, x_width, conv_op_cache)
        x, x_height, x_width = self.layer1_module3(x, device, batch_size, x_height, x_width, conv_op_cache)
        if self.batch_size == 20 and is_wormhole_b0():
            x = ttnn.reallocate(x)
            x = ttnn.to_memory_config(x, ops_parallel_config["layer2_module1_input"])

        x, x_height, x_width = self.layer2_module1(x, device, batch_size, x_height, x_width, conv_op_cache)
        x, x_height, x_width = self.layer2_module2(x, device, batch_size, x_height, x_width, conv_op_cache)
        x, x_height, x_width = self.layer2_module3(x, device, batch_size, x_height, x_width, conv_op_cache)
        x, x_height, x_width = self.layer2_module4(x, device, batch_size, x_height, x_width, conv_op_cache)

        # do reshard before layer3
        x = ttnn.to_memory_config(x, ops_parallel_config["layer3_module1_input"])
        x, x_height, x_width = self.layer3_module1(x, device, batch_size, x_height, x_width, conv_op_cache)
        x, x_height, x_width = self.layer3_module2(x, device, batch_size, x_height, x_width, conv_op_cache)
        x, x_height, x_width = self.layer3_module3(x, device, batch_size, x_height, x_width, conv_op_cache)
        x, x_height, x_width = self.layer3_module4(x, device, batch_size, x_height, x_width, conv_op_cache)
        x, x_height, x_width = self.layer3_module5(x, device, batch_size, x_height, x_width, conv_op_cache)
        x, x_height, x_width = self.layer3_module6(
            x,
            device,
            batch_size,
            x_height,
            x_width,
            conv_op_cache,
            eltwise_binary_out_in_place=False,
        )

        # do reshard before layer4
        x = ttnn.to_memory_config(x, ops_parallel_config["layer4_module1_input"])
        x, x_height, x_width = self.layer4_module1(
            x, device, batch_size, x_height, x_width, conv_op_cache, reshard_if_not_optimal=True
        )
        x, x_height, x_width = self.layer4_module2(x, device, batch_size, x_height, x_width, conv_op_cache)
        x, x_height, x_width = self.layer4_module3(x, device, batch_size, x_height, x_width, conv_op_cache)

        unpadded_shape = x.shape_without_padding()
        x = ttnn.untilize_with_unpadding(
            x,
            output_tensor_end=(
                unpadded_shape[0] - 1,
                unpadded_shape[1] - 1,
                unpadded_shape[2] - 1,
                unpadded_shape[3] - 1,
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        x = ttnn.reshape(
            x,
            (
                self.batch_size,
                x.get_legacy_shape()[1],
                (int)(x.get_legacy_shape()[2] / self.batch_size),
                x.get_legacy_shape()[3],
            ),
        )

        grid_size = (8, 4)
        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(grid_size[0] - 1, grid_size[1] - 1),
                )
            }
        )
        shard_shape = [
            x.volume() // x.get_legacy_shape()[-1],
            x.get_legacy_shape()[-1] // (grid_size[0] * grid_size[1]),
        ]
        shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR, False)
        width_sharded_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec
        )
        x = ttnn.to_memory_config(x, width_sharded_mem_config)
        unpadded_shape = x.get_legacy_shape()
        padded_shape = [
            unpadded_shape[0],
            unpadded_shape[1],
            _nearest_32(unpadded_shape[2]),
            _nearest_32(unpadded_shape[3]),
        ]
        x = ttnn.tilize_with_val_padding(
            x,
            padded_shape,
            0,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        x = self.avgpool(x, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)

        unpadded_shape_end = [
            x.get_legacy_shape()[0] - 1,
            x.get_legacy_shape()[1] - 1,
            1 - 1,
            x.get_legacy_shape()[3] - 1,
        ]
        x = ttnn.untilize_with_unpadding(
            x,
            output_tensor_end=unpadded_shape_end,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )

        x = ttnn.reshape(
            x,
            (
                1,
                x.get_legacy_shape()[1],
                self.batch_size * x.get_legacy_shape()[2],
                x.get_legacy_shape()[3],
            ),
        )

        unpadded_shape = x.get_legacy_shape()
        padded_shape = [
            unpadded_shape[0],
            unpadded_shape[1],
            _nearest_32(unpadded_shape[2]),
            _nearest_32(unpadded_shape[3]),
        ]

        x = ttnn.tilize_with_val_padding(
            x,
            padded_shape,
            0,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        x = self.fc(x)
        desired_shape = list(x.shape_without_padding())
        desired_shape[-1] = 1000
        x = ttnn.untilize_with_unpadding(
            x,
            output_tensor_end=(
                desired_shape[0] - 1,
                desired_shape[1] - 1,
                desired_shape[2] - 1,
                desired_shape[3] - 1,
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        x = ttnn.reshape(
            x,
            (
                self.batch_size,
                x.get_legacy_shape()[1],
                (int)(x.get_legacy_shape()[2] / self.batch_size),
                x.get_legacy_shape()[3],
            ),
        )

        return x
