# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from models.utility_functions import (
    is_grayskull,
    is_wormhole_b0,
    pad_and_fold_conv_activation_for_unity_stride,
)

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

    def __init__(
        self,
        device,
        parameters,
        reader_patterns_cache,
        batch_size,
        input_height,
        input_width,
        stride,
        sharded_memory_config_type,
        downsample=None,
        model_config=None,
        conv_2d=False,
        module_out_sharded=False,
    ) -> None:
        super().__init__()
        self.device = device
        self.model_config = model_config
        self.output_memory_config = sharded_memory_config_type if module_out_sharded else ttnn.L1_MEMORY_CONFIG
        self.out_in_place = module_out_sharded

        self.transpose_mcast = (not conv_2d) if is_wormhole_b0() else True

        conv1_in_channels = parameters.conv1.weight.shape[1]

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.stride = stride
        self.module_input_shape = [batch_size, input_height, input_width, conv1_in_channels]
        self.deallocate = True
        self.downsample_or_noop = downsample
        if self.downsample_or_noop is None:
            self.downsample_or_noop = do_nothing_op
            self.deallocate = False

        # 1x1 conv with stride 1 padding 0 is run using regular matmul
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
        parameters.conv1.weight = ttnn.from_device(parameters.conv1.weight)
        parameters.conv1.bias = ttnn.from_device(parameters.conv1.bias)
        out_channels = parameters.conv1.weight.shape[0]
        in_channels = parameters.conv1.weight.shape[1]
        conv1_config_override = {}
        # for module with stride == 2 (not 1), conv1 output shape != downsample output shape
        # the shard config calculated for conv1 will be different from downsample
        #  override conv1 shard config to use downsample shard config
        if stride == 2:
            assert downsample is not None
            ds_parallel_config = downsample.conv.get_parallelization_config()
            conv1_config_override = {
                "grid_size": (ds_parallel_config.grid_size.x, ds_parallel_config.grid_size.y),
                "num_cores_nhw": ds_parallel_config.num_cores_nhw,
            }
        self.conv1 = ttnn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dtype=model_config["ACTIVATIONS_DTYPE"],
            device=device,
            use_1d_systolic_array=not conv_2d,
            # transpose_mcast=self.transpose_mcast,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            reader_patterns_cache=reader_patterns_cache,
            weight=parameters.conv1.weight,
            bias=parameters.conv1.bias,
            math_fidelity=model_config["MATH_FIDELITY"],
            weights_dtype=model_config["WEIGHTS_DTYPE"],
            conv_blocking_and_parallelization_config_override=conv1_config_override,
            compute_kernel_config=compute_kernel_config,
            activation="relu",
        )

        move_utwh_output = False
        if self.deallocate and (
            self.module_input_shape[0] == 20 and self.module_input_shape[1] == 56 and self.module_input_shape[3] == 256
        ):
            move_utwh_output = True
        parameters.conv2.weight = ttnn.from_device(parameters.conv2.weight)
        parameters.conv2.bias = ttnn.from_device(parameters.conv2.bias)
        out_channels = parameters.conv2.weight.shape[0]
        in_channels = parameters.conv2.weight.shape[1]
        conv2_config_override = {}
        if is_grayskull():
            if out_channels == 64 and self.module_input_shape[1] == 56 and self.module_input_shape[0] == 20:
                conv2_config_override = {"act_block_h": 320}
        else:
            if (
                in_channels == 128
                and out_channels == 128
                and self.module_input_shape[1] == 56
                and self.module_input_shape[0] == 20
            ):
                conv2_config_override = {"act_block_h": 160}
        self.conv2 = ttnn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(stride, stride),
            padding=(1, 1),
            dtype=model_config["ACTIVATIONS_DTYPE"],
            device=device,
            use_1d_systolic_array=not conv_2d,
            # transpose_mcast=self.transpose_mcast,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            reader_patterns_cache=reader_patterns_cache,
            weight=parameters.conv2.weight,
            bias=parameters.conv2.bias,
            math_fidelity=model_config["MATH_FIDELITY"],
            weights_dtype=model_config["WEIGHTS_DTYPE"],
            reallocate_halo_output=move_utwh_output,
            deallocate_activation=True,
            conv_blocking_and_parallelization_config_override=conv2_config_override,
            compute_kernel_config=compute_kernel_config,
            activation="relu",
        )

        input_height = ((int)((input_height - 1) / stride)) + 1
        input_width = ((int)((input_width - 1) / stride)) + 1
        parameters.conv3.weight = ttnn.from_device(parameters.conv3.weight)
        parameters.conv3.bias = ttnn.from_device(parameters.conv3.bias)
        out_channels = parameters.conv3.weight.shape[0]
        in_channels = parameters.conv3.weight.shape[1]
        self.conv3 = ttnn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dtype=model_config["ACTIVATIONS_DTYPE"],
            device=device,
            use_1d_systolic_array=not conv_2d,
            # transpose_mcast=self.transpose_mcast,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            reader_patterns_cache=reader_patterns_cache,
            weight=parameters.conv3.weight,
            bias=parameters.conv3.bias,
            math_fidelity=model_config["MATH_FIDELITY"],
            weights_dtype=model_config["WEIGHTS_DTYPE"],
            conv_blocking_and_parallelization_config_override={},
            compute_kernel_config=compute_kernel_config,
        )
        assert self.conv1.conv.output_sharded_memory_config == self.conv2.conv.input_sharded_memory_config
        assert self.conv2.conv.output_sharded_memory_config == self.conv3.conv.input_sharded_memory_config
        if downsample is not None:
            assert downsample.conv.input_sharded_memory_config == self.conv1.conv.input_sharded_memory_config
            assert downsample.conv.output_sharded_memory_config == self.conv3.conv.output_sharded_memory_config
        self.module_output_height = input_height
        self.module_output_width = input_width
        self.run_downsample_before_conv2 = False
        if not (self.module_input_shape[1] == 56 and self.module_input_shape[3] == 64):
            self.run_downsample_before_conv2 = True

    def __call__(self, x):
        # logger.info("This module input shape - ", self.module_input_shape)
        # conv1 is 1x1 conv
        if is_wormhole_b0():
            if ttnn.get_memory_config(x) != self.conv1.conv.input_sharded_memory_config:
                x_n = ttnn.to_memory_config(x, self.conv1.conv.input_sharded_memory_config)
                ttnn.deallocate(x)
                x = x_n

        # print("Running conv1")
        out = self.conv1(x)

        if self.run_downsample_before_conv2:
            ds_out = self.downsample_or_noop(x)
            if self.deallocate:
                ttnn.deallocate(x)

        # print("Running conv2")
        out = self.conv2(out)
        # conv3 is 1x1 conv
        # print("Running conv3")
        out = self.conv3(out)

        if not self.run_downsample_before_conv2:
            ds_out = self.downsample_or_noop(x)
            if self.deallocate:
                ttnn.deallocate(x)
        if self.out_in_place:
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
                memory_config=self.output_memory_config,
            )
        ttnn.deallocate(ds_out)
        if self.module_input_shape[0] == 20 and self.module_input_shape[1] == 56 and self.module_input_shape[3] == 64:
            out = ttnn.move(out)

        return out


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
        conv_input_face_shape_hw = [224, 224]
        self.device = device
        self.conv_input_face_shape_hw = conv_input_face_shape_hw
        self.batch_size = batch_size
        self.model_config = model_config
        self.reader_patterns_cache = {}
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
        parameters.conv1.weight = ttnn.from_device(parameters.conv1.weight)
        parameters.conv1.bias = ttnn.from_device(parameters.conv1.bias)
        out_channels = parameters.conv1.weight.shape[0]
        in_channels = parameters.conv1.weight.shape[1]
        self.first_conv_padded_input_channels = 16 if not is_wormhole_b0() else 32
        conv1_config_override = {}
        if is_wormhole_b0():
            if batch_size == 16:
                conv1_config_override = {"act_block_h": 1568}
            elif batch_size == 20:
                conv1_config_override = {"act_block_h": 640}
        self.conv1 = ttnn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(4, 4),
            stride=(1, 1),
            padding=(0, 0),
            dtype=model_config["ACTIVATIONS_DTYPE"],
            device=device,
            use_1d_systolic_array=True,
            batch_size=batch_size,
            input_height=115,
            input_width=115,
            reader_patterns_cache=self.reader_patterns_cache,
            weight=parameters.conv1.weight,
            bias=parameters.conv1.bias,
            math_fidelity=model_config["MATH_FIDELITY"],
            weights_dtype=model_config["WEIGHTS_DTYPE"],
            use_shallow_conv_variant=not is_wormhole_b0(),
            deallocate_activation=True,
            padded_input_channels=self.first_conv_padded_input_channels,
            activation="relu",
            conv_blocking_and_parallelization_config_override=conv1_config_override,
            compute_kernel_config=compute_kernel_config,
        )

        self.max_pool_reader_patterns_cache = {}
        max_pool_parallel_config_override = {}
        if is_grayskull() and self.batch_size != 20:
            max_pool_parallel_config_override["grid_size"] = self.conv1.conv.grid_size
            max_pool_parallel_config_override["num_cores_nhw"] = self.conv1.conv.sliding_window_op_params.num_cores_nhw

        self.max_pool = ttnn.MaxPool2d(
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            dilation=(1, 1),
            dtype=ttnn.bfloat16,
            device=self.device,
            batch_size=self.batch_size,
            input_height=112,
            input_width=112,
            reader_patterns_cache=self.max_pool_reader_patterns_cache,
            deallocate_activation=True,
            parallel_config_override=max_pool_parallel_config_override,
            channels=out_channels,
        )

        # for Wh, batch size 20 run, max pool input sharded memory config != conv1 output sharded memory config
        if not is_wormhole_b0() or self.batch_size != 20:
            assert self.max_pool.max_pool.input_sharded_memory_config == self.conv1.conv.output_sharded_memory_config
        self.layer1, self.layer1_output_height, self.layer1_output_width = self._make_layer(
            parameters=parameters.layer1,
            planes=64,
            blocks=layers[0],
            stride=1,
            batch_size=batch_size,
            input_height=56,
            input_width=56,
            compute_kernel_config=compute_kernel_config,
            sharded_memory_config_type=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            module_out_sharded=True,
            model_config=model_config,
            conv_2d=False,
        )
        self.layer2, self.layer2_output_height, self.layer2_output_width = self._make_layer(
            parameters=parameters.layer2,
            planes=128,
            blocks=layers[1],
            stride=2,
            batch_size=batch_size,
            input_height=self.layer1_output_height,
            input_width=self.layer1_output_width,
            compute_kernel_config=compute_kernel_config,
            sharded_memory_config_type=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            module_out_sharded=True,
            model_config=model_config,
            conv_2d=False,
        )
        self.layer3, self.layer3_output_height, self.layer3_output_width = self._make_layer(
            parameters=parameters.layer3,
            planes=256,
            blocks=layers[2],
            stride=2,
            batch_size=batch_size,
            input_height=self.layer2_output_height,
            input_width=self.layer2_output_width,
            compute_kernel_config=compute_kernel_config,
            sharded_memory_config_type=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            module_out_sharded=False,
            model_config=model_config,
            conv_2d=True,
        )
        self.layer4, self.layer4_output_height, self.layer4_output_width = self._make_layer(
            parameters=parameters.layer4,
            planes=512,
            blocks=layers[3],
            stride=2,
            batch_size=batch_size,
            input_height=self.layer3_output_height,
            input_width=self.layer3_output_width,
            compute_kernel_config=compute_kernel_config,
            sharded_memory_config_type=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            module_out_sharded=True,
            model_config=model_config,
            conv_2d=True,
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
            weight=parameters.fc.weight,
            bias=parameters.fc.bias,
            output_mem_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            model_config=model_config,
            device=self.device,
            batch_size=batch_size,
            compute_kernel_config=compute_kernel_config,
        )  # num_classes = 1000
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

    def __del__(self):
        # Need to clear global configs for each Resnet run
        self.reader_patterns_cache.clear()
        self.max_pool_reader_patterns_cache.clear()

    def _make_layer(
        self,
        parameters,
        planes: int,
        blocks: int,
        stride: int,
        batch_size: int,
        input_height: int,
        input_width: int,
        compute_kernel_config,
        sharded_memory_config_type,
        module_out_sharded,
        model_config=None,
        conv_2d=False,
    ):
        if stride != 1 or self.inplanes != planes * resnet50Bottleneck.expansion:
            parameters[0].downsample.weight = ttnn.from_device(parameters[0].downsample.weight)
            parameters[0].downsample.bias = ttnn.from_device(parameters[0].downsample.bias)
            out_channels = parameters[0].downsample.weight.shape[0]
            in_channels = parameters[0].downsample.weight.shape[1]
            self.downsample = ttnn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(stride, stride),
                padding=(0, 0),
                dtype=model_config["ACTIVATIONS_DTYPE"],
                device=self.device,
                use_1d_systolic_array=not conv_2d,
                # transpose_mcast=(not conv_2d) if is_wormhole_b0() else True,
                batch_size=batch_size,
                input_height=input_height,
                input_width=input_width,
                reader_patterns_cache=self.reader_patterns_cache,
                weight=parameters[0].downsample.weight,
                bias=parameters[0].downsample.bias,
                math_fidelity=model_config["MATH_FIDELITY"],
                weights_dtype=model_config["WEIGHTS_DTYPE"],
                conv_blocking_and_parallelization_config_override={},
                compute_kernel_config=compute_kernel_config,
            )

        layers = []
        layers.append(
            resnet50Bottleneck(
                device=self.device,
                parameters=parameters[0],
                reader_patterns_cache=self.reader_patterns_cache,
                batch_size=batch_size,
                input_height=input_height,
                input_width=input_width,
                stride=stride,
                sharded_memory_config_type=sharded_memory_config_type,
                downsample=self.downsample,
                model_config=model_config,
                conv_2d=conv_2d,
                module_out_sharded=True,
            )
        )
        self.inplanes = planes * resnet50Bottleneck.expansion
        for block_num in range(1, blocks):
            previous_layer = layers[-1]
            input_height = previous_layer.module_output_height
            input_width = previous_layer.module_output_width
            layers.append(
                resnet50Bottleneck(
                    device=self.device,
                    parameters=parameters[block_num],
                    reader_patterns_cache=self.reader_patterns_cache,
                    batch_size=batch_size,
                    input_height=input_height,
                    input_width=input_width,
                    stride=1,
                    sharded_memory_config_type=sharded_memory_config_type,
                    model_config=model_config,
                    conv_2d=conv_2d,
                    module_out_sharded=True if block_num != blocks - 1 else module_out_sharded,
                )
            )
        return layers, layers[-1].module_output_height, layers[-1].module_output_width

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

        ## reshape to [1, 1, N*H*W, C]
        input_tensor = torch.reshape(input_tensor, (1, 1, -1, input_tensor.shape[-1]))
        input_num_cores_nhw = self.conv1.conv.get_num_cores_nhw()
        input_tensor_height_snapped_to_tile = (
            self.conv1.conv.input_sharded_memory_config.shard_spec.shape[0] * input_num_cores_nhw
        )
        assert self.first_conv_padded_input_channels >= input_tensor.shape[3]
        input_tensor = torch.nn.functional.pad(
            input_tensor,
            (
                0,
                self.first_conv_padded_input_channels - input_tensor.shape[3],
                0,
                input_tensor_height_snapped_to_tile - input_tensor.shape[2],
                0,
                0,
            ),
        )
        input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16)
        return input_tensor

    def __call__(self, input_tensor) -> ttnn.Tensor:
        ## copy input to device sharded directly
        x = ttnn.to_device(
            input_tensor,
            device=self.device,
            memory_config=self.conv1.conv.input_sharded_memory_config,
        )

        x = self.conv1(x)
        # Relu is fused with conv1

        if self.batch_size == 20:
            x = ttnn.move(x)

        if is_wormhole_b0() and self.batch_size == 20:
            # TODO: fix the need to do the reshard here
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            x = ttnn.to_memory_config(x, self.max_pool.max_pool.input_sharded_memory_config)
        x = self.max_pool(x)

        x = ttnn.reshape(x, (1, 1, 56 * 56 * self.batch_size, 64))
        if is_wormhole_b0():
            # TODO: fix the need to do the reshard here
            x = ttnn.to_memory_config(x, self.layer1_module1.conv1.conv.input_sharded_memory_config)

        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, dtype=self.model_config["ACTIVATIONS_DTYPE"])

        if self.batch_size == 20 and not is_wormhole_b0():
            x = ttnn.move(x)

        x = self.layer1_module1(x)
        x = self.layer1_module2(x)
        x = self.layer1_module3(x)
        if self.batch_size == 20 and is_wormhole_b0():
            x = ttnn.move(x)

        x = self.layer2_module1(x)
        x = self.layer2_module2(x)
        x = self.layer2_module3(x)
        x = self.layer2_module4(x)

        # do reshard before layer3
        x = ttnn.to_memory_config(x, self.layer3_module1.conv1.conv.input_sharded_memory_config)
        x = self.layer3_module1(x)
        x = self.layer3_module2(x)
        x = self.layer3_module3(x)
        x = self.layer3_module4(x)
        x = self.layer3_module5(x)
        x = self.layer3_module6(x)

        # do reshard before layer4
        x = ttnn.to_memory_config(x, self.layer4_module1.conv1.conv.input_sharded_memory_config)
        x = self.layer4_module1(x)
        x = self.layer4_module2(x)
        x = self.layer4_module3(x)

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
        shard_grid = ttnn.experimental.tensor.CoreRangeSet(
            {
                ttnn.experimental.tensor.CoreRange(
                    ttnn.experimental.tensor.CoreCoord(0, 0),
                    ttnn.experimental.tensor.CoreCoord(grid_size[0] - 1, grid_size[1] - 1),
                )
            }
        )
        shard_shape = [
            x.volume() // x.get_legacy_shape()[-1],
            x.get_legacy_shape()[-1] // (grid_size[0] * grid_size[1]),
        ]
        shard_spec = ttnn.experimental.tensor.ShardSpec(
            shard_grid, shard_shape, ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR, False
        )
        width_sharded_mem_config = ttnn.types.MemoryConfig(
            ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec
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
