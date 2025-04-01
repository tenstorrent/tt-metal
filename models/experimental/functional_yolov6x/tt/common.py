# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import math

HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
BS = ttnn.TensorMemoryLayout.BLOCK_SHARDED
WS = ttnn.TensorMemoryLayout.WIDTH_SHARDED


class Yolov6x_Conv2D:
    def __init__(
        self,
        conv,
        conv_pth,
        bn=None,
        device=None,
        cache={},
        activation="",
        activation_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat8_b,
        use_1d_systolic_array=True,
        use_shallow_conv_variant=False,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        auto_shard=False,
        is_nhw_c=False,
        is_nhwc=False,
    ):
        self.is_nhw_c = is_nhw_c
        self.is_nhwc = is_nhwc
        self.conv = conv
        self.device = device
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.padding = conv.padding
        self.stride = conv.stride
        self.groups = conv.groups
        self.use_1d_systolic_array = use_1d_systolic_array
        self.deallocate_activation = False
        self.cache = cache
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
            math_approx_mode=True,
        )

        self.conv_config = ttnn.Conv2dConfig(
            dtype=activation_dtype,
            weights_dtype=weights_dtype,
            shard_layout=shard_layout if not auto_shard else None,
            deallocate_activation=self.deallocate_activation,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True if self.use_1d_systolic_array else False,
            activation=activation,
            input_channels_alignment=8 if use_shallow_conv_variant and not auto_shard else 32,
        )
        config_override = None
        if config_override and "act_block_h" in config_override and not auto_shard:
            self.conv_config.act_block_h_override = config_override["act_block_h"]

        if "bias" in conv_pth:
            bias = ttnn.from_device(conv_pth.bias)
            self.bias = bias
        else:
            self.bias = None

        weight = ttnn.from_device(conv_pth.weight)
        self.weight = weight

    def __call__(self, x):
        if self.is_nhw_c:
            input_height = int(math.sqrt(x.shape[2]))
            input_width = int(math.sqrt(x.shape[2]))
            batch_size = x.shape[0]
        elif self.is_nhwc:
            input_height = x.shape[1]
            input_width = x.shape[2]
            batch_size = x.shape[0]
        else:
            batch_size = x.shape[0]  # self.conv.batch_size
            input_height = self.conv.input_height
            input_width = self.conv.input_width

        [x, [output_height, output_width], [self.weight, self.bias]] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            input_height=input_height,
            input_width=input_width,
            batch_size=batch_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            conv_config=self.conv_config,
            conv_op_cache=self.cache,
            groups=self.groups,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        return x


class Yolov6x_Conv_T_2D:
    def __init__(
        self,
        conv,
        conv_pth,
        device=None,
        cache={},
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat8_b,
        use_1d_systolic_array=True,
        shard_layout=None,
        auto_shard=False,
        use_shallow_conv_variant=False,
        config_override=None,
    ):
        self.input_channels = conv.in_channels
        self.output_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.padding = conv.padding
        self.stride = conv.stride
        self.device = device
        if shard_layout is None and not auto_shard:
            shard_layout = (
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                if use_1d_systolic_array
                else ttnn.TensorMemoryLayout.BLOCK_SHARDED
            )
        self.conv_config = ttnn.Conv2dConfig(
            dtype=activations_dtype,
            weights_dtype=weights_dtype,
            shard_layout=shard_layout,
            input_channels_alignment=(
                16 if use_shallow_conv_variant or (self.input_channels == 16 and self.input_height == 115) else 32
            ),
            deallocate_activation=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            output_layout=ttnn.TILE_LAYOUT,
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        if config_override and "act_block_h" in config_override:
            self.conv_config.act_block_h_override = config_override["act_block_h"]

        if "bias" in conv_pth.conv_t:
            bias = ttnn.from_device(conv_pth.conv_t.bias)
            self.bias = bias
        else:
            self.bias = None

        self.weight = ttnn.from_device(conv_pth.conv_t.weight)

    def __call__(self, x):
        input_height = int(math.sqrt(x.shape[2]))
        input_width = int(math.sqrt(x.shape[2]))
        batch_size = x.shape[0]

        [tt_output_tensor_on_device, [out_height, out_width], [weights_device, bias_device]] = ttnn.conv_transpose2d(
            input_tensor=x,
            weight_tensor=self.weight,
            in_channels=self.input_channels,
            out_channels=self.output_channels,
            device=self.device,
            bias_tensor=self.bias,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=(0, 0),
            output_padding=(0, 0),
            dilation=(1, 1),
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            groups=1,
            mirror_kernel=True,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        return tt_output_tensor_on_device


def sharded_concat(input_tensors, num_cores=64, dim=3):  # expected input tensors to be in fp16, RM, same (h*w)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    in_shard_width = input_tensors[0].shape[-1]
    shard_height = (input_tensors[0].shape[2] + num_cores - 1) // num_cores
    input_sharded_memory_config = ttnn.create_sharded_memory_config(
        (shard_height, in_shard_width),
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    out_shard_width = 0
    for i in range(len(input_tensors)):
        out_shard_width += input_tensors[i].shape[-1]
        input_tensors[i] = ttnn.to_memory_config(input_tensors[i], input_sharded_memory_config)
    output_sharded_memory_config = ttnn.create_sharded_memory_config(
        (shard_height, out_shard_width),
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    output = ttnn.concat(input_tensors, dim, memory_config=output_sharded_memory_config)
    output = ttnn.sharded_to_interleaved(output, memory_config=ttnn.L1_MEMORY_CONFIG)

    return output


def deallocate_tensors(*tensors):
    for t in tensors:
        ttnn.deallocate(t)
