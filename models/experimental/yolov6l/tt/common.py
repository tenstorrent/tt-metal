# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import math

HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
BS = ttnn.TensorMemoryLayout.BLOCK_SHARDED
WS = ttnn.TensorMemoryLayout.WIDTH_SHARDED


class Yolov6l_Conv2D:
    def __init__(
        self,
        conv,
        conv_pth,
        bn=None,
        device=None,
        activation="",
        activation_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat8_b,
        use_1d_systolic_array=True,
        use_shallow_conv_variant=False,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        auto_shard=False,
        reshape=False,
        deallocate_activation=False,
        act_blocks=False,
        act_block_h=None,
        batch_size=1,
        return_height_width=False,
    ):
        self.conv = conv
        self.device = device
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.padding = conv.padding
        self.stride = conv.stride
        self.groups = conv.groups
        self.use_1d_systolic_array = use_1d_systolic_array
        self.deallocate_activation = deallocate_activation
        self.activation_dtype = activation_dtype
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
            math_approx_mode=True,
        )
        self.batch_size = batch_size
        self.return_height_width = return_height_width

        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=weights_dtype,
            shard_layout=shard_layout if not auto_shard else None,
            deallocate_activation=self.deallocate_activation,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True if self.use_1d_systolic_array else False,
            activation=activation,
        )
        if self.in_channels == 3:
            self.conv_config.act_block_h_override = 64
        self.reshape = reshape
        if act_block_h:
            self.conv_config.act_block_h_override = act_blocks

        if "bias" in conv_pth:
            bias = ttnn.from_device(conv_pth.bias)
            self.bias = bias
        else:
            self.bias = None

        weight = ttnn.from_device(conv_pth.weight)
        self.weight = weight

    def __call__(self, x):
        if x.shape[1] != 1:
            input_height = x.shape[1]
            input_width = x.shape[2]
        else:
            input_height = int(math.sqrt(x.shape[2]) // self.batch_size)
            input_width = int(math.sqrt(x.shape[2]) // self.batch_size)

        [output, [output_height, output_width], [self.weight, self.bias]] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            input_height=input_height,
            input_width=input_width,
            batch_size=self.batch_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            conv_config=self.conv_config,
            groups=self.groups,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.activation_dtype,
        )

        if self.reshape:
            output = ttnn.sharded_to_interleaved(output, ttnn.L1_MEMORY_CONFIG)
            output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
            output = ttnn.reshape(
                output, (x.shape[0], output_height, output_width, output.shape[3]), memory_config=ttnn.L1_MEMORY_CONFIG
            )
        if self.return_height_width:
            return output, output_height, output_width
        return output


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
        batch_size=1,
        reshape=False,
    ):
        self.batch_size = batch_size
        self.input_channels = conv.in_channels
        self.output_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.padding = conv.padding
        self.stride = conv.stride
        self.device = device
        self.reshape = reshape
        if shard_layout is None and not auto_shard:
            shard_layout = (
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                if use_1d_systolic_array
                else ttnn.TensorMemoryLayout.BLOCK_SHARDED
            )
        self.activations_dtype = activations_dtype
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=weights_dtype,
            shard_layout=shard_layout,
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
        if x.shape[1] != 1:
            input_height = x.shape[1]
            input_width = x.shape[2]
        else:
            input_height = int(math.sqrt(x.shape[2]) // self.batch_size)
            input_width = int(math.sqrt(x.shape[2]) // self.batch_size)

        [tt_output_tensor, [out_height, out_width], [self.weight, self.bias]] = ttnn.conv_transpose2d(
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
            batch_size=self.batch_size,
            input_height=input_height,
            input_width=input_width,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            groups=1,
            mirror_kernel=True,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.activations_dtype,
        )

        if self.reshape:
            tt_output_tensor = ttnn.sharded_to_interleaved(tt_output_tensor, ttnn.L1_MEMORY_CONFIG)
            tt_output_tensor = ttnn.to_layout(tt_output_tensor, ttnn.ROW_MAJOR_LAYOUT)
            tt_output_tensor = ttnn.reshape(
                tt_output_tensor,
                (x.shape[0], out_height, out_width, tt_output_tensor.shape[3]),
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        return tt_output_tensor
