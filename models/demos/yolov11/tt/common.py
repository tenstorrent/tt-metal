# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

import ttnn


class Yolov11Conv2D:
    def __init__(
        self,
        conv,
        conv_pth,
        bn=None,
        device=None,
        activation="",
        activation_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat8_b,
        reshard=False,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        is_detect=False,
        is_dfl=False,
        config_override=None,
        deallocate_activation=False,
    ):
        self.is_detect = is_detect
        self.activation = activation
        self.is_dfl = is_dfl
        self.conv = conv
        self.device = device
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.padding = conv.padding
        self.stride = conv.stride
        self.groups = conv.groups
        self.reshard = reshard
        self.deallocate_activation = deallocate_activation
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
            math_approx_mode=True,
        )
        self.activation_dtype = activation_dtype
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=weights_dtype,
            shard_layout=shard_layout,
            deallocate_activation=self.deallocate_activation,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True if self.reshard else False,
            activation=self.activation,
        )
        if config_override and "act_block_h" in config_override:
            self.conv_config.act_block_h_override = config_override["act_block_h"]

        if "bias" in conv_pth:
            bias = ttnn.from_device(conv_pth.bias)
            self.bias = bias
        else:
            self.bias = None

        weight = ttnn.from_device(conv_pth.weight)
        self.weight = weight

    def __call__(self, x):
        if self.is_detect:
            input_height = int(math.sqrt(x.shape[2]))
            input_width = int(math.sqrt(x.shape[2]))
            batch_size = x.shape[0]
        elif self.is_dfl:
            input_height = x.shape[1]
            input_width = x.shape[2]
            batch_size = x.shape[0]
        else:
            batch_size = self.conv.batch_size
            input_height = self.conv.input_height
            input_width = self.conv.input_width

        kernel_size = [self.kernel_size[0], self.kernel_size[1]]
        stride = [self.stride[0], self.stride[1]]
        padding = [self.padding[0], self.padding[1]]
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
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            conv_config=self.conv_config,
            groups=self.groups,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.activation_dtype,
        )
        hw = output_height * output_width
        if x.shape[2] != hw:
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
            x = x[:, :, :hw, :]
        return x


def sharded_concat(input_tensors, num_cores=64, dim=3, to_interleaved=True):
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
    if to_interleaved:
        output = ttnn.sharded_to_interleaved(output, memory_config=ttnn.L1_MEMORY_CONFIG)

    return output


# for input tensor's whose shape is different from each other
def sharded_concat_2(
    input_tensor_1, input_tensor_2, num_cores=64, shard_grid_coord_min=0, shard_grid_coord_max=7, dim=-1
):
    if input_tensor_1.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
        input_tensor_1 = ttnn.to_layout(input_tensor_1, ttnn.ROW_MAJOR_LAYOUT)

    if input_tensor_2.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
        input_tensor_2 = ttnn.to_layout(input_tensor_2, ttnn.ROW_MAJOR_LAYOUT)

    shard_height = (input_tensor_1.shape[2] + num_cores - 1) // num_cores

    input_sharded_memory_config_1 = ttnn.create_sharded_memory_config(
        (shard_height, input_tensor_1.shape[-1]),
        core_grid=ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(shard_grid_coord_min, shard_grid_coord_min),
                    ttnn.CoreCoord(shard_grid_coord_max, shard_grid_coord_max),
                )
            }
        ),
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    input_sharded_memory_config_2 = ttnn.create_sharded_memory_config(
        (shard_height, input_tensor_2.shape[-1]),
        core_grid=ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(shard_grid_coord_min, shard_grid_coord_min),
                    ttnn.CoreCoord(shard_grid_coord_max, shard_grid_coord_max),
                )
            }
        ),
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    input_tensor_1 = ttnn.to_memory_config(input_tensor_1, input_sharded_memory_config_1)
    input_tensor_2 = ttnn.to_memory_config(input_tensor_2, input_sharded_memory_config_2)
    out_sharded_memory_config_ = ttnn.create_sharded_memory_config(
        (shard_height, input_tensor_1.shape[-1] + input_tensor_2.shape[-1]),
        core_grid=ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(shard_grid_coord_min, shard_grid_coord_min),
                    ttnn.CoreCoord(shard_grid_coord_max, shard_grid_coord_max),
                )
            }
        ),
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    output = ttnn.concat((input_tensor_1, input_tensor_2), dim, memory_config=out_sharded_memory_config_)
    return output


class TtnnConv:
    def __init__(
        self,
        device,
        parameter,
        conv_pt,
        enable_act=True,
        is_detect=False,
        reshard=False,
        activation="",
        deallocate_activation=False,
    ):
        self.enable_act = enable_act
        if self.enable_act:
            activation = "silu"
        self.conv = Yolov11Conv2D(
            parameter.conv,
            conv_pt.conv,
            device=device,
            is_detect=is_detect,
            reshard=reshard,
            activation=activation,
            deallocate_activation=deallocate_activation,
        )

    def __call__(self, device, x):
        x = self.conv(x)
        return x


def deallocate_tensors(*tensors):
    for t in tensors:
        ttnn.deallocate(t)
