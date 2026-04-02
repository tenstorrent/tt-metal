# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

import ttnn
from models.common.utility_functions import roundup32
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores


class Yolov11sConv2D:
    def __init__(
        self,
        conv,
        conv_pth,
        bn=None,
        device=None,
        activation=None,
        activation_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat8_b,
        reshard=False,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        is_detect=False,
        is_dfl=False,
        config_override=None,
        deallocate_activation=False,
        split_weights=False,
        enable_act_double_buffer=True,
        enable_weights_double_buffer=True,
        core_count=None,
    ):
        self.is_detect = is_detect
        self.activation = activation
        self.is_dfl = is_dfl
        self.conv = conv
        self.device = device
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        if split_weights:
            self.out_channels = self.out_channels // 2
        self.kernel_size = conv.kernel_size
        if hasattr(conv.padding, "__len__"):
            if len(conv.padding) == 2:
                self.padding = (conv.padding[0], conv.padding[0], conv.padding[1], conv.padding[1])
            elif len(conv.padding) == 4:
                self.padding = (conv.padding[0], conv.padding[1], conv.padding[2], conv.padding[3])
            else:
                raise ValueError("Padding should be a scalar or a list of 2 or 4 elements")
        else:
            p = conv.padding
            self.padding = (p, p, p, p)
        self.stride = conv.stride
        self.groups = conv.groups
        self.reshard = reshard
        self.deallocate_activation = deallocate_activation
        self.core_count = core_count
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            # packer_l1_acc=True,
            packer_l1_acc=False if self.is_detect else True,
            math_approx_mode=True,
        )
        self.activation_dtype = activation_dtype
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=weights_dtype,
            shard_layout=shard_layout,
            deallocate_activation=self.deallocate_activation,
            enable_act_double_buffer=enable_act_double_buffer,
            reshard_if_not_optimal=True if self.reshard else False,
            activation=self.activation,
            enable_weights_double_buffer=enable_weights_double_buffer,
            output_layout=ttnn.TILE_LAYOUT,
        )
        if config_override:
            if "act_block_h" in config_override:
                self.conv_config.act_block_h_override = config_override["act_block_h"]
            if "act_block_w_div" in config_override:
                self.conv_config.act_block_w_div = config_override["act_block_w_div"]
            if "enable_act_double_buffer" in config_override:
                self.conv_config.enable_act_double_buffer = config_override["enable_act_double_buffer"]
            if "enable_weights_double_buffer" in config_override:
                self.conv_config.enable_weights_double_buffer = config_override["enable_weights_double_buffer"]
            if config_override.get("config_tensors_in_dram"):
                self.conv_config.config_tensors_in_dram = True
        self.weight = ttnn.from_device(conv_pth.weight)
        self.bias = ttnn.from_device(conv_pth.bias) if "bias" in conv_pth else None
        if self.core_count is not None:
            shard_grid = get_shard_grid_from_num_cores(self.core_count, self.device)
            self.conv_config.core_grid = shard_grid
            self.conv_config.override_sharding_config = True

    def __call__(self, x, output_rm_needed=False, to_interleaved=False):
        if self.is_detect:
            h = int(math.sqrt(x.shape[2]))
            input_height = h
            input_width = h
            batch_size = x.shape[0]
        elif self.is_dfl:
            input_height = x.shape[1]
            input_width = x.shape[2]
            batch_size = x.shape[0]
        else:
            batch_size = self.conv.batch_size
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
            groups=self.groups,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.activation_dtype,
            slice_config=ttnn.Conv2dL1FullSliceConfig,
        )
        hw = output_height * output_width
        if to_interleaved:
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
            x = x[:, :, :hw, :]
        else:
            if x.shape[2] != hw and output_rm_needed:
                x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
                x = x[:, :, :hw, :]
        return x


def reshard_if_possible(x, core_grid=None):  # reshards if shard_spec is not multiples of 32
    if x.is_sharded() and (
        x.memory_config().shard_spec.shape[0] % 32 != 0 or x.memory_config().shard_spec.shape[1] % 32 != 0
    ):
        aligned_h, aligned_w = roundup32(x.memory_config().shard_spec.shape[0]), roundup32(
            x.memory_config().shard_spec.shape[1]
        )
        resharded_memory_config = ttnn.create_sharded_memory_config(
            shape=(aligned_h, aligned_w),
            core_grid=x.memory_config().shard_spec.grid if core_grid is None else core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=x.memory_config().shard_spec.orientation,
            use_height_and_width_as_shard_shape=True,
        )
        x = ttnn.to_memory_config(x, resharded_memory_config)
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


# for input tensor's whose shard_w is different from each other
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
        activation=None,
        deallocate_activation=False,
        split_weights=False,
        config_override=None,
        enable_act_double_buffer=True,
        enable_weights_double_buffer=True,
        core_count=None,
    ):
        self.enable_act = enable_act
        if self.enable_act:
            activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
        merged = dict(config_override) if config_override else {}
        if enable_act_double_buffer is not None:
            merged["enable_act_double_buffer"] = enable_act_double_buffer
        if enable_weights_double_buffer is not None:
            merged["enable_weights_double_buffer"] = enable_weights_double_buffer
        co = merged if merged else None
        self.conv = Yolov11sConv2D(
            parameter.conv,
            conv_pt.conv,
            device=device,
            is_detect=is_detect,
            reshard=reshard,
            activation=activation,
            deallocate_activation=deallocate_activation,
            split_weights=split_weights,
            config_override=co,
            enable_act_double_buffer=enable_act_double_buffer,
            enable_weights_double_buffer=enable_weights_double_buffer,
            core_count=core_count,
        )

    def __call__(self, device, x, output_rm_needed=False, to_interleaved=False):
        x = self.conv(x, output_rm_needed, to_interleaved)
        return x


def deallocate_tensors(*tensors):
    for t in tensors:
        ttnn.deallocate(t)
