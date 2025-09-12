# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_cnn.tt.builder import (
    AutoShardedStrategyConfiguration,
    BlockShardedStrategyConfiguration,
    Conv2dConfiguration,
    DeviceDescriptor,
    HeightShardedStrategyConfiguration,
    TtConv2d,
    WidthShardedStrategyConfiguration,
)


def to_sharding_strategy(conv_param):
    if conv_param.shard_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        return HeightShardedStrategyConfiguration(
            reshard_if_not_optimal=conv_param.reshard_if_not_optimal,
            act_block_h_override=conv_param.act_block_h if conv_param.act_block_h is not None else 0,
        )
    elif conv_param.shard_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        return WidthShardedStrategyConfiguration(
            reshard_if_not_optimal=conv_param.reshard_if_not_optimal,
        )
    elif conv_param.shard_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        return BlockShardedStrategyConfiguration(
            reshard_if_not_optimal=conv_param.reshard_if_not_optimal,
        )
    else:
        return AutoShardedStrategyConfiguration()


def create_conv2d_from_params(device, params, conv_pth, activation=""):
    weight = ttnn.from_device(conv_pth.weight)
    bias = ttnn.from_device(conv_pth.bias) if conv_pth.bias else None

    sharding_strategy = to_sharding_strategy(params)
    config = Conv2dConfiguration(
        input_height=params.input_height,
        input_width=params.input_width,
        in_channels=params.in_channels,
        out_channels=params.out_channels,
        batch_size=params.batch_size,
        kernel_size=params.kernel_size,
        stride=params.stride,
        padding=params.padding,
        groups=params.groups,
        dilation=params.dilation,
        activation=activation,
        weights_dtype=ttnn.bfloat8_b,
        output_dtype=params.dtype,
        weight=weight,
        bias=bias,
        sharding_strategy=sharding_strategy,
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
        enable_act_double_buffer=True,
        enable_split_reader=True if params.shard_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED else False,
        deallocate_activation=params.deallocate_activation,
        reallocate_halo_output=True,
    )

    device_descriptor = DeviceDescriptor(device, grid_size=(8, 8))
    conv2d = TtConv2d(config, device_descriptor)
    return conv2d
