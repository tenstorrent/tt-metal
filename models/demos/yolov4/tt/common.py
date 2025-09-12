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
    weight, bias = ttnn.from_device(conv_pth.weight), ttnn.from_device(conv_pth.bias) if conv_pth.bias else None
    sharding_strategy = to_sharding_strategy(params)
    config = Conv2dConfiguration.from_model_args(
        conv2d_args=params,
        weights=weight,
        bias=bias,
        sharding_strategy=sharding_strategy,
        activation=activation,
        weights_dtype=ttnn.bfloat8_b,
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
        enable_act_double_buffer=True,
        enable_split_reader=True if params.shard_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED else False,
        deallocate_activation=params.deallocate_activation,
        reallocate_halo_output=True,
    )
    device_descriptor = DeviceDescriptor(device, grid_size=(8, 8))
    return TtConv2d(config, device_descriptor)
