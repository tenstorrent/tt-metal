# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.utility_functions import (
    is_wormhole_b0,
    is_grayskull,
)


def resnet_basic_block(x, *, parameters):
    identity = x

    # Relu and bn1 are fused with conv1
    conv1 = parameters.conv1(x)

    # Relu and bn2 are fused with conv1
    conv2 = parameters.conv2(conv1)
    ttnn.deallocate(conv1)

    if "downsample" in parameters and parameters.downsample is not None:
        identity = parameters.downsample(x)
        ttnn.deallocate(x)

    identity = ttnn.reshape(identity, conv2.shape)
    out = ttnn.add_and_apply_activation(conv2, identity, activation="relu", memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(conv2)
    if x is not identity:
        ttnn.deallocate(identity)

    return out


def create_sharded_mem_config(x, is_1d, core_grid, strategy, orientation, halo, use_height_and_width_as_shard_shape):
    mem_layout = ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED if is_1d else ttnn.types.TensorMemoryLayout.BLOCK_SHARDED
    core_grid = ttnn.CoreGrid(x=12, y=9)
    shard_grid = ttnn.experimental.tensor.CoreRangeSet(
        {
            ttnn.experimental.tensor.CoreRange(
                ttnn.experimental.tensor.CoreCoord(0, 0),
                ttnn.experimental.tensor.CoreCoord(core_grid.x - 1, core_grid.y - 1),
            )
        }
    )
    num_cores_nhw = core_grid.x * core_grid.y
    if is_1d:
        shard_shape = x.shape[0] * x.shape[1] * x.shape[2] // num_cores_nhw
    else:
        shard_shape = x.shape[1] * x.shape[1] * x.shape[2] // core_grid.y, x.shape[3] // core_grid.x
    shard_spec = ttnn.experimental.tensor.ShardSpec(
        shard_grid, shard_shape, ttnn.experimental.tensor.ShardOrientation.COL_MAJOR, False
    )
    return ttnn.types.MemoryConfig(mem_layout, ttnn.types.BufferType.L1, shard_spec)


def do_reshard(output_tensor, input_mem_config):
    if ttnn.get_memory_config(output_tensor) != input_mem_config:
        output_tensor = ttnn.to_memory_config(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_memory_config(output_tensor, input_mem_config)
    return output_tensor


def resnet_bottleneck_block(x, parameters, layer=None, module=None, device=None):
    conv1 = parameters.conv1(x)
    conv1 = do_reshard(conv1, parameters.conv2.conv.input_sharded_memory_config)

    identity = x

    conv2 = parameters.conv2(conv1)
    if conv1.is_allocated():
        ttnn.deallocate(conv1)

    conv3 = parameters.conv3(conv2)
    ttnn.deallocate(conv2)

    conv3_mem_config = ttnn.get_memory_config(conv3)
    # if layer is not None and layer >= 3:
    #     conv3 = ttnn.to_memory_config(conv3, ttnn.DRAM_MEMORY_CONFIG)

    if "downsample" in parameters and parameters.downsample is not None:
        identity = do_reshard(identity, parameters.downsample.conv.input_sharded_memory_config)
        if layer is not None and layer == 2:
            if x.is_allocated() and x is not identity:
                ttnn.deallocate(x)
            if module >= 2:
                identity = ttnn.experimental.tensor.move_sharded(identity)
        identity = parameters.downsample(identity)

    if layer is not None and layer >= 3:
        conv3 = ttnn.to_memory_config(conv3, conv3_mem_config)
    conv3 = ttnn.reshape(conv3, identity.shape)
    mem_config = ttnn.get_memory_config(conv3)
    # if layer == 4 and module == 3:
    #     mem_config = ttnn.L1_MEMORY_CONFIG
    out = ttnn.add_and_apply_activation(conv3, identity, activation="relu", memory_config=mem_config)
    ttnn.deallocate(conv3)

    if x is not identity:
        ttnn.deallocate(identity)

    if (layer is not None and module is not None) and (
        (layer == 1 and module == 1)
        or (layer == 1 and module == 2 and is_grayskull())
        or (layer == 1 and module == 3 and is_grayskull())
    ):
        out = ttnn.experimental.tensor.move_sharded(out)

    return out
