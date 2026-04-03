# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import ttnn


def adjust_shapes_for_testing(config, mesh_device):
    """Scale TP dimension for smaller meshes. sp_dim (per-device seq len) is always correct."""
    _, n_tp_devices = mesh_device.shape
    if n_tp_devices != 4:
        config.dim = config.dim // (4 // n_tp_devices)


def get_input_mem_config(config, mesh_shape):
    shard_height = (config.sp_dim + config.num_cores - 1) // config.num_cores
    shard_height = ((shard_height + 31) // 32) * 32
    shard_width = (config.dim + mesh_shape[1] - 1) // mesh_shape[1]
    return ttnn.create_sharded_memory_config(
        shape=(shard_height, shard_width),
        core_grid=config.core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
