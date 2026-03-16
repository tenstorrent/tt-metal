# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import ttnn


def adjust_shapes_for_testing(config, mesh_device):
    """Scale input dimensions for smaller meshes to preserve per-device shapes."""
    n_sp_devices, n_tp_devices = mesh_device.shape
    if n_sp_devices != 32:
        config.max_seq_len = config.max_seq_len // (32 // n_sp_devices)
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


def calculate_average_recall(predicted_experts, reference_experts):
    recall = 0
    for i in range(predicted_experts.shape[0]):
        pred_set = set(e.item() for e in predicted_experts[i])
        ref_set = set(e.item() for e in reference_experts[i])
        recall += len(pred_set & ref_set) / len(ref_set) if ref_set else 0
    return recall / predicted_experts.shape[0]
