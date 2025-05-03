# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import os

from models.demos.wormhole.mamba.reference.args import ModelMode

MAMBA_MAX_SEQUENCE_LEN = 128


def create_model_config(batch_size, hidden_size, mode=ModelMode.DECODE, seq_len=1):
    configs = {}
    latent = 32
    configs["max_seq_length"] = MAMBA_MAX_SEQUENCE_LEN
    configs["core_grid_row"] = 5
    configs["core_grid_col"] = 8
    configs["latent_size"] = latent
    configs["mode"] = mode
    configs["seq_len"] = seq_len
    configs["batch_size"] = batch_size
    configs["num_users"] = 32  # fixing the number of users to 32 throughout the model
    configs["current_user"] = 0  # fixing the number of users to 32 throughout the model

    if mode == ModelMode.DECODE:
        outer_dim = batch_size
        assert batch_size == 32, "Batch size must be 32 for decode model"
    elif mode == ModelMode.PREFILL:
        outer_dim = seq_len
        assert batch_size == 1, "Batch size must be 1 for prefill model"
        assert seq_len % 32 == 0, "Sequence length must be a multiple of 32 for prefill model"
    else:
        raise ValueError(f"Invalid model mode: {mode}")

    configs["outer_dim"] = outer_dim

    configs["sharded_h"] = ttnn.create_sharded_memory_config(
        shape=(1, 1, outer_dim, hidden_size),
        core_grid=ttnn.CoreGrid(y=get_nearest_core_grid_row(hidden_size), x=configs["core_grid_col"]),
        strategy=ttnn.ShardStrategy.WIDTH,
    )
    configs["sharded_d"] = ttnn.create_sharded_memory_config(
        shape=(1, 1, outer_dim, hidden_size * 2),
        core_grid=ttnn.CoreGrid(y=get_nearest_core_grid_row(hidden_size * 2), x=configs["core_grid_col"]),
        strategy=ttnn.ShardStrategy.WIDTH,
    )
    configs["sharded_dn"] = ttnn.create_sharded_memory_config(
        shape=(1, 1, outer_dim, hidden_size * 2 * latent),
        core_grid=ttnn.CoreGrid(y=configs["core_grid_row"], x=configs["core_grid_col"]),
        strategy=ttnn.ShardStrategy.WIDTH,
    )
    configs["sharded_scan"] = ttnn.create_sharded_memory_config(
        shape=(1, 1, outer_dim, hidden_size * 2 * latent),
        core_grid=ttnn.CoreGrid(y=8, x=8),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    configs["sharded_prev_hidden"] = ttnn.create_sharded_memory_config(
        shape=(1, 1, 1, hidden_size * 2 * latent),
        core_grid=ttnn.CoreGrid(y=8, x=8),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    configs["SHARDED_NORM_PRGM_CFG"] = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[configs["core_grid_col"], get_nearest_core_grid_row(hidden_size)],
        subblock_w=(hidden_size // (configs["core_grid_col"] * get_nearest_core_grid_row(hidden_size))) // 32,
        block_h=outer_dim // 32,
        block_w=(hidden_size // (configs["core_grid_col"] * get_nearest_core_grid_row(hidden_size))) // 32,
        inplace=False,
    )
    configs["dtype"] = {"activations": ttnn.bfloat8_b, "weights": ttnn.bfloat4_b}
    return configs


def get_model_config(configs, model_config_str):
    return configs[model_config_str]


def get_weights_cache_path(model_version, cache_dir="/tmp"):
    cache_path = os.path.join(cache_dir, model_version)
    return cache_path


def get_nearest_core_grid_row(tensor_width, core_grid_row=8, core_grid_col=8):
    core_grid_size = core_grid_row * core_grid_col
    tile_width = 32
    num_tiles = tensor_width // tile_width
    if num_tiles <= core_grid_size:
        return num_tiles // core_grid_col
    else:
        for i in range(core_grid_size, 0, -1):
            if num_tiles % i == 0:
                return i // core_grid_col
