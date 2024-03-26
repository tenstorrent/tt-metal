# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import ttnn


def create_model_config(num_users, hidden_size):
    configs = {}

    row = 4
    col = 8
    latent = 32
    orientation = ttnn.ShardOrientation.ROW_MAJOR

    # num_users, hidden_size*2
    configs["sharded"] = ttnn.L1_MEMORY_CONFIG
    '''
    ttnn.create_sharded_memory_config(
        shape=(1, 1, num_users, hidden_size*2 // (row * col)),
        core_grid=ttnn.CoreGrid(y=row, x=col),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=orientation,
        use_height_and_width_as_shard_shape=True,
    )
    '''

    configs["sharded_large"] = ttnn.L1_MEMORY_CONFIG
    '''
    ttnn.create_sharded_memory_config(
        shape=(1, 1, num_users, hidden_size*2 * latent // (row * col)),
        core_grid=ttnn.CoreGrid(y=row, x=col),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=orientation,
        use_height_and_width_as_shard_shape=True,
    )
    '''
    configs["sharded_rank"] = ttnn.L1_MEMORY_CONFIG
    '''
    ttnn.create_sharded_memory_config(
        shape=(1, 1, hidden_size*2 // 32, hidden_size*2 // (row * col)),
        core_grid=ttnn.CoreGrid(y=row, x=col),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=orientation,
        use_height_and_width_as_shard_shape=True,
    )
    '''
    return configs


def get_model_config(configs, model_config_str):
    return configs[model_config_str]
