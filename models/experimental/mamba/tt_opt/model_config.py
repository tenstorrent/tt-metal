# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def create_model_config(batch_size, hidden_size):
    configs = {}
    row = 5
    col = 8
    latent = 16
    configs["sharded_d"] = ttnn.create_sharded_memory_config(
        shape=(1, 1, batch_size, hidden_size * 2),
        core_grid=ttnn.CoreGrid(y=row, x=col),
        strategy=ttnn.ShardStrategy.WIDTH,
    )
    configs["sharded_dn"] = ttnn.create_sharded_memory_config(
        shape=(1, 1, batch_size, hidden_size * 2 * latent),
        core_grid=ttnn.CoreGrid(y=row, x=col),
        strategy=ttnn.ShardStrategy.WIDTH,
    )
    return configs


def get_model_config(configs, model_config_str):
    return configs[model_config_str]
