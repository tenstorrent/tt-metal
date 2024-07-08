# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
import ttnn.experimental.tensor as tt_tensor
from tests.ttnn.utils_for_testing import check_with_pcc


def fold_torch(input_tensor, stride_h, stride_w):
    N, H, W, C = input_tensor.shape

    reshaped = input_tensor.reshape(N, H // stride_h, stride_h, W // stride_w, stride_w, C)
    transposed = reshaped.permute(0, 1, 3, 2, 4, 5)
    return transposed.reshape(N, H // stride_h, W // stride_w, C * stride_h * stride_w)


parameters = {
    "N": [1, 8],
    "H": [2, 4, 6, 8],
    "W": [2, 4, 6, 8],
    "C": [8, 16, 32],
    "shard_strategy": [
        None,
        ttnn.ShardStrategy.HEIGHT,
        ttnn.ShardStrategy.WIDTH,
    ],
    "stride_h": [1, 2, 3, 4],
    "stride_w": [1, 2, 3, 4],
}


def skip(N, H, W, C, shard_strategy, stride_h, stride_w):
    if shard_strategy == ttnn.ShardStrategy.WIDTH:
        return True, "Unsupported shard strategy"

    if H % stride_h != 0 or W % stride_w != 0:
        return True, "Invalid stride"

    if shard_strategy:
        memory_config = ttnn.create_sharded_memory_config(
            shape=(N, H, W, C),
            core_grid=ttnn.CoreGrid(y=2, x=1),
            strategy=shard_strategy,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

        shard_shape = memory_config.shard_spec.shape
        if shard_shape[0] % (W * stride_h * stride_w) != 0:
            return True, "Invalid shard shape"

    return False, ""


def run(N, H, W, C, shard_strategy, stride_h, stride_w, *, device):
    shape = (N, H, W, C)
    torch_input_tensor = torch.randn(shape, dtype=torch.float32)
    torch_output = fold_torch(torch_input_tensor, stride_h, stride_w)
    torch_output = torch_output.reshape(1, 1, -1, C * stride_h * stride_w)

    if shard_strategy:
        memory_config = ttnn.create_sharded_memory_config(
            shape=shape,
            core_grid=ttnn.CoreGrid(y=2, x=1),
            strategy=shard_strategy,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
    else:
        memory_config = ttnn.DRAM_MEMORY_CONFIG

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=memory_config,
    )

    output = tt_tensor.fold(input_tensor, stride_h, stride_w)
    output = ttnn.to_torch(output)

    return check_with_pcc(torch_output, output, 0.999)
