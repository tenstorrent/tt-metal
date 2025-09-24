# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.yolov11.common import YOLOV11_L1_SMALL_SIZE

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


def p(x, a="x"):
    print(f"{a}'s  shape: {x.shape,x.padded_shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV11_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "use_in_place, numeric_stable",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_softmax_interleaved(device, use_in_place, numeric_stable):
    torch_input = torch.randn((1, 8400, 64), dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    ttnn_input = ttnn.reshape(ttnn_input, (ttnn_input.shape[0], ttnn_input.shape[1], 4, 16))
    p(ttnn_input, "input")
    if use_signpost:
        signpost(header=f"use_in_place:{use_in_place}, numeric_stable:{numeric_stable} ")
    if use_in_place:
        ttnn_input = ttnn.softmax_in_place(ttnn_input, dim=-1, numeric_stable=numeric_stable)
    else:
        ttnn_input = ttnn.softmax(ttnn_input, dim=-1, numeric_stable=numeric_stable)
    p(ttnn_input, "outttt")
    ttnn_input_torch = ttnn.to_torch(ttnn_input)


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV11_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "use_in_place, numeric_stable",
    [
        (True, True),
        (True, False),
        (False, True),  # --> buffer clash due to sharded format if use_in_place=False
        (False, False),  # --> buffer clash due to sharded format if use_in_place=False
    ],
)
def test_softmax_sharded(device, use_in_place, numeric_stable):
    torch_input = torch.randn((1, 8400, 64), dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    ttnn_input = ttnn.reshape(ttnn_input, (ttnn_input.shape[0], ttnn_input.shape[1], 4, 16))
    p(ttnn_input, "input")
    core_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(7, 7),
            )
        }
    )
    input_sharded_memory_config = ttnn.create_sharded_memory_config(
        (8416, 32),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    ttnn_input = ttnn.to_memory_config(ttnn_input, input_sharded_memory_config)
    p(ttnn_input, "sharded_input")
    program_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        subblock_w=1,  # or appropriate value for your tensor
        block_h=263,  # height blocks
        block_w=1,  # width blocks (16/32 = 0.5, so use 1)
    )
    if use_signpost:
        signpost(header=f"use_in_place:{use_in_place}, numeric_stable:{numeric_stable} ")
    if use_in_place:
        ttnn_input = ttnn.softmax_in_place(
            ttnn_input, dim=-1, numeric_stable=numeric_stable
        )  # ,program_config=program_config)
    else:
        ttnn_input = ttnn.softmax(ttnn_input, dim=-1, numeric_stable=numeric_stable)  # ,program_config=program_config)
    p(ttnn_input, "outttt")
    ttnn_input_torch = ttnn.to_torch(ttnn_input)
