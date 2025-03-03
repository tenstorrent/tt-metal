# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand_inf

Y, X = (8, 8)


def run_tests(
    input_shape,
    dtype,
    dlayout,
    tensor_memory_layout,
    byffer_type,
    shard_grid,
    shard_shape,
    shard_orientation,
    torch_op,
    ttnn_op,
    gen_infs,
    device,
):
    random.seed(0)
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    if gen_infs:
        torch_input_tensor_a = gen_rand_inf(input_shape, low=-100, high=100)
    else:
        torch_input_tensor_a = torch.Tensor(size=input_shape).uniform_(-50, 50).to(torch.bfloat16)

    torch_output_tensor = torch_input_tensor_a

    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation)
    sharded_config = ttnn.MemoryConfig(tensor_memory_layout, byffer_type, shard_spec)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=dtype,
        layout=dlayout,
        device=device,
        memory_config=sharded_config,
    )

    output_tensor = input_tensor_a
    output_tensor = ttnn.to_torch(output_tensor)

    [passed, message] = check_with_pcc(torch_output_tensor, output_tensor, 0.999)
    assert passed, f"PCC={message}"


test_sweep_args = [
    (
        (256, 2, 5, 1536),  # Tensor shape
        ttnn.bfloat16,  # Tensor dtype
        ttnn.TILE_LAYOUT,  # Tensor layout
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),  # core grid
        [320, 192],  # shard shape
        ttnn.ShardOrientation.COL_MAJOR,
    ),
    (
        (256, 2, 5, 1536),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),  # core grid
        [320, 192],
        ttnn.ShardOrientation.ROW_MAJOR,
    ),
    (
        (256, 2, 5, 1536),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),  # core grid
        [320, 192],
        ttnn.ShardOrientation.COL_MAJOR,
    ),
    (
        (1, 256, 2, 2304),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),  # core grid
        [64, 288],
        ttnn.ShardOrientation.COL_MAJOR,
    ),
    (
        (1, 256, 2, 2304),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),  # core grid
        [64, 288],
        ttnn.ShardOrientation.ROW_MAJOR,
    ),
    (
        (1, 256, 2, 2304),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),  # core grid
        [64, 288],
        ttnn.ShardOrientation.COL_MAJOR,
    ),
    (
        (32, 4, 8, 768),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),  # core grid
        [128, 96],
        ttnn.ShardOrientation.COL_MAJOR,
    ),
    (
        (32, 4, 8, 768),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),  # core grid
        [128, 96],
        ttnn.ShardOrientation.ROW_MAJOR,
    ),
    (
        (32, 4, 8, 768),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),  # core grid
        [128, 96],
        ttnn.ShardOrientation.COL_MAJOR,
    ),
    (
        (1, 25, 160, 32),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),  # core grid
        [32, 160],
        ttnn.ShardOrientation.COL_MAJOR,
    ),
    (
        (1, 25, 160, 32),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),  # core grid
        [32, 160],
        ttnn.ShardOrientation.COL_MAJOR,
    ),
    (
        (1, 2, 1248, 32),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),  # core grid
        [32, 1248],
        ttnn.ShardOrientation.COL_MAJOR,
    ),
    (
        (1, 2, 1248, 32),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),  # core grid
        [32, 1248],
        ttnn.ShardOrientation.COL_MAJOR,
    ),
    (
        (1, 2, 1472, 32),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),  # core grid
        [32, 1472],
        ttnn.ShardOrientation.COL_MAJOR,
    ),
    (
        (1, 2, 1472, 32),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),  # core grid
        [32, 1472],
        ttnn.ShardOrientation.COL_MAJOR,
    ),
    (
        (2, 1, 224, 128),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),  # core grid
        [128, 224],
        ttnn.ShardOrientation.COL_MAJOR,
    ),
]


def nop(x, memory_config=None):
    return x


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, tensor_memory_layout, byffer_type, shard_grid, shard_shape, shard_orientation",
    (test_sweep_args),
)
def test_eltwise_nop(
    input_shape,
    dtype,
    dlayout,
    tensor_memory_layout,
    byffer_type,
    shard_grid,
    shard_shape,
    shard_orientation,
    device,
):
    run_tests(
        input_shape,
        dtype,
        dlayout,
        tensor_memory_layout,
        byffer_type,
        shard_grid,
        shard_shape,
        shard_orientation,
        nop,
        nop,
        False,
        device,
    )
