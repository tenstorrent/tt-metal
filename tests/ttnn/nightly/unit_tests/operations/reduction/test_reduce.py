# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import ttnn
from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_equal,
    comp_pcc,
)


@pytest.mark.parametrize("N", [8, 16])
@pytest.mark.parametrize("in_sharded", [True], ids=["in0_sharded"])
@pytest.mark.parametrize("out_sharded", [True], ids=["out_sharded"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_sharded_reduce_h(N, in_sharded, out_sharded, dtype, device, function_level_defaults):
    grid_size = (8, 4)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    C = 1
    H = 64
    W = 2048

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    out_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    x = torch.randn((N, C, H, W)).bfloat16()

    xt = ttnn.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttnn.TILE_LAYOUT,
    ).to(
        device,
        interleaved_mem_config,
    )

    if in_sharded:
        xt = ttnn.interleaved_to_sharded(
            xt,
            grid_size,
            [N * C * H, W // (grid_size[0] * grid_size[1])],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.ShardOrientation.COL_MAJOR,
        )

    yt = ttnn.max(xt, 2, memory_config=out_mem_config)

    if out_sharded:
        yt = ttnn.sharded_to_interleaved(
            yt,
            interleaved_mem_config,
        )

    tt_got_back = yt.cpu().to_torch()

    y = torch.amax(x, 2)

    if dtype == ttnn.bfloat16:
        passing, output = comp_equal(y, tt_got_back)
    else:
        passing, output = comp_pcc(y, tt_got_back, 0.999)
    logger.info(output)

    assert passing
