# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
from loguru import logger
import ttnn
import pytest
import torch
import math
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)


def rms_norm(x, dim, gamma, beta, eps):
    return x * torch.rsqrt(x.pow(2).mean([-i for i in range(1, len(dim) + 1)], keepdim=True) + eps) * gamma + beta


PREFETCHER_NOC1_GRID = [
    (6, 6),
    (6, 7),
    (6, 9),
    (6, 0),
    (6, 1),
    (6, 2),
    (6, 4),
    (6, 5),
    (5, 5),
    (5, 6),
    (5, 7),
    (5, 9),
    (5, 0),
    (5, 1),
    (5, 2),
    (5, 4),
    (1, 4),
    (1, 5),
    (1, 9),
    (1, 0),
    (2, 0),
    (2, 4),
    (2, 5),
    (2, 9),
]


@pytest.mark.parametrize(
    "num_devices_fractured, input_dim, input_core_grid, output_core_grid",
    [
        (4, 8192, ttnn.CoreGrid(x=2, y=8), PREFETCHER_NOC1_GRID),  # TG llama use case; 4 tiles per core input
        (4, 8192, ttnn.CoreGrid(x=2, y=8), None),
    ],
)
@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4)],
    indirect=True,
)
def test_layernorm_perf(mesh_device, num_devices_fractured, input_dim, input_core_grid, output_core_grid):
    torch.manual_seed(1234)

    num_cores = input_core_grid.num_cores
    dim = int(
        math.ceil(input_dim / num_devices_fractured / num_cores / 32) * num_devices_fractured * num_cores * 32
    )  # padded
    input_shape = (1, 1, 32, dim)
    if isinstance(input_core_grid, ttnn.CoreGrid):
        input_core_range_set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(input_core_grid.x - 1, input_core_grid.y - 1)),
            ]
        )
    else:
        input_core_range_set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(x, y),
                    ttnn.CoreCoord(x, y),
                )
                for x, y in input_core_grid
            ]
        )
    size_per_device = dim // num_devices_fractured
    # Input memory config
    input_memory_config = ttnn.create_sharded_memory_config(
        shape=(
            input_shape[0] * input_shape[1] * input_shape[2],
            input_shape[3] // num_devices_fractured // input_core_range_set.num_cores(),
        ),
        core_grid=input_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    # Create input tensor with input memory config
    input_tensor_torch = torch.randn(input_shape)
    gamma_torch = torch.randn((1, 1, 1, input_shape[3]))
    input_tensor = ttnn.as_tensor(
        input_tensor_torch,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device=mesh_device, dims=(None, 3), mesh_shape=list(mesh_device.shape)),
        layout=ttnn.TILE_LAYOUT,
        memory_config=input_memory_config,
    )
    gamma_tensor = ttnn.as_tensor(
        gamma_torch.reshape([1, 1, dim // 32, 32]),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 2), mesh_shape=list(mesh_device.shape)),
    )
    ln_prg_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(input_core_grid.x, input_core_grid.y),
        subblock_w=1,
        block_h=1,
        block_w=(size_per_device // num_cores) // 32,
        inplace=False,
    )
    ln_sharded_stats_memcfg = ttnn.create_sharded_memory_config(
        shape=[1, 1, 32, 32 * num_devices_fractured],
        core_grid=ttnn.CoreGrid(y=1, x=1),
        strategy=ttnn.ShardStrategy.WIDTH,
    )
    # Run distributed rmsnorm part 1
    tt_stats = ttnn.rms_norm_pre_all_gather(input_tensor, program_config=ln_prg_cfg)

    # All gather stats
    tt_stats = ttnn.all_gather(
        tt_stats,
        3,
        num_links=1,
        cluster_axis=1,
        mesh_device=mesh_device,
        memory_config=ln_sharded_stats_memcfg,
        topology=ttnn.Topology.Linear,
    )

    # Output memory config
    if output_core_grid is None:
        output_core_grid = input_core_grid

    if isinstance(output_core_grid, ttnn.CoreGrid):
        output_core_range_set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(output_core_grid.x - 1, output_core_grid.y - 1)),
            ]
        )
    else:
        output_core_range_set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(x, y),
                    ttnn.CoreCoord(x, y),
                )
                for x, y in output_core_grid
            ]
        )
    padded_out_w = math.ceil(input_shape[3] / num_devices_fractured / output_core_range_set.num_cores() / 32) * 32
    output_memory_config = ttnn.create_sharded_memory_config(
        shape=(
            input_shape[0] * input_shape[1] * input_shape[2],
            padded_out_w,
        ),
        core_grid=output_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Run distributed rmsnorm part 2
    tt_out = ttnn.rms_norm_post_all_gather(
        input_tensor,
        epsilon=1e-05,
        weight=gamma_tensor,
        program_config=ln_prg_cfg,
        stats=tt_stats,
        memory_config=output_memory_config,
        dtype=ttnn.bfloat8_b,
    )

    tt_stats.deallocate(True)
    tt_out_torch = ttnn.to_torch(
        tt_out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 3), mesh_shape=(8, 4))
    )[0].unsqueeze(0)

    ref_lnorm = rms_norm(input_tensor_torch, [3], gamma_torch, torch.zeros_like(gamma_torch), 1e-5)
    passing, output = comp_pcc(tt_out_torch, ref_lnorm, 0.999)
    logger.info(output)

    assert passing
