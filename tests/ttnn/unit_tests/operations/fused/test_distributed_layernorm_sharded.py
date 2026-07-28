# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Single-device tests for the sharded distributed layer_norm / rms_norm op path.

A "distributed" norm is a three-stage, multi-device flow: each device reduces partial statistics over
its slice of the width (*_pre_all_gather), an all-gather exchanges those stats so every device holds the
global set, then each device normalizes (*_post_all_gather). These tests "simulate" that flow on a
single device: the width slices that would live on separate devices are sharded across cores (or
processed as chunks) on one device, and the all-gather is stood in for by resharding/concatenating the
per-slice statistics onto a single core. No real multi-device execution or collective runs — only the
on-device pre/post all-gather op path is exercised.
"""
import ttnn
import torch
import pytest
import math
from loguru import logger

from models.common.utility_functions import (
    skip_for_wormhole_b0,
)
from tests.ttnn.utils_for_testing import assert_numeric_metrics
from tests.ttnn.unit_tests.operations.fused.sharded_test_utils import (
    make_sharded_norm_mem_config,
    to_poisoned_sharded,
)

from models.common.utility_functions import tt2torch_tensor

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


def rms_norm(x, gamma, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gamma


def layer_norm(x, gamma, eps):
    return (x - x.mean(-1, keepdim=True)) * torch.rsqrt(x.var(-1, keepdim=True) + eps) * gamma


def create_input_and_weight_tensors(input_width, num_devices, seed, mean, std):
    torch.manual_seed(seed)
    input_shape = (1, 1, 32, input_width * num_devices)
    weights_shape = (1, 1, 1, input_width * num_devices)

    torch_input_tensor = torch.normal(mean, std, size=input_shape, dtype=torch.bfloat16)
    torch_weight = torch.normal(mean, std, size=weights_shape, dtype=torch.bfloat16)

    torch_input_chunks = torch.chunk(torch_input_tensor, num_devices, dim=-1)
    torch_weight_chunks = torch.chunk(torch_weight, num_devices, dim=-1)

    return torch_input_tensor, torch_weight, torch_input_chunks, torch_weight_chunks


def create_tt_tensors(
    torch_chunk, device, df, core_grid, input_width, is_weight=False, grid_offset=ttnn.CoreCoord(0, 0)
):
    tt_tensor = ttnn.from_torch(
        torch_chunk,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG if is_weight else ttnn.L1_MEMORY_CONFIG,
        dtype=df,
    )

    with device.cache_entries_counter.measure():
        if not is_weight:
            core_range = ttnn.CoreRange(
                grid_offset, ttnn.CoreCoord(core_grid[0] + grid_offset.x - 1, core_grid[1] + grid_offset.y - 1)
            )
            tt_sharded_config = ttnn.create_sharded_memory_config(
                shape=(32, input_width // (core_grid[0] * core_grid[1])),
                core_grid=ttnn.CoreRangeSet(
                    {
                        core_range,
                    }
                ),
                strategy=ttnn.ShardStrategy.WIDTH,
                use_height_and_width_as_shard_shape=True,
            )
            tt_tensor = ttnn.to_memory_config(tt_tensor, memory_config=tt_sharded_config)

        return tt_tensor


def compute_pre_allgather_stats(tt_input_tensor, core_grid, input_width, is_rmsnorm, residual_input_tensor=None):
    SHARDED_NORM_PRGM_CFG = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[core_grid[0], core_grid[1]],
        subblock_w=(input_width // (core_grid[0] * core_grid[1])) // 32,
        block_h=1,
        block_w=(input_width // (core_grid[0] * core_grid[1])) // 32,
        inplace=False,
    )

    if is_rmsnorm:
        return ttnn.rms_norm_pre_all_gather(
            tt_input_tensor, residual_input_tensor=residual_input_tensor, program_config=SHARDED_NORM_PRGM_CFG
        )
    else:
        return ttnn.layer_norm_pre_all_gather(
            tt_input_tensor, residual_input_tensor=residual_input_tensor, program_config=SHARDED_NORM_PRGM_CFG
        )


def compute_post_allgather_output(
    tt_input_tensor, tt_weights, tt_stats_tensor, eps, is_rmsnorm, core_grid, input_width, output_df, out_memory_config
):
    SHARDED_NORM_PRGM_CFG = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(core_grid[0], core_grid[1]),
        subblock_w=(input_width // (core_grid[0] * core_grid[1])) // 32,
        block_h=1,
        block_w=(input_width // (core_grid[0] * core_grid[1])) // 32,
        inplace=False,
    )

    if is_rmsnorm:
        return ttnn.rms_norm_post_all_gather(
            tt_input_tensor,
            epsilon=eps,
            weight=tt_weights,
            program_config=SHARDED_NORM_PRGM_CFG,
            stats=tt_stats_tensor,
            dtype=output_df,
            memory_config=out_memory_config,
        )
    else:
        return ttnn.layer_norm_post_all_gather(
            tt_input_tensor,
            epsilon=eps,
            weight=tt_weights,
            program_config=SHARDED_NORM_PRGM_CFG,
            stats=tt_stats_tensor,
            dtype=output_df,
            memory_config=out_memory_config,
        )


def compute_reference_output(torch_input_tensor, torch_weight, is_rmsnorm, eps):
    if is_rmsnorm:
        return rms_norm(torch_input_tensor, torch_weight, eps=eps)
    else:
        return torch.nn.functional.layer_norm(
            torch_input_tensor,
            (torch_input_tensor.shape[-1],),
            weight=torch_weight.squeeze(0).squeeze(0).squeeze(0),
            eps=eps,
        )


def create_output_memory_config(output_core_grid, input_shape):
    if isinstance(output_core_grid, tuple):
        output_core_range_set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(output_core_grid[0] - 1, output_core_grid[1] - 1)),
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
    padded_out_w = math.ceil(input_shape[3] / output_core_range_set.num_cores() / 32) * 32
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

    return output_memory_config


def run_pre_allgather_layernorm(
    device,
    input_width,
    num_devices,
    is_rmsnorm,
    input_df,
    seed,
    mean,
    std,
    core_grid,
    min_pcc_ex,
    max_atol_ex,
    min_pcc_ex2,
    max_atol_ex2,
    min_pcc_residual_add=0.9997,
    fuse_residual=False,
):
    torch_input_tensor, _, torch_input_chunks, _ = create_input_and_weight_tensors(
        input_width, num_devices, seed, mean, std
    )

    if fuse_residual:
        torch_residual_input_tensor, _, torch_residual_input_chunks, _ = create_input_and_weight_tensors(
            input_width, num_devices, seed + 100, mean, std
        )

    for d in range(num_devices):
        tt_input_tensor = create_tt_tensors(torch_input_chunks[d], device, input_df, core_grid, input_width)
        if fuse_residual:
            tt_residual_input_tensor = create_tt_tensors(
                torch_residual_input_chunks[d], device, input_df, core_grid, input_width
            )
            torch_input_chunks = list(torch_input_chunks)
            torch_input_chunks[d] = torch_input_chunks[d] + torch_residual_input_chunks[d]
        else:
            tt_residual_input_tensor = None

        with device.cache_entries_counter.measure():
            tt_pre_allgather_output = compute_pre_allgather_stats(
                tt_input_tensor, core_grid, input_width, is_rmsnorm, tt_residual_input_tensor
            )
            tt_pre_allgather_torch = ttnn.to_torch(tt_pre_allgather_output).to(torch.bfloat16)
            if fuse_residual:
                tt_residual_add_output = ttnn.to_torch(tt_input_tensor).to(torch.bfloat16)
                assert_numeric_metrics(
                    torch_input_chunks[d],
                    tt_residual_add_output,
                    pcc_threshold=min_pcc_residual_add,
                    rtol=0.05,
                    atol=0.063,
                    frobenius_threshold=0.05,
                )

            if is_rmsnorm:
                tt_ex2 = tt_pre_allgather_torch[..., :1]
                torch_ex2 = torch.mean(torch_input_chunks[d] ** 2, dim=-1, keepdim=True)
                assert_numeric_metrics(
                    torch_ex2,
                    tt_ex2,
                    pcc_threshold=min_pcc_ex2,
                    rtol=0.05,
                    atol=max_atol_ex2,
                    frobenius_threshold=0.15,
                )
            else:
                tt_ex = tt_pre_allgather_torch[..., :1]
                tt_ex2 = tt_pre_allgather_torch[..., 32:33]
                torch_ex = torch.mean(torch_input_chunks[d], dim=-1, keepdim=True)
                torch_ex2 = torch.mean(torch_input_chunks[d] ** 2, dim=-1, keepdim=True)
                assert_numeric_metrics(
                    torch_ex,
                    tt_ex,
                    pcc_threshold=min_pcc_ex,
                    rtol=0.05,
                    atol=max_atol_ex,
                    frobenius_threshold=0.15,
                )
                assert_numeric_metrics(
                    torch_ex2,
                    tt_ex2,
                    pcc_threshold=min_pcc_ex2,
                    rtol=0.05,
                    atol=max_atol_ex2,
                    frobenius_threshold=0.15,
                )

    assert device.cache_entries_counter.total == 2, "Program cache not working as expected"
    logger.info("Pre-allgather layernorm test passed for all devices")


@pytest.mark.parametrize("is_rmsnorm", [True, False])
@pytest.mark.parametrize("seed", [0, 1234])
@pytest.mark.parametrize("input_width", [2048])
@pytest.mark.parametrize("num_devices", [4, 8])
@pytest.mark.parametrize("input_df", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize(("mean", "std"), ([0, 1],))
@pytest.mark.parametrize("core_grid", ((8, 4),))
@pytest.mark.parametrize(("min_pcc_ex", "max_atol_ex"), [(0.9997, 0.01)])
@pytest.mark.parametrize("min_pcc_residual_add", [0.997])
@pytest.mark.parametrize(
    "min_pcc_ex2",
    [
        0.982,
    ],
)
@pytest.mark.parametrize(("fuse_residual", "max_atol_ex2"), [(False, 0.04), (True, 0.09)])
def test_pre_allgather_layernorm(
    device,
    input_width,
    num_devices,
    is_rmsnorm,
    input_df,
    seed,
    mean,
    std,
    core_grid,
    min_pcc_ex,
    max_atol_ex,
    min_pcc_ex2,
    max_atol_ex2,
    min_pcc_residual_add,
    fuse_residual,
):
    run_pre_allgather_layernorm(
        device,
        input_width,
        num_devices,
        is_rmsnorm,
        input_df,
        seed,
        mean,
        std,
        core_grid,
        min_pcc_ex,
        max_atol_ex,
        min_pcc_ex2,
        max_atol_ex2,
        min_pcc_residual_add,
        fuse_residual,
    )


@pytest.mark.parametrize("is_rmsnorm", [True, False])
@pytest.mark.parametrize("seed", [0, 1234])
@pytest.mark.parametrize("input_width", [1024])
@pytest.mark.parametrize("num_devices", [4, 8])
@pytest.mark.parametrize("input_df", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize(("mean", "std"), ([0, 1],))
@pytest.mark.parametrize("core_grid", ((1, 4),))
@pytest.mark.parametrize(("min_pcc_ex", "max_atol_ex"), [(0.9997, 0.01)])
@pytest.mark.parametrize(("min_pcc_ex2", "max_atol_ex2"), [(0.986, 0.04)])
def test_pre_allgather_layernorm_1d_reduce(
    device,
    input_width,
    num_devices,
    is_rmsnorm,
    input_df,
    seed,
    mean,
    std,
    core_grid,
    min_pcc_ex,
    max_atol_ex,
    min_pcc_ex2,
    max_atol_ex2,
):
    run_pre_allgather_layernorm(
        device,
        input_width,
        num_devices,
        is_rmsnorm,
        input_df,
        seed,
        mean,
        std,
        core_grid,
        min_pcc_ex,
        max_atol_ex,
        min_pcc_ex2,
        max_atol_ex2,
    )


@pytest.mark.parametrize("is_rmsnorm", [True, False])
@pytest.mark.parametrize("seed", [0, 1234])
@pytest.mark.parametrize("eps", [1e-6])
@pytest.mark.parametrize(("min_pcc", "max_atol"), ((0.9997, 0.45),))
@pytest.mark.parametrize("input_width", [2048])
@pytest.mark.parametrize("num_devices", [4, 8])
@pytest.mark.parametrize("input_df", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("output_df", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("weights_df", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize(("mean", "std"), ([0, 1],))
@pytest.mark.parametrize("core_grid", ((8, 2),))
def test_post_allgather_layernorm(
    device,
    input_width,
    num_devices,
    is_rmsnorm,
    input_df,
    output_df,
    weights_df,
    seed,
    eps,
    mean,
    std,
    min_pcc,
    max_atol,
    core_grid,
):
    torch_input_tensor, torch_weight, torch_input_chunks, torch_weight_chunks = create_input_and_weight_tensors(
        input_width, num_devices, seed, mean, std
    )

    torch_output_tensor = compute_reference_output(torch_input_tensor, torch_weight, is_rmsnorm, eps)
    torch_output_chunks = torch.chunk(torch_output_tensor, num_devices, dim=-1)

    # Compute distributed statistics
    device_stats_list = []
    for d in range(num_devices):
        if is_rmsnorm:
            local_ex2 = torch.mean(torch_input_chunks[d] ** 2, dim=-1, keepdim=True)
            local_ex2 = torch.nn.functional.pad(local_ex2, (0, 31), "constant", 0)
            device_stats_list.append(local_ex2)

        else:
            local_ex = torch.mean(torch_input_chunks[d], dim=-1, keepdim=True)
            local_ex = torch.nn.functional.pad(local_ex, (0, 31), "constant", 0)
            local_ex2 = torch.mean(torch_input_chunks[d] ** 2, dim=-1, keepdim=True)
            local_ex2 = torch.nn.functional.pad(local_ex2, (0, 31), "constant", 0)
            device_stats_list.append(local_ex)
            device_stats_list.append(local_ex2)

    device_stats = torch.cat(device_stats_list, dim=-1)
    tt_device_stats = ttnn.from_torch(
        device_stats, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    # shard to 1 core
    tt_stats_sharded_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, 32, tt_device_stats.padded_shape[-1]),
        core_grid=ttnn.CoreGrid(y=1, x=1),
        strategy=ttnn.ShardStrategy.WIDTH,
    )
    tt_device_stats = ttnn.to_memory_config(tt_device_stats, memory_config=tt_stats_sharded_config)

    for d in range(num_devices):
        tt_input_tensor = create_tt_tensors(torch_input_chunks[d], device, input_df, core_grid, input_width)
        tt_weights = create_tt_tensors(
            torch_weight_chunks[d], device, weights_df, core_grid, input_width, is_weight=True
        )
        tt_output_tensor = compute_post_allgather_output(
            tt_input_tensor, tt_weights, tt_device_stats, eps, is_rmsnorm, core_grid, input_width, output_df, None
        )
        tt_output_torch = ttnn.to_torch(tt_output_tensor).to(torch.bfloat16)

        assert_numeric_metrics(
            torch_output_chunks[d],
            tt_output_torch,
            pcc_threshold=min_pcc,
            rtol=0.05,
            atol=max_atol,
            frobenius_threshold=0.15,
        )

    logger.info("Post-allgather layernorm test passed for all devices")


@pytest.mark.parametrize("is_rmsnorm", [True, False])
@pytest.mark.parametrize("seed", [0, 1234])
@pytest.mark.parametrize("eps", [1e-6])
@pytest.mark.parametrize(("min_pcc", "max_atol"), ((0.9997, 0.45),))
@pytest.mark.parametrize("input_width", [2048])
@pytest.mark.parametrize("num_devices", [1])
@pytest.mark.parametrize("input_df", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("weights_df", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize(("mean", "std"), ([0, 1],))
@pytest.mark.parametrize(
    "core_grid, grid_offset, output_core_grid",
    [
        ((8, 4), ttnn.CoreCoord(0, 0), (8, 4)),
        ((4, 4), ttnn.CoreCoord(1, 0), (4, 4)),
        ((8, 2), ttnn.CoreCoord(0, 0), (8, 3)),
        ((2, 4), ttnn.CoreCoord(0, 0), (4, 2)),
    ],
)
def test_simulated_distributed_layernorm(
    device,
    input_width,
    num_devices,
    is_rmsnorm,
    input_df,
    weights_df,
    seed,
    eps,
    mean,
    std,
    min_pcc,
    max_atol,
    core_grid,
    grid_offset,
    output_core_grid,
):
    # Create input and weight tensors
    torch_input_tensor, torch_weight, torch_input_chunks, torch_weight_chunks = create_input_and_weight_tensors(
        input_width, num_devices, seed, mean, std
    )

    if output_core_grid is None:
        output_core_grid = core_grid
    out_memory_config = create_output_memory_config(output_core_grid, torch_input_chunks[0].shape)

    # Compute reference output
    torch_output_tensor = compute_reference_output(torch_input_tensor, torch_weight, is_rmsnorm, eps)
    torch_output_chunks = torch.chunk(torch_output_tensor, num_devices, dim=-1)

    # Simulate multi-device pre-allgather computation
    tt_pre_allgather_outputs = []
    for d in range(num_devices):
        tt_input_tensor = create_tt_tensors(
            torch_input_chunks[d], device, input_df, core_grid, input_width, grid_offset=grid_offset
        )
        tt_pre_allgather_output = compute_pre_allgather_stats(tt_input_tensor, core_grid, input_width, is_rmsnorm)
        tt_pre_allgather_outputs.append(tt_pre_allgather_output)

    # Extract and concatenate statistics from pre-allgather outputs
    tt_stats_list = []
    for tt_pre_allgather_output in tt_pre_allgather_outputs:
        tt_pre_allgather_output = ttnn.to_memory_config(tt_pre_allgather_output, memory_config=ttnn.L1_MEMORY_CONFIG)
        tt_stats_list.append(tt_pre_allgather_output)

    tt_global_stats = ttnn.concat(tt_stats_list, -1)
    # shard to 1 core
    tt_stats_sharded_config = ttnn.create_sharded_memory_config(
        shape=(32, tt_global_stats.padded_shape[-1]),
        core_grid=ttnn.CoreRangeSet([ttnn.CoreRange(grid_offset, grid_offset)]),
        strategy=ttnn.ShardStrategy.WIDTH,
        use_height_and_width_as_shard_shape=True,
    )
    tt_global_stats = ttnn.to_memory_config(tt_global_stats, memory_config=tt_stats_sharded_config)

    # Simulate multi-device post-allgather computation
    tt_output_chunks = []
    for d in range(num_devices):
        tt_input_tensor = create_tt_tensors(
            torch_input_chunks[d], device, input_df, core_grid, input_width, grid_offset=grid_offset
        )
        tt_weights = create_tt_tensors(
            torch_weight_chunks[d], device, weights_df, core_grid, input_width, is_weight=True
        )
        tt_output_tensor = compute_post_allgather_output(
            tt_input_tensor,
            tt_weights,
            tt_global_stats,
            eps,
            is_rmsnorm,
            core_grid,
            input_width,
            input_df,
            out_memory_config,
        )

        tt_output_chunks.append(ttnn.to_torch(tt_output_tensor).to(torch.bfloat16))

    # Concatenate output chunks
    tt_output_torch = torch.cat(tt_output_chunks, dim=-1)

    # Compare results
    assert_numeric_metrics(
        torch_output_tensor,
        tt_output_torch,
        pcc_threshold=min_pcc,
        rtol=0.05,
        atol=max_atol,
        frobenius_threshold=0.15,
    )


# Large out-of-distribution poison written into the implicit tile padding so that any read of the
# padded columns is observable: a statistic computed over the logical width is unaffected, while one
# that folds the padded columns in is grossly wrong.
_NON_TILE_ALIGNED_PAD_VALUE = 1000.0


def _run_simulated_distributed_norm_multi_core(device, is_rmsnorm, w, num_cores_w, eps, num_cores_h=1):
    """Simulated distributed sharded norm over a width w that is not a multiple of the tile size (32),
    split across a num_cores_w x num_cores_h grid of cores.

    Each core holds whole tiles, so the columns past w are padding on the final core. That padding is
    poisoned with _NON_TILE_ALIGNED_PAD_VALUE so any inclusion of padded value into statistics produces
    a grossly wrong result.

    num_cores_h picks the cross-core reduction: 1 is single-stage, >1 forms a 2D grid that uses the
    two-stage path (see should_use_two_stage_reduce in sharded_layernorm_factory_helpers.cpp).
    """
    tile_width = 32
    num_cores = num_cores_w * num_cores_h
    shard_wt = math.ceil(w / num_cores / tile_width)  # tiles per shard
    shard_w = shard_wt * tile_width
    physical_w = shard_w * num_cores  # logical w padded out to whole shards

    torch.manual_seed(0)
    torch_input_tensor = torch.normal(0.0, 1.0, size=(1, 1, 32, w), dtype=torch.bfloat16)
    torch_weight = torch.normal(0.0, 1.0, size=(1, 1, 1, w), dtype=torch.bfloat16)
    torch_golden = compute_reference_output(torch_input_tensor, torch_weight, is_rmsnorm, eps)

    core_range_set = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_w - 1, num_cores_h - 1))}
    )
    input_sharded_config = ttnn.create_sharded_memory_config(
        shape=(32, shard_w),
        core_grid=core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        use_height_and_width_as_shard_shape=True,
    )

    def make_poisoned_input():
        tt = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        tt = ttnn.to_memory_config(tt, memory_config=input_sharded_config)
        return ttnn.fill_implicit_tile_padding(tt, _NON_TILE_ALIGNED_PAD_VALUE)

    norm_prgm_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[num_cores_w, num_cores_h],
        subblock_w=shard_wt,
        block_h=1,
        block_w=shard_wt,
        inplace=False,
    )

    pre_all_gather = ttnn.rms_norm_pre_all_gather if is_rmsnorm else ttnn.layer_norm_pre_all_gather
    post_all_gather = ttnn.rms_norm_post_all_gather if is_rmsnorm else ttnn.layer_norm_post_all_gather

    tt_stats = pre_all_gather(make_poisoned_input(), program_config=norm_prgm_cfg)
    tt_stats = ttnn.to_memory_config(tt_stats, memory_config=ttnn.L1_MEMORY_CONFIG)
    stats_sharded_config = ttnn.create_sharded_memory_config(
        shape=(32, tt_stats.padded_shape[-1]),
        core_grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
        strategy=ttnn.ShardStrategy.WIDTH,
        use_height_and_width_as_shard_shape=True,
    )
    tt_stats = ttnn.to_memory_config(tt_stats, memory_config=stats_sharded_config)

    # Gamma is read as whole tiles per core, so it must span the full physical width; the columns beyond
    # the logical width only ever multiply discarded padding, so their values do not matter.
    torch_weight_padded = torch.nn.functional.pad(torch_weight, (0, physical_w - w))
    tt_weight = ttnn.from_torch(
        torch_weight_padded,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )

    out_memory_config = create_output_memory_config((num_cores_w, num_cores_h), torch_input_tensor.shape)
    tt_output = post_all_gather(
        make_poisoned_input(),
        epsilon=eps,
        weight=tt_weight,
        program_config=norm_prgm_cfg,
        stats=tt_stats,
        dtype=ttnn.bfloat16,
        memory_config=out_memory_config,
    )

    # Only the final core is partially valid, so the valid columns are contiguous and end at w.
    actual = ttnn.to_torch(tt_output).to(torch.bfloat16)[..., :w]
    assert_numeric_metrics(torch_golden, actual, pcc_threshold=0.999, rtol=0.05, atol=0.2, frobenius_threshold=0.05)


@pytest.mark.parametrize("is_rmsnorm", [True, False], ids=["rmsnorm", "layernorm"])
@pytest.mark.parametrize("w", [120, 240])
@pytest.mark.parametrize("num_cores_w", [2])
@pytest.mark.parametrize("eps", [1e-6])
def test_simulated_distributed_norm_multi_core_non_tile_aligned_width(device, is_rmsnorm, w, num_cores_w, eps):
    """Non-tile-aligned width split across a row of cores (num_cores_h=1, single-stage reduction), for
    both norms. Checks that the padding stays out of the per-shard statistics, so the normalized output
    matches the reference.
    """
    _run_simulated_distributed_norm_multi_core(device, is_rmsnorm=is_rmsnorm, w=w, num_cores_w=num_cores_w, eps=eps)


@pytest.mark.parametrize("is_rmsnorm", [True, False], ids=["rmsnorm", "layernorm"])
@pytest.mark.parametrize("w", [120, 240])
@pytest.mark.parametrize(("num_cores_w", "num_cores_h"), [(2, 2)])
@pytest.mark.parametrize("eps", [1e-6])
def test_simulated_distributed_norm_two_stage_reduce_non_tile_aligned_width(
    device, is_rmsnorm, w, num_cores_w, num_cores_h, eps
):
    """Non-tile-aligned width split across a 2D core grid (num_cores_w and num_cores_h both > 1), for
    both norms. The 2D grid drives the cross-core reduction through its two-stage path, which the
    single-stage row-of-cores variant does not reach; this checks the padding still stays out of the
    per-shard statistics there, so the output matches the reference.
    """
    _run_simulated_distributed_norm_multi_core(
        device, is_rmsnorm=is_rmsnorm, w=w, num_cores_w=num_cores_w, eps=eps, num_cores_h=num_cores_h
    )


def _run_simulated_distributed_norm(device, is_rmsnorm, w, eps):
    """Single-device simulated distributed (pre + post all-gather) sharded norm over width w.

    Computes per-shard statistics (pre-all-gather), reshards them onto a single core to stand in for
    the all-gather on one device, then normalizes (post-all-gather) and compares against the torch
    golden. The implicit tile padding is poisoned, so a path that folds the padded columns into the
    statistics is observably wrong. This helper can also be called for tile-aligned widths, which have
    no implicit padding, so the poison is a no-op there.
    """
    torch.manual_seed(0)
    h = 32
    padded_w = math.ceil(w / 32) * 32
    block_wt = padded_w // 32

    torch_input_tensor = torch.normal(0.0, 1.0, size=(1, 1, h, w), dtype=torch.bfloat16)
    torch_weight = torch.normal(0.0, 1.0, size=(1, 1, 1, w), dtype=torch.bfloat16)
    torch_golden = compute_reference_output(torch_input_tensor, torch_weight, is_rmsnorm, eps)

    # Single-core block-sharded config; the shard width is the tile-padded width.
    sharded_mem_config = make_sharded_norm_mem_config(num_cores_w=1, h=h, shard_w=padded_w)
    prgm_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[1, 1],
        subblock_w=1,
        block_h=1,
        block_w=block_wt,
        inplace=False,
    )

    def make_poisoned_input():
        return to_poisoned_sharded(device, torch_input_tensor, sharded_mem_config, _NON_TILE_ALIGNED_PAD_VALUE)

    # Pre-all-gather: per-shard statistics.
    tt_input_tensor = make_poisoned_input()
    if is_rmsnorm:
        tt_stats = ttnn.rms_norm_pre_all_gather(tt_input_tensor, program_config=prgm_cfg)
    else:
        tt_stats = ttnn.layer_norm_pre_all_gather(tt_input_tensor, program_config=prgm_cfg)

    # Single device, so the gathered stats are just this shard's stats, resharded to one core.
    tt_stats = ttnn.to_memory_config(tt_stats, memory_config=ttnn.L1_MEMORY_CONFIG)
    stats_sharded_config = ttnn.create_sharded_memory_config(
        shape=(32, tt_stats.padded_shape[-1]),
        core_grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
        strategy=ttnn.ShardStrategy.WIDTH,
        use_height_and_width_as_shard_shape=True,
    )
    tt_stats = ttnn.to_memory_config(tt_stats, memory_config=stats_sharded_config)

    # Post-all-gather: normalized output.
    tt_weight = create_tt_tensors(torch_weight, device, ttnn.bfloat16, (1, 1), w, is_weight=True)
    # Output has the same shape and sharding as the input; the op requires matching memory layouts.
    out_memory_config = sharded_mem_config
    tt_input_tensor = make_poisoned_input()
    if is_rmsnorm:
        tt_output = ttnn.rms_norm_post_all_gather(
            tt_input_tensor,
            epsilon=eps,
            weight=tt_weight,
            program_config=prgm_cfg,
            stats=tt_stats,
            dtype=ttnn.bfloat16,
            memory_config=out_memory_config,
        )
    else:
        tt_output = ttnn.layer_norm_post_all_gather(
            tt_input_tensor,
            epsilon=eps,
            weight=tt_weight,
            program_config=prgm_cfg,
            stats=tt_stats,
            dtype=ttnn.bfloat16,
            memory_config=out_memory_config,
        )

    # Discard padding before comparison.
    actual = ttnn.to_torch(tt_output).to(torch.bfloat16)[..., :w]

    assert_numeric_metrics(
        torch_golden,
        actual,
        pcc_threshold=0.999,
        rtol=0.05,
        atol=0.2,
        frobenius_threshold=0.05,
    )


def _run_simulated_distributed_norm_pre_all_gather_stats(device, is_rmsnorm, w, eps):
    """Single-device distributed pre-all-gather stats over a non-tile-aligned width.

    Checks the per-shard statistics directly (E[x] and E[x^2]) rather than the full pre + post
    pipeline, so it isolates the pre-all-gather reduction and avoids the unrelated single-core
    post-all-gather hang (see _SIMULATED_DISTRIBUTED_POST_ALL_GATHER_HANG). The statistics must be
    reduced over the logical width only; the implicit tile padding is poisoned so any inclusion of the
    padded columns is observable.
    """
    torch.manual_seed(0)
    h = 32
    padded_w = math.ceil(w / 32) * 32
    block_wt = padded_w // 32

    torch_input_tensor = torch.normal(0.0, 1.0, size=(1, 1, h, w), dtype=torch.bfloat16)
    # References reduce over the logical width only, so a correct kernel excludes the padding columns.
    torch_ex = torch.mean(torch_input_tensor.to(torch.float32), dim=-1, keepdim=True)
    torch_ex2 = torch.mean(torch_input_tensor.to(torch.float32) ** 2, dim=-1, keepdim=True)

    sharded_mem_config = make_sharded_norm_mem_config(num_cores_w=1, h=h, shard_w=padded_w)
    prgm_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[1, 1],
        subblock_w=1,
        block_h=1,
        block_w=block_wt,
        inplace=False,
    )

    tt_input_tensor = to_poisoned_sharded(device, torch_input_tensor, sharded_mem_config, _NON_TILE_ALIGNED_PAD_VALUE)

    if is_rmsnorm:
        tt_stats = ttnn.rms_norm_pre_all_gather(tt_input_tensor, program_config=prgm_cfg)
    else:
        tt_stats = ttnn.layer_norm_pre_all_gather(tt_input_tensor, program_config=prgm_cfg)
    stats = ttnn.to_torch(tt_stats).to(torch.float32)

    # Per-metric tolerances follow run_pre_allgather_layernorm (the tile-aligned pre-all-gather stats
    # test) for this same op and statistic; alignment does not change the per-element arithmetic. The
    # path is all-bf16 (fp32_dest_acc=False, HiFi4): the reduction scaler 1/N is bf16 and the result is
    # packed to bf16, two round-to-nearest steps of <= 2^-8 each (~2^-7, ~0.8% relative floor), with
    # fp16 DST accumulation over the row adding the rest.
    #   - rtol 0.05: per-element relative error envelope above that floor.
    #   - atol: absolute floor at each statistic's scale, 0.04 for the O(1) E[x^2] and 0.01 for the
    #     near-zero E[x].
    #   - frobenius 0.15: aggregate relative-norm safety net.
    #   - pcc: 0.982 for the near-constant E[x^2] (its low row-to-row variance makes PCC noise-
    #     sensitive, so the floor is loose) and 0.9997 for E[x].
    # Padding contamination is not subtle: folding the poisoned columns into the mean / mean of squares
    # dominates the reduction, landing orders of magnitude beyond these envelopes, so every metric fails
    # comfortably if the padding is not excluded.
    if is_rmsnorm:
        # Stats layout: E[x^2] in the first column.
        assert_numeric_metrics(
            torch_ex2, stats[..., :1], pcc_threshold=0.982, rtol=0.05, atol=0.04, frobenius_threshold=0.15
        )
    else:
        # Stats layout: one 32-wide tile per statistic, E[x] in column 0 and E[x^2] in column 32.
        assert_numeric_metrics(
            torch_ex, stats[..., :1], pcc_threshold=0.9997, rtol=0.05, atol=0.01, frobenius_threshold=0.15
        )
        assert_numeric_metrics(
            torch_ex2, stats[..., 32:33], pcc_threshold=0.982, rtol=0.05, atol=0.04, frobenius_threshold=0.15
        )


# The single-core (1x1 grid) simulated distributed post-all-gather flow hangs waiting on the
# cb_ex_global multicast. It reproduces for both tile-aligned and non-tile-aligned widths, so the hang
# is a distributed post-all-gather issue independent of non-tile-aligned padding handling. run=False
# keeps the hang from stalling CI.
_SIMULATED_DISTRIBUTED_POST_ALL_GATHER_HANG = (
    "Single-core simulated distributed post-all-gather hangs on the cb_ex_global multicast. Issue #48661"
)


@pytest.mark.xfail(reason=_SIMULATED_DISTRIBUTED_POST_ALL_GATHER_HANG, run=False)
@pytest.mark.parametrize("is_rmsnorm", [True, False])
@pytest.mark.parametrize("w", [64, 128])
@pytest.mark.parametrize("eps", [1e-6])
def test_simulated_distributed_norm_tile_aligned_width(device, is_rmsnorm, w, eps):
    """Tile-aligned single-core widths through the full pipeline. Shows the post-all-gather hang
    affects tile-aligned widths too, not just non-tile-aligned ones (see
    _SIMULATED_DISTRIBUTED_POST_ALL_GATHER_HANG).
    """
    _run_simulated_distributed_norm(device, is_rmsnorm, w, eps)


@pytest.mark.parametrize("is_rmsnorm", [True, False])
@pytest.mark.parametrize("w", [40, 72, 200])
@pytest.mark.parametrize("eps", [1e-6])
def test_simulated_distributed_norm_pre_all_gather_non_tile_aligned_width(device, is_rmsnorm, w, eps):
    """Non-tile-aligned single-core widths, for both norms. Checks the pre-all-gather per-shard
    statistics directly rather than the full pipeline, because the single-core post-all-gather hangs
    (see _SIMULATED_DISTRIBUTED_POST_ALL_GATHER_HANG).
    """
    _run_simulated_distributed_norm_pre_all_gather_stats(device, is_rmsnorm, w, eps)


@pytest.mark.xfail(reason=_SIMULATED_DISTRIBUTED_POST_ALL_GATHER_HANG, run=False)
@pytest.mark.parametrize("is_rmsnorm", [True, False])
@pytest.mark.parametrize("w", [40, 72, 200])
@pytest.mark.parametrize("eps", [1e-6])
def test_simulated_distributed_norm_non_tile_aligned_width(device, is_rmsnorm, w, eps):
    """Non-tile-aligned single-core widths through the full pipeline. Does not run: the single-core
    post-all-gather hangs (see _SIMULATED_DISTRIBUTED_POST_ALL_GATHER_HANG). The pre-all-gather padding
    exclusion is verified directly by test_simulated_distributed_norm_pre_all_gather_non_tile_aligned_width.
    """
    _run_simulated_distributed_norm(device, is_rmsnorm, w, eps)
