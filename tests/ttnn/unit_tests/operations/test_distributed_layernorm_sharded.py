# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import ttnn
import torch
import pytest
from loguru import logger

from models.utility_functions import (
    skip_for_wormhole_b0,
    comp_allclose_and_pcc,
    comp_pcc,
    comp_allclose,
)

from models.utility_functions import tt2torch_tensor, get_devices_for_t3000, skip_for_grayskull


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


def create_tt_tensors(torch_chunk, device, df, core_grid, input_width, is_weight=False):
    tt_tensor = ttnn.from_torch(
        torch_chunk,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG if is_weight else ttnn.L1_MEMORY_CONFIG,
        dtype=df,
    )

    if not is_weight:
        tt_sharded_config = ttnn.create_sharded_memory_config(
            shape=(1, 1, 32, input_width),
            core_grid=ttnn.CoreGrid(y=core_grid[0], x=core_grid[1]),
            strategy=ttnn.ShardStrategy.WIDTH,
        )
        tt_tensor = ttnn.to_memory_config(tt_tensor, memory_config=tt_sharded_config)

    return tt_tensor


def compute_pre_allgather_stats(tt_input_tensor, core_grid, input_width, is_rmsnorm):
    SHARDED_NORM_PRGM_CFG = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[core_grid[1], core_grid[0]],
        subblock_w=(input_width // (core_grid[0] * core_grid[1])) // 32,
        block_h=1,
        block_w=(input_width // (core_grid[0] * core_grid[1])) // 32,
        inplace=False,
    )

    if is_rmsnorm:
        return ttnn.rms_norm_pre_all_gather(tt_input_tensor, program_config=SHARDED_NORM_PRGM_CFG)
    else:
        return ttnn.layer_norm_pre_all_gather(tt_input_tensor, program_config=SHARDED_NORM_PRGM_CFG)


def compute_post_allgather_output(
    tt_input_tensor, tt_weights, tt_stats_tensor, eps, is_rmsnorm, core_grid, input_width
):
    SHARDED_NORM_PRGM_CFG = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(core_grid[1], core_grid[0]),
        subblock_w=(input_width // (core_grid[0] * core_grid[1])) // 32,
        block_h=1,
        block_w=(input_width // (core_grid[0] * core_grid[1])) // 32,
        inplace=False,
    )

    tt_sharded_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, 32, input_width),
        core_grid=ttnn.CoreGrid(y=core_grid[0], x=core_grid[1]),
        strategy=ttnn.ShardStrategy.WIDTH,
    )

    if is_rmsnorm:
        return ttnn.rms_norm_post_all_gather(
            tt_input_tensor,
            epsilon=eps,
            weight=tt_weights,
            program_config=SHARDED_NORM_PRGM_CFG,
            memory_config=tt_sharded_config,
            stats=tt_stats_tensor,
        )
    else:
        return ttnn.layer_norm_post_all_gather(
            tt_input_tensor,
            epsilon=eps,
            weight=tt_weights,
            program_config=SHARDED_NORM_PRGM_CFG,
            memory_config=tt_sharded_config,
            stats=tt_stats_tensor,
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


@skip_for_grayskull()
@pytest.mark.parametrize("is_rmsnorm", [True, False])
@pytest.mark.parametrize("seed", [0, 1234])
@pytest.mark.parametrize("input_width", [2048])
@pytest.mark.parametrize("num_devices", [4, 8])
@pytest.mark.parametrize("input_df", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize(("mean", "std"), ([0, 1],))
@pytest.mark.parametrize("core_grid", ((4, 8),))
@pytest.mark.parametrize(("min_pcc_ex", "max_atol_ex"), [(0.9997, 0.01)])
@pytest.mark.parametrize(("min_pcc_ex2", "max_atol_ex2"), [(0.987, 0.04)])
def test_pre_allgather_layernorm(
    device,
    use_program_cache,
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
    torch_input_tensor, _, torch_input_chunks, _ = create_input_and_weight_tensors(
        input_width, num_devices, seed, mean, std
    )

    for d in range(num_devices):
        tt_input_tensor = create_tt_tensors(torch_input_chunks[d], device, input_df, core_grid, input_width)
        tt_pre_allgather_output = compute_pre_allgather_stats(tt_input_tensor, core_grid, input_width, is_rmsnorm)
        tt_pre_allgather_torch = ttnn.to_torch(tt_pre_allgather_output).to(torch.bfloat16)

        if is_rmsnorm:
            tt_ex2 = tt_pre_allgather_torch[..., :1]
            torch_ex2 = torch.mean(torch_input_chunks[d] ** 2, dim=-1, keepdim=True)
            _, pcc_ex2 = comp_pcc(tt_ex2, torch_ex2, pcc=min_pcc_ex2)
            atol_delta_ex2 = torch.max(torch.abs(torch_ex2 - tt_ex2)).item()
            assert pcc_ex2 >= min_pcc_ex2, f"PCC of E(x^2) test failed: {pcc_ex2} (threshold: {min_pcc_ex2})"
            assert torch.allclose(
                tt_ex2, torch_ex2, atol=max_atol_ex2
            ), f"E(x^2) mismatch for device {d} (atol: {atol_delta_ex2})"
        else:
            tt_ex = tt_pre_allgather_torch[..., :1]
            tt_ex2 = tt_pre_allgather_torch[..., 32:33]
            torch_ex = torch.mean(torch_input_chunks[d], dim=-1, keepdim=True)
            torch_ex2 = torch.mean(torch_input_chunks[d] ** 2, dim=-1, keepdim=True)
            _, pcc_ex = comp_pcc(tt_ex, torch_ex, pcc=min_pcc_ex)
            _, pcc_ex2 = comp_pcc(tt_ex2, torch_ex2, pcc=min_pcc_ex2)
            atol_delta_ex = torch.max(torch.abs(torch_ex - tt_ex)).item()
            atol_delta_ex2 = torch.max(torch.abs(torch_ex2 - tt_ex2)).item()
            assert pcc_ex >= min_pcc_ex, f"PCC of E(x) test failed: {pcc_ex} (threshold: {min_pcc_ex})"
            assert pcc_ex2 >= min_pcc_ex2, f"PCC of E(x^2) test failed: {pcc_ex2} (threshold: {min_pcc_ex2})"
            assert torch.allclose(
                tt_ex, torch_ex, atol=max_atol_ex
            ), f"E(x) mismatch for device {d} (atol: {atol_delta_ex})"
            assert torch.allclose(
                tt_ex2, torch_ex2, atol=max_atol_ex2
            ), f"E(x^2) mismatch for device {d} (atol: {atol_delta_ex2})"

    logger.info("Pre-allgather layernorm test passed for all devices")


@skip_for_grayskull()
@pytest.mark.parametrize("is_rmsnorm", [True, False])
@pytest.mark.parametrize("seed", [0, 1234])
@pytest.mark.parametrize("eps", [1e-6])
@pytest.mark.parametrize(("min_pcc", "max_atol"), ((0.9997, 0.45),))
@pytest.mark.parametrize("input_width", [2048])
@pytest.mark.parametrize("num_devices", [4, 8])
@pytest.mark.parametrize("input_df", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("weights_df", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize(("mean", "std"), ([0, 1],))
@pytest.mark.parametrize("core_grid", ((4, 8),))
def test_post_allgather_layernorm(
    device,
    use_program_cache,
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
):
    torch_input_tensor, torch_weight, torch_input_chunks, torch_weight_chunks = create_input_and_weight_tensors(
        input_width, num_devices, seed, mean, std
    )

    torch_output_tensor = compute_reference_output(torch_input_tensor, torch_weight, is_rmsnorm, eps)
    torch_output_chunks = torch.chunk(torch_output_tensor, num_devices, dim=-1)

    # Compute global statistics
    if is_rmsnorm:
        global_ex2 = torch.mean(torch_input_tensor**2, dim=-1, keepdim=True)
        tt_global_stats = ttnn.from_torch(
            global_ex2, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
    else:
        global_ex = torch.mean(torch_input_tensor, dim=-1, keepdim=True)
        global_ex = torch.nn.functional.pad(global_ex, (0, 31), "constant", 0)
        global_ex2 = torch.mean(torch_input_tensor**2, dim=-1, keepdim=True)
        global_ex2 = torch.nn.functional.pad(global_ex2, (0, 31), "constant", 0)
        tt_global_stats = ttnn.concat(
            [
                ttnn.from_torch(global_ex, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG),
                ttnn.from_torch(
                    global_ex2, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
                ),
            ],
            -1,
        )

    # shard to 1 core
    tt_stats_sharded_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, 32, tt_global_stats.get_legacy_shape()[-1]),
        core_grid=ttnn.CoreGrid(y=1, x=1),
        strategy=ttnn.ShardStrategy.WIDTH,
    )
    tt_global_stats = ttnn.to_memory_config(tt_global_stats, memory_config=tt_stats_sharded_config)

    for d in range(num_devices):
        tt_input_tensor = create_tt_tensors(torch_input_chunks[d], device, input_df, core_grid, input_width)
        tt_weights = create_tt_tensors(
            torch_weight_chunks[d], device, weights_df, core_grid, input_width, is_weight=True
        )
        tt_output_tensor = compute_post_allgather_output(
            tt_input_tensor, tt_weights, tt_global_stats, eps, is_rmsnorm, core_grid, input_width
        )
        tt_output_torch = ttnn.to_torch(tt_output_tensor).to(torch.bfloat16)

        _, pcc_out = comp_pcc(torch_output_chunks[d], tt_output_torch, pcc=min_pcc)
        atol_delta = torch.max(torch.abs(torch_output_chunks[d] - tt_output_torch)).item()

        assert pcc_out >= min_pcc, f"PCC test failed for device {d}: {pcc_out} (threshold: {min_pcc})"
        assert atol_delta <= max_atol, f"Max Atol exceeded for device {d}: {atol_delta} (allowed: {max_atol})"

    logger.info("Post-allgather layernorm test passed for all devices")


@skip_for_grayskull()
@pytest.mark.parametrize("is_rmsnorm", [True, False])
@pytest.mark.parametrize("seed", [0, 1234])
@pytest.mark.parametrize("eps", [1e-6])
@pytest.mark.parametrize(("min_pcc", "max_atol"), ((0.9997, 0.45),))
@pytest.mark.parametrize("input_width", [2048])
@pytest.mark.parametrize("num_devices", [4])
@pytest.mark.parametrize("input_df", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("weights_df", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize(("mean", "std"), ([0, 1],))
@pytest.mark.parametrize("core_grid", ((4, 8),))
def test_simulated_distributed_layernorm(
    device,
    use_program_cache,
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
):
    # Create input and weight tensors
    torch_input_tensor, torch_weight, torch_input_chunks, torch_weight_chunks = create_input_and_weight_tensors(
        input_width, num_devices, seed, mean, std
    )

    # Compute reference output
    torch_output_tensor = compute_reference_output(torch_input_tensor, torch_weight, is_rmsnorm, eps)
    torch_output_chunks = torch.chunk(torch_output_tensor, num_devices, dim=-1)

    # Simulate multi-device pre-allgather computation
    tt_pre_allgather_outputs = []
    for d in range(num_devices):
        tt_input_tensor = create_tt_tensors(torch_input_chunks[d], device, input_df, core_grid, input_width)
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
        shape=(1, 1, 32, tt_global_stats.get_legacy_shape()[-1]),
        core_grid=ttnn.CoreGrid(y=1, x=1),
        strategy=ttnn.ShardStrategy.WIDTH,
    )
    tt_global_stats = ttnn.to_memory_config(tt_global_stats, memory_config=tt_stats_sharded_config)

    # Simulate multi-device post-allgather computation
    tt_output_chunks = []
    for d in range(num_devices):
        tt_input_tensor = create_tt_tensors(torch_input_chunks[d], device, input_df, core_grid, input_width)
        tt_weights = create_tt_tensors(
            torch_weight_chunks[d], device, weights_df, core_grid, input_width, is_weight=True
        )
        tt_output_tensor = compute_post_allgather_output(
            tt_input_tensor, tt_weights, tt_global_stats, eps, is_rmsnorm, core_grid, input_width
        )
        tt_output_chunks.append(ttnn.to_torch(tt_output_tensor).to(torch.bfloat16))

    # Concatenate output chunks
    tt_output_torch = torch.cat(tt_output_chunks, dim=-1)

    # Compare results
    _, pcc_out = comp_pcc(torch_output_tensor, tt_output_torch, pcc=min_pcc)
    all_close_passing = torch.allclose(torch_output_tensor, tt_output_torch, atol=max_atol, equal_nan=False)
    atol_delta = torch.max(torch.abs(torch_output_tensor - tt_output_torch)).item()

    assert pcc_out >= min_pcc, f"PCC test failed: {pcc_out} (threshold: {min_pcc})"
    assert atol_delta <= max_atol, f"Max Atol exceeded: {atol_delta} (allowed: {max_atol})"
