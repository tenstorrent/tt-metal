# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import ttnn
import torch
import pytest
from models.utility_functions import (
    skip_for_wormhole_b0,
    comp_allclose_and_pcc,
    comp_pcc,
    comp_allclose,
)

# create test for layer_norm for sharded input tensor [32, 2048]  and weight and bias tensors [2048]
# sharded on 32 cores


def rms_norm(x, gamma, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gamma


def layer_norm(x, gamma, eps):
    return (x - x.mean(-1, keepdim=True)) * torch.rsqrt(x.var(-1, keepdim=True) + eps) * gamma


@pytest.mark.parametrize("RMSNORM", [True, False])
@pytest.mark.parametrize("seed", [0, 2000, 50000])  # Test across 5 different seeds
@pytest.mark.parametrize("eps", [1e-6])
@pytest.mark.parametrize("min_pcc", [0.9997])
@pytest.mark.parametrize("max_atol", [0.38])
@pytest.mark.parametrize("input_width", [1024, 2048])
@pytest.mark.parametrize("input_df", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("weights_df", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize(("mean", "std"), ([0, 1], [10.0, 11.0], [100.0, 110.0]))
def test_sharded_layernorm(
    device, use_program_cache, input_width, RMSNORM, input_df, weights_df, seed, eps, mean, std, min_pcc, max_atol
):
    if RMSNORM:
        print("Testing RMSNorm")
    else:
        print("Testing LayerNorm")
    torch.manual_seed(seed)
    input_shape = (1, 1, 32, input_width)
    weights_shape = (1, 1, 1, input_width)

    torch_input_tensor = torch.normal(mean, std, size=input_shape, dtype=torch.bfloat16)
    torch_weight = torch.normal(mean, std, size=weights_shape, dtype=torch.bfloat16)

    print(f" Mean : {torch_input_tensor.mean()}, Var : {torch_input_tensor.var()}")

    if RMSNORM:
        torch_output_tensor = rms_norm(torch_input_tensor, torch_weight, eps=eps)
    else:
        torch_output_tensor = torch.nn.functional.layer_norm(
            torch_input_tensor, (input_width,), weight=torch_weight.squeeze(0).squeeze(0).squeeze(0), eps=eps
        )
        # torch_output_tensor = layer_norm(torch_input_tensor, torch_weight, eps=eps)

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=input_df,
    )
    # shard to 32 cores
    tt_sharded_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, 32, input_width),
        core_grid=ttnn.CoreGrid(y=2, x=8),
        strategy=ttnn.ShardStrategy.WIDTH,
    )
    tt_input_tensor = ttnn.experimental.tensor.interleaved_to_sharded(
        tt_input_tensor, sharded_mem_config=tt_sharded_config
    )

    SHARDED_NORM_PRGM_CFG = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[8, 2],
        subblock_w=(input_width // 16) // 32,
        block_h=1,
        block_w=(input_width // 16) // 32,
        inplace=False,
    )

    tt_weights = ttnn.as_tensor(
        torch_weight,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=weights_df,
        # cache_file_name="rms_weights_cache_1024",
    )

    if RMSNORM:
        tt_output_tensor = ttnn.rms_norm(
            tt_input_tensor,
            epsilon=eps,
            weight=tt_weights,
            program_config=SHARDED_NORM_PRGM_CFG,
            memory_config=tt_sharded_config,
        )
    else:
        tt_output_tensor = ttnn.layer_norm(
            tt_input_tensor,
            epsilon=eps,
            weight=tt_weights,
            program_config=SHARDED_NORM_PRGM_CFG,
            memory_config=tt_sharded_config,
        )
    tt_output_torch = ttnn.to_torch(tt_output_tensor).to(torch.bfloat16)

    pcc_passing, pcc_out = comp_pcc(torch_output_tensor, tt_output_torch, pcc=min_pcc)
    all_close_passing = torch.allclose(torch_output_tensor, tt_output_torch, atol=max_atol, equal_nan=False)
    atol_delta = torch.max(torch.abs(torch_output_tensor - tt_output_torch)).item()
    print(f"PCC: {pcc_out}")
    print(f"all_close : {all_close_passing}, Max ATOL: {atol_delta}")

    assert pcc_out >= min_pcc, f"PCC test failed: {pcc_out} (threshold: {min_pcc})"
    # assert atol_delta <= max_atol, f"Max Atol exceeded: {atol_delta} (allowed: {max_atol})"


@pytest.mark.parametrize("is_rmsnorm", [True, False])
@pytest.mark.parametrize("seed", [0])  # Test across 5 different seeds
@pytest.mark.parametrize("eps", [1e-6])
@pytest.mark.parametrize("min_pcc", [0.9997])
@pytest.mark.parametrize("max_atol", [0.38])
@pytest.mark.parametrize("input_width", [1024])
@pytest.mark.parametrize("input_df", [ttnn.bfloat16])
@pytest.mark.parametrize("weights_df", [ttnn.bfloat16])
@pytest.mark.parametrize(("mean", "std"), ([0, 1], [100, 101]))
def test_post_allgather_layernorm(
    all_devices,
    input_width,
    is_rmsnorm,
    input_df,
    weights_df,
    seed,
    eps,
    mean,
    std,
    min_pcc,
    max_atol,
):
    device = all_devices[0]

    if is_rmsnorm:
        print("Testing RMSNorm")
    else:
        print("Testing LayerNorm")
    torch.manual_seed(seed)
    input_shape = (1, 1, 32, input_width)
    weights_shape = (1, 1, 1, input_width)

    torch_input_tensor = torch.normal(mean, std, size=input_shape, dtype=torch.bfloat16)
    torch_weight = torch.normal(mean, std, size=weights_shape, dtype=torch.bfloat16)
    # torch_input_tensor = torch.ones(input_shape, dtype=torch.bfloat16)
    # torch_weight = torch.ones(weights_shape, dtype=torch.bfloat16)

    print(f" Mean : {torch_input_tensor.mean()}, Var : {torch_input_tensor.var()}")

    if is_rmsnorm:
        torch_output_tensor = rms_norm(torch_input_tensor, torch_weight, eps=eps)
    else:
        torch_output_tensor = torch.nn.functional.layer_norm(
            torch_input_tensor, (input_width,), weight=torch_weight.squeeze(0).squeeze(0).squeeze(0), eps=eps
        )

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=input_df,
    )
    # shard to 32 cores
    tt_sharded_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, 32, input_width),
        core_grid=ttnn.CoreGrid(y=2, x=8),
        strategy=ttnn.ShardStrategy.WIDTH,
    )
    tt_input_tensor = ttnn.experimental.tensor.interleaved_to_sharded(
        tt_input_tensor, sharded_mem_config=tt_sharded_config
    )

    SHARDED_NORM_PRGM_CFG = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[8, 2],
        subblock_w=(input_width // 16) // 32,
        block_h=1,
        block_w=(input_width // 16) // 32,
        inplace=False,
    )

    tt_weights = ttnn.as_tensor(
        torch_weight,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=weights_df,
    )

    # create E[x] and E[x^2] tensors
    sum_x_tensor = torch.sum(torch_input_tensor, dim=-1, keepdim=True).to(torch.bfloat16) / input_width

    sum_x2_tensor = torch.sum(torch_input_tensor**2, dim=-1, keepdim=True).to(torch.bfloat16) / input_width

    tt_sum_x_tensor = ttnn.from_torch(
        sum_x_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=input_df,
    )

    tt_sum_x2_tensor = ttnn.from_torch(
        sum_x2_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=input_df,
    )
    if is_rmsnorm:
        tt_stats_tensor = tt_sum_x2_tensor
        tt_stats_sharded_config = ttnn.create_sharded_memory_config(
            shape=(1, 1, 32, 32),
            core_grid=ttnn.CoreGrid(y=1, x=1),
            strategy=ttnn.ShardStrategy.WIDTH,
        )
        tt_stats_tensor = ttnn.experimental.tensor.interleaved_to_sharded(
            tt_stats_tensor, sharded_mem_config=tt_stats_sharded_config
        )

    else:
        tt_stats_tensor = ttnn.concat([tt_sum_x_tensor, tt_sum_x2_tensor], -1)
        # shard to 1 core
        tt_stats_sharded_config = ttnn.create_sharded_memory_config(
            shape=(1, 1, 32, 64),
            core_grid=ttnn.CoreGrid(y=1, x=1),
            strategy=ttnn.ShardStrategy.WIDTH,
        )
        tt_stats_tensor = ttnn.experimental.tensor.interleaved_to_sharded(
            tt_stats_tensor, sharded_mem_config=tt_stats_sharded_config
        )

    iterations = 1
    prev_tt_output_torch = None
    for iter in range(iterations):
        if is_rmsnorm:
            tt_output_tensor = ttnn.rmsnorm_post_all_gather(
                tt_input_tensor,
                epsilon=eps,
                weight=tt_weights,
                program_config=SHARDED_NORM_PRGM_CFG,
                memory_config=tt_sharded_config,
                stats=tt_stats_tensor,
            )
        else:
            tt_output_tensor = ttnn.layernorm_post_all_gather(
                tt_input_tensor,
                epsilon=eps,
                weight=tt_weights,
                program_config=SHARDED_NORM_PRGM_CFG,
                memory_config=tt_sharded_config,
                stats=tt_stats_tensor,
            )
        tt_output_torch = ttnn.to_torch(tt_output_tensor).to(torch.bfloat16)
        if iter == 0:
            _, pcc_out = comp_pcc(torch_output_tensor, tt_output_torch, pcc=min_pcc)
            all_close_passing = torch.allclose(torch_output_tensor, tt_output_torch, atol=max_atol, equal_nan=False)
            atol_delta = torch.max(torch.abs(torch_output_tensor - tt_output_torch)).item()
            print("torch_output_tensor", torch_output_tensor)
            print("tt_output_torch", tt_output_torch)
            print(f"PCC: {pcc_out}")
            print(f"all_close : {all_close_passing}, Max ATOL: {atol_delta}")
            assert pcc_out >= min_pcc, f"PCC test failed: {pcc_out} (threshold: {min_pcc})"
            # assert atol_delta <= max_atol, f"Max Atol exceeded: {atol_delta} (allowed: {max_atol})"

        else:
            all_close_passing = torch.allclose(prev_tt_output_torch, tt_output_torch, atol=0, equal_nan=False)
            atol_delta = torch.max(torch.abs(prev_tt_output_torch - tt_output_torch)).item()
            print(f"all_close_previous : {all_close_passing}, Max ATOL: {atol_delta}")
            assert all_close_passing, f"Max Atol exceeded for previous : {atol_delta} (allowed: {0})"
        prev_tt_output_torch = tt_output_torch


@pytest.mark.parametrize("is_rmsnorm", [False])
@pytest.mark.parametrize("seed", [0])  # Test across 5 different seeds
@pytest.mark.parametrize(("min_pcc_sumx", "min_pcc_sumx2"), ([0.9997, 0.995],))
@pytest.mark.parametrize("max_atol", [0.38])
@pytest.mark.parametrize("input_width", [1024])
@pytest.mark.parametrize("input_df", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize(("mean", "std"), ([0, 1],))
def test_pre_allgather_layernorm(
    device,
    input_width,
    is_rmsnorm,
    input_df,
    seed,
    mean,
    std,
    min_pcc_sumx,
    min_pcc_sumx2,
    max_atol,
):
    if is_rmsnorm:
        print("Testing RMSNorm")
    else:
        print("Testing LayerNorm")
    torch.manual_seed(seed)
    input_shape = (1, 1, 32, input_width)

    torch_input_tensor = torch.normal(mean, std, size=input_shape, dtype=torch.bfloat16)
    # torch_input_tensor = torch.ones(input_shape, dtype=torch.bfloat16)

    print(f" Mean : {torch_input_tensor.mean()}, Var : {torch_input_tensor.var()}")

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=input_df,
    )
    # shard to 32 cores
    tt_sharded_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, 32, input_width),
        core_grid=ttnn.CoreGrid(y=2, x=8),
        strategy=ttnn.ShardStrategy.WIDTH,
    )
    tt_input_tensor = ttnn.experimental.tensor.interleaved_to_sharded(
        tt_input_tensor, sharded_mem_config=tt_sharded_config
    )

    SHARDED_NORM_PRGM_CFG = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[8, 2],
        subblock_w=(input_width // 16) // 32,
        block_h=1,
        block_w=(input_width // 16) // 32,
        inplace=False,
    )

    # create sum[x] and sum[x^2] tensors
    sum_x_tensor = torch.sum(torch_input_tensor, dim=-1, keepdim=True).to(torch.bfloat16)  # [1, 1, 32, 1]

    sum_x2_tensor = torch.sum(torch_input_tensor**2, dim=-1, keepdim=True).to(torch.bfloat16)  # [1, 1, 32, 1]

    iterations = 200
    prev_tt_output_torch = None
    for iter in range(iterations):
        if is_rmsnorm:
            tt_output_tensor = ttnn.rmsnorm_pre_all_gather(tt_input_tensor, program_config=SHARDED_NORM_PRGM_CFG)
        else:
            tt_output_tensor = ttnn.layernorm_pre_all_gather(tt_input_tensor, program_config=SHARDED_NORM_PRGM_CFG)
        tt_output_torch = ttnn.to_torch(tt_output_tensor).to(torch.bfloat16)  # [1, 1, 32, 64]

        if is_rmsnorm:
            tt_ex2_torch = tt_output_torch[..., :1]
        else:
            tt_ex_torch = tt_output_torch[..., :1]
            tt_ex2_torch = tt_output_torch[..., 32:33]

        if iter == 0:
            if not is_rmsnorm:
                _, pcc_out1 = comp_pcc(sum_x_tensor, tt_ex_torch, pcc=min_pcc_sumx)
                all_close_passing = torch.allclose(sum_x_tensor, tt_ex_torch, atol=max_atol, equal_nan=False)
                atol_delta = torch.max(torch.abs(sum_x_tensor - tt_ex_torch)).item()
                print(f"PCC: {pcc_out1}")
                print(f"all_close : {all_close_passing}, Max ATOL: {atol_delta}")
                assert pcc_out1 >= min_pcc_sumx, f"PCC of Sum(x) test failed: {pcc_out1} (threshold: {min_pcc_sumx})"

            _, pcc_out2 = comp_pcc(sum_x2_tensor, tt_ex2_torch, pcc=min_pcc_sumx2)
            all_close_passing = torch.allclose(sum_x2_tensor, tt_ex2_torch, atol=max_atol, equal_nan=False)
            atol_delta = torch.max(torch.abs(sum_x2_tensor - tt_ex2_torch)).item()
            print(f"PCC: {pcc_out2}")
            print(f"all_close : {all_close_passing}, Max ATOL: {atol_delta}")
            assert pcc_out2 >= min_pcc_sumx2, f"PCC of Sum(x^2) test failed: {pcc_out2} (threshold: {min_pcc_sumx2})"
        else:
            all_close_passing = torch.allclose(prev_tt_output_torch, tt_output_torch, atol=0, equal_nan=False)
            atol_delta = torch.max(torch.abs(prev_tt_output_torch - tt_output_torch)).item()
            print(f"all_close_previous : {all_close_passing}, Max ATOL: {atol_delta}")
            assert all_close_passing, f"Max Atol exceeded for previous : {atol_delta} (allowed: {0})"
        prev_tt_output_torch = tt_output_torch

    # assert atol_delta <= max_atol, f"Max Atol exceeded: {atol_delta} (allowed: {max_atol})"
