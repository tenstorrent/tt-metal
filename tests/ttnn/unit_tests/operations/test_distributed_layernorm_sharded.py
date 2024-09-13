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

# create test for layer_norm for sharded input tensor [32, 2048]  and weight and bias tensors [2048]
# sharded on 32 cores


def rms_norm(x, gamma, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gamma


def layer_norm(x, gamma, eps):
    return (x - x.mean(-1, keepdim=True)) * torch.rsqrt(x.var(-1, keepdim=True) + eps) * gamma


@pytest.mark.parametrize("is_rmsnorm", [True, False])
@pytest.mark.parametrize("seed", [0, 2000, 50000])  # Test across 5 different seeds
@pytest.mark.parametrize("eps", [1e-6])
@pytest.mark.parametrize("min_pcc", [0.9997])
@pytest.mark.parametrize("max_atol", [0.38])
@pytest.mark.parametrize("input_width", [1024, 2048])
@pytest.mark.parametrize("input_df", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("weights_df", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize(("mean", "std"), ([0, 1], [10.0, 11.0], [100.0, 110.0]))
def test_sharded_layernorm(
    device, use_program_cache, input_width, is_rmsnorm, input_df, weights_df, seed, eps, mean, std, min_pcc, max_atol
):
    if is_rmsnorm:
        print("Testing RMSNorm")
    else:
        print("Testing LayerNorm")
    torch.manual_seed(seed)
    input_shape = (1, 1, 32, input_width)
    weights_shape = (1, 1, 1, input_width)

    torch_input_tensor = torch.normal(mean, std, size=input_shape, dtype=torch.bfloat16)
    torch_weight = torch.normal(mean, std, size=weights_shape, dtype=torch.bfloat16)

    print(f" Mean : {torch_input_tensor.mean()}, Var : {torch_input_tensor.var()}")

    if is_rmsnorm:
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
    tt_input_tensor = ttnn.to_memory_config(tt_input_tensor, memory_config=tt_sharded_config)

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

    if is_rmsnorm:
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
@pytest.mark.parametrize("seed", [1234])
@pytest.mark.parametrize(("min_pcc_Ex", "min_pcc_Ex2"), ([0.9997, 0.989],))
@pytest.mark.parametrize("max_atol", [0.38])
@pytest.mark.parametrize(
    "input_width, core_grid",
    [
        ([1024, (2, 8)]),
        ([2048, (8, 8)]),
    ],
)
@pytest.mark.parametrize("input_df", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize(("mean", "std"), ([0, 1],))
def test_pre_allgather_layernorm(
    all_devices,
    use_program_cache,
    input_width,
    core_grid,
    is_rmsnorm,
    input_df,
    seed,
    mean,
    std,
    min_pcc_Ex,
    min_pcc_Ex2,
    max_atol,
):
    device = all_devices[0]

    if is_rmsnorm:
        print("RMSNorm")
    else:
        print("LayerNorm")

    torch.manual_seed(seed)
    input_shape = (1, 1, 32, input_width)

    torch_input_tensor = torch.normal(mean, std, size=input_shape, dtype=torch.bfloat16)

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
        core_grid=ttnn.CoreGrid(y=core_grid[0], x=core_grid[1]),
        strategy=ttnn.ShardStrategy.WIDTH,
    )
    tt_input_tensor = ttnn.to_memory_config(tt_input_tensor, memory_config=tt_sharded_config)

    SHARDED_NORM_PRGM_CFG = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[core_grid[1], core_grid[0]],
        subblock_w=(input_width // (core_grid[0] * core_grid[1])) // 32,
        block_h=1,
        block_w=(input_width // (core_grid[0] * core_grid[1])) // 32,
        inplace=False,
    )

    # create E(x) and E(x^2) tensors
    Ex_tensor = torch.mean(torch_input_tensor, dim=-1, keepdim=True).to(torch.bfloat16)  # [1, 1, 32, 1]

    Ex2_tensor = torch.mean(torch_input_tensor**2, dim=-1, keepdim=True).to(torch.bfloat16)  # [1, 1, 32, 1]

    iterations = 10
    prev_tt_output_torch = None
    for iter in range(iterations):
        if is_rmsnorm:
            tt_output_tensor = ttnn.rms_norm_pre_all_gather(tt_input_tensor, program_config=SHARDED_NORM_PRGM_CFG)
        else:
            tt_output_tensor = ttnn.layer_norm_pre_all_gather(tt_input_tensor, program_config=SHARDED_NORM_PRGM_CFG)
        tt_output_torch = ttnn.to_torch(tt_output_tensor).to(torch.bfloat16)  # [1, 1, 32, 64]

        if is_rmsnorm:  # first tile contains E(xˆ2) in first column
            tt_ex2_torch = tt_output_torch[..., :1]
        else:  # first tile contains E(x) in first column (=index 0) and second tile contains E(xˆ2) in first column (=index 32)
            tt_ex_torch = tt_output_torch[..., :1]
            tt_ex2_torch = tt_output_torch[..., 32:33]

        if iter == 0:
            if not is_rmsnorm:
                _, pcc_out1 = comp_pcc(Ex_tensor, tt_ex_torch, pcc=min_pcc_Ex)
                all_close_passing = torch.allclose(Ex_tensor, tt_ex_torch, atol=max_atol, equal_nan=False)
                atol_delta = torch.max(torch.abs(Ex_tensor - tt_ex_torch)).item()
                print(f"PCC: {pcc_out1}")
                print(f"all_close : {all_close_passing}, Max ATOL: {atol_delta}")
                assert pcc_out1 >= min_pcc_Ex, f"PCC of E(x) test failed: {pcc_out1} (threshold: {min_pcc_Ex})"

            _, pcc_out2 = comp_pcc(Ex2_tensor, tt_ex2_torch, pcc=min_pcc_Ex2)
            all_close_passing = torch.allclose(Ex2_tensor, tt_ex2_torch, atol=max_atol, equal_nan=False)
            atol_delta = torch.max(torch.abs(Ex2_tensor - tt_ex2_torch)).item()
            print(f"PCC: {pcc_out2}")
            print(f"all_close : {all_close_passing}, Max ATOL: {atol_delta}")
            assert pcc_out2 >= min_pcc_Ex2, f"PCC of E(x^2) test failed: {pcc_out2} (threshold: {min_pcc_Ex2})"
        else:
            all_close_passing = torch.equal(prev_tt_output_torch, tt_output_torch)
            assert all_close_passing, f"Output is non-deterministic!"
        prev_tt_output_torch = tt_output_torch
