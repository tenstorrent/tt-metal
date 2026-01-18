# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc

from tracy.process_model_log import (
    get_latest_ops_log_filename,
    run_device_profiler,
)


def create_torch_input(L, in0_num_cores, E, M, K):
    """
    Create torch input tensor with unique integer values per layer/expert.

    Args:
        L: Number of layers
        in0_num_cores: Number of input cores
        E: Number of experts
        M: Sequence length
        K: Input dimension

    Returns:
        torch_input: Tensor of shape (L, in0_num_cores, E, M, K)
    """
    torch_input = torch.empty((L, in0_num_cores, E, M, K), dtype=torch.bfloat16)
    le_val = 1
    for layer in range(L):
        for expert in range(E):
            for k_chunk_id in range(K // 32):
                k_start, k_end = k_chunk_id * 32, k_chunk_id * 32 + 32
                chunk_value = le_val * 0.001 * k_chunk_id
                torch_input[layer, :, expert, :, k_start:k_end] = chunk_value
            le_val *= -1
    return torch_input


def create_torch_w0(L, E, K, N):
    """
    Create torch w0 weight tensor.

    Args:
        L: Number of layers
        E: Number of experts
        K: Input dimension
        N: Output dimension

    Returns:
        torch_w0: Tensor of shape (L, E, K, N)
    """
    torch_w0 = torch.empty((L, E, K, N), dtype=torch.bfloat16)
    # Use small values that are distinguishable in bfloat4_b range
    # Expert 0: 1 + k_val (1, 2, 3, ...)
    # Expert 1: 10 + k_val (10, 11, 12, ...)
    for l in range(L):
        for e in range(E):
            base_val = 1 + e * 9  # Expert 0: base=1, Expert 1: base=10
            for k_chunk in range(K // 32):
                k_start, k_end = k_chunk * 32, k_chunk * 32 + 32
                k_val = k_chunk % 8  # Keep k_val small (0-7) to stay in bf4 range
                for n_chunk in range(N // 32):
                    n_start, n_end = n_chunk * 32, n_chunk * 32 + 32
                    torch_w0[l, e, k_start:k_end, n_start:n_end] = base_val + k_val

    # torch_w0 = torch.randn((L, E, K, N), dtype=torch.bfloat16)
    return torch_w0


def create_torch_w1(L, E, K, N):
    """
    Create torch w1 weight tensor.

    Args:
        L: Number of layers
        E: Number of experts
        K: Input dimension
        N: Output dimension

    Returns:
        torch_w1: Tensor of shape (L, E, K, N)
    """
    torch_w1 = torch.empty((L, E, K, N), dtype=torch.bfloat16)
    # Use small values that are distinguishable in bfloat4_b range
    # Expert 0: 1 + k_val (1, 2, 3, ...)
    # Expert 1: 10 + k_val (10, 11, 12, ...)
    for l in range(L):
        for e in range(E):
            base_val = 1 + e * 9  # Expert 0: base=1, Expert 1: base=10
            for k_chunk in range(K // 32):
                k_start, k_end = k_chunk * 32, k_chunk * 32 + 32
                k_val = k_chunk % 8  # Keep k_val small (0-7) to stay in bf4 range
                for n_chunk in range(N // 32):
                    n_start, n_end = n_chunk * 32, n_chunk * 32 + 32
                    torch_w1[l, e, k_start:k_end, n_start:n_end] = base_val + k_val
    # w1 = torch.randn((L, E, K, N), dtype=torch.bfloat16)
    return -torch_w1


def create_torch_w2(L, E, N, K):
    """
    Create torch w2 weight tensor.

    Args:
        L: Number of layers
        E: Number of experts
        N: Intermediate dimension
        K: Output dimension

    Returns:
        torch_w2: Tensor of shape (L, E, N, K)
    """
    torch_w2 = torch.empty((L, E, N, K), dtype=torch.bfloat16)
    le_val = 1
    for l in range(L):
        for e in range(E):
            for n_chunk in range(N // 32):
                n_start, n_end = n_chunk * 32, n_chunk * 32 + 32
                n_val = 0.001 * n_chunk
                for k_chunk in range(K // 32):
                    k_start, k_end = k_chunk * 32, k_chunk * 32 + 32
                    k_val = k_chunk
                    torch_w2[l, e, n_start:n_end, k_start:k_end] = le_val + n_val + k_val
            le_val += 128
    # torch_w2 = torch.randn((L, E, N, K), dtype=torch.bfloat16)
    return torch_w2


def prepare_w0_w1_tensor(torch_w0, torch_w1, L, E, K, N, num_dram_banks):
    """
    Prepare the w0_w1 tensor by interleaving chunks of w0 and w1 width-wise.

    Args:
        torch_w0: Weight tensor of shape (L, E, K, N)
        torch_w1: Weight tensor of shape (L, E, K, N)
        L: Number of layers
        E: Number of experts
        K: Input dimension
        N: Output dimension

    Returns:
        torch_w0_w1_interleaved: Interleaved tensor of shape (L, E, K, 4096)
    """
    Nt = N // ttnn.TILE_SIZE  # 2048 / 32 = 64 chunks per tensor

    # Reshape to expose chunks: (L, E, K, N) -> (L, E, K, Nt, ttnn.TILE_SIZE)
    w0_chunks = torch_w0.view(L, E, K, Nt, ttnn.TILE_SIZE)
    w1_chunks = torch_w1.view(L, E, K, Nt, ttnn.TILE_SIZE)

    # Stack w0 and w1 chunks together: (L, E, K, Nt, 2, ttnn.TILE_SIZE)
    # This puts w0_chunk_i and w1_chunk_i adjacent to each other
    stacked = torch.stack([w0_chunks, w1_chunks], dim=4)

    # Reshape to interleave: (L, E, K, Nt * 2 * ttnn.TILE_SIZE) = (L, E, K, 4096)
    # The order will be: w0_chunk_0, w1_chunk_0, w0_chunk_1, w1_chunk_1, ...
    torch_w0_w1_interleaved = stacked.view(L, E, K, Nt, 2 * ttnn.TILE_SIZE)

    # Permute to move Nt before K: (L, E, K, Nt, 2*TILE) -> (L, E, Nt, K, 2*TILE)
    torch_w0_w1_permuted = torch_w0_w1_interleaved.permute(0, 1, 3, 2, 4)

    # Split Nt dimension into two groups: first 40 and next 24
    # Shape: (L, E, Nt, K, 2*TILE) -> group_1: (L, E, 40, K, 2*TILE), group_2: (L, E, 24, K, 2*TILE)
    group_1 = torch_w0_w1_permuted[:, :, :40, :, :]  # (L, E, 40, K, 64)
    group_2 = torch_w0_w1_permuted[:, :, 40:, :, :]  # (L, E, 24, K, 64)

    # Add Nt=6 padding to group_1: insert Nt=1 padding after every Nt=5 data
    group_1_per_bank = group_1.view(L, E, 8, 5, K, 2 * ttnn.TILE_SIZE)
    padding = torch.zeros(L, E, 8, 1, K, 2 * ttnn.TILE_SIZE, dtype=group_1.dtype)
    group1_with_pad = torch.cat([group_1_per_bank, padding], dim=3)  # (L, E, 8, 6, K, 64)
    group1_with_pad = group1_with_pad.view(L, E, -1, K, 2 * ttnn.TILE_SIZE)  # (L, E, 48, K, 64)

    all_groups = torch.cat([group1_with_pad, group_2], dim=2)  # (L, E, 48 + 24, K, 64)
    all_groups_per_bank = all_groups.view(L, E, 12, -1, K, 2 * ttnn.TILE_SIZE)  # (L, E, 12, 6, K, 64)
    all_groups_per_bank = all_groups_per_bank.permute(2, 0, 1, 3, 4, 5)  # (12, L, E, 6, K, 64)

    return all_groups_per_bank


def prepare_w2_tensor(torch_w2, L, E, N, K, num_dram_banks):
    """
    Prepare the w2 tensor by padding and reordering tiles.

    Args:
        torch_w2: Weight tensor of shape (L, E, N, K)
        L: Number of layers
        E: Number of experts
        N: Intermediate dimension
        K: Output dimension

    Returns:
        torch_w2_reordered: Reordered tensor of shape (L, E, N_padded, 7680)
    """
    Kt = K // ttnn.TILE_SIZE  # 7168 / 32 = 224 chunks per tensor

    # Group Kt in pairs of 2: (L, E, N, 112, 2 * 32)
    torch_w2_grouped = torch_w2.view(L, E, N, -1, 2 * ttnn.TILE_SIZE)

    # Permute to move Kt before N: (L, E, N, Kt, 2*TILE) -> (L, E, 112, N, 2*TILE)
    torch_w2_permuted = torch_w2_grouped.permute(0, 1, 3, 2, 4)

    # Split Kt dimension into two groups: first
    # Shape: (L, E, Kt, N, 2*TILE) -> group_1: (L, E, 80, N, 2*TILE), group_2: (L, E, 32, N, 2*TILE)
    group_1 = torch_w2_permuted[:, :, :80, :, :]  # (L, E, 80, N, 64)
    group_2 = torch_w2_permuted[:, :, 80:, :, :]  # (L, E, 32, N, 64)

    # Add Kt=2 padding to group_2: insert Kt=1 padding after every Kt=8 data
    group_2_per_bank = group_2.view(L, E, 4, 8, N, 2 * ttnn.TILE_SIZE)
    padding = torch.zeros(L, E, 4, 2, N, 2 * ttnn.TILE_SIZE, dtype=torch_w2.dtype)
    group2_with_pad = torch.cat([group_2_per_bank, padding], dim=3)  # (L, E, 4, 10, N, 64)
    group2_with_pad = group2_with_pad.view(L, E, -1, N, 2 * ttnn.TILE_SIZE)  # (L, E, 40, N, 64)

    all_groups = torch.cat([group_1, group2_with_pad], dim=2)  # (L, E, 80 + 40, N, 64)
    all_groups_per_bank = all_groups.view(L, E, num_dram_banks, -1, N, 2 * ttnn.TILE_SIZE)  # (L, E, 12, 10, N, 64)
    all_groups_per_bank = all_groups_per_bank.permute(2, 0, 1, 3, 4, 5)  # (12, L, E, 10, N, 64)

    # Pad "N" dimension to make it divisible by 7 tiles, since we read 7*2 tiles at a time.
    Nt = N // ttnn.TILE_SIZE  # 2048 / 32 = 64 chunks per tensor
    N_padding = math.ceil(Nt / 7) * 7 * ttnn.TILE_SIZE - N
    padding = torch.zeros(num_dram_banks, L, E, 10, N_padding, 2 * ttnn.TILE_SIZE, dtype=torch_w2.dtype)
    all_groups_per_bank = torch.cat([all_groups_per_bank, padding], dim=4)  # (12, L, E, 10, N + 192, 64)
    return all_groups_per_bank


def get_accuracy_metrics(torch_output, tt_output):
    _pcc_passed, pcc_val = comp_pcc(torch_output, tt_output)
    std = torch_output.std().item()
    relative_rmse_val = (torch.nn.functional.mse_loss(torch_output, tt_output).sqrt().item() / std) if std != 0 else 0.0
    return {
        "pcc": pcc_val,
        "relative_rmse": relative_rmse_val,
    }


def run_test_moe(device, M, K, N, check_accuracy, dump_outputs):
    logger.info(
        f"Running test_moe with M={M}, K={K}, N={N}, check_accuracy={check_accuracy}, dump_outputs={dump_outputs}"
    )

    # --------------------------------------------------------------------------
    # Shard grid
    # --------------------------------------------------------------------------
    # TODO(nsoraba): We can restrict this to just the cores that are used, replicating it over all cores for now.
    in0_core_grid = device.compute_with_storage_grid_size()
    in0_num_cores = in0_core_grid.x * in0_core_grid.y

    # output is L1 sharded, the exact number of cores are kind of flexible
    # It just needs to divide the output shape (M, N) evenly.oo
    out_core_grid = ttnn.CoreGrid(x=8, y=8)

    # --------------------------------------------------------------------------
    # Constants
    # --------------------------------------------------------------------------
    in0_dtype = ttnn.bfloat16
    w0_dtype = ttnn.bfloat4_b

    if check_accuracy:
        torch_input = torch.randn((M, K), dtype=torch.bfloat16)
        torch_w0 = torch.randn((K, N), dtype=torch.bfloat16)
        torch_w1 = torch.randn((K, N), dtype=torch.bfloat16)
        torch_w2 = torch.randn((N, K), dtype=torch.bfloat16)

    # Each core gets a copy of the original (E * M, K) input
    input_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=input_shape,
        core_grid=ttnn.CoreGrid(x=in0_core_grid.x, y=in0_core_grid.y),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    # Create WIDTH_SHARDED memory config for output (E * M, N)
    output_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=output_shape,
        core_grid=ttnn.CoreGrid(x=out_core_grid.x, y=out_core_grid.y),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    # ------------------------------------------------------------------------
    # Create DRAM shard spec for w0_w1
    # Tensor shape: (L, E, K, 4608) -> padded and reordered to (12, L, E, 6, K, 64)
    # ------------------------------------------------------------------------
    w0_w1_shard_height = L * E * 6 * K
    w0_w1_shard_width = 64

    w0_w1_shard_spec = ttnn.ShardSpec(
        dram_grid, (w0_w1_shard_height, w0_w1_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )

    w0_w1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w0_w1_shard_spec)

    # ------------------------------------------------------------------------
    # Create DRAM shard spec for w2
    # Tensor shape: (L, E, N, K) -> padded and reordered to (12, L, E, 10, N + 192, 64)
    # ------------------------------------------------------------------------
    w2_shard_height = L * E * 10 * (N + 192)
    w2_shard_width = 64

    w2_shard_spec = ttnn.ShardSpec(dram_grid, (w2_shard_height, w2_shard_width), ttnn.ShardOrientation.ROW_MAJOR)

    w2_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w2_shard_spec)

    # --------------------------------------------------------------------------
    # Prepare the tensors
    # --------------------------------------------------------------------------
    if check_accuracy:
        torch_input = create_torch_input(L, in0_num_cores, E, M, K)
        torch_w0 = create_torch_w0(L, E, K, N)
        torch_w1 = create_torch_w1(L, E, K, N)
        torch_w2 = create_torch_w2(L, E, N, K)

        # ------------------------------------------------------------------------
        # Prepare w0_w1 tensor (interleaved, padded, and reordered)
        torch_w0_w1_reordered = prepare_w0_w1_tensor(torch_w0, torch_w1, L, E, K, N, num_dram_banks)

        # Create tt_w0_w1 tensor with DRAM sharding
        tt_w0_w1 = ttnn.from_torch(
            torch_w0_w1_reordered,
            dtype=w0_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight0_1_shard_memory_config,
        )

        # ------------------------------------------------------------------------
        # Prepare w2 tensor (padded and reordered)
        torch_w2_reordered = prepare_w2_tensor(torch_w2, L, E, N, K, num_dram_banks)

        # Create tt_w2 tensor with DRAM sharding
        tt_w2 = ttnn.from_torch(
            torch_w2_reordered, dtype=w0_dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=w2_mem_config
        )
    else:
        tt_input = ttnn.empty(
            input_shape,
            dtype=in0_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=input_sharded_mem_config,
        )
        tt_weight0 = ttnn.empty((K, N), dtype=w0_dtype, device=device, layout=ttnn.TILE_LAYOUT)
        tt_weight1 = ttnn.empty((K, N), dtype=w0_dtype, device=device, layout=ttnn.TILE_LAYOUT)
        tt_weight2 = ttnn.empty((N, K), dtype=w0_dtype, device=device, layout=ttnn.TILE_LAYOUT)
        # Output is sharded (32, 2048) with each core having one tile (32x32)

    tt_output = ttnn.empty(
        output_shape,
        dtype=in0_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=output_sharded_mem_config,
    )

    # --------------------------------------------------------------------------
    # Run the operation
    # --------------------------------------------------------------------------
    # Collect accuracy metrics for all layers and experts
    all_accuracy_metrics = []

    for layer_id in range(L):
        if check_accuracy:
            tt_input = ttnn.from_torch(
                torch_input[layer_id],
                dtype=in0_dtype,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=input_sharded_mem_config,
            )

        tt_output = ttnn.experimental.moe(
            tt_input,
            w0_w1_tensor=tt_w0_w1,
            w2_tensor=tt_w2,
            output_tensor=tt_output,
            num_experts=E,
            layer_id=layer_id,
        )
        tt_to_torch_output = ttnn.to_torch(tt_output)

    # if check_accuracy:
    #     with torch.no_grad():
    #         # Reference calculation to match TT output shape (2*M, N) = (E*M, N)
    #         # Use first 2*M rows of input (one copy of the original replicated input)
    #         torch_input_2m = torch_input[: 2 * M]  # (2*M, K)

    #         # Compute gate activations for each expert
    #         # (2*M, K) @ (E, K, N) -> broadcasts to (E, 2*M, N)
    #         torch_w0_output = torch.nn.functional.silu(torch_input_2m @ torch_w0[layer_id])  # (E, 2*M, N)
    #         torch_w1_output = torch_input_2m @ torch_w1[layer_id]  # (E, 2*M, N)
    #         torch_intermediate = torch_w0_output * torch_w1_output  # (E, 2*M, N)

    #         # Reshape to match TT output (E*M, N) = (2*M, N)
    #         # Each expert produces M rows, stacked vertically
    #         torch_ref_output = torch_intermediate[:, :M, :].reshape(E * M, N)  # (2*M, N)

    #     # Calculate accuracy metrics for each expert
    #     for expert_id in range(E):
    #         expert_start = expert_id * M
    #         expert_end = (expert_id + 1) * M
    #         torch_expert_output = torch_ref_output[expert_start:expert_end, :]
    #         tt_expert_output = tt_to_torch_output[expert_start:expert_end, :]

    #         expert_metrics = get_accuracy_metrics(torch_expert_output, tt_expert_output)
    #         expert_metrics["layer_id"] = layer_id
    #         expert_metrics["expert_id"] = expert_id
    #         all_accuracy_metrics.append(expert_metrics)

    #         logger.info(
    #             f"Layer {layer_id}, Expert {expert_id}: PCC={expert_metrics['pcc']:.6f}, "
    #             f"Relative RMSE={expert_metrics['relative_rmse']:.6f}"
    #         )

    # if dump_outputs:
    #     torch.set_printoptions(profile="full")
    #     var2filename = {
    #         torch_w0_output: f"layer_{layer_id}_torch_w0_output.txt",
    #         torch_w1_output: f"layer_{layer_id}_torch_w1_output.txt",
    #         torch_intermediate: f"layer_{layer_id}_torch_intermediate.txt",
    #         torch_ref_output: f"layer_{layer_id}_torch_ref_output.txt",
    #         tt_to_torch_output: f"layer_{layer_id}_tt_output.txt",
    #     }
    #     for var, filename in var2filename.items():
    #         with open(filename, "w") as f:
    #             f.write(str(var))

    # if check_accuracy:
    #     # Aggregate metrics across all layers and experts
    #     min_pcc = min(m["pcc"] for m in all_accuracy_metrics)
    #     max_relative_rmse = max(m["relative_rmse"] for m in all_accuracy_metrics)
    #     return {
    #         "pcc": min_pcc,
    #         "relative_rmse": max_relative_rmse,
    #         "all_metrics": all_accuracy_metrics,
    #     }
    return {}


SHAPE2TIME = {
    (32, 7168, 2048, 2, 1): 290.0,
}


@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW,
            },
            id="dispatch_row",
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "M, K, N",
    SHAPE2TIME.keys(),
)
@pytest.mark.parametrize("check_accuracy", [True, False], ids=["check_accuracy_True", "check_accuracy_False"])
@pytest.mark.parametrize("dump_outputs", [True, False], ids=["dump_outputs_True", "dump_outputs_False"])
def test_moe(device, M, K, N, check_accuracy, dump_outputs):
    accuracy_metrics = run_test_moe(
        device,
        M,
        K,
        N,
        check_accuracy,
        dump_outputs,
    )

    # if check_accuracy:
    #     assert accuracy_metrics["pcc"] > 0.999_500
    #     assert accuracy_metrics["relative_rmse"] < 0.02


@pytest.mark.parametrize(
    "M, K, N",
    SHAPE2TIME.keys(),
)
@pytest.mark.parametrize("check_accuracy", [True, False], ids=["check_accuracy_True", "check_accuracy_False"])
@pytest.mark.parametrize("dump_outputs", [True, False], ids=["dump_outputs_True", "dump_outputs_False"])
def test_moe_performance(M, K, N, check_accuracy, dump_outputs):
    command = f"pytest tests/ttnn/nightly/unit_tests/operations/experimental/test_moe.py::test_moe[dump_outputs_{dump_outputs}-check_accuracy_{check_accuracy}-M={M}-K={K}-N={N}-dispatch_row]"
    run_device_profiler(command, "ttnn_moe_performance", device_analysis_types=["device_kernel_duration"])
    r = post_process_ops_log("ttnn_moe_performance", float_columns=["DEVICE KERNEL DURATION [ns]"])
    duration_us = r["DEVICE KERNEL DURATION [ns]"].mean() / 1000.0
    logger.info(f"Duration per layer: {duration_us / L} us")
    logger.info(f"Duration per layer per expert: {duration_us / L / E} us")
    logger.warning(f"Total Duration: {duration_us} us")

    bytes_per_tile = 512 + 64  # bfloat4_b
    tiles_per_txn = 14
    num_cores = 12

    Kt, Nt = math.ceil(K / ttnn.TILE_SIZE), math.ceil(N / ttnn.TILE_SIZE)
    w0_w1_padded_tiles_per_core = 2 * math.ceil(Nt / num_cores) * Kt
    w2_padded_tiles_per_core = 2 * math.ceil(Kt / num_cores / 2) * (math.ceil(Nt / 7) * 7)
    total_padded_tiles_per_core = w0_w1_padded_tiles_per_core + w2_padded_tiles_per_core

    total_bytes_transferred = L * E * num_cores * total_padded_tiles_per_core * bytes_per_tile
    realized_bandwidth = int(total_bytes_transferred / (duration_us * 1000))
    logger.warning(f"Realized Bandwidth: {realized_bandwidth} GB/s")

    total_tiles_0_1 = Kt * Nt
    total_tiles_2 = Nt * Kt
    total_tiles_per_core = 2 * total_tiles_0_1 + total_tiles_2
    total_bytes_used = L * E * total_tiles_per_core * bytes_per_tile
    bandwidth = int(total_bytes_used / (duration_us * 1000))
    logger.warning(f"Useful Bandwidth: {bandwidth} GB/s")

    assert (
        duration_us < SHAPE2TIME[(M, K, N)]
    ), f"Performance {duration_us} us is greater than expected {SHAPE2TIME[(M, K, N)]} us"


def post_process_ops_log(output_logs_subdir: str, float_columns: list[str]) -> dict[str, float]:
    filename = get_latest_ops_log_filename(output_logs_subdir)

    import pandas as pd

    df = pd.read_csv(filename)
    return {col: df[col].astype(float).to_numpy() for col in float_columns}
