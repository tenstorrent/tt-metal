# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import itertools
import math
import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc, comp_allclose

from tracy.process_model_log import (
    get_latest_ops_log_filename,
    run_device_profiler,
)

PCC_THRESHOLD = 0.988

# Even-indexed ring positions get 8 N-tiles, odd get 7 (GPT-OSS 2880x2880 distribution)
FULL_CORES = {0, 2, 4, 6, 8, 10}
PAD_CORES = {1, 3, 5, 7, 9, 11}


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
        torch_input: Tensor of shape (L, in0_num_cores, 2, M, K)
    """
    # torch_input = torch.empty((L, in0_num_cores, E, M, K), dtype=torch.bfloat16)
    # le_val = 1
    # for layer in range(L):
    #     for expert in range(E):
    #         for k_chunk_id in range(K // 32):
    #             k_start, k_end = k_chunk_id * 32, k_chunk_id * 32 + 32
    #             chunk_value = le_val * 0.001 * k_chunk_id
    #             torch_input[layer, :, expert, :, k_start:k_end] = chunk_value
    #         le_val *= -1
    # torch_input = 0.25 * 0.25 *torch.ones((L, in0_num_cores, E, M, K), dtype=torch.bfloat16)
    # torch_input = torch.empty((L, in0_num_cores, E, M, K), dtype=torch.bfloat16)
    # k_half = K // 2
    # # Interleave the positive and negatives
    # for i in range(K):
    #     if i % 2 == 0:
    #         torch_input[..., i] = 0.25
    #     else:
    #         torch_input[..., i] = -0.25
    # torch_input = (1 / 1024) * torch.ones((L, in0_num_cores, 2, M, K), dtype=torch.bfloat16)
    torch_input = torch.rand((L, E, M, K), dtype=torch.bfloat16) - 0.5
    torch_input = torch_input.unsqueeze(1).repeat(1, in0_num_cores, 1, 1, 1)
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
    # torch_w0 = torch.empty((L, E, K, N), dtype=torch.bfloat16)
    # le_val = 1
    # for l in range(L):
    #     for e in range(E):
    #         for k_chunk in range(K // 32):
    #             k_start, k_end = k_chunk * 32, k_chunk * 32 + 32
    #             k_val = k_chunk * 0.001
    #             for n_chunk in range(N // 32):
    #                 n_start, n_end = n_chunk * 32, n_chunk * 32 + 32
    #                 n_val = n_chunk
    #                 torch_w0[l, e, k_start:k_end, n_start:n_end] = (n_val + k_val) * le_val
    #         le_val *= -1

    torch_w0 = torch.rand((L, E, K, N), dtype=torch.bfloat16) - 0.5
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
    # torch_w1 = torch.empty((L, E, K, N), dtype=torch.bfloat16)
    # le_val = -1
    # for l in range(L):
    #     for e in range(E):
    #         for k_chunk in range(K // 32):
    #             k_start, k_end = k_chunk * 32, k_chunk * 32 + 32
    #             k_val = k_chunk * 0.001
    #             for n_chunk in range(N // 32):
    #                 n_start, n_end = n_chunk * 32, n_chunk * 32 + 32
    #                 n_val = n_chunk
    #                 torch_w1[l, e, k_start:k_end, n_start:n_end] = (n_val + k_val) * le_val
    #         le_val *= -1

    torch_w1 = torch.rand((L, E, K, N), dtype=torch.bfloat16) - 0.5
    return torch_w1


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
    # torch_w2 = torch.empty((L, E, N, K), dtype=torch.bfloat16)
    # le_val = 1
    # for l in range(L):
    #     for e in range(E):
    #         for n_chunk in range(N // 32):
    #             n_start, n_end = n_chunk * 32, n_chunk * 32 + 32
    #             n_val = 0.001 * n_chunk
    #             for k_chunk in range(K // 32):
    #                 k_start, k_end = k_chunk * 32, k_chunk * 32 + 32
    #                 k_val = k_chunk
    #                 torch_w2[l, e, n_start:n_end, k_start:k_end] = (n_val + k_val) * le_val
    #         le_val *= -1
    torch_w2 = torch.rand((L, E, N, K), dtype=torch.bfloat16) - 0.5
    return torch_w2


def prepare_w0_w1_tensor(torch_w0, torch_w1, L, E, K, N, ring2cores):
    """
    Prepare the w0_w1 tensor by interleaving w0 and w1 width-wise, distributing to cores.
    GPT-OSS: even cores get 8 N-tiles, odd get 7. TILES_PER_TXN=10, tiles_per_block=20.
    """
    Nt = N // ttnn.TILE_SIZE  # 90 tile-columns

    # Reshape to expose tile-columns: (L, E, K, Nt, TILE_SIZE)
    w0_chunks = torch_w0.view(L, E, K, Nt, ttnn.TILE_SIZE)
    w1_chunks = torch_w1.view(L, E, K, Nt, ttnn.TILE_SIZE)

    # Interleave w0/w1: (L, E, K, Nt, 2, TILE_SIZE) -> (L, E, K, Nt, 2*TILE_SIZE)
    stacked = torch.stack([w0_chunks, w1_chunks], dim=4)
    torch_w0_w1_interleaved = stacked.view(L, E, K, Nt, 2 * ttnn.TILE_SIZE)

    # Move Nt before K: (L, E, Nt, K, 2*TILE_SIZE)
    torch_w0_w1_permuted = torch_w0_w1_interleaved.permute(0, 1, 3, 2, 4)

    each_shard = []
    max_tiles_per_core = max(7 if ring2cores[rp][2] else 8 for rp in range(len(ring2cores)))

    # Distribute N-tiles to cores, padding to max_tiles_per_core
    start_tile = 0
    for ring_pos in range(len(ring2cores)):
        (_, _, pad_flag) = ring2cores[ring_pos]
        num_tiles = 7 if pad_flag else 8
        each_shard.append(torch_w0_w1_permuted[:, :, start_tile : start_tile + num_tiles, :, :])
        pad_count = max_tiles_per_core - num_tiles
        if pad_count > 0:
            each_shard.append(torch.zeros(L, E, pad_count, K, 2 * ttnn.TILE_SIZE, dtype=torch_w0_w1_permuted.dtype))
        start_tile += num_tiles

    # (L, E, 12*max_tiles, K, 2*TILE_SIZE) -> (12, L, E, max_tiles, K, 2*TILE_SIZE)
    torch_w0_w1_reordered = torch.cat(each_shard, dim=2)
    all_groups_per_bank = torch_w0_w1_reordered.view(L, E, 12, max_tiles_per_core, K, 2 * ttnn.TILE_SIZE)
    all_groups_per_bank = all_groups_per_bank.permute(2, 0, 1, 3, 4, 5)

    # Pair tiles: group into pairs of 2, making 4*TILE_SIZE columns
    pairs = max_tiles_per_core // 2  # 8/2=4
    torch_w0_w1_pair_2_tiles = all_groups_per_bank.view(12, L, E, pairs, 2, K, 2 * ttnn.TILE_SIZE)
    torch_w0_w1_pair_2_tiles = torch_w0_w1_pair_2_tiles.permute(0, 1, 2, 3, 5, 4, 6)
    torch_w0_w1_paired = torch_w0_w1_pair_2_tiles.reshape(12, L, E, pairs, K, 4 * ttnn.TILE_SIZE)

    return torch_w0_w1_paired


def prepare_w2_tensor(torch_w2, L, E, N, K, ring2cores):
    """
    Prepare the w2 tensor: distribute K-columns across cores, reorder N-rows for ring A2A.
    GPT-OSS: K=N=2880, each core gets 8 or 7 K-tiles. No N-padding (90/10=9 exact).
    """
    Kt = K // ttnn.TILE_SIZE  # 90
    Nt = N // ttnn.TILE_SIZE  # 90
    num_cores = len(ring2cores)

    tiles_per_core = [7 if ring2cores[rp][2] else 8 for rp in range(num_cores)]
    max_tiles = max(tiles_per_core)  # 8

    # Distribute K-columns to cores with padding
    each_shard = []
    start_col = 0
    for ring_pos in range(num_cores):
        ntiles = tiles_per_core[ring_pos]
        each_shard.append(torch_w2[:, :, :, start_col : start_col + ntiles * ttnn.TILE_SIZE])
        start_col += ntiles * ttnn.TILE_SIZE
        pad_count = max_tiles - ntiles
        if pad_count > 0:
            each_shard.append(torch.zeros(L, E, N, pad_count * ttnn.TILE_SIZE, dtype=torch_w2.dtype))

    torch_w2_distributed = torch.cat(each_shard, dim=-1)
    num_col_groups = max_tiles // 4  # 8/4=2
    all_groups = torch_w2_distributed.view(L, E, N, num_cores, num_col_groups, 4 * ttnn.TILE_SIZE)
    all_groups = all_groups.permute(3, 0, 1, 4, 2, 5)  # (12, L, E, 2, N, 4*TILE)

    # Group N into tiles for reordering
    N_grouped = all_groups.view(num_cores, L, E, num_col_groups, Nt, ttnn.TILE_SIZE, 4 * ttnn.TILE_SIZE)

    # Reorder N-tiles for ring A2A: each core needs N in reverse-ring order
    core_chunk_order = torch.tensor(list(reversed(range(num_cores)))).roll(1)
    chunk_sizes = [7 if ring2cores[rp][2] else 8 for rp in range(num_cores)]
    chunk_start_positions = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(torch.tensor(chunk_sizes, dtype=torch.int32), dim=0)]
    )

    each_shard = []
    for core_id in range(num_cores):
        each_chunk = []
        for chunk_id in core_chunk_order:
            start_pos = chunk_start_positions[chunk_id]
            end_pos = chunk_start_positions[chunk_id + 1]
            this_chunk = N_grouped[core_id, :, :, :, start_pos:end_pos, :, :]
            each_chunk.append(this_chunk)
        each_shard.append(torch.cat(each_chunk, dim=3))
        core_chunk_order = core_chunk_order.roll(1)

    N_reordered = torch.stack(each_shard)
    result = N_reordered.view(num_cores, L, E, num_col_groups, N, 4 * ttnn.TILE_SIZE)
    return result


def prepare_output_tensor(tt_output, E, M, K, ring2cores):
    """
    Extract valid output tiles per core.
    GPT-OSS: each core computes 8 output tiles, but pad cores only have 7 valid.
    """
    each_shard = []
    for ring_pos in range(len(ring2cores)):
        (_, _, pad_flag) = ring2cores[ring_pos]
        num_tiles = 7 if pad_flag else 8
        each_shard.append(tt_output[ring_pos, :, :, : num_tiles * ttnn.TILE_SIZE])

    result = torch.cat(each_shard, dim=-1)
    assert result.shape[-2:] == (M, K)
    return result


def get_accuracy_metrics(torch_output, tt_output):
    _pcc_passed, pcc_val = comp_pcc(torch_output, tt_output)
    std = torch_output.std().item()
    relative_rmse_val = (torch.nn.functional.mse_loss(torch_output, tt_output).sqrt().item() / std) if std != 0 else 0.0
    allclose_passed, allclose_val = comp_allclose(torch_output, tt_output)
    return {
        "pcc": pcc_val,
        "relative_rmse": relative_rmse_val,
        "allclose": allclose_passed,
        "allclose_val": allclose_val,
    }


def run_test_moe(device, M, K, N, E, L, check_accuracy, dump_outputs):
    logger.info(
        f"Running test_moe with M={M}, K={K}, N={N}, E={E}, L={L}, check_accuracy={check_accuracy}, dump_outputs={dump_outputs}"
    )

    # --------------------------------------------------------------------------
    # Shard grid
    # --------------------------------------------------------------------------
    in0_core_coords = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(device, 0)
    core2dram = {}
    for dram_bank_id, core_coords in enumerate(in0_core_coords):
        core2dram[core_coords] = dram_bank_id

    in0_num_cores = len(in0_core_coords)

    # Make a new list of core coords that are sorted in decreasing order by y coordinate and then x coordinate.
    in0_core_coords_sorted = sorted(in0_core_coords, key=lambda x: (x.y, x.x), reverse=True)

    ring2cores = {}
    for ring_pos, core_coord in enumerate(in0_core_coords_sorted):
        # key: ring_pos, value: (core_coord, dram_bank_id, pad_flag)
        ring2cores[ring_pos] = (core_coord, core2dram[core_coord], 1 if ring_pos in PAD_CORES else 0)

    in0_core_range = [ttnn.CoreRange(ring2cores[i][0], ring2cores[i][0]) for i in range(in0_num_cores)]
    in0_core_range_set = ttnn.CoreRangeSet(in0_core_range)

    # --------------------------------------------------------------------------
    # Constants
    # --------------------------------------------------------------------------
    in0_dtype = ttnn.bfloat16
    w0_dtype = ttnn.bfloat4_b
    num_dram_banks = 12

    dram_core_coords = [ttnn.CoreCoord(ring2cores[i][1], 0) for i in range(in0_num_cores)]
    dram_core_range = [ttnn.CoreRange(dram_core_coord, dram_core_coord) for dram_core_coord in dram_core_coords]
    dram_core_range_set = ttnn.CoreRangeSet(dram_core_range)

    # --------------------------------------------------------------------------
    # Tensor shapes and memory configurations
    # --------------------------------------------------------------------------
    # Define tensor shapes - same for both accuracy and performance testing
    input_shape = (in0_num_cores, E, M, K)

    in0_shard_spec = ttnn.ShardSpec(
        grid=in0_core_range_set,
        shard_shape=(E * M, K),  # E experts, M tokens each
        shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    # Each core gets a copy of the original (2 * M, K) input
    input_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in0_shard_spec
    )

    # ------------------------------------------------------------------------
    # Create DRAM shard spec for w0_w1
    # Tensor shape: (L, E, K, 4608) -> padded and reordered to (12, L, E, 6, K, 64)
    # ------------------------------------------------------------------------
    w0_w1_shard_height = L * E * 4 * K
    w0_w1_shard_width = 4 * ttnn.TILE_SIZE

    w0_w1_shard_spec = ttnn.ShardSpec(
        dram_core_range_set, (w0_w1_shard_height, w0_w1_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )

    w0_w1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w0_w1_shard_spec)

    # ------------------------------------------------------------------------
    # Create DRAM shard spec for w2
    # Tensor shape: (L, E, N, K) -> distributed to (12, L, E, 2, N, 128) for GPT-OSS
    # ------------------------------------------------------------------------
    w2_shard_height = L * E * 2 * N
    w2_shard_width = 4 * ttnn.TILE_SIZE

    w2_shard_spec = ttnn.ShardSpec(
        dram_core_range_set, (w2_shard_height, w2_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )

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
        torch_w0_w1_reordered = prepare_w0_w1_tensor(torch_w0, torch_w1, L, E, K, N, ring2cores)

        # Create tt_w0_w1 tensor with DRAM sharding
        tt_w0_w1 = ttnn.from_torch(
            torch_w0_w1_reordered,
            dtype=w0_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=w0_w1_mem_config,
        )

        # ------------------------------------------------------------------------
        # Prepare w2 tensor (padded and reordered)
        torch_w2_reordered = prepare_w2_tensor(torch_w2, L, E, N, K, ring2cores)

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
        tt_w0_w1 = ttnn.empty(
            [num_dram_banks] + w0_w1_shard_spec.shape,
            dtype=w0_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=w0_w1_mem_config,
        )
        tt_w2 = ttnn.empty(
            [num_dram_banks] + w2_shard_spec.shape,
            dtype=w0_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=w2_mem_config,
        )

    # --------------------------------------------------------------------------
    # Run the operation
    # --------------------------------------------------------------------------
    # Collect accuracy metrics for all layers and experts
    all_outputs = []
    all_accuracy_metrics = {}

    for layer_id in range(L):
        if check_accuracy:
            tt_input = ttnn.from_torch(
                torch_input[layer_id],
                dtype=in0_dtype,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=input_sharded_mem_config,
            )

        _tt_output = ttnn.experimental.moe_gpt_oss(
            tt_input,
            w0_w1_tensor=tt_w0_w1,
            w2_tensor=tt_w2,
            output_tensor=tt_input,
            num_experts=E,
            layer_id=layer_id,
        )

        # Output is produced in-place on the input tensor
        tt_raw_output = ttnn.to_torch(tt_input)
        tt_to_torch_output = prepare_output_tensor(tt_raw_output, E, M, K, ring2cores)
        all_outputs.append(tt_to_torch_output)

    tt_to_torch_outputs = torch.stack(all_outputs)

    if check_accuracy:
        with torch.no_grad():
            # Reference calculation to match TT output shape (2*M, N) = (E*M, N)
            # Use first 2*M rows of input (one copy of the original replicated input)
            torch_input_ref = torch_input[:, 0, ...]

            # Compute gate (W0) and up (W1) projections for each expert
            # (L, E, M, K) @ (L, E, K, N) -> (L, E, M, N)
            torch_w0_output_ref = torch_input_ref @ torch_w0
            torch_w1_output_ref = torch_input_ref @ torch_w1

            # SwiGLU activation: (clamp(up, +-7) + 1) * clamp(gate, max=7) * sigmoid(1.702 * clamp(gate, max=7))
            alpha = 1.702
            clamp_limit = 7.0
            gate_clamped = torch.clamp(torch_w0_output_ref, max=clamp_limit)
            up_clamped = torch.clamp(torch_w1_output_ref, min=-clamp_limit, max=clamp_limit)
            torch_intermediate_ref = (up_clamped + 1.0) * gate_clamped * torch.sigmoid(alpha * gate_clamped)

            # (L, E, M, N) @ (L, E, N, K) -> (L, E, M, K)
            torch_output_ref = torch_intermediate_ref @ torch_w2

        # Calculate accuracy metrics for each layer and expert
        for layer_id, expert_id in itertools.product(range(L), range(E)):
            torch_layer_output = torch_output_ref[layer_id, expert_id, :, :]
            tt_layer_output = tt_to_torch_outputs[layer_id, expert_id, :, :]

            layer_metrics = get_accuracy_metrics(torch_layer_output, tt_layer_output)
            all_accuracy_metrics[(layer_id, expert_id)] = layer_metrics

    if dump_outputs:
        torch.set_printoptions(profile="full")
        var2filename = {
            torch_w0_output_ref: f"torch_w0_output_ref.txt",
            torch_w1_output_ref: f"torch_w1_output_ref.txt",
            torch_intermediate_ref: f"torch_intermediate_ref.txt",
            torch_output_ref: f"torch_output_ref.txt",
            tt_to_torch_outputs: f"tt_output_act.txt",
        }

        for var, filename in var2filename.items():
            with open(filename, "w") as f:
                f.write(str(var))

    return all_accuracy_metrics


SHAPE2TIME = {
    (32, 2880, 2880, 4, 1): 225.0,
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
    "M, K, N, E, L",
    SHAPE2TIME.keys(),
)
@pytest.mark.parametrize("check_accuracy", [True, False], ids=["check_accuracy_True", "check_accuracy_False"])
@pytest.mark.parametrize("dump_outputs", [True, False], ids=["dump_outputs_True", "dump_outputs_False"])
def test_moe(device, M, K, N, E, L, check_accuracy, dump_outputs):
    accuracy_metrics = run_test_moe(
        device,
        M,
        K,
        N,
        E,
        L,
        check_accuracy,
        dump_outputs,
    )

    passing = True
    # Print the layers and experts that did not pass the PCC check
    for (layer_id, expert_id), metrics in accuracy_metrics.items():
        if metrics["pcc"] < PCC_THRESHOLD:
            passing = False
            logger.warning(f"Layer {layer_id}, Expert {expert_id}: PCC={metrics['pcc']:.6f}")
        else:
            logger.info(f"Layer {layer_id}, Expert {expert_id}: PCC={metrics['pcc']:.6f} (Passed)")

    assert passing, f"Some experts in some layers did not pass the PCC/Allclose check"


@pytest.mark.parametrize(
    "M, K, N, E, L",
    SHAPE2TIME.keys(),
)
@pytest.mark.parametrize("check_accuracy", [True, False], ids=["check_accuracy_True", "check_accuracy_False"])
@pytest.mark.parametrize("dump_outputs", [True, False], ids=["dump_outputs_True", "dump_outputs_False"])
def test_moe_performance(M, K, N, E, L, check_accuracy, dump_outputs):
    command = f"pytest tests/ttnn/nightly/unit_tests/operations/experimental/test_moe.py::test_moe[dump_outputs_{dump_outputs}-check_accuracy_{check_accuracy}-M={M}-K={K}-N={N}-E={E}-L={L}-dispatch_row]"
    run_device_profiler(command, "ttnn_moe_performance", device_analysis_types=["device_kernel_duration"])

    r = post_process_ops_log("ttnn_moe_performance", float_columns=["DEVICE KERNEL DURATION [ns]"])
    duration_us = r["DEVICE KERNEL DURATION [ns]"].sum() / 1000.0
    logger.info(f"Duration per layer: {duration_us / L} us")
    logger.info(f"Duration per layer per expert: {duration_us / L / E} us")
    logger.warning(f"Total Duration: {duration_us} us")

    bytes_per_tile = 512 + 64  # bfloat4_b
    tiles_per_txn = 10  # GPT-OSS
    num_cores = 12

    Kt, Nt = math.ceil(K / ttnn.TILE_SIZE), math.ceil(N / ttnn.TILE_SIZE)
    w0_w1_padded_tiles_per_core = 2 * math.ceil(Nt / num_cores) * Kt
    w2_padded_tiles_per_core = 4 * math.ceil(Kt / num_cores / 4) * Nt  # No N-padding for GPT-OSS
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
        duration_us < SHAPE2TIME[(M, K, N, E, L)]
    ), f"Performance {duration_us} us is greater than expected {SHAPE2TIME[(M, K, N, E, L)]} us"


def post_process_ops_log(output_logs_subdir: str, float_columns: list[str]) -> dict[str, float]:
    filename = get_latest_ops_log_filename(output_logs_subdir)

    import pandas as pd

    df = pd.read_csv(filename)
    return {col: df[col].astype(float).to_numpy() for col in float_columns}
