# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import itertools
import math
import pickle
import hashlib
import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc, comp_allclose

from tracy.process_model_log import (
    get_latest_ops_log_filename,
    run_device_profiler,
)

PCC_THRESHOLD = 0.98


def compute_tensor_checksum(tensor):
    """Compute SHA256 checksum for a tensor."""
    if isinstance(tensor, torch.Tensor):
        # Convert tensor to bytes for hashing
        torch_tensor = tensor.detach().cpu()
    else:
        raise ValueError(f"Unsupported tensor type: {type(tensor)}")

    # Convert BFloat16 and other unsupported dtypes to float32 for numpy conversion
    if torch_tensor.dtype == torch.bfloat16:
        torch_tensor = torch_tensor.float()
    elif torch_tensor.dtype not in [
        torch.float32,
        torch.float64,
        torch.int32,
        torch.int64,
        torch.int16,
        torch.int8,
        torch.uint8,
    ]:
        # Convert any other unsupported dtypes to float32
        torch_tensor = torch_tensor.float()

    tensor_bytes = torch_tensor.numpy().tobytes()
    return hashlib.sha256(tensor_bytes).hexdigest()


def load_or_create_checksum_dict(pickle_file="moe_input_checksums.pkl"):
    """Load checksum dictionary from pickle file, or create empty dict if file doesn't exist."""
    try:
        with open(pickle_file, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        return {}


def save_checksum_dict(checksum_dict, pickle_file="moe_input_checksums.pkl"):
    """Save checksum dictionary to pickle file."""
    with open(pickle_file, "wb") as f:
        pickle.dump(checksum_dict, f)


# Some cores have more tiles than others, but they are sprinkled around the ring for boundary alignment.
FULL_CORES_A = {0, 1, 8, 9}
PAD_CORES_A = {2, 3, 4, 5, 6, 7, 10, 11}

FULL_CORES_B = {0, 3, 6, 9}
PAD_CORES_B = {1, 2, 4, 5, 7, 8, 10, 11}

TOTAL_TOKENS = 512


def create_torch_input(L, in0_core_range_set, all_core_range_set, E, M, K):
    """
    Create torch input tensor with unique integer values per layer/expert.

    Args:
        L: Number of layers
        in0_core_range_set: core coordinates that receive input data shards
        all_core_range_set: all core coordinates
        E: Number of experts
        M: Sequence length
        K: Input dimension

    Returns:
        torch_input: Tensor of shape (L, all_core_range_set.num_cores(), 2, M, K)
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

    torch_input_ref = torch.rand((L, E, M, K), dtype=torch.bfloat16) - 0.5
    #     torch_input = torch.zeros((L, E, M, K), dtype=torch.bfloat16)
    #     for e in range(E):
    #         torch_input[:,e,:2] = torch.rand(K) - 0.5

    # torch_input[:,1,:,:] *= 2

    torch_input = torch_input_ref.unsqueeze(1).repeat(1, in0_core_range_set.num_cores(), 1, 1, 1)
    torch_input_shard_placed = torch.zeros([L, all_core_range_set.num_cores(), E, M, K], dtype=torch.bfloat16)

    # just to be really sure the input cores are ordered consistently
    sorted_all_cores = sorted(ttnn.corerange_to_cores(all_core_range_set), key=lambda x: (x.y, x.x))

    for l in range(L):
        for e in range(E):
            sidx = 0
            for idx, c in enumerate(sorted_all_cores):
                if in0_core_range_set.contains(c):
                    torch_input_shard_placed[l, idx, e, :, :] = torch_input[l, sidx, e, :, :]
                    sidx += 1

    return torch_input_shard_placed, torch_input_ref


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
    #     torch_w0 = torch.empty((L, E, K, N), dtype=torch.bfloat16)
    #     le_val = 1
    #     for l in range(L):
    #         for e in range(E):
    #             for k_chunk in range(K // 32):
    #                 k_start, k_end = k_chunk * 32, k_chunk * 32 + 32
    #                 k_val = k_chunk * 0.001
    #                 for n_chunk in range(N // 32):
    #                     n_start, n_end = n_chunk * 32, n_chunk * 32 + 32
    #                     n_val = n_chunk
    #                     torch_w0[l, e, k_start:k_end, n_start:n_end] = (n_val + k_val) * le_val
    #             le_val *= -1

    torch_w0 = torch.rand((L, E, K, N), dtype=torch.bfloat16) - 0.5
    #     torch_w0 = torch.zeros((L, E, K, N), dtype=torch.bfloat16)
    #
    #     k = min(K,N)
    #     torch_w0[..., torch.arange(k), torch.arange(k)] = 1

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
    #     torch_w1 = torch.empty((L, E, K, N), dtype=torch.bfloat16)
    #     le_val = -1
    #     for l in range(L):
    #         for e in range(E):
    #             for k_chunk in range(K // 32):
    #                 k_start, k_end = k_chunk * 32, k_chunk * 32 + 32
    #                 k_val = k_chunk * 0.001
    #                 for n_chunk in range(N // 32):
    #                     n_start, n_end = n_chunk * 32, n_chunk * 32 + 32
    #                     n_val = n_chunk
    #                     torch_w1[l, e, k_start:k_end, n_start:n_end] = (n_val + k_val) * le_val
    #             le_val *= -1

    torch_w1 = torch.rand((L, E, K, N), dtype=torch.bfloat16) - 0.5
    #  torch_w1 = torch.zeros((L, E, K, N), dtype=torch.bfloat16)
    #     k = min(K,N)
    #     torch_w1[..., torch.arange(k), torch.arange(k)] = 1

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
    #     le_val = 1
    #     for l in range(L):
    #         for e in range(E):
    #             for n_chunk in range(N // 32):
    #                 n_start, n_end = n_chunk * 32, n_chunk * 32 + 32
    #                 n_val = 0.001 * n_chunk
    #                 for k_chunk in range(K // 32):
    #                     k_start, k_end = k_chunk * 32, k_chunk * 32 + 32
    #                     k_val = k_chunk
    #                     torch_w2[l, e, n_start:n_end, k_start:k_end] = (n_val + k_val) * le_val
    #             le_val *= -1

    torch_w2 = torch.rand((L, E, N, K), dtype=torch.bfloat16) - 0.5

    #     torch_w2 = torch.zeros((L, E, N, K), dtype=torch.bfloat16) - 0.5
    #
    #     k = min(K,N)
    #     torch_w2[..., torch.arange(k), torch.arange(k)] = 1

    return torch_w2


def prepare_w0_w1_tensor(torch_w0, torch_w1, L, E, K, N, ring2cores):
    """
    Prepare the w0_w1 tensor by interleaving chunks of w0 and w1 width-wise.

    Args:
        torch_w0: Weight tensor of shape (L, E, K, N)
        torch_w1: Weight tensor of shape (L, E, K, N)
        L: Number of layers
        E: Number of experts
        K: Input dimension
        N: Output dimension
        ring2cores: Dictionary mapping ring position to (core_coord, dram_bank_id, pad_flag)

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

    each_shard = []

    # Pick appropriate number of column tiles for each core based on the ring position.
    start_tile = 0
    for ring_pos in range(len(ring2cores)):
        (_, _, pad_flag) = ring2cores[ring_pos]
        num_tiles = 5 if pad_flag else 6
        each_shard.append(torch_w0_w1_permuted[:, :, start_tile : start_tile + num_tiles, :, :])

        if pad_flag:
            each_shard.append(torch.zeros(L, E, 1, K, 2 * ttnn.TILE_SIZE, dtype=torch_w0_w1_permuted.dtype))
        start_tile += num_tiles

    torch_w0_w1_reordered = torch.cat(each_shard, dim=2)  # (L, E, 5 * 8 + 1 * 8 + 6 * 4, K, 64)
    all_groups_per_bank = torch_w0_w1_reordered.view(L, E, 12, -1, K, 2 * ttnn.TILE_SIZE)  # (L, E, 12, 6, K, 64)
    all_groups_per_bank = all_groups_per_bank.permute(2, 0, 1, 3, 4, 5)  # (12, L, E, 6, K, 64)

    # Let us further make the 6 as 3 and 64 as 128.
    torch_w0_w1_pair_2_tiles = all_groups_per_bank.view(12, L, E, 3, -1, K, 2 * ttnn.TILE_SIZE)
    # (12, L, E, 3, 2, K, 64) -> (12, L, E, 3, K, 2, 64)
    torch_w0_w1_pair_2_tiles = torch_w0_w1_pair_2_tiles.permute(0, 1, 2, 3, 5, 4, 6)
    torch_w0_w1_paired = torch_w0_w1_pair_2_tiles.reshape(12, L, E, 3, -1, 4 * ttnn.TILE_SIZE)

    return torch_w0_w1_paired


def prepare_w2_tensor(torch_w2, L, E, N, K, ring2cores):
    """
    Prepare the w2 tensor by padding and reordering tiles.

    Args:
        torch_w2: Weight tensor of shape (L, E, N, K)
        L: Number of layers
        E: Number of experts
        N: Intermediate dimension
        K: Output dimension
        ring2cores: Dictionary mapping ring position to (core_coord, dram_bank_id, pad_flag)

    Returns:
        torch_w2_reordered: Reordered tensor of shape (L, E, N_padded, 7680)
    """
    # Separate the tensor into 4 groups of 4 * 32 tiles and then 1 group of 2/3 * 32 tiles.
    each_shard = []

    start_col = 0
    for ring_pos in range(len(ring2cores)):
        (_, _, pad_flag) = ring2cores[ring_pos]
        last_group_tiles = 3 if pad_flag else 2
        last_group_pad_tiles = 1 if pad_flag else 2

        # Get the first 4 groups of 4 * 32 tiles.
        each_shard.append(torch_w2[:, :, :, start_col : start_col + 4 * 4 * ttnn.TILE_SIZE])
        start_col += 4 * 4 * ttnn.TILE_SIZE
        each_shard.append(torch_w2[:, :, :, start_col : start_col + last_group_tiles * ttnn.TILE_SIZE])
        start_col += last_group_tiles * ttnn.TILE_SIZE

        # Add padding for the last group.
        each_shard.append(torch.zeros(L, E, N, last_group_pad_tiles * ttnn.TILE_SIZE, dtype=torch_w2.dtype))

    torch_w2_reordered = torch.cat(each_shard, dim=-1)  # (L, E, N, 12 * (4 * 4 * 32 + 4 * 32))
    all_groups_per_bank = torch_w2_reordered.view(L, E, N, 12, -1, 4 * ttnn.TILE_SIZE)

    # (L, E, N, 12, 5, 128) -> (12, L, E, 5, N, 128)
    all_groups_per_bank = all_groups_per_bank.permute(3, 0, 1, 4, 2, 5)

    # Group N in terms of tiles first
    N_grouped = all_groups_per_bank.view(
        12, L, E, 5, -1, ttnn.TILE_SIZE, 4 * ttnn.TILE_SIZE
    )  # (12, L, E, 5, 64, 32, 128)

    # Figure out the order of N tiles based on the ring position.
    core_chunk_order = torch.tensor(list(reversed(range(len(ring2cores))))).roll(1)

    # Figure out the starting position for each chunk
    chunk_sizes = [5 if ring2cores[ring_pos][2] else 6 for ring_pos in range(len(ring2cores))]
    chunk_start_positions = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(torch.tensor(chunk_sizes, dtype=torch.int32), dim=0)]
    )

    each_shard = []
    # Assemble the number of such N tiles based on the ring position.
    for core_id in range(len(ring2cores)):
        each_chunk = []
        for chunk_id in core_chunk_order:
            start_pos = chunk_start_positions[chunk_id]
            end_pos = chunk_start_positions[chunk_id + 1]
            this_chunk = N_grouped[core_id, :, :, :, start_pos:end_pos, :, :]
            each_chunk.append(this_chunk)
        each_shard.append(torch.cat(each_chunk, dim=3))

        core_chunk_order = core_chunk_order.roll(1)

    N_reordered = torch.stack(each_shard).view(12, L, E, 5, -1, 4 * ttnn.TILE_SIZE)

    # Pad "N" dimension to make it divisible by 7 tiles, since we read 7 tiles at a time.
    Nt = N // ttnn.TILE_SIZE  # 2048 / 32 = 64 chunks per tensor
    N_padding = math.ceil(Nt / 7) * 7 * ttnn.TILE_SIZE - N
    padding = torch.zeros(12, L, E, 5, N_padding, 4 * ttnn.TILE_SIZE, dtype=torch_w2.dtype)
    all_groups_per_bank = torch.cat([N_reordered, padding], dim=4)  # (12, L, E, 5, N + 192, 128)
    return all_groups_per_bank


def prepare_output_tensor(tt_output, E, M, K, ring2cores):
    """
    Prepare the output tensor by padding and reordering tiles.

    Args:
        tt_output: Tensor of shape (num_cores, E, M, K)
        E: Number of experts
        M: Number of input features
        K: Number of output features
        ring2cores: Dictionary mapping ring position to (core_coord, dram_bank_id, pad_flag)

    Returns:
        torch_output: Tensor of shape (E, M, K)
    """
    each_shard = []

    for ring_pos in range(len(ring2cores)):
        (_, _, pad_flag) = ring2cores[ring_pos]
        num_tiles = 19 if pad_flag else 18
        each_shard.append(tt_output[ring_pos, :, :, : num_tiles * ttnn.TILE_SIZE])

    result = torch.cat(each_shard, dim=-1)
    assert result.shape == (2, M, K)
    return result


def prepare_output_tensor_from_combine_writer(
    raw_torch_output,
    all_core_range_set,
    output_shard_cores,
    output_shard_height_dim,
    output_shard_width_dim,
    E,
    M,
    K,
):
    # python in doesn't work as expected with list of CoreCoord
    def _output_shard_contains(core):
        for c in output_shard_cores:
            if core.x == c.x and core.y == c.y:
                return True
        return False

    torch.set_printoptions(profile="full")
    output_shards = []
    for i, c in enumerate(ttnn.corerange_to_cores(all_core_range_set, row_wise=True)):
        if _output_shard_contains(c):
            output_shards.append(raw_torch_output[i, :, :, :])

    output_core_shards = torch.stack(output_shards)

    output_shape = (
        output_shard_height_dim,
        output_shard_width_dim,
        E,
        TOTAL_TOKENS // output_shard_height_dim,
        K // output_shard_width_dim,
    )

    shaped_torch_output = output_core_shards.view(output_shape)

    #     for h in range(output_shard_height_dim):
    #         for w in range(output_shard_width_dim):
    #             for t in range(0,1):
    #                 for e in range(E):
    #                     print(f"{h=} {w=} {t=}  {e=}")
    #                     print(f"output shards: {shaped_torch_output[h,w,e,t,:32]}")

    shaped_torch_output = shaped_torch_output.permute([2, 0, 3, 1, 4]).reshape([E, TOTAL_TOKENS, K])
    torch_output = torch.zeros([E, M, K], dtype=torch.bfloat16)

    for e in range(E):
        active_tokens = M  # TODO use dynamic active tokens
        tokens_per_shard = math.ceil(active_tokens / output_shard_height_dim)
        for t in range(active_tokens):
            bt = t // tokens_per_shard
            ot = t % tokens_per_shard

            contrib = shaped_torch_output[e, bt * TOTAL_TOKENS // output_shard_height_dim + ot]

            torch_output[e, t] = contrib

    return torch_output


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

    torch.manual_seed(0)

    # --------------------------------------------------------------------------
    # Shard grid
    # --------------------------------------------------------------------------

    all_core_grid = device.compute_with_storage_grid_size()
    all_core_range_set = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(all_core_grid.x - 1, all_core_grid.y - 1),
            ),
        }
    )
    all_num_cores = all_core_range_set.num_cores()

    in0_core_coords = device.get_optimal_dram_bank_to_logical_worker_assignment(0)
    core2dram = {}
    for dram_bank_id, core_coords in enumerate(in0_core_coords):
        core2dram[core_coords] = dram_bank_id

    in0_num_cores = len(in0_core_coords)

    # Make a new list of core coords that are sorted in decreasing order by y coordinate and then x coordinate.
    in0_core_coords_sorted = sorted(in0_core_coords, key=lambda x: (x.y, x.x), reverse=True)

    ring2cores = {}
    for ring_pos, core_coord in enumerate(in0_core_coords_sorted):
        # key: ring_pos, value: (core_coord, dram_bank_id, pad_flag)
        ring2cores[ring_pos] = (core_coord, core2dram[core_coord], 1 if ring_pos in PAD_CORES_B else 0)

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
    # Output shard configuration
    # --------------------------------------------------------------------------

    output_height_shard_dim = 4
    output_width_shard_dim = 4

    unused_core_range_set = all_core_range_set.subtract(dram_core_range_set).subtract(in0_core_range_set)

    # Gaurantee core iteration order is consistent with ttnn.corerange_to_cores(all_core_range_set, row_wise=True)
    # even if output shard cares span multiple ranges
    sorted_output_shard_cores = sorted(
        ttnn.corerange_to_cores(unused_core_range_set, row_wise=True)[
            : output_height_shard_dim * output_width_shard_dim
        ],
        key=lambda x: (x.y, x.x),
    )

    # --------------------------------------------------------------------------
    # Tensor shapes and memory configurations
    # --------------------------------------------------------------------------
    # Define tensor shapes - same for both accuracy and performance testing
    input_shape = (all_num_cores, 2, M, K)

    in0_shard_spec = ttnn.ShardSpec(
        grid=all_core_range_set,
        shard_shape=(2 * M, K),  # Your shard dimensions.
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
    w0_w1_shard_height = L * E * 3 * K
    w0_w1_shard_width = 4 * ttnn.TILE_SIZE

    w0_w1_shard_spec = ttnn.ShardSpec(
        dram_core_range_set, (w0_w1_shard_height, w0_w1_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )

    w0_w1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w0_w1_shard_spec)

    # ------------------------------------------------------------------------
    # Create DRAM shard spec for w2
    # Tensor shape: (L, E, N, K) -> padded and reordered to (12, L, E, 5, N + 192, 128)
    # ------------------------------------------------------------------------
    w2_shard_height = L * E * 5 * (N + 192)
    w2_shard_width = 4 * ttnn.TILE_SIZE

    w2_shard_spec = ttnn.ShardSpec(
        dram_core_range_set, (w2_shard_height, w2_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )

    w2_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w2_shard_spec)

    # --------------------------------------------------------------------------
    # Prepare the tensors
    # --------------------------------------------------------------------------
    if check_accuracy:
        torch_input, torch_input_ref = create_torch_input(L, in0_core_range_set, all_core_range_set, E, M, K)
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

        tt_output = ttnn.experimental.moe(
            tt_input,
            w0_w1_tensor=tt_w0_w1,
            w2_tensor=tt_w2,
            output_tensor=tt_input,
            hidden_dim=K,
            num_experts=E,
            layer_id=layer_id,
            num_tokens_total=TOTAL_TOKENS,
            output_height_shard_dim=output_height_shard_dim,
            output_width_shard_dim=output_width_shard_dim,
            output_shard_cores=sorted_output_shard_cores,
        )

        # Output is produced in-place on the input tensor buffer, but output re-perceives it as RM
        tt_raw_output = ttnn.to_torch(tt_output)

        tt_to_torch_output = prepare_output_tensor_from_combine_writer(
            tt_raw_output,
            all_core_range_set,
            sorted_output_shard_cores,
            output_height_shard_dim,
            output_width_shard_dim,
            E,
            M,
            K,
        )

        all_outputs.append(tt_to_torch_output)

        # # Compute checksums for all inputs before calling moe (only when check_accuracy is True)
    #         seed = 0  # torch.manual_seed(0) was set earlier
    #         input_checksums = {
    #             "torch_input": compute_tensor_checksum(torch_input[layer_id]),
    #             "torch_w0_w1": compute_tensor_checksum(torch_w0_w1_reordered),
    #             "torch_w2": compute_tensor_checksum(torch_w2_reordered),
    #             "tt_to_torch_output": compute_tensor_checksum(tt_to_torch_output),
    #         }
    #
    #         # Load existing checksums and check against them
    #         checksum_dict = load_or_create_checksum_dict()
    #         if seed in checksum_dict:
    #             existing_checksums = checksum_dict[seed]
    #             for tensor_name, checksum in input_checksums.items():
    #                 if checksum != existing_checksums[tensor_name]:
    #                     continue
    #                     raise AssertionError(
    #                         f"Checksum mismatch for {tensor_name} with seed {seed}! "
    #                         f"Expected: {existing_checksums[tensor_name]}, Got: {checksum}"
    #                     )
    #         else:
    #             # New seed - add all checksums
    #             checksum_dict[seed] = input_checksums
    #
    #             # Save updated checksums
    #             save_checksum_dict(checksum_dict)

    tt_to_torch_outputs = torch.stack(all_outputs)

    if check_accuracy:
        with torch.no_grad():
            # Reference calculation to match TT output shape (2*M, N) = (E*M, N)

            # Compute gate activations for each expert
            # (L, E, M, K) @ (L, E, K, N) -> (L, E, M, N)
            torch_w0_output_ref = torch_input_ref @ torch_w0
            torch_silu_output_ref = torch_w0_output_ref  # torch.nn.functional.silu(torch_w0_output_ref)
            # (L, E, M, K) @ (L, E, K, N) -> (L, E, M, N)
            torch_w1_output_ref = torch_input_ref @ torch_w1
            torch_intermediate_ref = torch_silu_output_ref * torch_w1_output_ref  # (L, E, M, N)

            # (L, E, M, N) @ (L, E, N, K) -> (L, E, M, K)
            torch_output_ref = torch_intermediate_ref @ torch_w2

        # Calculate accuracy metrics for each layer and expert
        for layer_id, expert_id in itertools.product(range(L), range(E)):
            torch_layer_output = torch_output_ref[layer_id, expert_id, :, :]
            tt_layer_output = tt_to_torch_outputs[layer_id, expert_id, :, :]

            for t in range(torch_layer_output.shape[0]):
                print(f"{expert_id=} {t=}")
                print(f"{torch_layer_output[t,6656:]=}")
                print(f"{tt_layer_output[t,6656:]=}")

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
    (32, 7168, 2048, 2, 1): 225.0,
    # (32, 7168, 2048, 3, 1): 329.0,
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
@pytest.mark.parametrize("check_accuracy", [True])
@pytest.mark.parametrize("dump_outputs", [False])
# @pytest.mark.parametrize("check_accuracy", [True, False], ids=["check_accuracy_True", "check_accuracy_False"])
# @pytest.mark.parametrize("dump_outputs", [True, False], ids=["dump_outputs_True", "dump_outputs_False"])
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

    print(accuracy_metrics)

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
    tiles_per_txn = 14
    num_cores = 12

    Kt, Nt = math.ceil(K / ttnn.TILE_SIZE), math.ceil(N / ttnn.TILE_SIZE)
    w0_w1_padded_tiles_per_core = 2 * math.ceil(Nt / num_cores) * Kt
    w2_padded_tiles_per_core = 4 * math.ceil(Kt / num_cores / 4) * (math.ceil(Nt / 7) * 7)
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
