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

# Some cores have more tiles than others, but they are sprinkled around the ring for boundary alignment.
FULL_CORES_A = {0, 1, 8, 9}
PAD_CORES_A = {2, 3, 4, 5, 6, 7, 10, 11}

# In this case, we spread the tiles evenly around the ring, in groups of 3.
FULL_CORES_B = {0, 3, 6, 9}
PAD_CORES_B = {1, 2, 4, 5, 7, 8, 10, 11}

NUM_COMBINE_CORES_W = 4
NUM_COMBINE_CORES_H = 4
NUM_COMBINE_CORES = NUM_COMBINE_CORES_W * NUM_COMBINE_CORES_H

# These are the number of tokens written to each combine core before we round robin
NUM_TOKENS_PER_CORE = 8

FACE_WIDTH = 16
FACE_HEIGHT = 16

NUM_FACES_PER_TILE_WIDTH = ttnn.TILE_SIZE // FACE_WIDTH
NUM_FACES_PER_TILE_HEIGHT = ttnn.TILE_SIZE // FACE_HEIGHT


def flat_idx_to_tile_pos(flat_idx, num_tile_cols):
    """Decompose a flat buffer index into tile coordinates.

    ttnn stores tiles with the following hierarchy (innermost to outermost):
    c_f, r_f, f_x, f_y, t_x, t_y — where t_x wraps at num_tile_cols.

    Args:
        flat_idx: flat index into the raw buffer
        num_tile_cols: number of tile columns ttnn assumed (K // TILE_SIZE)
    """
    c_f = flat_idx % FACE_WIDTH
    r_f = (flat_idx // FACE_WIDTH) % FACE_HEIGHT
    f_x = (flat_idx // (FACE_WIDTH * FACE_HEIGHT)) % NUM_FACES_PER_TILE_WIDTH
    f_y = (flat_idx // (FACE_WIDTH * FACE_HEIGHT * NUM_FACES_PER_TILE_WIDTH)) % NUM_FACES_PER_TILE_HEIGHT
    t_x = (
        flat_idx // (FACE_WIDTH * FACE_HEIGHT * NUM_FACES_PER_TILE_WIDTH * NUM_FACES_PER_TILE_HEIGHT)
    ) % num_tile_cols
    t_y = flat_idx // (FACE_WIDTH * FACE_HEIGHT * NUM_FACES_PER_TILE_WIDTH * NUM_FACES_PER_TILE_HEIGHT * num_tile_cols)

    return t_y, t_x, f_y, f_x, r_f, c_f


def tile_pos_to_untilized_coord(t_y, t_x, f_y, f_x, r_f, c_f):
    """Convert tile coordinates to (row, col) in the untilized tensor that ttnn produced.

    Args:
        t_y, t_x, f_y, f_x, r_f, c_f: tile coordinates
    """
    row = t_y * ttnn.TILE_SIZE + f_y * FACE_HEIGHT + r_f
    col = t_x * ttnn.TILE_SIZE + f_x * FACE_WIDTH + c_f
    return row, col


def get_untilized_data(tt_output, E, M, K):
    """Recover row-major data (width=K//NUM_COMBINE_CORES_W) from a torch tensor
    that ttnn incorrectly untilized assuming tile layout with width K.

    The device wrote row-major data with width (K // NUM_COMBINE_CORES_W) into the
    buffer. ttnn.to_torch() treated the buffer as tile-layout with shape (E*M, K)
    and untilized it, producing a garbled (E*M, K) tensor. This function reverses
    that untilization to recover the original flat buffer, then reshapes it as
    row-major with the correct width.

    The total buffer has E*M*K elements. Reshaped at width_per_core, that gives
    (E*M*K // width_per_core) rows of row-major data.

    Args:
        tt_output: torch tensor of shape (E, M, K) from ttnn.to_torch()
        E: number of experts
        M: sequence length
        K: full output width (ttnn's assumed untilize width)
    """
    width_per_core = K // NUM_COMBINE_CORES_W
    num_tile_cols = K // ttnn.TILE_SIZE  # tile columns as ttnn assumed (224)
    total_elements = E * M * K
    num_rm_rows = total_elements // width_per_core
    output = tt_output.view(E * M, K)

    result = torch.empty((num_rm_rows, width_per_core), dtype=tt_output.dtype)
    for rm_row in range(num_rm_rows):
        for rm_col in range(width_per_core):
            # Flat buffer index for this row-major position
            flat_idx = rm_row * width_per_core + rm_col

            # Decompose flat_idx into tile coordinates (how ttnn interpreted it)
            t_y, t_x, f_y, f_x, r_f, c_f = flat_idx_to_tile_pos(flat_idx, num_tile_cols)

            # Convert tile coords to where ttnn placed this element after untilizing
            row, col = tile_pos_to_untilized_coord(t_y, t_x, f_y, f_x, r_f, c_f)

            result[rm_row, rm_col] = output[row, col]

    return result


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
    torch_input = torch.rand((L, 2, M, K), dtype=torch.bfloat16) - 0.5
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


def prepare_output_tensor(tt_output, E, M, K):
    """
    Prepare the output tensor by padding and reordering tiles.

    Args:
        tt_output: Tensor of shape (num_cores, E, M, K)
        E: Number of experts
        M: Number of input features
        K: Number of output features
    Returns:
        torch_output: Tensor of shape (E, M, K)
    """
    width_per_core = K // NUM_COMBINE_CORES_W

    # Get the values in the first 16 cores
    torch_write_out = tt_output[:NUM_COMBINE_CORES]

    untilized_data = []
    for core_id in range(NUM_COMBINE_CORES):
        untilized_data.append(get_untilized_data(torch_write_out[core_id], E, M, K))
    torch_write_out_rm = torch.stack(untilized_data).reshape(NUM_COMBINE_CORES, -1, width_per_core)

    # Put the cores in a 4x4 grid, in row major order
    torch_write_out_grid = torch_write_out_rm.view(NUM_COMBINE_CORES_H, NUM_COMBINE_CORES_W, -1, width_per_core)
    torch_write_out_w = torch_write_out_grid.permute(0, 2, 1, 3)
    torch_write_out_token = torch_write_out_w.reshape(NUM_COMBINE_CORES_H, -1, NUM_TOKENS_PER_CORE, K)
    torch_write_out_per_expert = torch_write_out_token.permute(1, 0, 2, 3).reshape(E, -1, K)
    return torch_write_out_per_expert[:, :M, :]
    # --------------------------------------------------------------------------
    # This works for the tilize, we want to keep this for testing.
    # each_shard = []

    # for ring_pos in range(len(ring2cores)):
    #     (_, _, pad_flag) = ring2cores[ring_pos]
    #     num_tiles = 19 if pad_flag else 18
    #     each_shard.append(tt_output[ring_pos, :, :, : num_tiles * ttnn.TILE_SIZE])

    # result = torch.cat(each_shard, dim=-1)
    # assert result.shape == (2, M, K)
    # return result
    # --------------------------------------------------------------------------

    # # View it as a row major tensor on each core
    # tt_output_a = tt_output.view(len(ring2cores) + len(combine_core_coords), E, M * K)

    # expert_shards = []

    # for expert in range(E):
    #     each_shard = []
    #     for ring_pos in range(len(ring2cores)):
    #         (_, _, pad_flag) = ring2cores[ring_pos]
    #         num_tiles = 19 if pad_flag else 18
    #         untilized_data = get_untilized_data(tt_output[ring_pos, expert])
    #         each_shard.append(untilized_data[:, : num_tiles * ttnn.TILE_SIZE])
    #     expert_shards.append(torch.cat(each_shard, dim=-1))

    # result = torch.stack(expert_shards)
    # return result
    # --------------------------------------------------------------------------


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
        ring2cores[ring_pos] = (core_coord, core2dram[core_coord], 1 if ring_pos in PAD_CORES_B else 0)

    in0_core_range = [ttnn.CoreRange(ring2cores[i][0], ring2cores[i][0]) for i in range(in0_num_cores)]
    in0_core_range_set = ttnn.CoreRangeSet(in0_core_range)

    # Now pick the last 16 cores as the combine cores, skipping the ones in in0_core_coords
    out_num_cores = NUM_COMBINE_CORES
    all_core_grid = device.compute_with_storage_grid_size()

    dram_coords = [(coord.x, coord.y) for coord in in0_core_coords]

    combine_core_xy = []
    for x_coord, y_coord in itertools.product(range(all_core_grid.x - 1, -1, -1), range(all_core_grid.y - 1, -1, -1)):
        if (x_coord, y_coord) not in dram_coords:
            combine_core_xy.append((x_coord, y_coord))

        if len(combine_core_xy) == out_num_cores:
            break
    else:
        raise ValueError(f"Did not find {out_num_cores} combine cores")

    combine_core_coords = [ttnn.CoreCoord(x_coord, y_coord) for x_coord, y_coord in combine_core_xy]

    # --------------------------------------------------------------------------
    # Constants
    # --------------------------------------------------------------------------
    in0_dtype = ttnn.bfloat16
    w0_dtype = ttnn.bfloat4_b
    num_dram_banks = 12

    dram_core_coords = [ttnn.CoreCoord(ring2cores[i][1], 0) for i in range(in0_num_cores)]
    dram_core_range = [ttnn.CoreRange(dram_core_coord, dram_core_coord) for dram_core_coord in dram_core_coords]
    dram_core_range_set = ttnn.CoreRangeSet(dram_core_range)

    combine_core_range = [
        ttnn.CoreRange(combine_core_coord, combine_core_coord) for combine_core_coord in combine_core_coords
    ]
    combine_core_range_set = ttnn.CoreRangeSet(combine_core_range)

    input_and_combine_core_range_set = ttnn.CoreRangeSet(combine_core_range + in0_core_range)
    total_num_cores = in0_num_cores + out_num_cores

    # --------------------------------------------------------------------------
    # Tensor shapes and memory configurations
    # --------------------------------------------------------------------------
    # Define tensor shapes - same for both accuracy and performance testing
    input_shape = (total_num_cores, 2, M, K)

    in0_shard_spec = ttnn.ShardSpec(
        grid=input_and_combine_core_range_set,
        shard_shape=(2 * M, K),
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
        torch_input = create_torch_input(L, total_num_cores, E, M, K)
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

        _tt_output = ttnn.experimental.moe(
            tt_input,
            w0_w1_tensor=tt_w0_w1,
            w2_tensor=tt_w2,
            output_tensor=tt_input,
            num_experts=E,
            layer_id=layer_id,
            output_shard_core_ranges=combine_core_range_set,
        )

        # Output is produced in-place on the input tensor
        tt_raw_output = ttnn.to_torch(tt_input)
        tt_to_torch_output = prepare_output_tensor(tt_raw_output, E, M, K)

        all_outputs.append(tt_to_torch_output)

    tt_to_torch_outputs = torch.stack(all_outputs)

    if check_accuracy:
        with torch.no_grad():
            # Reference calculation to match TT output shape (2*M, N) = (E*M, N)
            # Use first 2*M rows of input (one copy of the original replicated input)
            torch_input_ref = torch_input[:, 0, ...]

            # Compute gate activations for each expert
            # (L, E, M, K) @ (L, E, K, N) -> (L, E, M, N)
            torch_w0_output_ref = torch_input_ref @ torch_w0
            torch_silu_output_ref = torch.nn.functional.silu(torch_w0_output_ref)
            # (L, E, M, K) @ (L, E, K, N) -> (L, E, M, N)
            torch_w1_output_ref = torch_input_ref @ torch_w1
            torch_intermediate_ref = torch_silu_output_ref * torch_w1_output_ref  # (L, E, M, N)

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
    (32, 7168, 2048, 2, 1): 234.0,
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
