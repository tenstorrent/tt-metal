# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
import math
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc, comp_allclose
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3YarnRotaryEmbedding

PCC_THRESHOLD = 0.996

"""
The kernel reads 7 tiles per transaction.
"""
W_TILES_PER_TXN = 7
WQ_B_K = 1536
WQ_B_N = 16 * 192
WQ_B_K_TILES_PER_TXN = 7
WQ_B_N_TILES_PER_GROUP = 4

NUM_HEADS = 16
Q_NOPE_K = 128
Q_NOPE_N = 512

"""
# SMALL_N_TILES_PER_CORE and LARGE_N_TILES_PER_CORE explain core groupings used for splitting the N dimension
#
# The MLA WQKV-AB kernel shards the N dimension (output channels) across multiple device cores. To maximize memory bandwidth
# and balance workloads, the N tiles are split unevenly: 'small' cores handle a smaller number of N tiles, and 'large' cores
# handle a larger number. Typically, half the cores are assigned SMALL_N_TILES_PER_CORE (here, 5), and the other half are assigned
# LARGE_N_TILES_PER_CORE (here, 6):
#
#   - num_cores // 2 cores will process SMALL_N_TILES_PER_CORE = 5 N tiles each
#   - (num_cores - num_cores // 2) cores will process LARGE_N_TILES_PER_CORE = 6 N tiles each
#
# This uneven sharding allows for better balancing in cases where N is not exactly divisible by the core count, ensuring all required tiles are processed.
#
# MAX_N_TILES_PER_CORE gives the maximum tiles any core may process (6 in this setup).

"""
SMALL_N_TILES_PER_CORE = 5
LARGE_N_TILES_PER_CORE = 6
MAX_N_TILES_PER_CORE = max(SMALL_N_TILES_PER_CORE, LARGE_N_TILES_PER_CORE)


def create_torch_input(L, in0_num_cores, M, K):
    """
    Create torch input tensor with random values per layer.

    Args:
        L: Number of layers
        in0_num_cores: Number of input cores the tensor is replicated across
        M: Number of tokens
        K: Hidden dimension

    Returns:
        torch_input: Tensor of shape (L, in0_num_cores, M, K)
    """
    torch_input = torch.rand((L, 1, M, K), dtype=torch.bfloat16) - 0.5
    torch_input = torch_input.repeat(1, in0_num_cores, 1, 1)
    return torch_input


def create_torch_w(L, K, N):
    """
    Create torch weight tensor with random values.

    Args:
        L: Number of layers
        K: Hidden dimension
        N: Output dimension

    Returns:
        torch_w: Tensor of shape (L, K, N)
    """
    torch_w = torch.rand((L, K, N), dtype=torch.bfloat16) - 0.5
    return torch_w


def create_torch_q_nope(L, NUM_HEADS, Q_NOPE_K, Q_NOPE_N):
    """
    Create torch q_nope tensor with random values.

    Args:
        L: Number of layers
        NUM_HEADS: Number of heads
        Q_NOPE_K: Hidden dimension
        Q_NOPE_N: Output dimension

    Returns:
        torch_q_nope: Tensor of shape (L, NUM_HEADS, Q_NOPE_K, Q_NOPE_N)
    """
    torch_q_nope = torch.rand((L, NUM_HEADS, Q_NOPE_K, Q_NOPE_N), dtype=torch.bfloat16) - 0.5
    return torch_q_nope


def create_torch_rope(hf_config):
    """
    Create fused DeepSeek sin/cos table in torch.

    Returns:
        torch_rope: Tensor of shape (1, 1, max_seq_len, 64) with fused [cos(32) | sin(32)].
    """
    args = {
        "dim": hf_config.qk_rope_head_dim,
        "max_position_embeddings": hf_config.max_seq_len,
        "base": hf_config.rope_theta * 1.0,
        "device": "cpu",
        "scaling_factor": hf_config.rope_scaling["factor"],
        "original_max_position_embeddings": hf_config.rope_scaling["original_max_position_embeddings"],
        "beta_fast": hf_config.rope_scaling["beta_fast"],
        "beta_slow": hf_config.rope_scaling["beta_slow"],
        "mscale": hf_config.rope_scaling["mscale"],
        "mscale_all_dim": hf_config.rope_scaling["mscale_all_dim"],
    }

    reference_rope = DeepseekV3YarnRotaryEmbedding(**args)

    # [max_seq_len, dim], where dim is [t1, .., td//2, t1, .., td//2]
    # Same data is repeated for cos and sin, so we can just take the first half.
    cos = reference_rope.cos_cached[:, : hf_config.qk_rope_head_dim // 2]
    sin = reference_rope.sin_cached[:, : hf_config.qk_rope_head_dim // 2]

    return cos, sin


def prepare_rope_tensor(torch_rope, num_dram_banks):
    """
    Replicate the rope table for every DRAM bank so each worker can read from its local bank.

    Args:
        torch_rope: Tuple of tensors of shape (1, 1, max_seq_len, head_dim//2) with cos and sin.
        num_dram_banks: Number of DRAM banks / worker cores.

    Returns:
        Tensor of shape (num_dram_banks, 1, max_seq_len * 2, 32) in the layout TT expects.
    """
    cos, sin = torch_rope

    # Rearrange value within each half
    # The first 8 values are in even positions, and the last 8 values are in odd positions
    cos_8 = cos.view(-1, 2, 2, ttnn.TILE_SIZE // 4)
    cos_8_interleaved = cos_8.permute(0, 1, 3, 2)

    # Now view this as a single 16 row
    cos_16 = cos_8_interleaved.reshape(-1, 2, ttnn.TILE_SIZE // 2)

    # Do the same for sin
    sin_8 = sin.view(-1, 2, 2, ttnn.TILE_SIZE // 4)
    sin_8_interleaved = sin_8.permute(0, 1, 3, 2)
    sin_16 = sin_8_interleaved.reshape(-1, 2, ttnn.TILE_SIZE // 2)

    # Now put the cos and sin together
    cos_sin = torch.stack([cos_16, sin_16], dim=-2)

    # Group positions in pairs
    cos_sin_pairs = cos_sin.view(-1, 2, 2, 2, 16)

    # Put values from each pair for first half before second half
    cos_sin_pairs_first = cos_sin_pairs.permute(0, 2, 1, 3, 4)

    # Now collapse all intemediate dimensions to height
    cos_sin_pairs_first = cos_sin_pairs_first.reshape(1, -1, ttnn.TILE_SIZE // 2)
    return cos_sin_pairs_first.repeat(num_dram_banks, 1, 1)


def n_tiles_for_core(core_id, num_cores):
    """
    Get the number of N tiles for a given core.
    Args:
        core_id: Core ID
        num_cores: Number of cores

    Returns:
        Number of N tiles for the core
    """
    return SMALL_N_TILES_PER_CORE if core_id < (num_cores // 2) else LARGE_N_TILES_PER_CORE


def prepare_w_tensor(torch_w, L, K, N, num_dram_banks):
    """
    Prepare w tensor shards for mla_wqkv_ab pure matmul path.

    Args:
        torch_w: Weight tensor of shape (L, K, N)
        L: Number of layers
        K: Input dimension
        N: Output dimension
        num_dram_banks: Number of DRAM banks / worker cores

    Returns:
        Tensor of shape (num_dram_banks, L, H, W) laid out as 7-tile packets.
    """
    Kt, Nt = math.ceil(K / ttnn.TILE_SIZE), math.ceil(N / ttnn.TILE_SIZE)
    w_tile_view = torch_w.view(L, Kt, ttnn.TILE_SIZE, Nt, ttnn.TILE_SIZE)

    each_shard = []
    max_w_tiles_per_core = Kt * MAX_N_TILES_PER_CORE

    # We don't need padding in height dimension, if this is true
    assert max_w_tiles_per_core % W_TILES_PER_TXN == 0

    current_n_tile = 0
    for core_id in range(num_dram_banks):
        n_tiles_this_core = n_tiles_for_core(core_id, num_dram_banks)
        this_core_tiles = w_tile_view[:, :, :, current_n_tile : current_n_tile + n_tiles_this_core, :]
        current_n_tile += n_tiles_this_core

        # Put this_core_tiles together, in the last dimension.
        this_core_tiles = this_core_tiles.reshape(L, Kt, ttnn.TILE_SIZE, n_tiles_this_core * ttnn.TILE_SIZE)

        # Pad smaller-N cores so all DRAM shards keep a uniform shape.
        pad_tiles = LARGE_N_TILES_PER_CORE - n_tiles_this_core
        this_core_padding = torch.zeros(L, Kt, ttnn.TILE_SIZE, pad_tiles * ttnn.TILE_SIZE, dtype=torch_w.dtype)
        this_core_data = torch.cat([this_core_tiles, this_core_padding], dim=-1)

        each_shard.append(this_core_data)

    assert current_n_tile == Nt
    return torch.stack(each_shard, dim=1)


def prepare_wq_b_tensor(torch_wq_b, L, K, N, num_dram_banks):
    """
    Prepare wq_b tensor shards for 7x4 (KxN tiles) read order.

    Args:
        torch_wq_b: Weight tensor of shape (L, K, N)
        L: Number of layers
        K: Input dimension
        N: Output dimension
        num_dram_banks: Number of DRAM banks / worker cores

    Returns:
        Tensor of shape (num_dram_banks, L, H, W), where:
          - W = 4 tiles (128)
          - H = (N_tiles_per_core/4) * padded_K_tiles * 32
        and padded_K_tiles is K padded up to a multiple of 7 tiles.
    """
    Kt, Nt = math.ceil(K / ttnn.TILE_SIZE), math.ceil(N / ttnn.TILE_SIZE)
    n_tiles_per_core = Nt // num_dram_banks

    # K must be padded to full 7-tile transactions (e.g., 48 -> 49 tiles).
    padded_k_tiles = math.ceil(Kt / WQ_B_K_TILES_PER_TXN) * WQ_B_K_TILES_PER_TXN
    n_groups_per_core = n_tiles_per_core // WQ_B_N_TILES_PER_GROUP

    w_tile_view = torch_wq_b.view(L, Kt, ttnn.TILE_SIZE, Nt, ttnn.TILE_SIZE)
    each_shard = []
    current_n_tile = 0

    for _ in range(num_dram_banks):
        this_core_tiles = w_tile_view[:, :, :, current_n_tile : current_n_tile + n_tiles_per_core, :]

        # Reorder as requested:
        # 1) Consume one 4-tile-wide N group at a time.
        # 2) For each group, pack all K tiles (padded to 7-tile transaction multiple).
        # 3) Stack next 4-tile-wide group below the previous one.
        this_core_groups = []
        for group_idx in range(n_groups_per_core):
            group_n_start = group_idx * WQ_B_N_TILES_PER_GROUP
            group_n_end = group_n_start + WQ_B_N_TILES_PER_GROUP
            group_tiles = this_core_tiles[:, :, :, group_n_start:group_n_end, :].reshape(
                L, Kt, ttnn.TILE_SIZE, WQ_B_N_TILES_PER_GROUP * ttnn.TILE_SIZE
            )

            pad_k_tiles = padded_k_tiles - Kt
            group_padding = torch.zeros(
                L, pad_k_tiles, ttnn.TILE_SIZE, WQ_B_N_TILES_PER_GROUP * ttnn.TILE_SIZE, dtype=torch_wq_b.dtype
            )
            this_core_groups.append(torch.cat([group_tiles, group_padding], dim=1))

        each_shard.append(torch.cat(this_core_groups, dim=1))
        current_n_tile += n_tiles_per_core

    return torch.stack(each_shard, dim=1)


def prepare_q_nope_tensor(torch_q_nope, L, NUM_HEADS, Q_NOPE_K, Q_NOPE_N, num_dram_banks):
    """
    Prepare q_nope tensor shards for 4x8 (KxN/2 tiles) read order.
    Args:
        torch_q_nope: Weight tensor of shape (L, NUM_HEADS, Q_NOPE_K, Q_NOPE_N)
        L: Number of layers
        NUM_HEADS: Number of heads
        Q_NOPE_K: Number of q_nope keys
        Q_NOPE_N: Number of q_nope values
        num_dram_banks: Number of DRAM banks / worker cores

    Returns:
        Tensor of shape (num_dram_banks, L, H, W), where:
          - W = 8 tiles (128)
          - H = (N_tiles_per_core/8) * padded_K_tiles * 32
        and padded_K_tiles is K padded up to a multiple of 7 tiles.
    """
    Kt, Nt = math.ceil(Q_NOPE_K / ttnn.TILE_SIZE), math.ceil(Q_NOPE_N / ttnn.TILE_SIZE)
    heads_per_core = NUM_HEADS // 8

    each_shard = []
    current_head_idx = 0
    for _ in range(8):
        this_core_tiles = torch_q_nope[:, current_head_idx : current_head_idx + heads_per_core, :, :]
        half_width_view = this_core_tiles.reshape(L, heads_per_core, Kt, ttnn.TILE_SIZE, Nt // 8, 8 * ttnn.TILE_SIZE)

        # [L, 2, Kt, 32, Nt // 2, 8 * 32] -> [L, 2, Nt // 2, 32, Kt, 8 * 32]
        this_core_tiles_kfirst = half_width_view.permute(0, 1, 4, 2, 3, 5)
        each_shard.append(this_core_tiles_kfirst.reshape(L, -1, 8 * 32))

        current_head_idx += heads_per_core

    # Pad for the other banks
    for _ in range(num_dram_banks - 8):
        each_shard.append(torch.zeros(each_shard[-1].shape, dtype=torch_q_nope.dtype))

    return torch.stack(each_shard, dim=1)


def prepare_output_tensor(tt_output, num_dram_banks):
    """
    Prepare output by extracting the valid 5/6 N tiles per core.

    Args:
        tt_output: Tensor of shape (M, num_dram_banks * 6 * ttnn.TILE_SIZE)
        num_dram_banks: Number of DRAM banks / worker cores

    Returns:
        Tensor of shape (M, N)
    """
    each_shard = []
    shard_width = MAX_N_TILES_PER_CORE * ttnn.TILE_SIZE

    core_offset = 0
    for core_id in range(num_dram_banks):
        n_tiles_this_core = n_tiles_for_core(core_id, num_dram_banks)
        each_shard.append(tt_output[:, core_offset : core_offset + n_tiles_this_core * ttnn.TILE_SIZE])
        core_offset += MAX_N_TILES_PER_CORE * ttnn.TILE_SIZE
    return torch.cat(each_shard, dim=1)


def get_accuracy_metrics(torch_output, tt_output):
    _pcc_passed, pcc_val = comp_pcc(torch_output, tt_output)
    std = torch_output.std().item()
    relative_rmse_val = (torch.nn.functional.mse_loss(torch_output, tt_output).sqrt().item() / std) if std != 0 else 0.0
    allclose_passed, allclose_val = comp_allclose(torch_output, tt_output, rtol=2e-2, atol=1e-1)
    return {
        "pcc": pcc_val,
        "relative_rmse": relative_rmse_val,
        "allclose": allclose_passed,
        "allclose_val": allclose_val,
    }


def run_test_mla_wqkv_ab(device, M, K, N, L, pos, check_accuracy, dump_outputs):
    torch.manual_seed(0)

    logger.info(
        f"Running test_mla_wqkv_ab with M={M}, K={K}, N={N}, L={L}, pos={pos}, check_accuracy={check_accuracy}, dump_outputs={dump_outputs}"
    )

    # --------------------------------------------------------------------------
    # Shard grid
    # --------------------------------------------------------------------------
    in0_core_coords = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)

    in0_num_cores = len(in0_core_coords)
    in0_core_range = [ttnn.CoreRange(in0_core_coord, in0_core_coord) for in0_core_coord in in0_core_coords]
    in0_core_range_set = ttnn.CoreRangeSet(in0_core_range)

    # --------------------------------------------------------------------------
    # Constants
    # --------------------------------------------------------------------------
    in0_dtype = ttnn.bfloat16
    w_dtype = ttnn.bfloat8_b
    rope_dtype = ttnn.bfloat16
    num_dram_banks = len(in0_core_coords)

    dram_core_coords = [ttnn.CoreCoord(dram_bank_id, 0) for dram_bank_id in range(num_dram_banks)]
    dram_core_range = [ttnn.CoreRange(dram_core_coord, dram_core_coord) for dram_core_coord in dram_core_coords]
    dram_core_range_set = ttnn.CoreRangeSet(dram_core_range)

    # --------------------------------------------------------------------------
    # Tensor shapes and memory configurations
    # --------------------------------------------------------------------------
    # Define tensor shapes - same for both accuracy and performance testing
    input_shape = (in0_num_cores, M, K)

    in0_shard_spec = ttnn.ShardSpec(
        grid=in0_core_range_set,
        shard_shape=(M, K),
        shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    # Each core gets a copy of the original (M, K) input
    input_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in0_shard_spec
    )

    # ------------------------------------------------------------------------
    # Create DRAM shard spec for w_a
    # Tensor shape: (L, K, N) -> sharded over cores with fixed max shard shape.
    # ------------------------------------------------------------------------
    w_a_shard_height = K
    w_a_shard_width = MAX_N_TILES_PER_CORE * ttnn.TILE_SIZE

    w_a_shard_spec = ttnn.ShardSpec(
        dram_core_range_set, (w_a_shard_height, w_a_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )
    w_a_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w_a_shard_spec)

    # ------------------------------------------------------------------------
    # Create DRAM shard spec for wq_b
    # Tensor shape: (L, K, N) -> sharded over cores with fixed max shard shape.
    # ------------------------------------------------------------------------
    wq_b_shard_height = 2 * 49 * ttnn.TILE_SIZE
    wq_b_shard_width = 4 * ttnn.TILE_SIZE
    wq_b_shard_spec = ttnn.ShardSpec(
        dram_core_range_set, (wq_b_shard_height, wq_b_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )
    wq_b_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, wq_b_shard_spec)

    # ------------------------------------------------------------------------
    # Create DRAM shard spec for q_nope
    # Tensor shape: (L, K, N) -> sharded over cores with fixed max shard shape.
    # ------------------------------------------------------------------------
    q_nope_shard_height = 2 * (NUM_HEADS // 8) * Q_NOPE_K
    q_nope_shard_width = Q_NOPE_N // 2
    q_nope_shard_spec = ttnn.ShardSpec(
        dram_core_range_set, (q_nope_shard_height, q_nope_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )
    q_nope_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, q_nope_shard_spec
    )

    # ------------------------------------------------------------------------
    # Create DRAM shard spec for rope table.
    # Tensor shape: (max_seq_len, 128) -> replicated over 12 DRAM banks
    # ------------------------------------------------------------------------
    deepseek_config_path = (
        Path(__file__).resolve().parents[6] / "models" / "demos" / "deepseek_v3" / "reference" / "config.json"
    )
    with open(deepseek_config_path) as f:
        deepseek_config = json.load(f)
    rope_cfg = SimpleNamespace(
        qk_rope_head_dim=deepseek_config["qk_rope_head_dim"],
        max_seq_len=deepseek_config["max_position_embeddings"],
        rope_theta=float(deepseek_config["rope_theta"]),
        rope_scaling=deepseek_config["rope_scaling"],
    )

    rope_shard_height = rope_cfg.max_seq_len * rope_cfg.qk_rope_head_dim // (ttnn.TILE_SIZE // 2)
    rope_shard_width = ttnn.TILE_SIZE // 2
    rope_shard_spec = ttnn.ShardSpec(
        dram_core_range_set, (rope_shard_height, rope_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )
    rope_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, rope_shard_spec)

    # ------------------------------------------------------------------------
    # Create DRAM shard spec for output
    # Tensor shape: (M, N_padded) -> one fixed-width output shard per core.
    # ------------------------------------------------------------------------
    output_shard_height = M
    output_shard_width = MAX_N_TILES_PER_CORE * ttnn.TILE_SIZE
    output_shard_spec = ttnn.ShardSpec(
        in0_core_range_set, (output_shard_height, output_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, output_shard_spec)
    tt_output = ttnn.empty(
        (M, in0_num_cores * output_shard_width),
        dtype=in0_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=output_mem_config,
    )

    # ------------------------------------------------------------------------
    # Prepare the tensors
    # --------------------------------------------------------------------------
    if check_accuracy:
        torch_input = create_torch_input(L, in0_num_cores, M, K)
        torch_w_a = create_torch_w(L, K, N)
        torch_wq_b = create_torch_w(L, WQ_B_K, WQ_B_N)
        torch_q_nope = create_torch_q_nope(L, NUM_HEADS, Q_NOPE_K, Q_NOPE_N)
        # ------------------------------------------------------------------------
        # Prepare w tensor (padded, and reordered)
        torch_w_a_reordered = prepare_w_tensor(torch_w_a, L, K, N, num_dram_banks)
        torch_wq_b_reordered = prepare_wq_b_tensor(torch_wq_b, L, WQ_B_K, WQ_B_N, num_dram_banks)
        torch_q_nope_reordered = prepare_q_nope_tensor(torch_q_nope, L, NUM_HEADS, Q_NOPE_K, Q_NOPE_N, num_dram_banks)

        tt_w_a = ttnn.from_torch(
            torch_w_a_reordered,
            dtype=w_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=w_a_mem_config,
        )
        tt_wq_b = ttnn.from_torch(
            torch_wq_b_reordered,
            dtype=w_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=wq_b_mem_config,
        )
        tt_q_nope = ttnn.from_torch(
            torch_q_nope_reordered,
            dtype=w_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=q_nope_mem_config,
        )

        # ------------------------------------------------------------------------
        # Create torch rope tensor
        # ------------------------------------------------------------------------
        torch_rope = create_torch_rope(rope_cfg)

        # ------------------------------------------------------------------------
        # Prepare rope tensor (reordered)
        torch_rope_reordered = prepare_rope_tensor(torch_rope, num_dram_banks)

        # Create tt_rope tensor with DRAM sharding
        tt_rope = ttnn.from_torch(
            torch_rope_reordered,
            dtype=rope_dtype,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=rope_mem_config,
        )

    else:
        tt_input = ttnn.empty(
            input_shape,
            dtype=in0_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=input_sharded_mem_config,
        )
        tt_w_a = ttnn.empty(
            (num_dram_banks, L, w_a_shard_height, w_a_shard_width),
            dtype=w_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=w_a_mem_config,
        )
        tt_wq_b = ttnn.empty(
            (num_dram_banks, L, wq_b_shard_height, wq_b_shard_width),
            dtype=w_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=wq_b_mem_config,
        )
        tt_q_nope = ttnn.empty(
            (num_dram_banks, L, wq_b_shard_height, wq_b_shard_width),
            dtype=w_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=q_nope_mem_config,
        )
        tt_rope = ttnn.empty(
            (num_dram_banks, rope_shard_height, rope_shard_width),
            dtype=rope_dtype,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=rope_mem_config,
        )

    # --------------------------------------------------------------------------
    # Run the operation
    # --------------------------------------------------------------------------
    # Collect accuracy metrics for all layers
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

        ttnn.experimental.deepseek.mla.mla_wqkv_ab(
            tt_input,
            w_a_tensor=tt_w_a,
            wq_b_tensor=tt_wq_b,
            q_nope_tensor=tt_q_nope,
            rope_tensor=tt_rope,
            output_tensor=tt_output,
            layer_id=layer_id,
            pos=pos,
        )

        tt_to_torch_output = ttnn.to_torch(tt_output)
        all_outputs.append(tt_to_torch_output)

    tt_to_torch_outputs = torch.stack(all_outputs)

    if check_accuracy:
        with torch.no_grad():
            torch_input_ref = torch_input[:, 0, ...]
            torch_mm_out = torch_input_ref @ torch_w_a

            # Apply rope to the last 64 values of each row
            k_pe_rope_input = torch_mm_out[:, :, -64:]

            # View it has a complex number, where real and complex are interleaved
            k_pe_rope_input_c = torch.view_as_complex(k_pe_rope_input.float().reshape(L, M, -1, 2))

            # View the rope table as a complex number, where real and complex are interleaved
            cos, sin = torch_rope
            cos_pos, sin_pos = cos[pos], sin[pos]
            cos_sin_pos = torch.stack([cos_pos, sin_pos], dim=-1)
            rot_matrix_c = torch.view_as_complex(cos_sin_pos.float())

            # Apply the rotation matrix to the k_pe_rope_input
            k_pe_rope_output_c = k_pe_rope_input_c * rot_matrix_c

            # View the output as a real number, where real and complex are interleaved
            k_pe_rope_output = torch.view_as_real(k_pe_rope_output_c).reshape(L, M, -1)

            # Do an in-place update of the last 64 values of each row
            torch_mm_out[:, :, -64:] = k_pe_rope_output

            # Apply RMS Norm to the q_norm and k_norm tensors in place
            q_norm_input = torch_mm_out[:, :, :1536]
            k_norm_input = torch_mm_out[:, :, 1536 : 1536 + 512]
            q_norm_output = torch.nn.functional.rms_norm(q_norm_input, normalized_shape=(1536,), eps=1e-5)
            k_norm_output = torch.nn.functional.rms_norm(k_norm_input, normalized_shape=(512,), eps=1e-5)
            torch_mm_out[:, :, :1536] = q_norm_output
            torch_mm_out[:, :, 1536 : 1536 + 512] = k_norm_output

        # Calculate accuracy metrics for each layer
        for layer_id in range(L):
            torch_ref_layer = torch_mm_out[layer_id, :, :]
            tt_layer_output = tt_to_torch_outputs[layer_id, :, :]
            tt_values = prepare_output_tensor(tt_layer_output, num_dram_banks)
            layer_metrics = get_accuracy_metrics(torch_ref_layer, tt_values)
            all_accuracy_metrics[layer_id] = layer_metrics

    if dump_outputs:
        torch.set_printoptions(profile="full")
        var2filename = {
            tt_to_torch_outputs: "tt_wqkv_ab_output_act.txt",
        }
        if check_accuracy:
            var2filename[torch_mm_out] = "torch_wqkv_ab_output_ref.txt"

        for var, filename in var2filename.items():
            with open(filename, "w") as f:
                f.write(str(var))

    return all_accuracy_metrics


SHAPE2TIME = {
    (32, 7168, 2112, 1, 0): 97,
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
    "M, K, N, L, pos",
    SHAPE2TIME.keys(),
)
@pytest.mark.parametrize("check_accuracy", [True, False], ids=["check_accuracy_True", "check_accuracy_False"])
@pytest.mark.parametrize("dump_outputs", [False], ids=["dump_outputs_False"])
def test_mla_wqkv_ab(device, M, K, N, L, pos, check_accuracy, dump_outputs):
    accuracy_metrics = run_test_mla_wqkv_ab(
        device,
        M,
        K,
        N,
        L,
        pos,
        check_accuracy,
        dump_outputs,
    )

    passing = True
    # Print the layers that did not pass the PCC check
    for layer_id, metrics in accuracy_metrics.items():
        if metrics["pcc"] < PCC_THRESHOLD:
            passing = False
            logger.warning(f"Layer {layer_id}: PCC={metrics['pcc']:.6f}")
        else:
            logger.info(f"Layer {layer_id}: PCC={metrics['pcc']:.6f} (Passed)")

    assert passing, f"Some layers did not pass the PCC/Allclose check"
