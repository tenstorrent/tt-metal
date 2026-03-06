# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Weight loading and management for throughput-optimized MoE experts.

This module handles loading and sharding expert weights across devices for
the all_to_all-based throughput experts implementation.
"""

import math
from dataclasses import dataclass

import torch

import ttnn
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name

from .config import ThroughputExpertConfig

# ---------------------------------------------------------------------------
# Fused MoE kernel constants (GPT-OSS: K=N=2880, 12 DRAM banks)
# ---------------------------------------------------------------------------
_FUSED_NUM_CORES = 12
_FUSED_FULL_CORES = {0, 1, 4, 5, 8, 9}  # 8 tiles per core
_FUSED_PAD_CORES = {2, 3, 6, 7, 10, 11}  # 7 tiles per core
_FUSED_MAX_TILES_PER_CORE = 8


@dataclass(frozen=True)
class ThroughputExpertWeights:
    """Container for throughput expert weight tensors.

    Weights are sharded across devices such that each device holds
    num_experts_per_device complete expert weight sets.

    When use_fused_gate_up=True, only w2, w2_bias, w1_w3_fused, and w1_w3_bias_fused are loaded.
    When use_fused_gate_up=False, only w1, w2, w3, w1_bias, w2_bias, w3_bias are loaded.

    Attributes:
        w2: Down projection weights [num_experts_per_device, intermediate_size, hidden_size]
        w2_bias: Down projection bias [num_experts_per_device, 1, hidden_size]
        w1: Optional gate projection weights [num_experts_per_device, hidden_size, intermediate_size]
        w3: Optional up projection weights [num_experts_per_device, hidden_size, intermediate_size]
        w1_bias: Optional gate projection bias [num_experts_per_device, 1, intermediate_size]
        w3_bias: Optional up projection bias [num_experts_per_device, 1, intermediate_size]
        w1_w3_fused: Optional fused gate+up weights [num_experts_per_device, hidden_size, 2*intermediate_size]
        w1_w3_bias_fused: Optional fused gate+up bias [num_experts_per_device, 1, 2*intermediate_size]
    """

    # Required weights (always loaded)
    w2: ttnn.Tensor  # Down projection
    w2_bias: ttnn.Tensor  # Down projection bias

    # Unfused weights (loaded only when use_fused_gate_up=False)
    w1: ttnn.Tensor = None  # Gate projection
    w3: ttnn.Tensor = None  # Up projection
    w1_bias: ttnn.Tensor = None  # Gate projection bias
    w3_bias: ttnn.Tensor = None  # Up projection bias

    # Fused weights (loaded only when use_fused_gate_up=True)
    w1_w3_fused: ttnn.Tensor = None  # Fused gate+up projection
    w1_w3_bias_fused: ttnn.Tensor = None  # Fused gate+up bias


def _shard_experts_by_device(
    weights: torch.Tensor,
    num_devices: int,
    mesh_device,
    dtype: ttnn.DataType,
    cache_file_name: str = None,
    shard_dim: int = 1,
) -> ttnn.Tensor:
    """Shard expert weights across devices.

    Each device gets a subset of experts (num_experts / num_devices experts per device).

    Args:
        weights: Full weight tensor (expert dimension will be sharded)
        num_devices: Number of devices to shard across
        mesh_device: TTNN mesh device
        dtype: Weight data type
        cache_file_name: Optional cache file path
        shard_dim: Dimension to shard across (default: 1 for weights, 3 for bias)

    Returns:
        Sharded ttnn.Tensor where each device has its local experts
    """
    # Shard along the expert dimension
    # Each device gets consecutive experts
    return ttnn.as_tensor(
        weights,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        mesh_mapper=ttnn.ShardTensorToMesh(
            mesh_device,
            dim=shard_dim,  # Shard expert dim across all devices
        ),
        cache_file_name=cache_file_name,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def load_throughput_expert_weights(
    mesh_device,
    config: ThroughputExpertConfig,
    state_dict: dict,
    weight_dtype: ttnn.DataType = ttnn.bfloat4_b,
    tensor_cache_path: str = None,
) -> ThroughputExpertWeights:
    """Load and shard expert weights for throughput mode.

    Args:
        mesh_device: TTNN mesh device
        config: Throughput expert configuration
        state_dict: Dictionary containing expert weights
        weight_dtype: Data type for weights (default: bfloat4_b)
        tensor_cache_path: Optional path for weight caching

    Returns:
        ThroughputExpertWeights with sharded tensors
    """
    num_experts = config.num_experts
    num_devices = config.num_devices
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size

    # Extract weights from state dict
    # Assume state_dict has keys like "gate_up_proj" (fused) and "down_proj"
    # or individual "gate_proj", "up_proj", "down_proj"

    if "gate_up_proj" in state_dict:
        # Fused gate/up projection - unfuse
        gate_up = state_dict["gate_up_proj"]
        # gate_up shape: [num_experts, hidden_size, 2 * intermediate_size] (interleaved)
        w1 = gate_up[..., ::2].reshape(1, num_experts, hidden_size, intermediate_size)
        w3 = gate_up[..., 1::2].reshape(1, num_experts, hidden_size, intermediate_size)

        # Unfuse bias if present
        if "gate_up_proj_bias" in state_dict:
            gate_up_bias = state_dict["gate_up_proj_bias"]
            w1_bias = gate_up_bias[..., ::2].reshape(1, num_experts, 1, intermediate_size)
            w3_bias = gate_up_bias[..., 1::2].reshape(1, num_experts, 1, intermediate_size)
        else:
            w1_bias = torch.zeros(1, num_experts, 1, intermediate_size)
            w3_bias = torch.zeros(1, num_experts, 1, intermediate_size)
    else:
        # Separate gate and up projections
        w1 = state_dict["gate_proj"].reshape(1, num_experts, hidden_size, intermediate_size)
        w3 = state_dict["up_proj"].reshape(1, num_experts, hidden_size, intermediate_size)
        w1_bias = state_dict.get("gate_proj_bias", torch.zeros(1, num_experts, 1, intermediate_size)).reshape(
            1, num_experts, 1, intermediate_size
        )
        w3_bias = state_dict.get("up_proj_bias", torch.zeros(1, num_experts, 1, intermediate_size)).reshape(
            1, num_experts, 1, intermediate_size
        )

    # Load down projection weights (always needed)
    w2 = state_dict["down_proj"].reshape(1, num_experts, intermediate_size, hidden_size)
    w2_bias = state_dict.get("down_proj_bias", torch.zeros(1, num_experts, 1, hidden_size)).reshape(
        1, num_experts, 1, hidden_size
    )
    if config.pad_w2:
        # Pad w2 output dimension for tile alignment in CCL operations.
        # Without padding, local_hidden = hidden_size / TP may not be tile-aligned (e.g., 2880/8 = 360),
        # causing CCL to do expensive Untilize->Pad->Tilize cycles internally.
        hidden_size = config.hidden_size
        scattered_local_hidden = hidden_size // mesh_device.shape[1]
        padded_scattered_local_hidden = ((scattered_local_hidden + 31) // 32) * 32  # Round up to tile boundary
        w2_pad_size = (padded_scattered_local_hidden * mesh_device.shape[1]) - hidden_size

        if w2_pad_size > 0 and mesh_device.shape[1] > 1:
            # Pad the output dimension of w2 weight: [input_dim, hidden_size] -> [input_dim, padded_hidden]
            w2 = torch.nn.functional.pad(w2, (0, w2_pad_size), "constant", value=0.0)
            # Pad bias similarly
            w2_bias = torch.nn.functional.pad(w2_bias, (0, w2_pad_size), "constant", value=0.0)

        # Use unique cache key when padding is applied
        w2_cache_suffix = f"_padded{w2_pad_size}" if w2_pad_size > 0 and mesh_device.shape[1] > 1 else ""
    else:
        w2_cache_suffix = ""

    w2_tt = _shard_experts_by_device(
        w2,
        num_devices,
        mesh_device,
        weight_dtype,
        get_cache_file_name(tensor_cache_path, f"throughput_w2{w2_cache_suffix}"),
    )

    w2_bias_tt = _shard_experts_by_device(
        w2_bias,
        num_devices,
        mesh_device,
        ttnn.bfloat16,
        get_cache_file_name(tensor_cache_path, "throughput_w2_bias"),
        shard_dim=-3,  # Expert dimension after reshape
    )

    # Load either fused or unfused weights based on config
    w1_tt = None
    w3_tt = None
    w1_bias_tt = None
    w3_bias_tt = None
    w1_w3_fused_tt = None
    w1_w3_bias_fused_tt = None

    if config.use_fused_gate_up:
        # ======================================================================
        # FUSED MODE: Load only fused weights to save memory
        # ======================================================================
        # Fuse w1 and w3 along the output dimension
        # w1: [1, num_experts, hidden_size, intermediate_size]
        # w3: [1, num_experts, hidden_size, intermediate_size]
        # -> w1_w3_fused: [1, num_experts, hidden_size, 2*intermediate_size]
        weights_cache_suffix = "throughput_w1_w3_fused"
        bias_cache_suffix = "throughput_w1_w3_bias_fused"
        w1_w3_fused = torch.cat([w1, w3], dim=-1)

        # Fuse biases
        # w1_bias: [1, num_experts, 1, intermediate_size]
        # w3_bias: [1, num_experts, 1, intermediate_size]
        # -> w1_w3_bias_fused: [1, num_experts, 1, 2*intermediate_size]
        w1_w3_bias_fused = torch.cat([w1_bias, w3_bias], dim=-1)

        if config.pad_w1_w3:
            ideal_core_grid_size = 64
            fused_hidden_size = 2 * intermediate_size
            fused_hidden_size_t = fused_hidden_size / ttnn.TILE_SIZE
            required_fused_hidden_size = (
                math.ceil(fused_hidden_size_t / ideal_core_grid_size) * ideal_core_grid_size * ttnn.TILE_SIZE
            )
            pad_size = int(required_fused_hidden_size - fused_hidden_size)
            w1_w3_fused = torch.nn.functional.pad(w1_w3_fused, (0, pad_size))
            w1_w3_bias_fused = torch.nn.functional.pad(w1_w3_bias_fused, (0, pad_size))
            weights_cache_suffix += f"_padded{pad_size}"
            bias_cache_suffix += f"_padded{pad_size}"

        w1_w3_fused_tt = _shard_experts_by_device(
            w1_w3_fused,
            num_devices,
            mesh_device,
            weight_dtype,
            get_cache_file_name(tensor_cache_path, weights_cache_suffix),
        )

        w1_w3_bias_fused_tt = _shard_experts_by_device(
            w1_w3_bias_fused,
            num_devices,
            mesh_device,
            ttnn.bfloat16,
            get_cache_file_name(tensor_cache_path, bias_cache_suffix),
            shard_dim=-3,  # Expert dimension after reshape
        )
    else:
        # ======================================================================
        # UNFUSED MODE: Load individual gate and up projection weights
        # ======================================================================
        w1_tt = _shard_experts_by_device(
            w1,
            num_devices,
            mesh_device,
            weight_dtype,
            get_cache_file_name(tensor_cache_path, "throughput_w1"),
        )

        w3_tt = _shard_experts_by_device(
            w3,
            num_devices,
            mesh_device,
            weight_dtype,
            get_cache_file_name(tensor_cache_path, "throughput_w3"),
        )

        # Load and shard bias (use bfloat16 for better precision)
        w1_bias_tt = _shard_experts_by_device(
            w1_bias,
            num_devices,
            mesh_device,
            ttnn.bfloat16,
            get_cache_file_name(tensor_cache_path, "throughput_w1_bias"),
            shard_dim=-3,  # Expert dimension after reshape
        )

        w3_bias_tt = _shard_experts_by_device(
            w3_bias,
            num_devices,
            mesh_device,
            ttnn.bfloat16,
            get_cache_file_name(tensor_cache_path, "throughput_w3_bias"),
            shard_dim=-3,  # Expert dimension after reshape
        )

    return ThroughputExpertWeights(
        w2=w2_tt,
        w2_bias=w2_bias_tt,
        w1=w1_tt,
        w3=w3_tt,
        w1_bias=w1_bias_tt,
        w3_bias=w3_bias_tt,
        w1_w3_fused=w1_w3_fused_tt,
        w1_w3_bias_fused=w1_w3_bias_fused_tt,
    )


# ===========================================================================
# Fused MoE kernel weight preparation
# ===========================================================================


def _tiles_for_core(ring_pos: int) -> int:
    """Return the number of valid tiles for a core at a given ring position."""
    return 7 if ring_pos in _FUSED_PAD_CORES else 8


def _prepare_w0_b0_w1_b1_tensor(
    torch_w0: torch.Tensor,
    torch_b0: torch.Tensor,
    torch_w1: torch.Tensor,
    torch_b1: torch.Tensor,
    L: int,
    E: int,
    K: int,
    N: int,
    ring2cores: dict,
) -> torch.Tensor:
    """Interleave, shard, and pad w0/w1 weights (with bias) for the fused MoE kernel.

    Takes w0 (gate) and w1 (up) weights of shape ``(L, E, K, N)`` along with
    their bias tensors of shape ``(L, E, K_b, N)`` and produces a tensor of
    shape ``(12, L, E, 4, K_new, 128)`` ready for DRAM HEIGHT_SHARDED placement,
    where ``K_new = K + K_b`` (e.g. 2880 + 32 = 2912).

    Each core receives a shard of ``(L, E, K_new, _FUSED_MAX_TILES_PER_CORE)``
    column tiles, with zero-padding applied to cores that receive fewer than
    ``_FUSED_MAX_TILES_PER_CORE`` real column tiles.

    Args:
        torch_w0: Gate weight tensor ``(L, E, K, N)``.
        torch_b0: Gate bias tensor ``(L, E, K_b, N)``.
        torch_w1: Up weight tensor ``(L, E, K, N)``.
        torch_b1: Up bias tensor ``(L, E, K_b, N)``.
        L: Number of layers.
        E: Number of experts.
        K: Input / hidden dimension (2880).
        N: Intermediate dimension (2880).
        ring2cores: Ring-position → ``(core_coord, dram_bank_id, pad_flag)`` mapping.

    Returns:
        Tensor of shape ``(num_cores, L, E, groups_per_core, K_new, 4*TILE_SIZE)``.
    """
    num_cores = len(ring2cores)
    torch_w0_b0 = torch.cat([torch_w0, torch_b0], dim=2)
    torch_w1_b1 = torch.cat([torch_w1, torch_b1], dim=2)
    K_new = torch_w0_b0.shape[2]  # K + K_b, e.g. 2912
    Nt = N // ttnn.TILE_SIZE

    w0_b0_chunks = torch_w0_b0.view(L, E, K_new, Nt, ttnn.TILE_SIZE)
    w1_b1_chunks = torch_w1_b1.view(L, E, K_new, Nt, ttnn.TILE_SIZE)
    stacked = torch.stack([w0_b0_chunks, w1_b1_chunks], dim=4)
    w0_b0_w1_b1_interleaved = stacked.view(L, E, K_new, Nt, 2 * ttnn.TILE_SIZE)
    w0_b0_w1_b1_permuted = w0_b0_w1_b1_interleaved.permute(0, 1, 3, 2, 4)

    each_shard = []
    start_tile = 0
    for ring_pos in range(num_cores):
        num_tiles = _tiles_for_core(ring_pos)
        shard = w0_b0_w1_b1_permuted[:, :, start_tile : start_tile + num_tiles, :, :]
        start_tile += num_tiles

        if num_tiles < _FUSED_MAX_TILES_PER_CORE:
            pad_tiles = _FUSED_MAX_TILES_PER_CORE - num_tiles
            padding = torch.zeros(L, E, pad_tiles, K_new, 2 * ttnn.TILE_SIZE, dtype=torch_w0.dtype)
            shard = torch.cat([shard, padding], dim=2)

        each_shard.append(shard)

    w0_b0_w1_b1_reordered = torch.cat(each_shard, dim=2)
    groups_per_core = _FUSED_MAX_TILES_PER_CORE // 2  # 4

    all_groups_per_bank = w0_b0_w1_b1_reordered.view(
        L, E, num_cores, _FUSED_MAX_TILES_PER_CORE, K_new, 2 * ttnn.TILE_SIZE
    )
    all_groups_per_bank = all_groups_per_bank.permute(2, 0, 1, 3, 4, 5)

    w0_b0_w1_b1_pair_2_tiles = all_groups_per_bank.view(num_cores, L, E, groups_per_core, 2, K_new, 2 * ttnn.TILE_SIZE)
    w0_b0_w1_b1_pair_2_tiles = w0_b0_w1_b1_pair_2_tiles.permute(0, 1, 2, 3, 5, 4, 6)
    w0_b0_w1_b1_paired = w0_b0_w1_b1_pair_2_tiles.reshape(num_cores, L, E, groups_per_core, K_new, 4 * ttnn.TILE_SIZE)

    # indices = torch.arange(K_new, dtype=w0_b0_w1_b1_paired.dtype)
    # indices[-32:] = -1
    # w0_b0_w1_b1_paired = indices.view(1, 1, 1, 1, K_new, 1).expand_as(w0_b0_w1_b1_paired)

    return w0_b0_w1_b1_paired


def _prepare_w2_b2_tensor(
    torch_w2: torch.Tensor,
    torch_b2: torch.Tensor,
    L: int,
    E: int,
    N: int,
    K: int,
    ring2cores: dict,
) -> torch.Tensor:
    """Shard, pad, and reorder w2 weights (with bias) for the fused MoE kernel.

    Takes w2 (down) weight of shape ``(L, E, N, K)`` and its bias tensor of
    shape ``(L, E, N_b, K)`` and produces a tensor of shape
    ``(12, L, E, 2, N_new, 128)`` ready for DRAM HEIGHT_SHARDED placement,
    where ``N_new = N + N_b`` (e.g. 2880 + 32 = 2912).

    Args:
        torch_w2: Down weight tensor ``(L, E, N, K)``.
        torch_b2: Down bias tensor ``(L, E, N_b, K)``.
        L: Number of layers.
        E: Number of experts.
        N: Intermediate dimension (2880).
        K: Output / hidden dimension (2880).
        ring2cores: Ring-position → ``(core_coord, dram_bank_id, pad_flag)`` mapping.

    Returns:
        Tensor of shape ``(num_cores, L, E, 2, N_new, 4*TILE_SIZE)``.
    """
    num_cores = len(ring2cores)
    torch_w2_b2 = torch.cat([torch_w2, torch_b2], dim=2)
    N_new = torch_w2_b2.shape[2]  # N + N_b, e.g. 2912
    each_shard = []

    start_col = 0
    for ring_pos in range(num_cores):
        (_, _, pad_flag) = ring2cores[ring_pos]

        if pad_flag:
            each_shard.append(torch_w2_b2[:, :, :, start_col : start_col + 4 * ttnn.TILE_SIZE])
            start_col += 4 * ttnn.TILE_SIZE
            each_shard.append(torch_w2_b2[:, :, :, start_col : start_col + 3 * ttnn.TILE_SIZE])
            start_col += 3 * ttnn.TILE_SIZE
            each_shard.append(torch.zeros(L, E, N_new, 1 * ttnn.TILE_SIZE, dtype=torch_w2.dtype))
        else:
            each_shard.append(torch_w2_b2[:, :, :, start_col : start_col + 4 * ttnn.TILE_SIZE])
            start_col += 4 * ttnn.TILE_SIZE
            each_shard.append(torch_w2_b2[:, :, :, start_col : start_col + 4 * ttnn.TILE_SIZE])
            start_col += 4 * ttnn.TILE_SIZE

    w2_b2_reordered = torch.cat(each_shard, dim=-1)
    all_groups_per_bank = w2_b2_reordered.view(L, E, N_new, num_cores, 2, 4 * ttnn.TILE_SIZE)
    all_groups_per_bank = all_groups_per_bank.permute(3, 0, 1, 4, 2, 5)

    Nt = N_new // ttnn.TILE_SIZE
    w2_b2_grouped = all_groups_per_bank.view(num_cores, L, E, 2, Nt, ttnn.TILE_SIZE, 4 * ttnn.TILE_SIZE)

    core_chunk_order = torch.tensor(list(reversed(range(num_cores)))).roll(1)
    chunk_sizes = [_tiles_for_core(i) for i in range(num_cores)]
    chunk_start_positions = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(torch.tensor(chunk_sizes, dtype=torch.int32), dim=0)]
    )

    each_shard = []
    for core_id in range(num_cores):
        each_chunk = []
        for chunk_id in core_chunk_order:
            start_pos = chunk_start_positions[chunk_id]
            end_pos = chunk_start_positions[chunk_id + 1]
            this_chunk = w2_b2_grouped[core_id, :, :, :, start_pos:end_pos, :, :]
            each_chunk.append(this_chunk)
        each_shard.append(torch.cat(each_chunk, dim=3))
        core_chunk_order = core_chunk_order.roll(1)

    w2_b2_paired = torch.stack(each_shard).view(num_cores, L, E, 2, -1, 4 * ttnn.TILE_SIZE)

    torch.set_printoptions(profile="full")
    torch.set_printoptions(sci_mode=False)
    with open(f"w2.txt", "w") as f:
        f.write(str(w2_b2_paired[0, 0, 0, 0, :, 0]))
    return w2_b2_paired


def _build_ring2cores(device) -> dict:
    """Build the ring-position → core mapping from device DRAM bank assignment.

    Args:
        device: A single ttnn device.

    Returns:
        Dictionary mapping ring position (int) to
        ``(core_coord, dram_bank_id, pad_flag)`` where *pad_flag* is 1 for
        cores with 7 tiles and 0 for cores with 8 tiles.
    """
    in0_core_coords = device.get_optimal_dram_bank_to_logical_worker_assignment(0)
    core2dram = {core_coords: dram_bank_id for dram_bank_id, core_coords in enumerate(in0_core_coords)}

    in0_core_coords_sorted = sorted(in0_core_coords, key=lambda x: (x.y, x.x), reverse=True)

    ring2cores = {}
    for ring_pos, core_coord in enumerate(in0_core_coords_sorted):
        ring2cores[ring_pos] = (
            core_coord,
            core2dram[core_coord],
            1 if ring_pos in _FUSED_PAD_CORES else 0,
        )
    return ring2cores


def prepare_fused_moe_kernel_weights(
    device,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    weight_dtype=ttnn.bfloat4_b,
) -> tuple:
    """Prepare weight tensors for the fused MoE kernel (``moe_gpt``).

    Transforms gate (w0), up (w1) and down (w2) projection weights from their
    natural shapes into the DRAM-sharded, interleaved, ring-ordered format
    required by the fused kernel.

    Args:
        device: A single ttnn device.
        w0: Gate projection weights ``[E, K, N]``.
        w1: Up projection weights ``[E, K, N]``.
        w2: Down projection weights ``[E, N, K]``.
        weight_dtype: Weight data type (default ``ttnn.bfloat4_b``).

    Returns:
        ``(tt_w0_w1, tt_w2, ring2cores)`` where *tt_w0_w1* and *tt_w2* are
        DRAM HEIGHT_SHARDED ttnn tensors and *ring2cores* is the mapping dict.
    """
    E, K, N = w0.shape
    L = 1  # single layer

    ring2cores = _build_ring2cores(device)
    num_cores = len(ring2cores)

    # Add L dimension: (E, K, N) -> (1, E, K, N)
    torch_w0 = w0.unsqueeze(0)
    torch_w1 = w1.unsqueeze(0)
    torch_w2 = w2.unsqueeze(0)

    # --- w0_w1 ---
    torch_w0_w1_reordered = _prepare_w0_w1_tensor(torch_w0, torch_w1, L, E, K, N, ring2cores)

    groups_per_core = _FUSED_MAX_TILES_PER_CORE // 2
    w0_w1_shard_height = L * E * groups_per_core * K
    w0_w1_shard_width = 4 * ttnn.TILE_SIZE

    dram_core_coords = [ttnn.CoreCoord(ring2cores[i][1], 0) for i in range(num_cores)]
    dram_core_range = [ttnn.CoreRange(c, c) for c in dram_core_coords]
    dram_core_range_set = ttnn.CoreRangeSet(dram_core_range)

    w0_w1_shard_spec = ttnn.ShardSpec(
        dram_core_range_set, (w0_w1_shard_height, w0_w1_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )
    w0_w1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w0_w1_shard_spec)

    tt_w0_w1 = ttnn.from_torch(
        torch_w0_w1_reordered,
        dtype=weight_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=w0_w1_mem_config,
    )

    # --- w2 ---
    torch_w2_reordered = _prepare_w2_tensor(torch_w2, L, E, N, K, ring2cores)

    w2_shard_height = L * E * 2 * N
    w2_shard_width = 4 * ttnn.TILE_SIZE

    w2_shard_spec = ttnn.ShardSpec(
        dram_core_range_set, (w2_shard_height, w2_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )
    w2_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w2_shard_spec)

    tt_w2 = ttnn.from_torch(
        torch_w2_reordered,
        dtype=weight_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=w2_mem_config,
    )

    return tt_w0_w1, tt_w2, ring2cores


def prepare_fused_moe_kernel_input(
    device,
    activation: torch.Tensor,
    ring2cores: dict,
) -> ttnn.Tensor:
    """Prepare an activation tensor for the fused MoE kernel.

    Replicates the activation across all 12 cores and returns an L1
    HEIGHT_SHARDED tensor.

    Args:
        device: A single ttnn device.
        activation: Input tensor ``[E, M, K]``.
        ring2cores: Mapping from :func:`prepare_fused_moe_kernel_weights`.

    Returns:
        L1 HEIGHT_SHARDED ttnn tensor with shape ``(12, E, M, K)`` and
        shard shape ``(E*M, K)``.
    """
    E, M, K = activation.shape
    num_cores = len(ring2cores)

    # Replicate across cores: (E, M, K) -> (num_cores, E, M, K)
    torch_input = activation.unsqueeze(0).repeat(num_cores, 1, 1, 1)

    in0_core_range = [ttnn.CoreRange(ring2cores[i][0], ring2cores[i][0]) for i in range(num_cores)]
    in0_core_range_set = ttnn.CoreRangeSet(in0_core_range)

    in0_shard_spec = ttnn.ShardSpec(
        grid=in0_core_range_set,
        shard_shape=(E * M, K),
        shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    input_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in0_shard_spec
    )

    return ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=input_sharded_mem_config,
    )


def extract_fused_moe_kernel_output(
    tt_output: ttnn.Tensor,
    E: int,
    M: int,
    K: int,
    ring2cores: dict,
) -> torch.Tensor:
    """Extract the fused MoE kernel output back to a torch tensor.

    Converts the L1 sharded output tensor to ``(E, M, K)`` by extracting the
    valid tiles per core (7 or 8) and concatenating along K.

    Args:
        tt_output: The L1 sharded output tensor from ``moe_gpt``.
        E: Number of experts.
        M: Sequence length / tokens.
        K: Hidden dimension.
        ring2cores: Mapping from :func:`prepare_fused_moe_kernel_weights`.

    Returns:
        Torch tensor of shape ``(E, M, K)``.
    """
    tt_raw_output = ttnn.to_torch(tt_output)

    each_shard = []
    for ring_pos in range(len(ring2cores)):
        (_, _, pad_flag) = ring2cores[ring_pos]
        num_tiles = 7 if pad_flag else 8
        each_shard.append(tt_raw_output[ring_pos, :, :, : num_tiles * ttnn.TILE_SIZE])

    result = torch.cat(each_shard, dim=-1)
    assert result.shape == (E, M, K), f"Expected shape {(E, M, K)}, got {result.shape}"
    return result
