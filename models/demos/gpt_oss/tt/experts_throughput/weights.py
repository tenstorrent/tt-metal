# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Weight loading and management for throughput-optimized MoE experts.

This module handles loading and sharding expert weights across devices for
the all_to_all-based throughput experts implementation.
"""

import math
from dataclasses import dataclass

import torch
from ttnn.operations.ccl import MoEActivationFunction

import ttnn
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name

from .config import ThroughputExpertConfig


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

    if state_dict:
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
    else:
        w1 = None
        w3 = None
        w1_bias = None
        w3_bias = None
        w2 = None
        w2_bias = None

    # Compute w2 cache suffix based on config (independent of state_dict)
    w2_cache_suffix = ""
    if config.pad_w2:
        # Pad w2 output dimension for tile alignment in CCL operations.
        # Without padding, local_hidden = hidden_size / TP may not be tile-aligned (e.g., 2880/8 = 360),
        # causing CCL to do expensive Untilize->Pad->Tilize cycles internally.
        scattered_local_hidden = hidden_size // mesh_device.shape[1]
        padded_scattered_local_hidden = ((scattered_local_hidden + 31) // 32) * 32  # Round up to tile boundary
        w2_pad_size = (padded_scattered_local_hidden * mesh_device.shape[1]) - hidden_size
        # Use unique cache key when padding is applied
        w2_cache_suffix = f"_padded{w2_pad_size}" if w2_pad_size > 0 and mesh_device.shape[1] > 1 else ""

        if state_dict and w2_pad_size > 0 and mesh_device.shape[1] > 1:
            # Pad the output dimension of w2 weight: [input_dim, hidden_size] -> [input_dim, padded_hidden]
            w2 = torch.nn.functional.pad(w2, (0, w2_pad_size), "constant", value=0.0)
            w2_bias = torch.nn.functional.pad(w2_bias, (0, w2_pad_size), "constant", value=0.0)

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

        if config.pad_w1_w3:
            ideal_core_grid_size = 64
            fused_hidden_size = 2 * intermediate_size
            fused_hidden_size_t = fused_hidden_size / ttnn.TILE_SIZE
            required_fused_hidden_size = (
                math.ceil(fused_hidden_size_t / ideal_core_grid_size) * ideal_core_grid_size * ttnn.TILE_SIZE
            )
            pad_size = int(required_fused_hidden_size - fused_hidden_size)
            weights_cache_suffix += f"_padded{pad_size}"
            bias_cache_suffix += f"_padded{pad_size}"
        else:
            pad_size = 0

        if state_dict:
            # Fuse biases
            # w1_bias: [1, num_experts, 1, intermediate_size]
            # w3_bias: [1, num_experts, 1, intermediate_size]
            # -> w1_w3_bias_fused: [1, num_experts, 1, 2*intermediate_size]
            w1_w3_fused = torch.cat([w1, w3], dim=-1)
            w1_w3_bias_fused = torch.cat([w1_bias, w3_bias], dim=-1)
            if pad_size > 0:
                w1_w3_fused = torch.nn.functional.pad(w1_w3_fused, (0, pad_size))
                w1_w3_bias_fused = torch.nn.functional.pad(w1_w3_bias_fused, (0, pad_size))
        else:
            w1_w3_fused = None
            w1_w3_bias_fused = None

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
# FusedMoeComputeConfig factory
# ===========================================================================


def _get_moe_mux_core_range_set(mesh_device, output_height_shard_dim, hidden_size):
    """Determine mux core range set for moe_compute based on combine core placement."""
    from ttnn._experimental.moe_compute_utils import auto_output_width_shard_dim

    output_width_shard_dim = auto_output_width_shard_dim(hidden_size)
    combine_cores = ttnn.experimental.get_moe_combine_cores(
        mesh_device,
        output_height_shard_dim,
        output_width_shard_dim,
        hidden_size,
    )
    combine_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in combine_cores])

    compute_grid = mesh_device.compute_with_storage_grid_size()
    last_combine = combine_cores[-1]
    mux_start = ttnn.CoreCoord(last_combine.x + 1, combine_cores[0].y)
    mux_end = ttnn.CoreCoord(min(last_combine.x + 2, compute_grid.x - 1), last_combine.y)
    return ttnn.CoreRangeSet([ttnn.CoreRange(mux_start, mux_end)])


def create_fused_moe_compute_config(
    mesh_device,
    config: "ThroughputExpertConfig",
    state_dict: dict,
    tokens_per_device: int,
    weight_dtype=ttnn.bfloat4_b,
    cluster_axis: int = 0,
    num_links: int = 4,
    tensor_cache_path: str = None,
):
    """Create a FusedMoeComputeConfig with pre-allocated resources for the fused decode flow.

    Builds all tensors and semaphores required by fused_decode_forward:
      all_to_all_dispatch_metadata → moe_compute (Full mode, with integrated combine)

    Uses global expert IDs (0..num_experts-1) and a global mapping tensor of shape
    [total_devices, num_experts] with mapping[d, e] = e // experts_per_device (all rows
    identical). Weights for this device's experts_per_device local experts are loaded
    from state_dict and prepared using ttnn.experimental moe_compute_utils.

    Args:
        mesh_device: TTNN mesh device (e.g., (4,8) Galaxy mesh).
        config: ThroughputExpertConfig with num_experts (total=128), hidden_size, etc.
        state_dict: Experts state dict with keys 'gate_up_proj'/'gate_proj'/'up_proj',
            'down_proj'. Shapes: [num_experts_total, K, {2N or N}] for gate/up,
            [num_experts_total, N, K] for down.
        tokens_per_device: Tokens per device per ring (M, e.g., 32 for 128 total / 4 rows).
        weight_dtype: Weight quantization type (default: bfloat4_b).
        cluster_axis: Mesh axis for the ring dispatch (default: 0, ring along rows).
        num_links: Number of fabric links for all CCL ops (default: 4).

    Returns:
        FusedMoeComputeConfig with all pre-allocated tensors and semaphores.
    """
    from .config import FusedMoeComputeConfig

    ring_devices = mesh_device.shape[cluster_axis]
    total_devices = mesh_device.get_num_devices()
    experts_per_ring = config.num_experts_per_device * ring_devices  # e.g., 1*4=4 (20b) or 4*4=16 (120b)
    experts_per_device = config.num_experts_per_device  # e.g., 1 (20b) or 4 (120b)
    total_tokens = tokens_per_device * ring_devices  # e.g., 32 * 4 = 128
    K = config.hidden_size
    N = config.intermediate_size
    E = experts_per_device
    L = 1  # single-layer mode
    mesh_cols = total_devices // ring_devices
    experts_per_cluster = config.num_experts // mesh_cols
    OUTPUT_HEIGHT_SHARD_DIM = 4

    # --- Extract expert weights from state_dict ---
    if state_dict:
        if "gate_up_proj" in state_dict:
            gate_up_all = state_dict["gate_up_proj"]  # [num_experts, K, 2N]
            w0_all = gate_up_all[..., ::2].contiguous().float()  # [num_experts, K, N]
            w1_all = gate_up_all[..., 1::2].contiguous().float()  # [num_experts, K, N]
        else:
            w0_all = state_dict["gate_proj"].contiguous().float()
            w1_all = state_dict["up_proj"].contiguous().float()
        w2_all = state_dict["down_proj"].contiguous().float()  # [num_experts, N, K]

        # Extract bias tensors
        if "gate_up_proj_bias" in state_dict:
            gate_up_bias = state_dict["gate_up_proj_bias"]
            b0_all = gate_up_bias[..., ::2].contiguous().float()  # [num_experts, N]
            b1_all = gate_up_bias[..., 1::2].contiguous().float()  # [num_experts, N]
        elif "gate_up_proj" in state_dict:
            b0_all = torch.zeros(w0_all.shape[0], N, dtype=torch.float32)
            b1_all = torch.zeros(w1_all.shape[0], N, dtype=torch.float32)
        else:
            b0_all = state_dict.get("gate_proj_bias", torch.zeros(w0_all.shape[0], N)).contiguous().float()
            b1_all = state_dict.get("up_proj_bias", torch.zeros(w1_all.shape[0], N)).contiguous().float()
        b2_all = state_dict.get("down_proj_bias", torch.zeros(w2_all.shape[0], K)).contiguous().float()

        # Organize per-device: each device at (row r, col c) owns experts
        # [c*epc + r*E : c*epc + (r+1)*E] where epc = experts_per_cluster.
        # The C++ prep functions handle bias padding internally (L, E, N) → (L, E, 32, N).
        w0_rows, w1_rows, w2_rows = [], [], []
        b0_rows, b1_rows, b2_rows = [], [], []
        for r in range(ring_devices):
            w0_cols, w1_cols, w2_cols = [], [], []
            b0_cols, b1_cols, b2_cols = [], [], []
            for c in range(mesh_cols):
                start = c * experts_per_cluster + r * E
                w0_cols.append(w0_all[start : start + E].unsqueeze(0))  # [1, E, K, N]
                w1_cols.append(w1_all[start : start + E].unsqueeze(0))
                w2_cols.append(w2_all[start : start + E].unsqueeze(0))  # [1, E, N, K]
                b0_cols.append(b0_all[start : start + E].unsqueeze(0))  # [1, E, N]
                b1_cols.append(b1_all[start : start + E].unsqueeze(0))
                b2_cols.append(b2_all[start : start + E].unsqueeze(0))  # [1, E, K]
            w0_rows.append(torch.cat(w0_cols, dim=0))
            w1_rows.append(torch.cat(w1_cols, dim=0))
            w2_rows.append(torch.cat(w2_cols, dim=0))
            b0_rows.append(torch.cat(b0_cols, dim=0))
            b1_rows.append(torch.cat(b1_cols, dim=0))
            b2_rows.append(torch.cat(b2_cols, dim=0))
        torch_w0 = torch.cat(w0_rows, dim=0).reshape(ring_devices * mesh_cols, L, E, K, N)
        torch_w1 = torch.cat(w1_rows, dim=0).reshape(ring_devices * mesh_cols, L, E, K, N)
        torch_w2 = torch.cat(w2_rows, dim=0).reshape(ring_devices * mesh_cols, L, E, N, K)
        torch_b0 = torch.cat(b0_rows, dim=0).reshape(ring_devices * mesh_cols, L, E, N)
        torch_b1 = torch.cat(b1_rows, dim=0).reshape(ring_devices * mesh_cols, L, E, N)
        torch_b2 = torch.cat(b2_rows, dim=0).reshape(ring_devices * mesh_cols, L, E, K)
    else:
        torch_w0, torch_w1, torch_w2 = None, None, None
        torch_b0, torch_b1, torch_b2 = None, None, None

    # --- Prepare weights using moe_compute generalized utils ---
    weight_mem_configs = ttnn.experimental.get_weight_mem_configs(
        mesh_device,
        num_layers=L,
        experts_per_device=E,
        hidden_size=K,
        intermediate_size=N,
        has_bias=True,
    )

    def _upload_raw(torch_tensor):
        if torch_tensor is None:
            return None
        return ttnn.from_torch(
            torch_tensor,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )

    tt_w0_raw = _upload_raw(torch_w0)
    tt_w1_raw = _upload_raw(torch_w1)
    tt_w2_raw = _upload_raw(torch_w2)
    tt_b0_raw = _upload_raw(torch_b0)
    tt_b1_raw = _upload_raw(torch_b1)
    tt_b2_raw = _upload_raw(torch_b2)

    tt_w0_w1_prepped = ttnn.experimental.prepare_w0_w1_tensor_with_bias(
        tt_w0_raw,
        tt_w1_raw,
        tt_b0_raw,
        tt_b1_raw,
        L=L,
        E=E,
        K=K,
        N=N,
    )
    ttnn.deallocate(tt_w0_raw)
    ttnn.deallocate(tt_w1_raw)
    ttnn.deallocate(tt_b0_raw)
    ttnn.deallocate(tt_b1_raw)

    tt_w0_w1 = ttnn.experimental.quantize_weights_via_host(
        tt_w0_w1_prepped,
        dtype=weight_dtype,
        memory_config=weight_mem_configs.w0_w1,
    )
    ttnn.deallocate(tt_w0_w1_prepped)

    tt_w2_prepped = ttnn.experimental.prepare_w2_tensor_with_bias(
        tt_w2_raw,
        tt_b2_raw,
        L=L,
        E=E,
        N=N,
        K=K,
    )
    ttnn.deallocate(tt_w2_raw)
    ttnn.deallocate(tt_b2_raw)

    tt_w2 = ttnn.experimental.quantize_weights_via_host(
        tt_w2_prepped,
        dtype=weight_dtype,
        memory_config=weight_mem_configs.w2,
    )
    ttnn.deallocate(tt_w2_prepped)

    # --- Expert routing mapping: [total_devices, num_experts] uint16 ---
    num_experts = config.num_experts
    mapping_row = torch.zeros(num_experts, dtype=torch.int16)
    for e in range(num_experts):
        cluster_id = e // experts_per_cluster
        expert_within_cluster = e % experts_per_cluster
        device_within_cluster = expert_within_cluster // experts_per_device
        mapping_row[e] = device_within_cluster * mesh_cols + cluster_id
    mapping_data = mapping_row.unsqueeze(0).repeat(total_devices, 1)

    tt_dispatch_mapping = ttnn.from_torch(
        mapping_data,
        device=mesh_device,
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_expert_mapping = ttnn.from_torch(
        mapping_data,
        device=mesh_device,
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # --- Pre-allocate dispatch output tensors ---
    mux_core_range_set = _get_moe_mux_core_range_set(mesh_device, OUTPUT_HEIGHT_SHARD_DIM, K)
    tilize_drain_core_coord = ttnn.experimental.get_moe_tilize_drain_core(
        mesh_device,
        OUTPUT_HEIGHT_SHARD_DIM,
        0,
        K,
        mux_core_range_set=mux_core_range_set,
    )
    dispatch_drain_core = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(tilize_drain_core_coord.x, tilize_drain_core_coord.y),
                ttnn.CoreCoord(tilize_drain_core_coord.x, tilize_drain_core_coord.y),
            )
        }
    )

    tt_dispatch_sparse = ttnn.from_torch(
        torch.zeros(ring_devices, total_tokens, K, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=tuple(mesh_device.shape)),
    )

    dispatch_metadata_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            dispatch_drain_core,
            [total_tokens, config.num_experts_per_tok],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    tt_dispatch_indices = ttnn.from_torch(
        torch.zeros(ring_devices, total_tokens, config.num_experts_per_tok, dtype=torch.int16),
        dtype=ttnn.uint16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=dispatch_metadata_mem_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=tuple(mesh_device.shape)),
    )
    tt_dispatch_scores = ttnn.from_torch(
        torch.zeros(ring_devices, total_tokens, config.num_experts_per_tok, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=dispatch_metadata_mem_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=tuple(mesh_device.shape)),
    )

    # --- Global semaphores ---
    compute_grid = mesh_device.compute_with_storage_grid_size()
    all_worker_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1))}
    )
    dispatch_semaphore = ttnn.create_global_semaphore(mesh_device, all_worker_cores, 0)
    combine_semaphore = ttnn.create_global_semaphore(mesh_device, all_worker_cores, 0)

    # --- Pre-allocate combine output ---
    K_sel = config.num_experts_per_tok
    tt_combine_preallocated = ttnn.from_torch(
        torch.zeros(K_sel, tokens_per_device, K, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    return FusedMoeComputeConfig(
        tt_w0_w1=tt_w0_w1,
        tt_w2=tt_w2,
        tt_dispatch_mapping=tt_dispatch_mapping,
        tt_expert_mapping=tt_expert_mapping,
        dispatch_sparse=tt_dispatch_sparse,
        dispatch_indices=tt_dispatch_indices,
        dispatch_scores=tt_dispatch_scores,
        dispatch_semaphore=dispatch_semaphore,
        combine_preallocated=tt_combine_preallocated,
        combine_semaphore=combine_semaphore,
        cluster_axis=cluster_axis,
        output_height_shard_dim=OUTPUT_HEIGHT_SHARD_DIM,
        mux_core_range_set=mux_core_range_set,
        has_bias=True,
        activation_type=MoEActivationFunction.SWIGLU,
        num_links=num_links,
    )
