# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Generic expert weight loading and management."""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.gpt_oss.config import MeshConfig, Mode
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name

from .config import ExpertConfig


@dataclass(frozen=True)  # ✅ Make immutable to prevent accidental modification
class ExpertWeights:
    """Container for expert weight tensors - immutable after creation"""

    gate_proj: ttnn.Tensor
    up_proj: ttnn.Tensor
    down_proj: ttnn.Tensor
    gate_proj_bias: ttnn.Tensor
    up_proj_bias: ttnn.Tensor
    down_proj_bias: ttnn.Tensor
    intermediate_size_per_device: int


def load_expert_weights(
    mesh_device,
    config: ExpertConfig,
    state_dict,
    mesh_config: MeshConfig,
    weight_dtype=ttnn.bfloat4_b,
    tensor_cache_path=None,
) -> ExpertWeights:
    """
    Load and shard expert weights.

    Args:
        mesh_device: TTNN mesh device
        config: Expert configuration
        state_dict: Dictionary with expert weights
        mesh_config: Mesh parallelization configuration
        weight_dtype: Data type for weights
        tensor_cache_path: Optional path for weight caching

    Returns:
        ExpertWeights with loaded and sharded tensors
    """
    # Calculate sharded dimensions
    intermediate_size_per_device = mesh_config.shard_size(config.intermediate_size, mode=Mode.DECODE)

    if state_dict:
        # Extract gate and up projections from fused weight
        gate_proj = state_dict["gate_up_proj"][..., ::2].reshape(
            1, config.num_experts, config.hidden_size, config.intermediate_size
        )
        up_proj = state_dict["gate_up_proj"][..., 1::2].reshape(
            1, config.num_experts, config.hidden_size, config.intermediate_size
        )
        gate_proj_bias = state_dict["gate_up_proj_bias"][..., ::2].reshape(
            1, config.num_experts, config.intermediate_size
        )
        up_proj_bias = state_dict["gate_up_proj_bias"][..., 1::2].reshape(
            1, config.num_experts, config.intermediate_size
        )
    else:
        gate_proj = None
        up_proj = None
        gate_proj_bias = None
        up_proj_bias = None
    # Get mesh mappers
    col_mesh_mapper = mesh_config.column_parallel(mesh_device)
    row_mesh_mapper = mesh_config.row_parallel(mesh_device)

    # Load gate projection
    gate_proj_tt = ttnn.as_tensor(
        gate_proj,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=col_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, "gate_proj"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Load up projection
    up_proj_tt = ttnn.as_tensor(
        up_proj,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=col_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, "up_proj"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    bias_dtype = ttnn.bfloat16
    # Load gate bias
    gate_proj_bias_tt = ttnn.as_tensor(
        gate_proj_bias,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=bias_dtype,
        mesh_mapper=col_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"gate_proj_bias"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Load up bias
    up_proj_bias_tt = ttnn.as_tensor(
        up_proj_bias,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=bias_dtype,
        mesh_mapper=col_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"up_proj_bias"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Load down projection
    if state_dict:
        down_proj = state_dict["down_proj"].reshape(1, config.num_experts, config.intermediate_size, config.hidden_size)
        down_proj_bias = state_dict["down_proj_bias"].reshape(1, config.num_experts, config.hidden_size)
        # Handle row-parallel bias (must not be replicated across TP devices)
        if mesh_config.decode.tp > 1:
            down_proj_bias = torch.cat(
                [down_proj_bias] + [torch.zeros_like(down_proj_bias)] * (mesh_config.decode.tp - 1), dim=-1
            )
    else:
        down_proj = None
        down_proj_bias = None

    down_proj_tt = ttnn.as_tensor(
        down_proj,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=row_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, "down_proj"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    down_proj_bias_tt = ttnn.as_tensor(
        down_proj_bias,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=bias_dtype,
        mesh_mapper=col_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"down_proj_bias"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return ExpertWeights(
        gate_proj=gate_proj_tt,
        up_proj=up_proj_tt,
        down_proj=down_proj_tt,
        gate_proj_bias=gate_proj_bias_tt,
        up_proj_bias=up_proj_bias_tt,
        down_proj_bias=down_proj_bias_tt,
        intermediate_size_per_device=intermediate_size_per_device,
    )


# Decode-only expert weights for the streamed MoE (experts/stream.py). Each TP shard of the intermediate dimension is
# zero-padded up to a multiple of DECODE_INTERMEDIATE_ALIGN so the gate|up column pairs split evenly over the DRAM
# banks (720 -> 768 = 24 tiles = 3 pairs per bank for gpt-oss-20b at TP=4). Padded gate/up columns are zero (weight
# and bias), so their SwiGLU output is exactly zero and the padded down rows contribute nothing.
DECODE_INTERMEDIATE_ALIGN = 256


@dataclass(frozen=True)
class DecodeExpertWeights:
    # Per device: packed [gate | up] (+ bias) and down (+ bias, real on the first TP device only) of every expert, in
    # the per-DRAM-bank streamed layouts of experts/stream.py.
    gate_up_stream: ttnn.Tensor
    down_stream: ttnn.Tensor
    intermediate_padded: int


def load_decode_expert_weights(
    mesh_device,
    config: ExpertConfig,
    state_dict,
    mesh_config: MeshConfig,
    weight_dtype=ttnn.bfloat4_b,
    tensor_cache_path=None,
) -> DecodeExpertWeights:
    from .stream import NBIAS, as_stream_tensor, columns_per_bank, stream_down_layout, stream_gate_up_layout

    tp = mesh_config.decode.tp
    inter_local = config.intermediate_size // tp
    inter_pad = -(-inter_local // DECODE_INTERMEDIATE_ALIGN) * DECODE_INTERMEDIATE_ALIGN
    E, H = config.num_experts, config.hidden_size
    banks = mesh_device.dram_grid_size().x
    tile = ttnn.TILE_SIZE

    def pad_shards(t, dim):
        """Split t along dim into tp shards, zero-pad each shard to inter_pad, concatenate."""
        chunks = torch.chunk(t, tp, dim=dim)
        pad = [0, 0] * (t.dim() - 1 - (dim % t.dim())) + [0, inter_pad - inter_local]
        return torch.cat([torch.nn.functional.pad(c, pad) for c in chunks], dim=dim)

    gu_layouts = dn_layouts = None
    if state_dict:
        gate_up = state_dict["gate_up_proj"]
        gate_up_bias = state_dict["gate_up_proj_bias"]
        gate = pad_shards(gate_up[..., ::2].reshape(1, E, H, config.intermediate_size), -1)
        up = pad_shards(gate_up[..., 1::2].reshape(1, E, H, config.intermediate_size), -1)
        gate_bias = pad_shards(gate_up_bias[..., ::2].reshape(E, 1, config.intermediate_size), -1)
        up_bias = pad_shards(gate_up_bias[..., 1::2].reshape(E, 1, config.intermediate_size), -1)
        down = pad_shards(state_dict["down_proj"].reshape(1, E, config.intermediate_size, H), -2)
        down_bias = state_dict["down_proj_bias"].reshape(E, 1, H)
        # Row-parallel down: the bias is added once, on the first TP device.
        down_bias = torch.cat([down_bias] + [torch.zeros_like(down_bias)] * (tp - 1), dim=-1)
        # One streamed layout per TP shard: [E, H, 2 * I_pad] = [gate shard | up shard] and its [E, 2 * I_pad] bias;
        # [E, I_pad, H] down rows and the [E, H] bias.
        gu_layouts = [
            stream_gate_up_layout(torch.cat([g[0], u[0]], -1), torch.cat([gb[:, 0], ub[:, 0]], -1), banks)
            for g, u, gb, ub in zip(
                torch.chunk(gate, tp, -1),
                torch.chunk(up, tp, -1),
                torch.chunk(gate_bias, tp, -1),
                torch.chunk(up_bias, tp, -1),
            )
        ]
        dn_layouts = [
            stream_down_layout(w[0], b[:, 0], banks)
            for w, b in zip(torch.chunk(down, tp, -2), torch.chunk(down_bias, tp, -1))
        ]

    suffix = f"decode_i{inter_pad}_b{banks}_nb{NBIAS}"
    rows = {
        "gate_up": E * 2 * inter_pad // tile // banks * (H // tile + 1) * tile,
        "down": E * columns_per_bank(H, banks) * (inter_pad // tile + 1) * tile,
    }
    return DecodeExpertWeights(
        gate_up_stream=as_stream_tensor(
            mesh_device,
            gu_layouts,
            rows["gate_up"],
            weight_dtype,
            get_cache_file_name(tensor_cache_path, f"gate_up_stream_{suffix}"),
        ),
        down_stream=as_stream_tensor(
            mesh_device,
            dn_layouts,
            rows["down"],
            weight_dtype,
            get_cache_file_name(tensor_cache_path, f"down_stream_{suffix}"),
        ),
        intermediate_padded=inter_pad,
    )
