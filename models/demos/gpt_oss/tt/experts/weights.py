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


# Decode-only expert weights for the indexed sparse_matmul path (experts/decode.py: decode_forward_indexed).
# Each TP shard of the intermediate dimension is zero-padded up to a multiple of DECODE_INTERMEDIATE_ALIGN so the
# gate/up projections split into whole-tile columns over a rectangular core grid (720 -> 768 = 24 tiles = 8x3 cores
# for gpt-oss-20b at TP=4; the natural 23-tile padding only fits 12 cores). Padded gate/up columns are zero (weight
# and bias), so their SwiGLU output is exactly zero and the padded down rows contribute nothing.
DECODE_INTERMEDIATE_ALIGN = 256


@dataclass(frozen=True)
class DecodeExpertWeights:
    # gate and up share their input: packed as [gate | up] along N so one indexed matmul computes both.
    gate_up_proj: ttnn.Tensor  # [1, E, hidden, 2 * I_pad] per device
    down_proj: ttnn.Tensor  # [1, E, I_pad, hidden]
    gate_up_proj_bias: ttnn.Tensor  # [E, 1 (32), 2 * I_pad]: tile row e holds expert e's bias (indexed fused bias)
    down_proj_bias: ttnn.Tensor  # [E, 1 (32), hidden]; real values on the first TP device only (row-parallel)
    intermediate_padded: int


def load_decode_expert_weights(
    mesh_device,
    config: ExpertConfig,
    state_dict,
    mesh_config: MeshConfig,
    weight_dtype=ttnn.bfloat4_b,
    tensor_cache_path=None,
) -> DecodeExpertWeights:
    tp = mesh_config.decode.tp
    inter_local = config.intermediate_size // tp
    inter_pad = -(-inter_local // DECODE_INTERMEDIATE_ALIGN) * DECODE_INTERMEDIATE_ALIGN
    E, H = config.num_experts, config.hidden_size

    def pad_shards(t, dim):
        """Split t along dim into tp shards, zero-pad each shard to inter_pad, concatenate."""
        chunks = torch.chunk(t, tp, dim=dim)
        pad = [0, 0] * (t.dim() - 1 - (dim % t.dim())) + [0, inter_pad - inter_local]
        return torch.cat([torch.nn.functional.pad(c, pad) for c in chunks], dim=dim)

    if state_dict:
        gate_up = state_dict["gate_up_proj"]
        gate_up_bias = state_dict["gate_up_proj_bias"]
        gate = pad_shards(gate_up[..., ::2].reshape(1, E, H, config.intermediate_size), -1)
        up = pad_shards(gate_up[..., 1::2].reshape(1, E, H, config.intermediate_size), -1)
        gate_bias = pad_shards(gate_up_bias[..., ::2].reshape(E, 1, config.intermediate_size), -1)
        up_bias = pad_shards(gate_up_bias[..., 1::2].reshape(E, 1, config.intermediate_size), -1)

        def pack(g, u):
            """Per TP shard: [gate shard | up shard] so each device holds its packed [.., 2 * I_pad] block."""
            return torch.cat([t for pair in zip(torch.chunk(g, tp, -1), torch.chunk(u, tp, -1)) for t in pair], -1)

        gate_up = pack(gate, up)
        gate_up_bias = pack(gate_bias, up_bias)
        down = pad_shards(state_dict["down_proj"].reshape(1, E, config.intermediate_size, H), -2)
        down_bias = state_dict["down_proj_bias"].reshape(E, 1, H)
        down_bias = torch.cat([down_bias] + [torch.zeros_like(down_bias)] * (tp - 1), dim=-1)
    else:
        gate_up = gate_up_bias = down = down_bias = None

    col = mesh_config.column_parallel(mesh_device)
    row = mesh_config.row_parallel(mesh_device)
    suffix = f"decode_i{inter_pad}"

    def load(t, name, dtype, mapper):
        return ttnn.as_tensor(
            t,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            mesh_mapper=mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, f"{name}_{suffix}"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return DecodeExpertWeights(
        gate_up_proj=load(gate_up, "gate_up_proj", weight_dtype, col),
        down_proj=load(down, "down_proj", weight_dtype, row),
        gate_up_proj_bias=load(gate_up_bias, "gate_up_proj_bias", ttnn.bfloat16, col),
        down_proj_bias=load(down_bias, "down_proj_bias", ttnn.bfloat16, col),
        intermediate_padded=inter_pad,
    )
