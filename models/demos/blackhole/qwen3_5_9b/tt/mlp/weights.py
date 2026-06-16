# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Weight loading for the Qwen3.5 SwiGLU MLP (consumed by mlp.py).

w1/w3 are column-parallel (output dim sharded) and w2 row-parallel (input dim
sharded), so each device's down-projection emits a partial sum that mlp.py
reduces across the mesh. On a unit mesh each "shard" is the full weight, so
the same loader serves single-device runs.
"""
import os
from dataclasses import dataclass

import ttnn
from models.demos.blackhole.qwen3_5_9b.tt import tp_common as tpc


@dataclass(frozen=True)
class MLPWeights:
    """Per-device weight shards, transposed to [in, out] for ttnn.linear.
    Gate/up use bfloat4_b (halves DRAM bandwidth for the two biggest matmuls);
    down stays bfloat8_b because it sits on the critical accuracy path."""

    w1: ttnn.Tensor  # gate_proj, column-parallel, bfloat4_b
    w2: ttnn.Tensor  # down_proj, row-parallel, bfloat8_b
    w3: ttnn.Tensor  # up_proj, column-parallel, bfloat4_b


def load_mlp_weights(mesh_device, state_dict, tensor_cache_path=None) -> MLPWeights:
    """state_dict is the per-layer mlp substate: keys 'gate_proj.weight', 'down_proj.weight', 'up_proj.weight'."""

    if tensor_cache_path is not None:
        os.makedirs(tensor_cache_path, exist_ok=True)

    def cache(name):
        return str(tensor_cache_path / f"mlp.{name}.weight.tp") if tensor_cache_path else None

    # Shard across devices but keep each device's shard INTERLEAVED in DRAM so
    # a regular ttnn.linear serves both decode (M=1 tile) and prefill
    # (M=seq_len). DRAM-width-sharding the weights for a faster decode matmul
    # is a later optimization.
    return MLPWeights(
        w1=tpc.shard_w(
            state_dict["gate_proj.weight"],
            mesh_device,
            dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_path=cache("gate_proj"),
            dtype=ttnn.bfloat4_b,
        ),
        w3=tpc.shard_w(
            state_dict["up_proj.weight"],
            mesh_device,
            dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_path=cache("up_proj"),
            dtype=ttnn.bfloat4_b,
        ),
        w2=tpc.shard_w(
            state_dict["down_proj.weight"],
            mesh_device,
            dim=0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_path=cache("down_proj"),
            dtype=ttnn.bfloat8_b,
        ),
    )
