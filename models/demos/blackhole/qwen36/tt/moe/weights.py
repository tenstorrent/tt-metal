# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Weight loading for Qwen3.5-MoE routed experts.

The 35B-A3B checkpoint stores experts FUSED as 3D nn.Parameters (no `.weight`):
  mlp.experts.gate_up_proj  [E, 2*I, H]  (rows [:I]=gate, [I:]=up)
  mlp.experts.down_proj     [E, H, I]
Older per-expert checkpoints (`{i}.gate_proj.weight` etc.) are stacked into the same
fused layout up front, so a single code path handles both.

TP sharding across the (1,4) mesh: gate/up are column-parallel (shard the intermediate
dim), down is row-parallel (shard the intermediate dim); the expert dim (dim=1) is
REPLICATED. A reduce-scatter after down_proj (in decode/prefill) recombines the
row-parallel partials. gate/up are bfloat4_b, down is bfloat8_b (matching Qwen36MLP).
"""

from dataclasses import dataclass

import torch

import ttnn

TILE_SIZE = 32


@dataclass(frozen=True)
class ExpertWeights:
    gate_proj: ttnn.Tensor  # [1, E, H, I_per_device]
    up_proj: ttnn.Tensor  # [1, E, H, I_per_device]
    down_proj: ttnn.Tensor  # [1, E, I_per_device, H]
    intermediate_size_per_device: int
    gate_up_proj: ttnn.Tensor = None  # [1, E, H, 2*I_per_device] = concat(gate, up) on N


def _stack_per_expert(state_dict, intermediate_size):
    """Stack unfused per-expert weights into the fused [E,2I,H] / [E,H,I] layout."""
    gate_ups, downs = [], []
    i = 0
    while f"{i}.gate_proj.weight" in state_dict:
        g = state_dict[f"{i}.gate_proj.weight"]  # [I, H]
        u = state_dict[f"{i}.up_proj.weight"]  # [I, H]
        d = state_dict[f"{i}.down_proj.weight"]  # [H, I]
        gate_ups.append(torch.cat([g, u], dim=0))  # [2I, H]
        downs.append(d)
        i += 1
    assert gate_ups, "no fused `gate_up_proj` and no per-expert `{i}.gate_proj.weight` keys found"
    return {"gate_up_proj": torch.stack(gate_ups, 0), "down_proj": torch.stack(downs, 0)}


def load_expert_weights(
    mesh_device,
    config,
    state_dict,
    tensor_cache_path=None,
    gate_up_dtype=ttnn.bfloat4_b,
    down_dtype=ttnn.bfloat8_b,
) -> ExpertWeights:
    E = config.num_experts
    I = config.moe_intermediate_size
    tp = config.num_devices
    is_mesh = hasattr(mesh_device, "shape")

    gate_proj = up_proj = down_proj = None
    if state_dict:
        if "gate_up_proj" not in state_dict:
            state_dict = _stack_per_expert(state_dict, I)

        fused = state_dict["gate_up_proj"].to(torch.bfloat16)  # [E, 2I, H]
        gate_t = fused[:, :I, :]  # [E, I, H]
        up_t = fused[:, I:, :]  # [E, I, H]
        # Transpose to ttnn.linear (in, out) convention: [E, I, H] -> [1, E, H, I]
        gate_proj = gate_t.transpose(-2, -1).unsqueeze(0).contiguous()
        up_proj = up_t.transpose(-2, -1).unsqueeze(0).contiguous()
        # down: [E, H, I] -> [1, E, I, H]
        down_proj = state_dict["down_proj"].to(torch.bfloat16).transpose(-2, -1).unsqueeze(0).contiguous()

        # Pad the intermediate dim so each device's shard is a whole number of tiles.
        if tp > 1:
            per_device = I // tp
            padded_per_device = ((per_device + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
            pad_amount = padded_per_device * tp - I
            if pad_amount > 0:
                gate_proj = torch.nn.functional.pad(gate_proj, (0, pad_amount))  # last dim (I)
                up_proj = torch.nn.functional.pad(up_proj, (0, pad_amount))
                down_proj = torch.nn.functional.pad(down_proj, (0, 0, 0, pad_amount))  # dim -2 (I)

    per_device_intermediate = ((I // tp + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE if tp > 1 else I

    # column-parallel gate/up (shard last dim = intermediate), row-parallel down
    # (shard dim -2 = intermediate). Expert dim (1) stays replicated.
    if is_mesh and tp > 1:
        col_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=-1)
        row_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=-2)
    elif is_mesh:
        col_mapper = row_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    else:
        col_mapper = row_mapper = None

    tp_suffix = f"_tp{tp}" if tp > 1 else ""

    def _cache(name):
        return str(tensor_cache_path / f"moe.experts.{name}{tp_suffix}") if tensor_cache_path else None

    gate_proj_tt = ttnn.as_tensor(
        gate_proj,
        device=mesh_device,
        dtype=gate_up_dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=col_mapper,
        cache_file_name=_cache("gate_proj"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    up_proj_tt = ttnn.as_tensor(
        up_proj,
        device=mesh_device,
        dtype=gate_up_dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=col_mapper,
        cache_file_name=_cache("up_proj"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    down_proj_tt = ttnn.as_tensor(
        down_proj,
        device=mesh_device,
        dtype=down_dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=row_mapper,
        cache_file_name=_cache("down_proj"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Fused gate|up along N (each device already holds the SAME intermediate slice of both,
    # so a local concat yields [gate_slice | up_slice]). Prefill runs gate+up as ONE
    # sparse_matmul with N = 2*I_per_device, doubling the N-gridded core count (4->8), then
    # slices the two halves back apart — mathematically identical to two separate matmuls.
    gate_up_proj_tt = ttnn.concat([gate_proj_tt, up_proj_tt], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    return ExpertWeights(
        gate_proj=gate_proj_tt,
        up_proj=up_proj_tt,
        down_proj=down_proj_tt,
        intermediate_size_per_device=per_device_intermediate,
        gate_up_proj=gate_up_proj_tt,
    )
