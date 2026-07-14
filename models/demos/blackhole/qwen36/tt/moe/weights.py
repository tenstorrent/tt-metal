# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Weight loading for Qwen3.5-MoE routed experts.

The 35B-A3B checkpoint stores experts FUSED as 3D nn.Parameters (no `.weight`):
  mlp.experts.gate_up_proj  [E, 2*I, H]  (rows [:I]=gate, [I:]=up)
  mlp.experts.down_proj     [E, H, I]
Older per-expert checkpoints (`{i}.gate_proj.weight` etc.) are stacked into the same
fused layout up front, so a single code path handles both.

EXPERT-PARALLEL sharding across the (1,4) mesh: the expert dim (dim=1) is SHARDED
(num_experts/tp experts per device); the intermediate dim is FULL on every device. So
gate/up run one sparse_matmul with N = 2*full_intermediate (wider N -> more output tiles
-> more cores: 8 -> 32 on the 512-wide intermediate), and each device only computes its
own expert shard (constant per-device FLOP vs the old intermediate-parallel layout). down
produces the FULL hidden per expert; the reduce-scatter after down_proj (in decode/prefill)
then sums each device's expert-partial across the mesh (mathematically identical to the old
intermediate-partial sum, sum being associative). gate/up are bfloat4_b, down is bfloat8_b.
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

    if tp > 1:
        assert E % tp == 0, f"expert-parallel needs num_experts ({E}) divisible by tp ({tp})"

    # Intermediate stays FULL on every device (expert-parallel shards the expert dim, not
    # the intermediate), so no per-device intermediate tile-padding is needed (I is already
    # a tile multiple for the supported checkpoints).
    per_device_intermediate = I

    # Expert-parallel: shard the EXPERT dim (dim=1) for gate/up AND down; intermediate is
    # full on every device. (Old intermediate-parallel layout used dim=-1 / dim=-2.)
    if is_mesh and tp > 1:
        col_mapper = row_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=1)
    elif is_mesh:
        col_mapper = row_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    else:
        col_mapper = row_mapper = None

    # `_ep` marks the expert-parallel cache layout so it never collides with the older
    # intermediate-parallel (`_tp`) cached tensors on disk (different sharding = different bytes).
    tp_suffix = f"_ep{tp}" if tp > 1 else ""

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

    # Fused gate|up along N. Each device holds its expert shard of both at FULL intermediate,
    # so a local concat yields [gate | up] with N = 2*full_intermediate. gate+up run as ONE
    # sparse_matmul over that wide N (32 N-tiles -> 32 cores vs 8), then the two halves are
    # sliced back apart — mathematically identical to two separate matmuls.
    gate_up_proj_tt = ttnn.concat([gate_proj_tt, up_proj_tt], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    return ExpertWeights(
        gate_proj=gate_proj_tt,
        up_proj=up_proj_tt,
        down_proj=down_proj_tt,
        intermediate_size_per_device=per_device_intermediate,
        gate_up_proj=gate_up_proj_tt,
    )
