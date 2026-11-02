# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Weight loading for Gemma4 routed experts.

Keeps HF gate_up_proj [num_experts, 2*intermediate, hidden] fused ([up | gate] on N) and
shards whole experts over the (ep, tp) mesh axes for sparse_matmul (Qwen3.6 MoE layout).

No bias in Gemma4 experts (unlike GPT-OSS).
"""

import os
from dataclasses import dataclass

import torch

import ttnn
from models.demos.gemma4.utils.general_utils import get_cache_file_name

TILE_SIZE = 32


@dataclass(frozen=True)
class ExpertWeights:
    """Container for expert weight tensors — immutable after creation.

    Expert-parallel layout: every chip holds ``num_local_experts`` WHOLE experts (full intermediate
    width). The expert dim is split first over the ep axis (lanes) and then over the tp axis, so
    expert ``e`` lives on ep row ``e // (tp * E_loc)``, tp column ``(e % (tp * E_loc)) // E_loc``,
    local slot ``e % E_loc`` -- the same order ``ttnn.mesh_partition`` produces when the dense
    routing is partitioned over the ep axis and then the tp axis.
    """

    gate_up_proj: ttnn.Tensor  # [1, E_loc, hidden_size, 2 * intermediate]: [up | gate] on N
    down_proj: ttnn.Tensor  # [1, E_loc, intermediate, hidden_size]
    intermediate_size_per_device: int  # full intermediate (experts are not sliced)
    num_local_experts: int
    ep: int
    tp: int
    # [1, 1, E, E_loc] 0/1 selection of this chip's expert columns (None when the chip holds all experts):
    # routing @ select gives the local routing exactly (bf16 x 1.0), for any E_loc, where a tiled
    # per-device slice would need a tile-aligned width.
    expert_select: ttnn.Tensor = None


def _expert_parallel_factors(mesh_device, mesh_config):
    """(ep, tp, ep_axis, tp_axis) of the expert sharding for this mesh."""
    is_mesh = hasattr(mesh_device, "shape")
    if not is_mesh or mesh_config is None:
        return 1, 1, 0, 1
    tp = int(getattr(mesh_config, "tp", 1) or 1)
    ep = int(getattr(mesh_config, "ep", 1) or 1)
    tp_axis = int(getattr(mesh_config, "tp_axis", 1))
    ep_axis = 1 - tp_axis
    return ep, tp, ep_axis, tp_axis


def _refuse_placeholder_on_cache_miss(state_dict, cache_prefixes, weight_dtype):
    """A warm-cache placeholder state dict (``is_placeholder``) carries dataless ``torch.empty``
    tensors. ``ttnn.as_tensor`` only discards them on a cache HIT; on a miss it would convert the
    garbage and persist it as the cache entry (the 26B served zero expert outputs after the layout
    change renamed the expert tensorbins). Refuse up front so a renamed layout forces a cold load."""
    if not getattr(state_dict, "is_placeholder", False):
        return
    missing = [
        f"{prefix}_dtype_{weight_dtype.name}_layout_{ttnn.TILE_LAYOUT.name}.tensorbin"
        for prefix in cache_prefixes
        if prefix is None
        or not os.path.isfile(f"{prefix}_dtype_{weight_dtype.name}_layout_{ttnn.TILE_LAYOUT.name}.tensorbin")
    ]
    if missing:
        raise RuntimeError(
            "Expert weight cache miss with a warm-cache placeholder state dict; refusing to persist dataless "
            f"tensors. Cold-load the checkpoint (delete the .weights_complete marker or set the force env). Missing: {missing}"
        )


def load_expert_weights(
    mesh_device,
    config,
    state_dict,
    mesh_config=None,
    weight_dtype=ttnn.bfloat8_b,
    tensor_cache_path=None,
) -> ExpertWeights:
    """Load the routed experts for ``sparse_matmul`` in the expert-parallel layout.

    HF ``gate_up_proj`` [E, 2I, H] (rows [:I] gate, [I:] up) becomes a fused [1, E_loc, H, 2I]
    tensor ordered [up | gate] on N (the GLU convention: the second half is activated), and
    ``down_proj`` [E, H, I] becomes [1, E_loc, I, H]. Experts are sharded whole over the ep axis
    and then the tp axis; nothing is sliced along the intermediate, so the per-chip matmuls keep
    the full N and no padding is needed (Qwen3.6 MoE layout).
    """
    from models.demos.gemma4.tt.precision import dtype_to_str

    num_experts = config.num_experts
    hidden_size = config.hidden_size
    intermediate_size = config.moe_intermediate_size
    ep, tp, ep_axis, tp_axis = _expert_parallel_factors(mesh_device, mesh_config)
    assert num_experts % (ep * tp) == 0, f"num_experts {num_experts} not divisible by ep*tp {ep * tp}"
    num_local = num_experts // (ep * tp)
    is_mesh = hasattr(mesh_device, "shape")

    def _gate_up(fused):
        # [E, 2I, H] -> [E, H, 2I] ordered [up | gate] -> [ep, tp*E_loc, H, 2I]; the 2-D mesh mapper
        # splits dim 0 over the ep axis and dim 1 over the tp axis, so each chip receives its
        # [1, E_loc, H, 2I] block in natural expert order
        f = fused.to(torch.bfloat16)
        gate, up = f[:, :intermediate_size, :], f[:, intermediate_size:, :]
        t = torch.cat([up, gate], dim=1).transpose(-2, -1).contiguous()  # [E, H, 2I]
        return t.reshape(ep, tp * num_local, hidden_size, 2 * intermediate_size).contiguous()

    def _down(down):
        # [E, H, I] -> [E, I, H] -> [ep, tp*E_loc, I, H]
        t = down.to(torch.bfloat16).transpose(-2, -1).contiguous()
        return t.reshape(ep, tp * num_local, intermediate_size, hidden_size).contiguous()

    fused_src = state_dict.get("gate_up_proj") if state_dict else None
    down_src = state_dict.get("down_proj") if state_dict else None
    if is_mesh:
        # mesh axis 0 shards dims[0], axis 1 shards dims[1]; tensor dim 0 = ep, dim 1 = tp. An axis
        # whose factor is 1 replicates (None): a size-1 tensor dim cannot be split over it.
        ep_dim = 0 if ep > 1 else None
        tp_dim = 1 if tp > 1 else None
        dims = (tp_dim, ep_dim) if tp_axis == 0 else (ep_dim, tp_dim)
        mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=dims)
    else:
        mapper = None
    suffix = f"_ep{ep}tp{tp}_whole_{dtype_to_str(weight_dtype)}"
    _refuse_placeholder_on_cache_miss(
        state_dict,
        [get_cache_file_name(tensor_cache_path, f"{n}{suffix}") for n in ("gate_up_proj", "down_proj")],
        weight_dtype,
    )

    def _load(name, src, preprocess):
        # per chip: [1, E_loc, K, N]
        return ttnn.as_tensor(
            src,
            device=mesh_device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, f"{name}{suffix}"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            preprocess=preprocess,
        )

    gate_up_tt = _load("gate_up_proj", fused_src, _gate_up)
    down_tt = _load("down_proj", down_src, _down)
    expert_select = None
    if ep * tp > 1:
        # device (ep_r, tp_c) selects experts [b, b + E_loc) with b = (ep_r * tp + tp_c) * E_loc
        sel = torch.zeros(ep, tp, num_experts, num_local, dtype=torch.bfloat16)
        for r in range(ep):
            for c in range(tp):
                base = (r * tp + c) * num_local
                for j in range(num_local):
                    sel[r, c, base + j, j] = 1.0
        # per chip: [1, 1, E, E_loc] (dims: tensor dim 0 over the ep axis, dim 1 over the tp axis)
        expert_select = ttnn.from_torch(
            sel,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
    return ExpertWeights(
        gate_up_proj=gate_up_tt,
        down_proj=down_tt,
        intermediate_size_per_device=intermediate_size,
        num_local_experts=num_local,
        ep=ep,
        tp=tp,
        expert_select=expert_select,
    )
