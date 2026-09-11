# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""SiLU-gated (SwiGLU) MLP for the Gemma4-31B DFlash drafter's own 5 layers.

The drafter's config declares "hidden_act": "silu" (it's a Qwen3-style block,
see models/demos/gemma4/docs/dflash_design.md) -- NOT Gemma4's own GeGLU
("gelu_pytorch_tanh", models/demos/gemma4/tt/shared_mlp.py). down_proj(silu(gate)
* up), not down_proj(gelu(gate) * up), so SharedMLP is the wrong activation for
this module and is not reused here; the TP-sharding structure below (fused
column-parallel gate+up, row-parallel down + allreduce) mirrors it directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn
from models.demos.gemma4.tt.ccl import ccl_allreduce
from models.demos.gemma4.utils.general_utils import get_cache_file_name


@dataclass(frozen=True)
class DFlashMLPWeights:
    gate_up: ttnn.Tensor  # [1,1,hidden, 2*inter/tp], per-device layout [up_i | gate_i]
    down: ttnn.Tensor  # [1,1,inter/tp, hidden]
    inter_per_device: int


def load_dflash_mlp_weights(
    mesh_device,
    config,
    state_dict,
    mesh_config,
    weight_dtype=ttnn.bfloat8_b,
    tensor_cache_path=None,
) -> DFlashMLPWeights:
    """``state_dict``: {"gate_proj.weight", "up_proj.weight", "down_proj.weight"}
    (see weight_mapping.layer_mlp_state_dict)."""
    tp = mesh_config.tp if mesh_config else 1
    inter = config.intermediate_size
    assert inter % tp == 0, f"intermediate_size {inter} not divisible by tp={tp}"
    per_device = inter // tp
    assert per_device % ttnn.TILE_SIZE == 0, (
        f"per-device intermediate {per_device} not tile-aligned at tp={tp} -- "
        "the padding SharedMLP applies for this case is not implemented here"
    )

    if state_dict:
        gate_t = state_dict["gate_proj.weight"].transpose(-2, -1)  # [hidden, inter]
        up_t = state_dict["up_proj.weight"].transpose(-2, -1)  # [hidden, inter]
        down_t = state_dict["down_proj.weight"].transpose(-2, -1)  # [inter, hidden]
        if tp > 1:
            up_chunks = torch.chunk(up_t, tp, dim=-1)
            gate_chunks = torch.chunk(gate_t, tp, dim=-1)
            gate_up = torch.cat([torch.cat([up_chunks[i], gate_chunks[i]], dim=-1) for i in range(tp)], dim=-1)
        else:
            gate_up = torch.cat([up_t, gate_t], dim=-1)
        gate_up = gate_up.unsqueeze(0).unsqueeze(0)
        down_w = down_t.unsqueeze(0).unsqueeze(0)
    else:
        gate_up = None
        down_w = None

    col_mapper = mesh_config.column_parallel(mesh_device) if tp > 1 else None
    row_mapper = mesh_config.row_parallel(mesh_device) if tp > 1 else None
    tp_suffix = f"_tp{tp}" if tp > 1 else ""

    gate_up_tt = ttnn.as_tensor(
        gate_up,
        device=mesh_device,
        dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=col_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"mlp.gate_up{tp_suffix}"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    down_tt = ttnn.as_tensor(
        down_w,
        device=mesh_device,
        dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=row_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"mlp.down{tp_suffix}"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return DFlashMLPWeights(gate_up=gate_up_tt, down=down_tt, inter_per_device=per_device)


def dflash_mlp_forward(x, weights: DFlashMLPWeights, mesh_config=None, ccl_manager=None, compute_kernel_config=None):
    """down_proj(silu(gate) * up), TP-sharded (fused gate+up, row-parallel down + allreduce)."""
    gate_up = ttnn.linear(x, weights.gate_up, compute_kernel_config=compute_kernel_config)
    shard = weights.inter_per_device
    s = gate_up.shape[-2]
    up = ttnn.slice(gate_up, [0, 0, 0, 0], [1, 1, s, shard])
    gate = ttnn.slice(gate_up, [0, 0, 0, shard], [1, 1, s, 2 * shard])
    ttnn.deallocate(gate_up)

    activated = ttnn.silu(gate)
    hidden = ttnn.mul(activated, up)
    ttnn.deallocate(activated)
    ttnn.deallocate(gate)
    ttnn.deallocate(up)

    output = ttnn.linear(hidden, weights.down, compute_kernel_config=compute_kernel_config)
    ttnn.deallocate(hidden)

    if mesh_config is not None and mesh_config.tp > 1:
        output = ccl_allreduce(output, mesh_config, ccl_manager)
    return output
