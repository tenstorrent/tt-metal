# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sequence-parallel sliding-window prefill SDPA via all_gather composition.

Ring-SDPA-equivalent: ``all_gather`` K/V across SP + local SDPA with a global
causal+sliding mask.  Stand-in until ``ring_joint_scaled_dot_product_attention``
supports ``sliding_window_size`` / ``attention_sink``.
"""

import torch

import ttnn

from .config import ProgramConfig


def _build_sp_causal_sliding_mask(seq_len: int, sp: int, sliding_window: int) -> torch.Tensor:
    """Per-SP-row mask [sp, 1, seq_len, seq_total]: causal + sliding window on global indices."""
    seq_total = seq_len * sp
    rows_idx = torch.arange(seq_len, dtype=torch.float32).view(1, seq_len, 1)
    cols_idx = torch.arange(seq_total, dtype=torch.float32).view(1, 1, seq_total)
    row_offset = torch.arange(sp, dtype=torch.float32).view(sp, 1, 1) * seq_len
    q_abs = rows_idx + row_offset  # [sp, seq_len, 1]
    causal = cols_idx <= q_abs
    in_window = (q_abs - cols_idx) < sliding_window
    attend = causal & in_window
    base_mask = torch.where(attend, torch.tensor(0.0), torch.tensor(float("-inf"))).to(torch.bfloat16)
    return base_mask.unsqueeze(1)  # [sp, 1, seq_len, seq_total]


def sp_sliding_window_sdpa(
    tt_q,
    tt_k,
    tt_v,
    sinks,
    seq_len: int,
    sliding_window: int,
    mesh_config,
    mesh_device,
    program_config: ProgramConfig,
    ccl_manager,
):
    """
    Ring-SDPA-style composition for sliding-window SP prefill.

    Q stays SP-sharded. K/V are all-gathered on the SP axis so each device has
    the full sequence, then local SDPA runs with an explicit per-row causal +
    sliding-window mask (global token indices).
    """
    sp_factor = mesh_config.prefill.sp
    sp_axis = mesh_config.sp_axis
    seq_total = seq_len * sp_factor
    sdpa_compute_config = program_config.get_compute_kernel_config()

    tt_k_gathered = ttnn.all_gather(
        tt_k,
        dim=2,
        cluster_axis=sp_axis,
        num_links=ccl_manager.num_links,
        topology=ccl_manager.topology,
    )
    tt_v_gathered = ttnn.all_gather(
        tt_v,
        dim=2,
        cluster_axis=sp_axis,
        num_links=ccl_manager.num_links,
        topology=ccl_manager.topology,
    )
    tt_k.deallocate(True)
    tt_v.deallocate(True)

    mask_cpu = _build_sp_causal_sliding_mask(seq_len, sp_factor, sliding_window)
    mask_dims = [None, None]
    mask_dims[sp_axis] = 0
    tt_attn_mask = ttnn.as_tensor(
        mask_cpu,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=tuple(mask_dims)),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_sdpa_out = ttnn.transformer.scaled_dot_product_attention(
        tt_q,
        tt_k_gathered,
        tt_v_gathered,
        attn_mask=tt_attn_mask,
        is_causal=False,
        program_config=program_config.get_prefill_sdpa_config(mesh_device, seq_total),
        compute_kernel_config=sdpa_compute_config,
        attention_sink=sinks,
    )
    tt_q.deallocate(True)
    tt_k_gathered.deallocate(True)
    tt_v_gathered.deallocate(True)
    tt_attn_mask.deallocate(True)
    return tt_sdpa_out
