# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Arch dispatch for TP4 vision matmul program configs (N150×4 vs P150×4)."""

from __future__ import annotations

from ttnn.device import is_wormhole_b0


def tp4_matmul_pc(device, m_dim: int, k_dim: int, n_dim: int, **kwargs):
    if is_wormhole_b0(device):
        from models.experimental.tt_symbiote.modules.vision_tp4_wh import wh_tp4_matmul_pc

        return wh_tp4_matmul_pc(device, m_dim, k_dim, n_dim, **kwargs)
    from models.experimental.tt_symbiote.modules.vision_tp4_bh import bh_tp4_matmul_pc

    return bh_tp4_matmul_pc(device, m_dim, k_dim, n_dim, **kwargs)


def tp4_merger_fc1_pc(device):
    if is_wormhole_b0(device):
        from models.experimental.tt_symbiote.modules.vision_tp4_wh import wh_tp4_merger_fc1_pc

        return wh_tp4_merger_fc1_pc(device)
    from models.experimental.tt_symbiote.modules.vision_tp4_bh import bh_tp4_merger_fc1_pc

    return bh_tp4_merger_fc1_pc(device)


def tp4_merger_fc2_pc(device):
    if is_wormhole_b0(device):
        from models.experimental.tt_symbiote.modules.vision_tp4_wh import wh_tp4_merger_fc2_pc

        return wh_tp4_merger_fc2_pc(device)
    from models.experimental.tt_symbiote.modules.vision_tp4_bh import bh_tp4_merger_fc2_pc

    return bh_tp4_merger_fc2_pc(device)


def tp4_o_proj_pc(device, *, seq_len: int = 11264, ctx_dim: int = 384):
    if is_wormhole_b0(device):
        from models.experimental.tt_symbiote.modules.vision_tp4_wh import wh_tp4_o_proj_pc

        return wh_tp4_o_proj_pc(device, seq_len=seq_len, ctx_dim=ctx_dim)
    from models.experimental.tt_symbiote.modules.vision_tp4_bh import bh_tp4_o_proj_pc

    return bh_tp4_o_proj_pc(device, seq_len=seq_len, ctx_dim=ctx_dim)


def tp4_mlp_down_pc(device, *, seq_len: int = 11264, itp: int = 1056):
    if is_wormhole_b0(device):
        from models.experimental.tt_symbiote.modules.vision_tp4_wh import wh_tp4_mlp_down_pc

        return wh_tp4_mlp_down_pc(device, seq_len=seq_len, itp=itp)
    from models.experimental.tt_symbiote.modules.vision_tp4_bh import bh_tp4_mlp_down_pc

    return bh_tp4_mlp_down_pc(device, seq_len=seq_len, itp=itp)


def tp4_qkv_pc(device):
    if is_wormhole_b0(device):
        from models.experimental.tt_symbiote.modules.vision_tp4_wh import wh_tp4_qkv_pc

        return wh_tp4_qkv_pc(device)
    from models.experimental.tt_symbiote.modules.vision_tp4_bh import bh_tp4_qkv_pc

    return bh_tp4_qkv_pc(device)


def tp4_mlp_gate_up_pc(device):
    if is_wormhole_b0(device):
        from models.experimental.tt_symbiote.modules.vision_tp4_wh import wh_tp4_mlp_gate_up_pc

        return wh_tp4_mlp_gate_up_pc(device)
    from models.experimental.tt_symbiote.modules.vision_tp4_bh import bh_tp4_mlp_gate_up_pc

    return bh_tp4_mlp_gate_up_pc(device)
