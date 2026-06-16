# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TP4 vision matmul program configs (Wormhole N150×4 / DP2×TP4 only).

Thin pass-throughs to the hardware-swept ``wh_tp4_*`` configs in
``vision_tp4_wh.py``. Blackhole support was removed -- this stack targets
Wormhole TP4 (DP2×TP4) only.
"""

from __future__ import annotations


def tp4_matmul_pc(device, m_dim: int, k_dim: int, n_dim: int, **kwargs):
    from models.experimental.tt_symbiote.modules.vision_tp4_wh import wh_tp4_matmul_pc

    return wh_tp4_matmul_pc(device, m_dim, k_dim, n_dim, **kwargs)


def tp4_merger_fc1_pc(device):
    from models.experimental.tt_symbiote.modules.vision_tp4_wh import wh_tp4_merger_fc1_pc

    return wh_tp4_merger_fc1_pc(device)


def tp4_merger_fc2_pc(device, *, seq_len: int = 2816, k: int = 6144, n: int = 384):
    from models.experimental.tt_symbiote.modules.vision_tp4_wh import wh_tp4_merger_fc2_pc

    return wh_tp4_merger_fc2_pc(device, seq_len=seq_len, k=k, n=n)


def tp4_o_proj_pc(device, *, seq_len: int = 11264, ctx_dim: int = 384):
    from models.experimental.tt_symbiote.modules.vision_tp4_wh import wh_tp4_o_proj_pc

    return wh_tp4_o_proj_pc(device, seq_len=seq_len, ctx_dim=ctx_dim)


def tp4_mlp_down_pc(device, *, seq_len: int = 11264, itp: int = 1056):
    from models.experimental.tt_symbiote.modules.vision_tp4_wh import wh_tp4_mlp_down_pc

    return wh_tp4_mlp_down_pc(device, seq_len=seq_len, itp=itp)


def tp4_qkv_pc(device):
    from models.experimental.tt_symbiote.modules.vision_tp4_wh import wh_tp4_qkv_pc

    return wh_tp4_qkv_pc(device)


def tp4_mlp_gate_up_pc(device):
    from models.experimental.tt_symbiote.modules.vision_tp4_wh import wh_tp4_mlp_gate_up_pc

    return wh_tp4_mlp_gate_up_pc(device)


def tp4_patch_embed_pc(device):
    from models.experimental.tt_symbiote.modules.vision_tp4_wh import wh_tp4_patch_embed_pc

    return wh_tp4_patch_embed_pc(device)


def tp4_sdpa_pc(device):
    from models.experimental.tt_symbiote.modules.vision_tp4_wh import wh_tp4_sdpa_pc

    return wh_tp4_sdpa_pc(device)
