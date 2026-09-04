# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: on-device RoPE vs the torch reference.

Not a row in the recipe's D2 table — the table folds RoPE into the attention test. It is broken out
here because RoPE is the single highest-risk block for this model: the donor implements YaRN, Llama
needs llama3 smooth-ramp scaling, and the error the substitution produces is a phase drift that grows
with position. It is invisible at short sequence length and collapses long-context K PCC, so it is
tested at long positions on purpose.

Two device paths are covered:
  * ``rotary_embedding_llama`` with per-chunk replicated cos/sin (the one-shot path);
  * ``rotary_embedding_indexed`` with the whole-cache block-cyclic SP-sharded cos/sin (the chunked
    path), which must agree with the same reference at the same global positions.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.llama3_1_8b_d_p.reference import model as ref
from models.demos.llama3_1_8b_d_p.tt import rope as tt_rope
from models.demos.llama3_1_8b_d_p.tt.attention.operations import apply_rope

from ..test_factory import (
    concat_sp,
    dev0,
    llama_config,
    make_mesh_config,
    parametrize_mesh_with_fabric,
    replicate,
    shard_seq_on_sp,
)

PCC = 0.99


def _reference_rope(x_hf, cfg, offset):
    """HF half-split RoPE at global positions ``[offset, offset+S)``."""
    cos, sin = ref.build_cos_sin_hf(x_hf.shape[-2], cfg, offset=offset)
    out, _ = ref.apply_rotary_pos_emb(x_hf, x_hf.clone(), cos, sin)
    return out


def _to_meta(x_hf, head_dim):
    """Interleave the two HF halves of each head — the inverse of ``meta_to_hf_head_perm``."""
    a, b = x_hf[..., : head_dim // 2], x_hf[..., head_dim // 2 :]
    return torch.stack([a, b], dim=-1).flatten(-2)


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@pytest.mark.parametrize("seq_len", [128, 4096], ids=["s128", "s4096"])
@pytest.mark.parametrize("offset", [0, 65536], ids=["pos0", "pos64k"])
def test_rope_one_shot_vs_ref(mesh_device, device_params, seq_len, offset, reset_seeds):
    """``rotary_embedding_llama`` on Meta-swizzled Q vs HF half-split RoPE on plain Q.

    ``offset=65536`` is the case that separates llama3 scaling from YaRN: at position 0 every scaling
    rule agrees, so a short test at pos0 cannot catch a wrong rope.
    """
    cfg = llama_config()
    hd = cfg.head_dim
    n_heads = 8
    x_hf = torch.randn(1, n_heads, seq_len, hd)
    reference = _reference_rope(x_hf, cfg, offset)

    cos, sin = ref.build_cos_sin_meta(seq_len, cfg, offset=offset)
    rope_mats = [replicate(mesh_device, cos), replicate(mesh_device, sin)]
    trans = tt_rope.build_transformation_mat(mesh_device)

    x_tt = replicate(mesh_device, _to_meta(x_hf, hd))
    out_meta = dev0(apply_rope(x_tt, rope_mats, trans)).reshape(1, n_heads, seq_len, hd)
    out_hf = ref.meta_to_hf_head_perm(out_meta, hd)

    passing, pcc = comp_pcc(reference, out_hf, PCC)
    logger.info(f"rope one-shot s={seq_len} offset={offset}: {pcc}")
    assert passing, f"PCC fail: {pcc}"


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
@pytest.mark.parametrize("chunk_size, cached_len", [(4096, 0), (4096, 4096), (4096, 65536)], ids=["c0", "c1", "c16"])
def test_rope_indexed_sp_vs_ref(mesh_device, device_params, chunk_size, cached_len, reset_seeds):
    """``rotary_embedding_indexed`` over the whole-cache block-cyclic SP cos/sin.

    The op derives each chip's start row on-device from ``kv_actual_global`` plus the chip's SP mesh
    coordinate. Reassembled across SP rows, chunk ``n`` must carry exactly the rope for global
    positions ``[cached_len, cached_len + chunk_size)`` — the invariant the KV cache depends on.
    """
    cfg = llama_config()
    hd = cfg.head_dim
    max_seq_len = 131072
    mesh_config = make_mesh_config(mesh_device)
    sp = mesh_config.sp
    n_heads_local = 8

    rope_mats = tt_rope.build_indexed_rope(
        mesh_device, head_dim=hd, max_seq_len=max_seq_len, chunk_size=chunk_size, sp_axis=mesh_config.sp_axis
    )
    trans = tt_rope.build_transformation_mat(mesh_device)

    x_hf = torch.randn(1, n_heads_local, chunk_size, hd)
    reference = _reference_rope(x_hf, cfg, cached_len)

    # This chunk is SP-sharded contiguously: row i holds tokens [i*chunk_local, (i+1)*chunk_local).
    x_tt = shard_seq_on_sp(mesh_device, _to_meta(x_hf, hd), mesh_config)
    out_tt = apply_rope(x_tt, rope_mats, trans, kv_actual_global=cached_len, cluster_axis=mesh_config.sp_axis)
    out_meta = concat_sp(mesh_device, out_tt, mesh_config).reshape(1, n_heads_local, chunk_size, hd)
    out_hf = ref.meta_to_hf_head_perm(out_meta, hd)

    passing, pcc = comp_pcc(reference, out_hf, PCC)
    logger.info(f"rope indexed chunk={chunk_size} cached_len={cached_len}: {pcc}")
    assert passing, f"PCC fail: {pcc}"
