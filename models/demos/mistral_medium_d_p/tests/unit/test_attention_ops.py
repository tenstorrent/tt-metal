# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DEVICE (1 chip): attention primitive ops in isolation.

The head split, the RoPE apply and the head concat, each checked on its own so that a failure in
``test_attention_vs_ref.py`` can be localised instead of bisected. Shapes are the per-chip ones the
TP=4 target actually produces (24 Q + 2 KV heads), even though the mesh here is 1x1 — the ops are
per-device, so a 1-chip box can exercise the real per-chip geometry.

Run:  pytest models/demos/mistral_medium_d_p/tests/unit/test_attention_ops.py
"""

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral_medium_d_p.tt.attention.operations import apply_rope, concat_heads, split_qkv_heads_prefill
from models.demos.mistral_medium_d_p.tt.rope import build_transformation_mat
from models.demos.mistral_medium_d_p.tt.rope_tables import build_hf_cos_sin, build_yarn_cos_sin

from ..test_factory import parametrize_mesh_with_fabric, replicate
from .shapes import HEAD_DIM, YARN, per_chip

SEQ = 256
PC = per_chip(4)  # 24 Q heads, 2 KV heads — the real per-chip geometry at TP=4


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
def test_split_qkv_heads(mesh_device, device_params, reset_seeds):
    """``nlp_create_qkv_heads`` must cut the fused 3584-wide QKV into 24 Q + 2 K + 2 V heads.

    The per-device fused layout is ``[Q(24*128) | K(2*128) | V(2*128)]``; getting the split point
    wrong silently mixes V into K, which SDPA will happily consume.
    """
    torch.manual_seed(0)
    n_q, n_kv = PC["n_q"], PC["n_kv"]
    q = torch.randn(1, 1, SEQ, n_q * HEAD_DIM)
    k = torch.randn(1, 1, SEQ, n_kv * HEAD_DIM)
    v = torch.randn(1, 1, SEQ, n_kv * HEAD_DIM)
    fused = torch.cat([q, k, v], dim=-1)
    assert fused.shape[-1] == PC["qkv"] == 3584

    tt_q, tt_k, tt_v = split_qkv_heads_prefill(replicate(fused, mesh_device), n_q, n_kv)

    for name, tt, ref, n_heads in (("q", tt_q, q, n_q), ("k", tt_k, k, n_kv), ("v", tt_v, v, n_kv)):
        got = ttnn.to_torch(ttnn.get_device_tensors(tt)[0])
        assert tuple(got.shape) == (1, n_heads, SEQ, HEAD_DIM), f"{name}: got {tuple(got.shape)}"
        want = ref.reshape(1, SEQ, n_heads, HEAD_DIM).transpose(1, 2)
        passing, pcc = comp_pcc(want, got, 0.999)
        assert passing, f"{name} head split wrong: {pcc}"
    logger.info(f"qkv split OK: {n_q} Q + {n_kv} K + {n_kv} V heads from {PC['qkv']}")


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
def test_rope_apply_matches_hf_convention(mesh_device, device_params, reset_seeds):
    """The device RoPE (Meta interleaved) must equal HF ``rotate_half`` on swizzled inputs.

    The weight swizzle / table-layout equivalence is proven on the host in
    ``test_checkpoint_ingest.py``; this checks the DEVICE op honours it.
    """
    torch.manual_seed(0)
    n_q = PC["n_q"]
    x = torch.randn(1, n_q, SEQ, HEAD_DIM) * 0.5

    cos_hf, sin_hf = build_hf_cos_sin(SEQ, HEAD_DIM, **YARN)

    def rotate_half(t):
        a, b = t[..., : t.shape[-1] // 2], t[..., t.shape[-1] // 2 :]
        return torch.cat([-b, a], dim=-1)

    ref_hf = x * cos_hf + rotate_half(x) * sin_hf
    # Meta interleaves the frequency pairs, so column 2i/2i+1 here is column i / i+half there.
    ref = torch.stack([ref_hf[..., : HEAD_DIM // 2], ref_hf[..., HEAD_DIM // 2 :]], dim=-1).flatten(-2)

    # ...and the input must be swizzled the same way the q/k projections are at load time.
    x_meta = torch.stack([x[..., : HEAD_DIM // 2], x[..., HEAD_DIM // 2 :]], dim=-1).flatten(-2)
    cos_meta, sin_meta = build_yarn_cos_sin(SEQ, HEAD_DIM, **YARN)

    out_tt = apply_rope(
        replicate(x_meta, mesh_device),
        [replicate(cos_meta, mesh_device), replicate(sin_meta, mesh_device)],
        build_transformation_mat(mesh_device),
    )
    out = ttnn.to_torch(ttnn.get_device_tensors(out_tt)[0])

    passing, pcc = comp_pcc(ref, out, 0.99)
    logger.info(f"device RoPE vs HF convention: {pcc}")
    assert passing, f"RoPE PCC fail: {pcc}"


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
def test_concat_heads(mesh_device, device_params, reset_seeds):
    """``nlp_concat_heads``: [1, 24, s, 128] -> [1, 1, s, 3072] (= the per-chip o_proj input)."""
    torch.manual_seed(0)
    n_q = PC["n_q"]
    x = torch.randn(1, n_q, SEQ, HEAD_DIM)

    out_tt = concat_heads(replicate(x, mesh_device))
    out = ttnn.to_torch(ttnn.get_device_tensors(out_tt)[0])

    assert tuple(out.shape) == (1, 1, SEQ, n_q * HEAD_DIM) == (1, 1, SEQ, PC["hidden"])
    want = x.transpose(1, 2).reshape(1, 1, SEQ, n_q * HEAD_DIM)
    passing, pcc = comp_pcc(want, out, 0.999)
    logger.info(f"concat heads -> {n_q * HEAD_DIM} (o_proj K per chip): {pcc}")
    assert passing, f"concat_heads PCC fail: {pcc}"
