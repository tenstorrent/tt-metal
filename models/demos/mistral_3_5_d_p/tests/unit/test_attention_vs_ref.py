# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The whole attention block vs a torch reference: QKV proj -> head split -> RoPE -> causal SDPA ->
o_proj. Pattern: ``minimax_m3/tests/unit/test_attention_vs_ref.py``.

Mistral's block is dense GQA (96 Q / 8 KV, head_dim 128), full rotary with YaRN, full-causal on
every layer, and NO projection bias, NO QK-norm, NO attention sink — so what is under test is the
head split, the projection sharding and the o_proj CCL tail, exactly as the donor entry says.

Two things the test deliberately holds fixed so it measures attention and nothing else:

  * **Same random weights on both sides.** The TT module is fed the q/k projections through the
    PRODUCTION ``convert_hf_qkv_to_meta_format`` swizzle (which ``tt/model_config.py`` also applies),
    because the device rope is the Meta-interleaved one; the reference keeps HF convention. Feeding
    the TT side unswizzled weights is the single most likely way to get a plausible-but-wrong
    attention, so the swizzle goes through the same helper the model uses.
  * **Shared cos/sin.** Both sides are driven from one YaRN table (this package's
    ``build_yarn_cos_sin`` for the Meta form, its halves-concatenated twin for HF), so a wrong rope
    CONSTANT cannot pass here and then fail later — that table is pinned against transformers in
    ``test_reference_config.py``.

Target mesh (SP=4 x TP=8), one-shot: the activation is replicated, each device computes its 12 Q
heads over the whole sequence, and the row-parallel o_proj is closed by the TP all-reduce.

NOTE on what this test canNOT catch: a block-output PCC is a weak detector of a rope-CONVENTION
error. Both q and k are rotated by the same per-pair angles, so a wrong channel pairing largely
cancels inside q.k^T; measured here, dropping the Meta swizzle entirely still scores ~0.998 on random
weights. The sensitive check is element-wise post-RoPE K, which is what
``test_kv_cache_write_vs_ref.py`` does (and where the unswizzled negative control lives).
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral_3_5_d_p.reference import model as reference
from models.demos.mistral_3_5_d_p.reference.mistral_config import MistralMedium35Config as C
from models.demos.mistral_3_5_d_p.reference.mistral_config import reduced_text_config
from models.demos.mistral_3_5_d_p.spec import SPEC
from models.demos.mistral_3_5_d_p.tt.attention import Attention, AttentionConfig, ProgramConfig
from models.demos.mistral_3_5_d_p.tt.rope import build_transformation_mat, build_yarn_cos_sin, yarn_inv_freq

from ..test_factory import build_mesh_and_ccl, from_device_replicated, meta_swizzle, parametrize_mesh, to_device

NQ, NKV, HEAD_DIM, HIDDEN = C.NUM_ATTENTION_HEADS, C.NUM_KEY_VALUE_HEADS, C.HEAD_DIM, C.HIDDEN_SIZE


def build_cos_sin(seq_len, *, offset=0):
    """``(cos_hf, sin_hf)`` [S, head_dim] for the reference and ``(cos_meta, sin_meta)``
    [1,1,S,head_dim] for the TT module — one YaRN table, two conventions.

    ``offset`` shifts the positions the table covers. The non-indexed rope op applies whatever table
    it is handed, so an offset table is how a test reaches large rotation angles without running a
    long sequence (see ``test_attention_needs_meta_swizzled_qk``).
    """
    inv_freq, mscale = yarn_inv_freq(
        HEAD_DIM,
        C.ROPE_THETA,
        C.YARN_FACTOR,
        C.YARN_ORIG_MAX_POS,
        C.YARN_BETA_FAST,
        C.YARN_BETA_SLOW,
        C.YARN_TRUNCATE,
    )
    freqs = torch.outer(torch.arange(offset, offset + seq_len).float(), inv_freq)
    cos_half, sin_half = torch.cos(freqs) * mscale, torch.sin(freqs) * mscale
    cos_hf = torch.cat([cos_half, cos_half], dim=-1)
    sin_hf = torch.cat([sin_half, sin_half], dim=-1)
    cos_meta = torch.stack([cos_half, cos_half], dim=-1).flatten(-2)[None, None]
    sin_meta = torch.stack([sin_half, sin_half], dim=-1).flatten(-2)[None, None]
    if offset == 0:
        # At offset 0 the production builder must agree exactly; assert it rather than trusting the
        # local re-derivation, so this helper cannot drift from tt/rope.py.
        prod_cos, prod_sin = build_yarn_cos_sin(
            seq_len,
            HEAD_DIM,
            rope_theta=C.ROPE_THETA,
            yarn_factor=C.YARN_FACTOR,
            yarn_orig_max_pos=C.YARN_ORIG_MAX_POS,
            yarn_beta_fast=C.YARN_BETA_FAST,
            yarn_beta_slow=C.YARN_BETA_SLOW,
            truncate=C.YARN_TRUNCATE,
        )
        assert torch.allclose(prod_cos, cos_meta) and torch.allclose(prod_sin, sin_meta)
        cos_meta, sin_meta = prod_cos, prod_sin
    return (cos_hf, sin_hf), (cos_meta, sin_meta)


def random_attention_weights(hidden=HIDDEN, scale=0.02):
    """HF ``[out, in]`` q/k/v/o. No biases — Ministral3 builds all four with ``bias=False``."""
    return {
        "q_proj.weight": torch.randn(NQ * HEAD_DIM, hidden) * scale,
        "k_proj.weight": torch.randn(NKV * HEAD_DIM, hidden) * scale,
        "v_proj.weight": torch.randn(NKV * HEAD_DIM, hidden) * scale,
        "o_proj.weight": torch.randn(hidden, NQ * HEAD_DIM) * scale,
    }


def build_attention(
    mesh_device,
    mesh_config,
    ccl,
    hf_state,
    *,
    max_seq_len,
    weight_dtype=ttnn.bfloat16,
    sequence_parallel=False,
    swizzle=True,
):
    """The production Attention with Meta-swizzled q/k and a replicated rope transformation matrix.

    ``swizzle=False`` skips ``convert_hf_qkv_to_meta_format`` — only for the negative control in
    ``test_kv_cache_write_vs_ref.py``, which needs an otherwise-identical module.
    """
    attn_config = AttentionConfig(
        hidden_size=HIDDEN,
        num_heads=NQ,
        num_kv_heads=NKV,
        head_dim=HEAD_DIM,
        max_seq_len=max_seq_len,
        rms_norm_eps=C.RMS_NORM_EPS,
        sequence_parallel=sequence_parallel,
    )
    return Attention(
        mesh_device=mesh_device,
        config=attn_config,
        state_dict=meta_swizzle(hf_state, HEAD_DIM) if swizzle else hf_state,
        ccl_manager=ccl,
        mesh_config=mesh_config,
        program_config=ProgramConfig(),
        layer_idx=0,
        transformation_mats={"prefill": build_transformation_mat(mesh_device)},
        weight_dtype=weight_dtype,
    )


@parametrize_mesh()
@pytest.mark.parametrize("seq_len", [128, 512], ids=["s128", "s512"])
def test_attention_prefill_vs_ref(mesh_device, device_params, seq_len, reset_seeds):
    """One-shot dense-GQA prefill attention vs the torch reference, random weights, real dims."""
    hf_state = random_attention_weights()
    x = torch.randn(1, seq_len, HIDDEN) * 0.1
    (cos_hf, sin_hf), (cos_meta, sin_meta) = build_cos_sin(seq_len)

    hf_config = reduced_text_config()
    ref = reference.attention_reference(
        x, hf_state, hf_config, cos_sin=(cos_hf.unsqueeze(0), sin_hf.unsqueeze(0))
    ).output

    mesh_config, ccl = build_mesh_and_ccl(mesh_device)
    attn = build_attention(mesh_device, mesh_config, ccl, hf_state, max_seq_len=max(seq_len, 128))

    rope_mats = [to_device(cos_meta, mesh_device), to_device(sin_meta, mesh_device)]
    x_tt = to_device(x.reshape(1, 1, seq_len, HIDDEN), mesh_device)

    tt_out = attn(x_tt, rope_mats=rope_mats, position_idx=None, kv_cache=None)
    out = from_device_replicated(tt_out, (1, seq_len, HIDDEN))

    passing, pcc = comp_pcc(ref, out, SPEC.pcc)
    logger.info(f"attention prefill seq={seq_len} tp={mesh_config.tp}: pcc={pcc}")
    assert passing, f"attention PCC fail (seq={seq_len}): {pcc}"
