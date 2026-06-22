# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU: validate the MLA weight-absorption identity (compressed-latent attention == standard
attention) before porting to the ttnn paged flash-MLA op (A6).

Standard MLA caches expanded k/v [H, qk] and does normal attention. The compressed-latent form caches
only the kv_a latent [kv_lora + rope] (MQA, 12.8x smaller) and absorbs kv_b into q/out:
  scores_h = [q_nope_h @ Wk_h ; q_rope_h] · [kv_pass ; k_rot]      (Wk_h: kv_b k-part [qk_nope,kv_lora])
  out_h    = (softmax(scores_h) @ kv_pass) @ Wv_h                  (Wv_h: kv_b v-part [kv_lora,v_head])
This test reconstructs both paths in torch from the reference weights and asserts they match — the
recipe the ttnn build (paged_flash_multi_latent_attention_decode + wkv_b1/wkv_b2) must implement.
"""
import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

from models.common.utility_functions import comp_pcc
from models.demos.mistral4.tests.m4_text_reference import load_m4_text_reference


def _rms(x, w, eps):
    return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


@pytest.mark.timeout(0)
def test_m4_mla_absorb_cpu():
    ckpt = os.environ["HF_MODEL"]
    cfg = AutoConfig.from_pretrained(ckpt).text_config
    H, nope, rope, vd, kvl = (
        cfg.num_attention_heads,
        cfg.qk_nope_head_dim,
        cfg.qk_rope_head_dim,
        cfg.v_head_dim,
        cfg.kv_lora_rank,
    )
    model, _, _ = load_m4_text_reference(ckpt, n_layers=1)
    model = model.float()  # CPU math check in fp32 (avoid bf16 Linear dtype clashes)
    sa = model.model.layers[0].self_attn
    eps = cfg.rms_norm_eps

    torch.manual_seed(0)
    S = 8
    mla_in = torch.randn(1, S, cfg.hidden_size) * 0.02
    rot = model.model.rotary_emb(mla_in.float(), torch.arange(S)[None])
    cos, sin = rot[0][0].float(), rot[1][0].float()  # [S, rope] (HF interleaved cos/sin halves)

    def rope_apply(x):  # x [H,S,rope] interleaved -> apply HF rotary (rotate_half on de-interleaved)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        xd = torch.cat([x1, x2], dim=-1)
        rh = torch.cat([-xd[..., rope // 2 :], xd[..., : rope // 2]], dim=-1)
        return xd * cos + rh * sin

    with torch.no_grad():
        x = mla_in.float()
        # q path
        q = sa.q_b_proj(_rms(sa.q_a_proj(x), sa.q_a_layernorm.weight.float(), eps))
        q = q.view(1, S, H, nope + rope).transpose(1, 2)  # [1,H,S,qk]
        q_nope, q_rope = q[..., :nope], rope_apply(q[..., nope:][0])  # q_rope [H,S,rope]
        q_nope = q_nope[0]  # [H,S,nope]
        # kv path
        kv_a = sa.kv_a_proj_with_mqa(x)[0]  # [S, kvl+rope]
        kv_pass = _rms(kv_a[:, :kvl], sa.kv_a_layernorm.weight.float(), eps)  # [S, kvl]
        k_rot = rope_apply(kv_a[:, kvl:][None].expand(1, S, rope))[0]  # [S, rope] (shared)
        kv_b = sa.kv_b_proj.weight.float().view(H, nope + vd, kvl)  # [H, nope+vd, kvl]
        Wk = kv_b[:, :nope, :]  # [H, nope, kvl]  (k-part)
        Wv = kv_b[:, nope:, :]  # [H, vd, kvl]   (v-part)

        # --- standard attention (expanded) ---
        k_nope = torch.einsum("sk,hdk->hsd", kv_pass, Wk)  # [H,S,nope]
        k = torch.cat([k_nope, k_rot[None].expand(H, S, rope)], dim=-1)  # [H,S,qk]
        v = torch.einsum("sk,hdk->hsd", kv_pass, Wv)  # [H,S,vd]
        qx = torch.cat([q_nope, q_rope], dim=-1)  # [H,S,qk]
        scale = (nope + rope) ** -0.5
        mask = torch.triu(torch.full((S, S), float("-inf")), 1)
        std = torch.softmax(torch.einsum("hsd,htd->hst", qx, k) * scale + mask, -1)
        std_out = torch.einsum("hst,htd->hsd", std, v)  # [H,S,vd]

        # --- absorbed compressed-latent attention ---
        q_lat_nope = torch.einsum("hsn,hnk->hsk", q_nope, Wk)  # [H,S,kvl]  (Wk: [H,nope,kvl])
        q_lat = torch.cat([q_lat_nope, q_rope], dim=-1)  # [H,S,kvl+rope]
        kv_lat = torch.cat([kv_pass, k_rot], dim=-1)  # [S, kvl+rope] (MQA, shared)
        sc = torch.softmax(torch.einsum("hsd,td->hst", q_lat, kv_lat) * scale + mask, -1)
        ctx = torch.einsum("hst,tk->hsk", sc, kv_pass)  # [H,S,kvl]
        abs_out = torch.einsum("hsk,hdk->hsd", ctx, Wv)  # [H,S,vd]

    passing, msg = comp_pcc(std_out, abs_out, 0.999)
    logger.info(f"MLA absorption identity (compressed-latent vs standard): {msg}")
    assert passing, f"absorption math mismatch: {msg}"
