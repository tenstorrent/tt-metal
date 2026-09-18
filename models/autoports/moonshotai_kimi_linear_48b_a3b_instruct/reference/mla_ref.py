# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""NoPE multi-head latent attention oracle (fp32), expanded and absorbed forms, with an explicit latent cache."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F

from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.reference.config import KimiLinearConfig


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * w.float()


def mla_latent(hidden: torch.Tensor, weights: Mapping[str, torch.Tensor], cfg: KimiLinearConfig) -> torch.Tensor:
    """[B,T,H] -> cached latent [B,T,576] = norm(kv_a[:512]) ++ k_pe(64) (never rotated: NoPE)."""
    kv = F.linear(hidden.float(), weights["kv_a_proj_with_mqa.weight"].float())
    lat, pe = kv[..., : cfg.kv_lora_rank], kv[..., cfg.kv_lora_rank :]
    lat = rms_norm(lat, weights["kv_a_layernorm.weight"], cfg.rms_norm_eps)
    return torch.cat([lat, pe], dim=-1)


def mla_forward_reference(
    hidden: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    cfg: KimiLinearConfig,
    past_latent: torch.Tensor | None = None,
    absorbed: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal MLA over ``past_latent`` (if any) followed by the new tokens.

    Returns (output [B,T,hidden] fp32, full latent cache [B,P+T,576]).
    expanded: k = [kv_b_k(latent) | k_pe], v = kv_b_v(latent) per head, softmax(q k^T / sqrt(192)).
    absorbed: q_lat = q_nope @ Wk_b^T (per head, 128->512) ++ q_pe, scores against the raw latent [576],
              o = (attn @ latent[:512]) @ Wv_b^T. Both must agree numerically.
    """
    H, dn, dr, dv = cfg.num_attention_heads, cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim
    R = cfg.kv_lora_rank
    x = hidden.float()
    B, T, _ = x.shape
    q = F.linear(x, weights["q_proj.weight"].float()).view(B, T, H, dn + dr)
    q_nope, q_pe = q[..., :dn], q[..., dn:]
    new_lat = mla_latent(x, weights, cfg)
    lat = new_lat if past_latent is None else torch.cat([past_latent.float(), new_lat], dim=1)
    P = lat.shape[1] - T
    wkv_b = weights["kv_b_proj.weight"].float().view(H, dn + dv, R)  # [H, 256, 512]
    wk_b, wv_b = wkv_b[:, :dn, :], wkv_b[:, dn:, :]  # [H,128,512], [H,128,512]
    scale = (dn + dr) ** -0.5
    mask = torch.ones(T, P + T, dtype=torch.bool).tril(diagonal=P)  # causal incl. past
    if absorbed:
        q_lat = torch.einsum("bthd,hdr->bthr", q_nope, wk_b)  # [B,T,H,512]
        q_full = torch.cat([q_lat, q_pe], dim=-1)  # [B,T,H,576]
        scores = torch.einsum("bthr,bsr->bhts", q_full, lat) * scale
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = scores.softmax(-1)
        o_lat = torch.einsum("bhts,bsr->bthr", attn, lat[..., :R])  # [B,T,H,512]
        o = torch.einsum("bthr,hdr->bthd", o_lat, wv_b)  # [B,T,H,128]
    else:
        kv = torch.einsum("bsr,hdr->bshd", lat[..., :R], wkv_b)  # [B,S,H,256]
        k_nope, v = kv[..., :dn], kv[..., dn:]
        k = torch.cat([k_nope, lat[..., R:].unsqueeze(2).expand(-1, -1, H, -1)], dim=-1)  # [B,S,H,192]
        scores = torch.einsum("bthd,bshd->bhts", q, k) * scale
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = scores.softmax(-1)
        o = torch.einsum("bhts,bshd->bthd", attn, v)
    out = F.linear(o.reshape(B, T, H * dv), weights["o_proj.weight"].float())
    return out, lat
