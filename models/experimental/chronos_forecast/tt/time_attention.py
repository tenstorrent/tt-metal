# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""TimeSelfAttention (encoder sublayer 1) weights and host RoPE cache; TtEncoderBlock runs it on TtMhaCore.

reference : models/experimental/chronos_forecast/reference/chronos2/layers.py
    x = x + TimeSelfAttention(x)  # RMSNorm, RoPE Q/K, SDPA scale 1.0, mask (B,H,T,T)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.experimental.chronos_forecast.tt.mha_core import TtMhaWeights


@dataclass(frozen=True)
class TtTimeAttentionWeights:
    """Host-side weights using ``nn.Linear`` convention: (out_features, in_features)."""

    wqkv: torch.Tensor  # (3 * inner, d) — Wq/Wk/Wv concatenated along out dim
    wo: torch.Tensor  # (d, inner)
    rms_weight: torch.Tensor  # (d,)
    inv_freq: torch.Tensor  # (Dh // 2,) RoPE buffer (not trained)
    num_heads: int
    head_dim: int
    eps: float = 1e-6

    @classmethod
    def from_torch_layer(cls, layer) -> "TtTimeAttentionWeights":
        """Extract weights from a reference ``TimeSelfAttention`` (or matching module)."""
        mha = layer.self_attention
        wqkv = torch.cat([mha.q.weight.detach(), mha.k.weight.detach(), mha.v.weight.detach()], dim=0).clone()
        return cls(
            wqkv=wqkv,
            wo=mha.o.weight.detach().clone(),
            rms_weight=layer.layer_norm.weight.detach().clone(),
            inv_freq=mha.rope_embed.inv_freq.detach().clone(),
            num_heads=mha.n_heads,
            head_dim=mha.kv_proj_dim,
            eps=layer.layer_norm.variance_epsilon,
        )

    def to_mha(self) -> TtMhaWeights:
        """Shared-core view (drops the RoPE buffer, which the core never sees)."""
        return TtMhaWeights(
            wqkv=self.wqkv,
            wo=self.wo,
            rms_weight=self.rms_weight,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            eps=self.eps,
        )


def build_rope_cache(position_ids: torch.Tensor, inv_freq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Host RoPE cos/sin (fp32). position_ids (B,T), inv_freq (Dh//2) -> (cos, sin) (B,T,Dh)."""
    with torch.no_grad():
        inv_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        pos_expanded = position_ids[:, None, :].float()
        freqs = (inv_expanded.float() @ pos_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()
