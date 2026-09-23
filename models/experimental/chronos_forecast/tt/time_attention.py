# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Single-chip TTNN TimeSelfAttention (encoder sublayer 1). Wrapper over TtMhaCore.

reference : models/experimental/chronos_forecast/reference/chronos2/layers.py
    x = x + TimeSelfAttention(x)  # RMSNorm, RoPE Q/K, SDPA scale 1.0, mask (B,H,T,T)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.experimental.chronos_forecast.tt.mha_core import TtMhaCore, TtMhaWeights, maybe_upload_mask


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


class TtTimeAttention:
    """TTNN time self-attention. Weights move host -> device once in ``__init__``."""

    def __init__(self, device, weights: TtTimeAttentionWeights):
        self.device = device
        self.weights = weights
        self.core = TtMhaCore(device, weights.to_mha())

    def forward(
        self,
        x_host: torch.Tensor,
        cos_host: torch.Tensor,
        sin_host: torch.Tensor,
        mask_host: torch.Tensor,
    ) -> torch.Tensor:
        """Host (B,T,d) + cos/sin (B,T,Dh) + mask (1,1,T,T) -> host (B,T,d) float for PCC."""
        import ttnn

        b, t, _d = x_host.shape
        x = ttnn.from_torch(
            x_host.detach().to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # cos/sin (B,T,Dh) -> (B,1,T,Dh) broadcast over heads.
        cos = ttnn.unsqueeze(
            ttnn.from_torch(
                cos_host.detach().to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            1,
        )
        sin = ttnn.unsqueeze(
            ttnn.from_torch(
                sin_host.detach().to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            1,
        )
        # Upload the mask unless it is an all-zero small-seq identity (see
        # maybe_upload_mask); callers must guard the deallocate below.
        mask = maybe_upload_mask(self.device, mask_host, seq_len=t)
        out = self.core(x, mask, cos, sin)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        if mask is not None:
            ttnn.deallocate(mask)
        if out.memory_config() != x.memory_config():
            out = ttnn.to_memory_config(out, x.memory_config())
        y = ttnn.add(x, out, memory_config=x.memory_config())
        # ttnn.linear promotes 3D host inputs to 4D on device; restore (B,T,d),
        # drop seq tile padding on host; return float for PCC.
        host = ttnn.to_torch(y).float()
        if host.dim() == 4 and host.shape[0] == 1:
            host = host.squeeze(0)
        return host[:, :t, :]

    __call__ = forward
