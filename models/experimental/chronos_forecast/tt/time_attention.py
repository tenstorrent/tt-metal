# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Single-chip TTNN TimeSelfAttention for Chronos-2 (encoder sublayer 1).

Device-only. Thin wrapper over the shared :class:`TtMhaCore`: builds RoPE
cos/sin on host, delegates norm/QKV/attention/projection to the core, adds
the residual. Data starts on host and moves host -> device inside ``forward``.

Oracle: ``models/experimental/chronos_forecast/reference/chronos2/layers.py``
``TimeSelfAttention`` (eval mode)::

    x_norm = RMSNorm(x)                        # (B, T, d), T5-style, no bias
    Q/K/V = x_norm @ Wq/Wk/Wv (no bias)        # -> (B, H, T, Dh)
    Q', K' = RoPE(Q, K, position_ids)          # V untouched
    ctx = softmax(Q' @ K'.T * 1.0 + mask) @ V  # scale is 1.0, NOT 1/sqrt(Dh)
    out = merge(ctx) @ Wo (no bias)            # (B, T, d)
    return x + out
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.experimental.chronos_forecast.tt.mha_core import TtMhaCore, TtMhaWeights


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
        wqkv = torch.cat(
            [mha.q.weight.detach(), mha.k.weight.detach(), mha.v.weight.detach()], dim=0
        ).clone()
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


def build_rope_cache(
    position_ids: torch.Tensor, inv_freq: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Host RoPE cos/sin replicating ``Chronos2RotaryEmbedding.forward`` (fp32).

    Args:
        position_ids: (B, T) long.
        inv_freq: (Dh // 2,) float.

    Returns:
        (cos, sin), each (B, T, Dh) float32.
    """
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
        """Forward starting from host inputs. Returns host torch (float32) for PCC.

        Args:
            x_host: (B, T, d) float.
            cos_host / sin_host: (B, T, Dh) float32 from :func:`build_rope_cache`.
            mask_host: (1, 1, T, T) additive (0 valid / large-negative invalid).
        """
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
        mask = ttnn.from_torch(
            mask_host.detach().to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = self.core(x, mask, cos, sin)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        ttnn.deallocate(mask)
        if out.memory_config() != x.memory_config():
            out = ttnn.to_memory_config(out, x.memory_config())
        y = ttnn.add(x, out, memory_config=x.memory_config())
        ttnn.deallocate(x)
        ttnn.deallocate(out)
        # Drop seq tile padding on host; return float for PCC.
        return ttnn.to_torch(y).float()[:, :t, :]

    __call__ = forward
