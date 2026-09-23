# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Single-chip TTNN GroupSelfAttention (encoder sublayer 2). Wrapper over TtMhaCore.

reference : models/experimental/chronos_forecast/reference/chronos2/layers.py
    x = x + GroupSelfAttention(x)  # no RoPE, batch-axis, mask (T,1,B,B)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange

from models.experimental.chronos_forecast.tt.mha_core import TtMhaCore, TtMhaWeights, maybe_upload_mask


@dataclass(frozen=True)
class TtGroupAttentionWeights:
    """Host-side weights using ``nn.Linear`` convention: (out_features, in_features)."""

    wqkv: torch.Tensor  # (3 * inner, d) — Wq/Wk/Wv concatenated along out dim
    wo: torch.Tensor  # (d, inner)
    rms_weight: torch.Tensor  # (d,)
    num_heads: int
    head_dim: int
    eps: float = 1e-6

    @classmethod
    def from_torch_layer(cls, layer) -> "TtGroupAttentionWeights":
        """Extract weights from a reference ``GroupSelfAttention`` (or matching module)."""
        mha = layer.self_attention
        wqkv = torch.cat([mha.q.weight.detach(), mha.k.weight.detach(), mha.v.weight.detach()], dim=0).clone()
        return cls(
            wqkv=wqkv,
            wo=mha.o.weight.detach().clone(),
            rms_weight=layer.layer_norm.weight.detach().clone(),
            num_heads=mha.n_heads,
            head_dim=mha.kv_proj_dim,
            eps=layer.layer_norm.variance_epsilon,
        )

    def to_mha(self) -> TtMhaWeights:
        """Shared-core view."""
        return TtMhaWeights(
            wqkv=self.wqkv,
            wo=self.wo,
            rms_weight=self.rms_weight,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            eps=self.eps,
        )


def build_group_mask(
    group_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Host group-time mask. group_ids (B,), attn (B,T) binary -> (T,1,B,B) additive."""
    with torch.no_grad():
        group_mask = group_ids[:, None] == group_ids[None, :]
        group_time_mask = torch.einsum("qb,bt->qbt", group_mask, attention_mask)
        group_time_mask = rearrange(group_time_mask, "q b t -> t 1 q b")
        return ((1.0 - group_time_mask.float()) * torch.finfo(dtype).min).to(dtype)


class TtGroupAttention:
    """TTNN group self-attention. Weights move host -> device once in ``__init__``."""

    def __init__(self, device, weights: TtGroupAttentionWeights):
        self.device = device
        self.weights = weights
        self.core = TtMhaCore(device, weights.to_mha())

    def forward(self, x_host: torch.Tensor, mask_host: torch.Tensor) -> torch.Tensor:
        """Host (B,T,d) + mask (T,1,B,B) -> host (B,T,d) float for PCC."""
        import ttnn

        b, t, _d = x_host.shape
        x = ttnn.from_torch(
            x_host.detach().to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Transpose batch/time: attention runs along the batch axis.
        x_flip = ttnn.permute(x, (1, 0, 2))
        # Group attention runs along batch, so its seq length is B.
        mask = maybe_upload_mask(self.device, mask_host, seq_len=b)
        out = self.core(x_flip, mask)
        ttnn.deallocate(x_flip)
        if mask is not None:
            ttnn.deallocate(mask)
        # Flip back; residual against the ORIGINAL (B, T, d).
        back = ttnn.permute(out, (1, 0, 2))
        ttnn.deallocate(out)
        if back.memory_config() != x.memory_config():
            back = ttnn.to_memory_config(back, x.memory_config())
        y = ttnn.add(x, back, memory_config=x.memory_config())
        # ttnn.linear promotes 3D host inputs to 4D on device; restore (B,T,d),
        # drop seq tile padding on host; return float for PCC.
        host = ttnn.to_torch(y).float()
        if host.dim() == 4 and host.shape[0] == 1:
            host = host.squeeze(0)
        return host[:, :t, :]

    __call__ = forward
