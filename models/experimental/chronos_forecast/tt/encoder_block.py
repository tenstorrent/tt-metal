# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
x = x + TimeSelfAttention(x)    # RoPE, mask (B,H,T,T)
x = x + GroupSelfAttention(x)   # no RoPE, mask (T,1,B,B), batch-axis
x = x + MLP(RMSNorm(x))         # Wi (d->d_ff, relu), Wo (d_ff->d), no bias
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.experimental.chronos_forecast.tt.group_attention import TtGroupAttentionWeights
from models.experimental.chronos_forecast.tt.mha_core import TtMhaCore
from models.experimental.chronos_forecast.tt.time_attention import TtTimeAttentionWeights


@dataclass(frozen=True)
class TtEncoderBlockWeights:
    """Host-side weights using ``nn.Linear`` convention: (out_features, in_features)."""

    time: TtTimeAttentionWeights
    group: TtGroupAttentionWeights
    ff_wi: torch.Tensor  # (d_ff, d)
    ff_wo: torch.Tensor  # (d, d_ff)
    ff_rms_weight: torch.Tensor  # (d,)
    ff_eps: float = 1e-6

    @classmethod
    def from_torch_block(cls, block) -> "TtEncoderBlockWeights":
        """Extract weights from a reference ``Chronos2EncoderBlock``.

        ``block.layer`` is [TimeSelfAttention, GroupSelfAttention, FeedForward].
        """
        time_layer, group_layer, ff_layer = block.layer[0], block.layer[1], block.layer[2]
        return cls(
            time=TtTimeAttentionWeights.from_torch_layer(time_layer),
            group=TtGroupAttentionWeights.from_torch_layer(group_layer),
            ff_wi=ff_layer.mlp.wi.weight.detach().clone(),
            ff_wo=ff_layer.mlp.wo.weight.detach().clone(),
            ff_rms_weight=ff_layer.layer_norm.weight.detach().clone(),
            ff_eps=ff_layer.layer_norm.variance_epsilon,
        )


class TtEncoderBlock:
        self.device = device
        self.weights = weights
        self.time_core = TtMhaCore(device, weights.time.to_mha())
        self.group_core = TtMhaCore(device, weights.group.to_mha())
        self._ff = self._move_ff_weights_to_device(device, weights)

    @staticmethod
    def _move_ff_weights_to_device(device, weights: TtEncoderBlockWeights):
        import ttnn

        def _weight(out_in: torch.Tensor):
            # ttnn.linear expects (in, out); torch nn.Linear stores (out, in).
            t = out_in.detach().to(torch.float32).t().contiguous()
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        rms_w = ttnn.from_torch(
            weights.ff_rms_weight.detach().to(torch.float32).reshape(1, -1).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return (_weight(weights.ff_wi), _weight(weights.ff_wo), rms_w)

    def forward(
        self,
        x_host: torch.Tensor,
        cos_host: torch.Tensor,
        sin_host: torch.Tensor,
        time_mask_host: torch.Tensor,
        group_mask_host: torch.Tensor,
    ) -> torch.Tensor:
        """Forward starting from host inputs. Returns host torch (float32) for PCC.

        Args:
            x_host: (B, T, d) float.
            cos_host / sin_host: (B, T, Dh) float32 from ``build_rope_cache``.
            time_mask_host: (1, 1, T, T) additive.
            group_mask_host: (T, 1, B, B) additive.
        """
        import ttnn

        ff_wi, ff_wo, ff_rms = self._ff
        _b, t, _d = x_host.shape

        def _upload(m: torch.Tensor):
            return ttnn.from_torch(
                m.detach().to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        x = _upload(x_host)
        cos = ttnn.unsqueeze(_upload(cos_host), 1)
        sin = ttnn.unsqueeze(_upload(sin_host), 1)
        time_mask = _upload(time_mask_host)
        group_mask = _upload(group_mask_host)

        # Sublayer 1: time attention + residual
        out = self.time_core(x, time_mask, cos, sin)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        ttnn.deallocate(time_mask)
        x = self._residual_add(x, out)

        # Sublayer 2: group attention (batch-axis) + residual vs original layout
        x_flip = ttnn.permute(x, (1, 0, 2))
        out = self.group_core(x_flip, group_mask)
        ttnn.deallocate(x_flip)
        ttnn.deallocate(group_mask)
        back = ttnn.permute(out, (1, 0, 2))
        ttnn.deallocate(out)
        x = self._residual_add(x, back)

        # Sublayer 3: feedforward (inline) + residual
        n = ttnn.rms_norm(x, epsilon=self.weights.ff_eps, weight=ff_rms)
        h = ttnn.linear(n, ff_wi, activation="relu", memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(n)
        m = ttnn.linear(h, ff_wo, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(h)
        x = self._residual_add(x, m)

        # Single download, sliced to T; return float for PCC
        host = ttnn.to_torch(x).float()[:, :t, :]
        ttnn.deallocate(x)
        return host

    @staticmethod
    def _residual_add(x, out):
        import ttnn

        if out.memory_config() != x.memory_config():
            out = ttnn.to_memory_config(out, x.memory_config())
        y = ttnn.add(x, out, memory_config=x.memory_config())
        ttnn.deallocate(x)
        ttnn.deallocate(out)
        return y

    __call__ = forward
