# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
reference : models/experimental/chronos_forecast/reference/chronos2/model.py
    x = x + TimeSelfAttention(x)    # RoPE, mask (B,H,T,T)
    x = x + GroupSelfAttention(x)   # no RoPE, mask (T,1,B,B), batch-axis
    x = x + MLP(RMSNorm(x))         # Wi (d->d_ff, relu), Wo (d_ff->d), no bias
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.experimental.chronos_forecast.tt import program_configs
from models.experimental.chronos_forecast.tt.group_attention import TtGroupAttentionWeights
from models.experimental.chronos_forecast.tt.mha_core import TtMhaCore, maybe_upload_mask
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
    """TTNN encoder block. Weights move host -> device once in ``__init__``."""

    def __init__(
        self, device, weights: TtEncoderBlockWeights, precision: program_configs.TtChronosPrecision | None = None
    ):
        self.device = device
        self.weights = weights
        self.precision = precision or program_configs.TtChronosPrecision()
        self.time_core = TtMhaCore(device, weights.time.to_mha(), precision=self.precision)
        self.group_core = TtMhaCore(
            device, weights.group.to_mha(), enable_diagonal_v_path=True, precision=self.precision
        )
        self._ff = self._move_ff_weights_to_device(device, weights, self.precision.weight_dtype())

    @staticmethod
    def _move_ff_weights_to_device(device, weights: TtEncoderBlockWeights, weight_dtype):
        import ttnn

        def _weight(out_in: torch.Tensor):
            # ttnn.linear expects (in, out); torch nn.Linear stores (out, in).
            t = out_in.detach().to(torch.float32).t().contiguous()
            return ttnn.from_torch(
                t,
                dtype=weight_dtype,
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

    def forward_device(
        self, x, cos, sin, time_mask, group_mask, *, diagonal_group_attention: bool = False, memory_config=None
    ):
        """Device (B,T,d) + cos/sin (1 or B,1,T,Dh) + masks -> device (B,T,d); caller owns it.

        ``memory_config`` places the per-token path (time attention, diagonal
        group attention, FF) intermediates; the default is DRAM.
        """
        import ttnn

        mem = ttnn.DRAM_MEMORY_CONFIG if memory_config is None else memory_config
        ff_wi, ff_wo, ff_rms = self._ff

        # Sublayer 1: time attention + residual
        out = self.time_core(x, time_mask, cos, sin, memory_config=mem)
        x = self._residual_add(x, out)

        # Sublayer 2: group attention (batch-axis) + residual vs original layout
        if diagonal_group_attention:
            x = self._residual_add(x, self.group_core.forward_diagonal_group(x, memory_config=mem))
        else:
            x_flip = ttnn.permute(x, (1, 0, 2))
            out = self.group_core(x_flip, group_mask)
            ttnn.deallocate(x_flip)
            back = ttnn.permute(out, (1, 0, 2))
            ttnn.deallocate(out)
            x = self._residual_add(x, back)

        # Sublayer 3: feedforward (inline) + residual
        ff_fidelity = self.precision.ff_math_fidelity()
        n = ttnn.rms_norm(x, epsilon=self.weights.ff_eps, weight=ff_rms, memory_config=mem)
        h = program_configs.linear(
            n,
            ff_wi,
            activation="relu",
            math_fidelity=ff_fidelity,
            dtype=self.precision.ff_hidden_dtype(),
            memory_config=mem,
        )
        ttnn.deallocate(n)
        m = program_configs.linear(
            h, ff_wo, math_fidelity=ff_fidelity, dtype=self.precision.sublayer_out_dtype(), memory_config=mem
        )
        ttnn.deallocate(h)
        x = self._residual_add(x, m)
        return x

    def forward(
        self,
        x_host: torch.Tensor,
        cos_host: torch.Tensor,
        sin_host: torch.Tensor,
        time_mask_host: torch.Tensor,
        group_mask_host: torch.Tensor,
    ) -> torch.Tensor:
        """Host (B,T,d) + cos/sin + masks -> host (B,T,d) float for PCC."""
        import ttnn

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
        # Time masks attend over T; group masks attend over B (post-flip seq).
        time_mask = maybe_upload_mask(self.device, time_mask_host, seq_len=t)
        group_mask = maybe_upload_mask(self.device, group_mask_host, seq_len=_b)

        x = self.forward_device(x, cos, sin, time_mask, group_mask)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        if time_mask is not None:
            ttnn.deallocate(time_mask)
        if group_mask is not None:
            ttnn.deallocate(group_mask)

        # Single download, sliced to T; return float for PCC. ttnn.linear
        # promotes 3D host inputs to 4D on device, so restore (B,T,d) first.
        host = ttnn.to_torch(x).float()
        if host.dim() == 4 and host.shape[0] == 1:
            host = host.squeeze(0)
        host = host[:, :t, :]
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
