# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Single-chip TTNN Chronos2Encoder (N x block + final norm). One upload/download.

reference : models/experimental/chronos_forecast/reference/chronos2/model.py
    h = block(h) for each block  # time + group + FF, masks/RoPE reused
    h = final_layer_norm(h)      # eval: dropouts are no-ops
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.experimental.chronos_forecast.tt.encoder_block import TtEncoderBlock, TtEncoderBlockWeights
from models.experimental.chronos_forecast.tt.mha_core import maybe_upload_mask
from models.experimental.chronos_forecast.tt.program_configs import TtChronosPrecision


@dataclass(frozen=True)
class TtEncoderWeights:
    """Host-side weights: per-block weights + final norm."""

    blocks: tuple  # tuple[TtEncoderBlockWeights, ...]
    final_rms_weight: torch.Tensor  # (d,)
    final_eps: float = 1e-6

    @classmethod
    def from_torch_encoder(cls, encoder) -> "TtEncoderWeights":
        """Extract weights from a reference ``Chronos2Encoder``."""
        blocks = tuple(TtEncoderBlockWeights.from_torch_block(b) for b in encoder.block)
        return cls(
            blocks=blocks,
            final_rms_weight=encoder.final_layer_norm.weight.detach().clone(),
            final_eps=encoder.final_layer_norm.variance_epsilon,
        )


class TtEncoder:
    """TTNN encoder. Weights move host -> device once in ``__init__``."""

    def __init__(self, device, weights: TtEncoderWeights, precision: TtChronosPrecision | None = None):
        self.device = device
        self.weights = weights
        self.blocks = [TtEncoderBlock(device, w, precision) for w in weights.blocks]
        self._final_norm = self._move_final_norm_to_device(device, weights)

    @staticmethod
    def _move_final_norm_to_device(device, weights: TtEncoderWeights):
        import ttnn

        return ttnn.from_torch(
            weights.final_rms_weight.detach().to(torch.float32).reshape(1, -1).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward_device(self, x, cos, sin, time_mask, group_mask, *, diagonal_group_attention: bool = False):
        """Device (B,T,d) + cos/sin (1 or B,1,T,Dh) + masks -> device (B,T,d); caller owns it."""
        import ttnn

        for block in self.blocks:
            x = block.forward_device(
                x,
                cos,
                sin,
                time_mask,
                group_mask,
                diagonal_group_attention=diagonal_group_attention,
            )
        x = ttnn.rms_norm(x, epsilon=self.weights.final_eps, weight=self._final_norm)
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

        # ttnn.linear promotes 3D host inputs to 4D on device; restore (B,T,d).
        host = ttnn.to_torch(x).float()
        if host.dim() == 4 and host.shape[0] == 1:
            host = host.squeeze(0)
        host = host[:, :t, :]
        ttnn.deallocate(x)
        return host

    __call__ = forward
