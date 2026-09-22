# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Single-chip TTNN Chronos2Encoder (N x EncoderBlock + final RMSNorm).

Device-only. Activations stay on device across blocks: one host upload, one
download. Masks and RoPE cos/sin are uploaded once and reused by every block.

Oracle: ``models/experimental/chronos_forecast/reference/chronos2/model.py``
``Chronos2Encoder`` (eval mode)::

    h = dropout(embeds)              # no-op in eval
    for block in blocks:
        h = block(h, position_ids, time_mask, group_mask)
    h = final_layer_norm(h)
    h = dropout(h)                   # no-op in eval
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.experimental.chronos_forecast.tt.encoder_block import TtEncoderBlock, TtEncoderBlockWeights


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

    def __init__(self, device, weights: TtEncoderWeights):
        self.device = device
        self.weights = weights
        self.blocks = [TtEncoderBlock(device, w) for w in weights.blocks]
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

    def forward_device(self, x, cos, sin, time_mask, group_mask):
        """Device-to-device encoder. Borrowed tensors are NOT deallocated.

        Args:
            x: device (B, T, d) TILE.
            cos / sin: device (B, 1, T, Dh) TILE.
            time_mask: device (1, 1, T, T) TILE + DRAM.
            group_mask: device (T, 1, B, B) TILE + DRAM.

        Returns:
            Device (B, T, d) tensor; caller owns it.
        """
        import ttnn

        for block in self.blocks:
            x = block.forward_device(x, cos, sin, time_mask, group_mask)
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
        """Forward starting from host inputs. Returns host torch (float32) for PCC.

        Args:
            x_host: (B, T, d) float.
            cos_host / sin_host: (B, T, Dh) float32 from ``build_rope_cache``.
            time_mask_host: (1, 1, T, T) additive.
            group_mask_host: (T, 1, B, B) additive.
        """
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
        time_mask = _upload(time_mask_host)
        group_mask = _upload(group_mask_host)

        x = self.forward_device(x, cos, sin, time_mask, group_mask)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        ttnn.deallocate(time_mask)
        ttnn.deallocate(group_mask)

        host = ttnn.to_torch(x).float()[:, :t, :]
        ttnn.deallocate(x)
        return host

    __call__ = forward
