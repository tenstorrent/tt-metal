# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Single-chip TTNN ResidualBlock (input + output patch embeddings).

reference : models/experimental/chronos_forecast/reference/chronos2/layers.py
    hid = relu(hidden(x))     # (B, P, in) -> (B, P, h)
    out = output(hid)         # (B, P, h)  -> (B, P, out)
    return out + residual(x)  # skip projection, no norm, eval
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.experimental.chronos_forecast.tt import program_configs
from models.experimental.chronos_forecast.tt.program_configs import TILE


@dataclass(frozen=True)
class TtResidualBlockWeights:
    """Host-side weights using ``nn.Linear`` convention: (out_features, in_features)."""

    hidden_weight: torch.Tensor  # (h, 48)
    hidden_bias: torch.Tensor  # (h,)
    output_weight: torch.Tensor  # (out, h)
    output_bias: torch.Tensor  # (out,)
    residual_weight: torch.Tensor  # (out, 48)
    residual_bias: torch.Tensor  # (out,)
    act_fn_name: str = "relu"

    @classmethod
    def from_torch_block(cls, block) -> "TtResidualBlockWeights":
        """Extract weights from a reference ``ResidualBlock`` (or any matching module)."""
        return cls(
            hidden_weight=block.hidden_layer.weight.detach().clone(),
            hidden_bias=block.hidden_layer.bias.detach().clone(),
            output_weight=block.output_layer.weight.detach().clone(),
            output_bias=block.output_layer.bias.detach().clone(),
            residual_weight=block.residual_layer.weight.detach().clone(),
            residual_bias=block.residual_layer.bias.detach().clone(),
            act_fn_name="relu",
        )


def padded_in_features(in_features: int) -> int:
    """Input width after zero-padding to whole tiles (48 patch features -> 64)."""
    if in_features < TILE:
        return in_features
    return -(-in_features // TILE) * TILE


def pad_input_features(x: torch.Tensor, in_features: int) -> torch.Tensor:
    """Zero-pad the last dim of a host input to ``padded_in_features(in_features)``."""
    pad = padded_in_features(in_features) - x.shape[-1]
    return torch.nn.functional.pad(x, (0, pad)) if pad > 0 else x


class TtResidualBlock:
    """TTNN residual block. Weights move host -> device once in ``__init__``."""

    def __init__(self, device, weights: TtResidualBlockWeights):
        self.device = device
        self.weights = weights
        self.in_features = weights.hidden_weight.shape[1]
        self._tt = self._move_weights_to_device(device, weights)

    @staticmethod
    def _move_weights_to_device(device, weights: TtResidualBlockWeights):
        import ttnn

        in_features = weights.hidden_weight.shape[1]

        def _weight(out_in: torch.Tensor, *, pad_in: bool = False):
            if pad_in:
                out_in = pad_input_features(out_in, in_features)
            # ttnn.linear expects (in, out); torch nn.Linear stores (out, in).
            t = out_in.detach().to(torch.float32).t().contiguous()
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        def _bias(b: torch.Tensor):
            t = b.detach().to(torch.float32).reshape(1, 1, 1, -1).contiguous()
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        return (
            _weight(weights.hidden_weight, pad_in=True),
            _bias(weights.hidden_bias),
            _weight(weights.output_weight),
            _bias(weights.output_bias),
            _weight(weights.residual_weight, pad_in=True),
            _bias(weights.residual_bias),
        )

    def forward(self, x_host: torch.Tensor) -> torch.Tensor:
        """Forward starting from host input. Returns host torch (float32) for PCC."""
        import ttnn

        x = ttnn.from_torch(
            pad_input_features(x_host.detach(), self.in_features).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = self.forward_device(x, deallocate_input=True)
        # ttnn.linear promotes 3D host inputs to 4D on device; restore (B,T,C).
        host = ttnn.to_torch(out).float()
        if host.dim() == 4 and host.shape[0] == 1:
            host = host.squeeze(0)
        ttnn.deallocate(out)
        return host

    def forward_device(self, x, *, deallocate_input: bool = False):
        """Run the residual MLP entirely on device.

        ``x`` is borrowed by default so address-stable trace inputs can be
        refreshed and replayed. Set ``deallocate_input`` for owned temporaries.
        Inputs should already be zero-padded with ``pad_input_features``.
        """
        import ttnn

        hidden_w, hidden_b, output_w, output_b, residual_w, residual_b = self._tt
        if x.shape[-1] != padded_in_features(self.in_features):
            raise ValueError(
                f"expected input width {padded_in_features(self.in_features)} (pad_input_features), got {x.shape[-1]}"
            )
        # Main path: 48 -> h (fused relu) -> out.
        hidden_act = program_configs.linear(x, hidden_w, bias=hidden_b, activation="relu")
        main = program_configs.linear(hidden_act, output_w, bias=output_b)
        ttnn.deallocate(hidden_act)
        # Skip projection path: 48 -> out.
        skip = program_configs.linear(x, residual_w, bias=residual_b)
        if deallocate_input:
            ttnn.deallocate(x)
        if skip.memory_config() != main.memory_config():
            skip = ttnn.to_memory_config(skip, main.memory_config())
        out = ttnn.add(main, skip, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(main)
        ttnn.deallocate(skip)
        return out

    __call__ = forward
