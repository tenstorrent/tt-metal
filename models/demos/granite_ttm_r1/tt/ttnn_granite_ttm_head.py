# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from models.demos.granite_ttm_r1.tt.common import TorchModuleFallback


class TtnnGraniteTTMHead:
    """Implements TinyTimeMixerForPredictionHead with TTNN.

    Input shape:  [B, C, P, d_dec]
    Output shape: [B, forecast_len, C]

    Forward pass:
        1. Flatten last two dims: [B, C, P, d_dec] -> [B, C, P*d_dec]
        2. Linear:  [B, C, P*d_dec] -> [B, C, forecast_len]
        3. Permute: [B, C, forecast_len] -> [B, forecast_len, C]
    """

    def __init__(self, *, parameters=None, config=None, torch_module=None):
        self._params = parameters
        self._config = config
        self._fallback = TorchModuleFallback(torch_module) if torch_module is not None and parameters is None else None

    def __call__(self, hidden_states, *, device=None, past_values=None, future_values=None, **kwargs):
        if self._params is not None:
            return self._forward_ttnn(hidden_states, device=device)
        if self._fallback is not None:
            return self._fallback(hidden_states, device=device, **kwargs)
        return hidden_states

    def _forward_ttnn(self, hidden_states, *, device):
        import ttnn

        # hidden_states: [B, C, P, d_dec]
        B, C, P, d = hidden_states.shape

        # Flatten last two dims: [B, C, P, d_dec] -> [B, C, P*d_dec]
        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        hidden_states = ttnn.reshape(hidden_states, [B, C, P * d])
        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)

        # Linear: [B, C, P*d_dec] -> [B, C, forecast_len]
        hidden_states = ttnn.linear(
            hidden_states,
            self._params.base_forecast_block.weight,
            bias=self._params.base_forecast_block.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )

        # Permute: [B, C, forecast_len] -> [B, forecast_len, C]
        hidden_states = ttnn.permute(hidden_states, (0, 2, 1))
        return hidden_states
