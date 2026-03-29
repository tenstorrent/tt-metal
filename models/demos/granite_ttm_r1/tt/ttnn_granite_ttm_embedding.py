# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from models.demos.granite_ttm_r1.tt.common import TorchModuleFallback


class TtnnGraniteTTMEmbedding:
    """Wraps the backbone.encoder.patcher Linear layer (input projection from patch_length to d_model).

    When ``parameters`` is provided the layer is executed via ttnn; otherwise a
    ``TorchModuleFallback`` is used when ``torch_module`` is supplied.
    """

    def __init__(self, *, parameters=None, config=None, torch_module=None):
        self._params = parameters
        self._config = config
        # Use TorchModuleFallback only when no pre-processed TTNN parameters are available.
        self._fallback = TorchModuleFallback(torch_module) if torch_module is not None and parameters is None else None

    def __call__(self, hidden_states, *, device=None, **kwargs):
        if self._params is not None:
            return self._forward_ttnn(hidden_states, device=device)
        if self._fallback is not None:
            return self._fallback(hidden_states, device=device, **kwargs)
        return hidden_states

    def _forward_ttnn(self, hidden_states, *, device):
        import ttnn

        return ttnn.linear(
            hidden_states,
            self._params.weight,
            bias=self._params.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
