# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from models.demos.granite_ttm_r1.ttnn.common import TorchModuleFallback


class TtnnGraniteTTMEmbedding:
    def __init__(self, torch_module=None):
        self._fallback = TorchModuleFallback(torch_module) if torch_module is not None else None

    def __call__(self, hidden_states, *, device=None, **kwargs):
        if self._fallback is None:
            return hidden_states
        return self._fallback(hidden_states, device=device, **kwargs)
