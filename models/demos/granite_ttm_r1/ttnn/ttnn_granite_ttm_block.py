# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from models.demos.granite_ttm_r1.ttnn.common import TorchModuleFallback
from models.demos.granite_ttm_r1.ttnn.ttnn_granite_ttm_channel_mixer import TtnnGraniteTTMChannelMixer
from models.demos.granite_ttm_r1.ttnn.ttnn_granite_ttm_time_mixer import TtnnGraniteTTMTimeMixer


class TtnnGraniteTTMBlock:
    def __init__(self, torch_block=None, time_mixer=None, channel_mixer=None):
        self._fallback = TorchModuleFallback(torch_block) if torch_block is not None else None
        self.time_mixer = TtnnGraniteTTMTimeMixer(time_mixer)
        self.channel_mixer = TtnnGraniteTTMChannelMixer(channel_mixer)

    def __call__(self, hidden_states, *, device=None, **kwargs):
        if self._fallback is not None:
            return self._fallback(hidden_states, device=device, **kwargs)

        hidden_states = self.time_mixer(hidden_states, device=device, **kwargs)
        hidden_states = self.channel_mixer(hidden_states, device=device, **kwargs)
        return hidden_states
