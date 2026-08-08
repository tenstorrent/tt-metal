# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from models.demos.granite_ttm_r1.tt.common import TorchModuleFallback
from models.demos.granite_ttm_r1.tt.ttnn_granite_ttm_channel_mixer import TtnnGraniteTTMChannelMixer
from models.demos.granite_ttm_r1.tt.ttnn_granite_ttm_time_mixer import TtnnGraniteTTMTimeMixer


class TtnnGraniteTTMBlock:
    """Implements a single TinyTimeMixerLayer (patch_mixer followed by feature_mixer).

    Construction modes:
    - TTNN path: pass ``parameters`` (preprocessed parameter tree) and ``config``.
      Sub-mixers are constructed with their respective TTNN parameters.
    - Torch fallback path: pass ``torch_block`` to wrap the whole layer in
      ``TorchModuleFallback``.
    - Manual sub-mixer path: pass ``time_mixer`` and/or ``channel_mixer`` torch modules
      directly; each sub-mixer is individually wrapped in its own fallback.
    """

    def __init__(self, *, parameters=None, config=None, torch_block=None, time_mixer=None, channel_mixer=None):
        if parameters is not None:
            # TTNN path: build sub-mixers from pre-processed parameter subtrees.
            self._fallback = None
            self.time_mixer = TtnnGraniteTTMTimeMixer(parameters=parameters.patch_mixer, config=config)
            self.channel_mixer = TtnnGraniteTTMChannelMixer(parameters=parameters.feature_mixer, config=config)
        elif torch_block is not None:
            # Whole-block torch fallback path.
            self._fallback = TorchModuleFallback(torch_block)
            self.time_mixer = None
            self.channel_mixer = None
        else:
            # Per-sub-mixer fallback path (or no-op if neither is supplied).
            self._fallback = None
            self.time_mixer = (
                TtnnGraniteTTMTimeMixer(torch_module=time_mixer)
                if time_mixer is not None
                else TtnnGraniteTTMTimeMixer()
            )
            self.channel_mixer = (
                TtnnGraniteTTMChannelMixer(torch_module=channel_mixer)
                if channel_mixer is not None
                else TtnnGraniteTTMChannelMixer()
            )

    def __call__(self, hidden_states, *, device=None, **kwargs):
        if self._fallback is not None:
            return self._fallback(hidden_states, device=device, **kwargs)
        hidden_states = self.time_mixer(hidden_states, device=device)
        hidden_states = self.channel_mixer(hidden_states, device=device)
        return hidden_states
