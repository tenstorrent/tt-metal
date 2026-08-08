# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from models.demos.granite_ttm_r1.tt.ttnn_granite_ttm_channel_mixer import TtnnGraniteTTMChannelMixer
from models.demos.granite_ttm_r1.tt.ttnn_granite_ttm_time_mixer import TtnnGraniteTTMTimeMixer


class TtnnGraniteTTMAdaptivePatchingBlock:
    """Implements a single TinyTimeMixerAdaptivePatchingBlock with TTNN.

    Input / output shape: [B, C, P, d_model]  (unchanged by this block)

    Internally the block:
      1. Expands to finer granularity:
             [B, C, P, d] → reshape → [B, C, P*factor, d//factor]
      2. Runs mixer_layers (time_mixer + channel_mixer per layer)
      3. Collapses back:
             [B, C, P*factor, d//factor] → reshape → [B, C, P, d]

    Attribute names follow tsfm_public naming:
      parameters.mixer_layers[i].patch_mixer   → TtnnGraniteTTMTimeMixer
      parameters.mixer_layers[i].feature_mixer → TtnnGraniteTTMChannelMixer
    """

    def __init__(self, *, parameters, config, adaptive_patch_factor: int):
        self._factor = adaptive_patch_factor
        self._mixer_layers = [
            (
                TtnnGraniteTTMTimeMixer(parameters=layer_params.patch_mixer, config=config),
                TtnnGraniteTTMChannelMixer(parameters=layer_params.feature_mixer, config=config),
            )
            for layer_params in parameters.mixer_layers
        ]

    def __call__(self, hidden_states, *, device=None):
        import ttnn

        B, C, P, d = hidden_states.shape
        factor = self._factor

        # Expand to finer granularity: [B, C, P, d] → [B, C, P*factor, d//factor]
        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        hidden_states = ttnn.reshape(hidden_states, [B, C, P * factor, d // factor])
        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)

        # Run each mixer layer (time_mixer then channel_mixer)
        for time_mixer, channel_mixer in self._mixer_layers:
            hidden_states = time_mixer(hidden_states, device=device)
            hidden_states = channel_mixer(hidden_states, device=device)

        # Collapse back: [B, C, P*factor, d//factor] → [B, C, P, d]
        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        hidden_states = ttnn.reshape(hidden_states, [B, C, P, d])
        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)

        return hidden_states


class TtnnGraniteTTMEncoderBlock:
    """Implements TinyTimeMixerBlock (the outer encoder wrapper) with TTNN.

    Wraps `adaptive_patching_levels` TtnnGraniteTTMAdaptivePatchingBlock instances
    corresponding to backbone.encoder.mlp_mixer_encoder.mixers[0..N-1].

    Each level has its own adaptive_patch_factor (e.g. 4, 2, 1 for TTM-R1).

    Input / output shape: [B, C, P, d_model]
    """

    def __init__(self, *, parameters, config, adaptive_patch_factors: list):
        self._blocks = [
            TtnnGraniteTTMAdaptivePatchingBlock(
                parameters=level_params,
                config=config,
                adaptive_patch_factor=factor,
            )
            for level_params, factor in zip(parameters.mixers, adaptive_patch_factors)
        ]

    def __call__(self, hidden_states, *, device=None):
        for block in self._blocks:
            hidden_states = block(hidden_states, device=device)
        return hidden_states
