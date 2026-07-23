# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `vibe_voice_semantic_tokenizer_model` of microsoft/VibeVoice-1.5B.

Reference submodule: `model.semantic_tokenizer`, a
`vibevoice.modular.modular_vibevoice_tokenizer.VibeVoiceSemanticTokenizerModel`
(encoder-only tokenizer, no decoder):

    def forward(self, audio, ...):
        encoder_output = self.encode(audio, ...)                # mean = encoder(audio).permute(0,2,1)
        sampled_latents, _ = self.sampling(encoder_output, dist_type='none')  # deterministic: returns mean as-is
        return None, sampled_latents

`sampling(dist_type='none')` hits `VibeVoiceTokenizerEncoderOutput.sample`'s
`else` branch (`return self.mean, self.std`) — no randomness here, unlike
the acoustic tokenizer's Gaussian sampling.
"""

from __future__ import annotations

import ttnn
from models.demos.vibevoice_1_5b._stubs.tokenizer_encoder import build as _build_tokenizer_encoder


def build(device, torch_module):
    """Bind the encoder's trained weights and return a native ttnn forward closure."""
    m = torch_module
    encoder_forward = _build_tokenizer_encoder(device, m.encoder)

    def forward(audio, *args, **kwargs):
        latents = encoder_forward(audio)  # [1, vae_dim, T'] channels-first
        sampled_latents = ttnn.transpose(latents, 1, 2)  # [1, T', vae_dim] == encode(...).mean
        return None, sampled_latents

    return forward


def vibe_voice_semantic_tokenizer_model(*args, **kwargs):
    raise RuntimeError(
        "vibe_voice_semantic_tokenizer_model requires build(device, torch_module) to bind trained weights; the bare callable has no parameters."
    )
