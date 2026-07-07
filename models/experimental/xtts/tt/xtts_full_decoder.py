# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN XTTS-v2 full ``HifiDecoder`` — reference audio + GPT latents -> waveform.

Composes the four phase modules into the complete GAN decoder, entirely on device:

    ref audio ─► mel frontend ─► speaker encoder ─► g          (conditioning)
    GPT latents ─────────────────────────────────► decoder(latents, g) ─► waveform

``speaker_embedding`` mirrors coqui's ``HifiDecoder.speaker_encoder(audio).unsqueeze(-1)``
(computed once per speaker); ``forward`` chains it into the latent-upsample + generator.
"""

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.xtts.tt.xtts_hifi_decoder import TtHifiDecoder
from models.experimental.xtts.tt.xtts_mel import TtMelFrontend
from models.experimental.xtts.tt.xtts_speaker_encoder import TtResNetSpeakerEncoder

COND_CHANNELS = 512


class TtXttsHifiDecoder(LightweightModule):
    """Full HifiDecoder. ``ref_wav`` is ``[1, L, 1]`` ROW_MAJOR, ``latents`` is
    ``[1, T, 1024]`` ROW_MAJOR channels-last; output is the ``[1, T_out, 1]`` waveform."""

    def __init__(self, device, ref_full):
        super().__init__()
        self.device = device
        self.mel_frontend = TtMelFrontend(device, ref_full.mel_frontend)
        self.speaker_encoder = TtResNetSpeakerEncoder(device, ref_full.speaker_encoder)
        self.decoder = TtHifiDecoder(device, ref_full.decoder.waveform_decoder.state_dict())

    def speaker_embedding(self, ref_wav):  # [1, L, 1] -> g [1, 1, 512] (channels-last)
        mel = self.mel_frontend(ref_wav)  # [1, 64, T] TILE
        g = self.speaker_encoder(mel)  # [1, 512]
        g = ttnn.reshape(g, [1, 1, COND_CHANNELS])
        return ttnn.to_layout(g, ttnn.ROW_MAJOR_LAYOUT)

    def forward(self, latents, ref_wav):
        return self.decoder(latents, self.speaker_embedding(ref_wav))
