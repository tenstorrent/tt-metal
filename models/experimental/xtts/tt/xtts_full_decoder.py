# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.xtts.config import COND_CHANNELS  # noqa: F401 — re-exported for callers
from models.experimental.xtts.tt.xtts_hifi_decoder import TtHifiDecoder
from models.experimental.xtts.tt.xtts_mel import TtMelFrontend
from models.experimental.xtts.tt.xtts_speaker_encoder import TtResNetSpeakerEncoder


class TtXttsHifiDecoder(LightweightModule):
    def __init__(self, device, ref_full):
        """Wire mel frontend, speaker encoder, and HiFi decoder."""
        super().__init__()
        self.device = device
        self.mel_frontend = TtMelFrontend(device, ref_full.mel_frontend)
        self.speaker_encoder = TtResNetSpeakerEncoder(device, ref_full.speaker_encoder)
        self.decoder = TtHifiDecoder(device, ref_full.decoder.waveform_decoder.state_dict())

    def speaker_embedding(self, ref_wav):
        """Compute speaker embedding from reference waveform."""
        mel = self.mel_frontend(ref_wav)
        g = self.speaker_encoder(mel)
        if mel.is_allocated():
            ttnn.deallocate(mel)
        g = ttnn.reshape(g, [1, 1, COND_CHANNELS])
        return ttnn.to_layout(g, ttnn.ROW_MAJOR_LAYOUT)

    def forward(self, latents, ref_wav):
        """Decode latents to waveform using speaker conditioning."""
        return self.decoder(latents, self.speaker_embedding(ref_wav))
