# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Reference (pure-PyTorch) XTTS-v2 end-to-end inference: text + ref audio -> waveform.

Composes the already-validated reference modules into the full model, the ground
truth for the TTNN top-level (``tt/xtts_inference.py``):

    ref audio -> cond mel  -> conditioning encoder ---> cond_latents [1, 32, 1024]
    text      -----------------------------------------> text ids
    (cond_latents, text ids) -> GPT greedy generate ---> codes + gpt_latents [1, T, 1024]
    ref audio -> speaker mel -> speaker encoder --------> g [1, 512, 1]
    (gpt_latents, g) -> latent upsample + HiFi-GAN -----> waveform

Two reference mels are involved (matching coqui): the conditioning path uses the
2048-fft / 80-mel spectrogram (``xtts_conditioning.wav_to_mel``) at 22.05 kHz, and
the speaker encoder uses the 512-fft / 64-mel frontend at 16 kHz. Both are supplied
by the caller (the conditioning mel as a tensor; the speaker path takes raw audio).
"""

import torch

from models.experimental.xtts.reference.xtts_conditioning import chunk_cond_mel, reference_conditioning
from models.experimental.xtts.reference.xtts_gpt_generate import START_AUDIO_TOKEN, greedy_generate
from models.experimental.xtts.reference.xtts_gpt_model import reference_gpt_model
from models.experimental.xtts.reference.xtts_hifi_decoder import XttsHifiDecoderFull


class XttsReference(torch.nn.Module):
    """Full XTTS-v2 reference. ``decoder_full`` bundles the speaker encoder + mel
    frontend + HiFi-GAN decoder (also reused to source the TTNN decoder weights)."""

    def __init__(self, state_dict):
        super().__init__()
        self.conditioning = reference_conditioning(state_dict)
        self.gpt = reference_gpt_model(state_dict)
        self.decoder_full = XttsHifiDecoderFull(state_dict)

    def _cond_latents(self, cond_mel):
        """coqui get_gpt_cond_latents: chunk the mel into gpt_cond_chunk_len windows,
        get_style_emb per chunk, average -> [1, 32, 1024]. Single chunk for short mels."""
        parts = [self.conditioning(m) for m in chunk_cond_mel(cond_mel)]  # each [1, 1024, 32]
        style = torch.stack(parts, dim=0).mean(dim=0) if len(parts) > 1 else parts[0]
        return style.transpose(1, 2)  # [1, 1024, 32] -> [1, 32, 1024]

    @torch.no_grad()
    def inference(self, text_ids, cond_mel, ref_wav_spk, max_new_tokens):
        """``text_ids`` are ``[START]/[STOP]``-wrapped (pass ``wrap_text=False``-style
        ids); ``cond_mel`` is the 80-mel ``[1, 80, s]``; ``ref_wav_spk`` is 16 kHz
        speaker audio ``[1, L]``. Returns ``(waveform [1, 1, T_out], codes, latents)``."""
        cond_latents = self._cond_latents(cond_mel)  # [1, 32, 1024]
        codes, latents = greedy_generate(
            self.gpt, text_ids, cond_latents, max_new_tokens=max_new_tokens, wrap_text=False
        )
        g = self.decoder_full.speaker_embedding(ref_wav_spk)  # [1, 512, 1]
        wav = self.decoder_full.decoder(latents, g)  # [1, 1, T_out]
        return wav, codes

    @torch.no_grad()
    def wav_from_codes(self, text_ids, cond_mel, ref_wav_spk, codes):
        """Teacher-forced torch decode of a *given* code sequence to waveform — for an
        apples-to-apples A/B against the device output on identical codes (rather than
        running an independent greedy generation that would diverge)."""
        cond_latents = self._cond_latents(cond_mel)
        codes_t = torch.as_tensor(codes, dtype=torch.long).reshape(1, -1)
        start = torch.full((1, 1), START_AUDIO_TOKEN, dtype=torch.long)
        mel_ids = torch.cat([start, codes_t], dim=1)  # [start, c_0, ..., c_{T-1}]
        latents = self.gpt(text_ids, mel_ids, cond_latents=cond_latents, return_latent=True)[:, 1:]  # drop start
        g = self.decoder_full.speaker_embedding(ref_wav_spk)
        return self.decoder_full.decoder(latents, g)
