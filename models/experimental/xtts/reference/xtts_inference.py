# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch

from models.experimental.xtts.reference.xtts_conditioning import chunk_cond_mel, reference_conditioning
from models.experimental.xtts.reference.xtts_gpt_generate import START_AUDIO_TOKEN, greedy_generate
from models.experimental.xtts.reference.xtts_gpt_model import reference_gpt_model
from models.experimental.xtts.reference.xtts_hifi_decoder import XttsHifiDecoderFull


class XttsReference(torch.nn.Module):
    def __init__(self, state_dict):
        """Wire conditioning, GPT, and HiFi-GAN decoder from a checkpoint."""
        super().__init__()
        self.conditioning = reference_conditioning(state_dict)
        self.gpt = reference_gpt_model(state_dict)
        self.decoder_full = XttsHifiDecoderFull(state_dict)

    def _cond_latents(self, cond_mel):
        """Encode mel chunks and average into conditioning latents."""
        parts = [self.conditioning(m) for m in chunk_cond_mel(cond_mel)]
        style = torch.stack(parts, dim=0).mean(dim=0) if len(parts) > 1 else parts[0]
        return style.transpose(1, 2)

    @torch.no_grad()
    def inference(self, text_ids, cond_mel, ref_wav_spk, max_new_tokens):
        """Run greedy GPT generation then decode waveform from latents."""
        cond_latents = self._cond_latents(cond_mel)
        codes, latents = greedy_generate(
            self.gpt, text_ids, cond_latents, max_new_tokens=max_new_tokens, wrap_text=False
        )
        g = self.decoder_full.speaker_embedding(ref_wav_spk)
        wav = self.decoder_full.decoder(latents, g)
        return wav, codes

    @torch.no_grad()
    def wav_from_codes(self, text_ids, cond_mel, ref_wav_spk, codes):
        """Decode waveform from provided audio codes via GPT latents."""
        cond_latents = self._cond_latents(cond_mel)
        codes_t = torch.as_tensor(codes, dtype=torch.long).reshape(1, -1)
        start = torch.full((1, 1), START_AUDIO_TOKEN, dtype=torch.long)
        mel_ids = torch.cat([start, codes_t], dim=1)
        latents = self.gpt(text_ids, mel_ids, cond_latents=cond_latents, return_latent=True)[:, 1:]
        g = self.decoder_full.speaker_embedding(ref_wav_spk)
        return self.decoder_full.decoder(latents, g)
