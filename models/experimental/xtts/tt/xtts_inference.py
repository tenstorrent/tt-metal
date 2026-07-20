# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN XTTS-v2 end-to-end inference: text + reference audio -> waveform.

Composes the ported modules into the full on-device model (mirrors
``reference/xtts_inference.py``):

    cond mel  -> TtXttsConditioning ---> cond_latents [1, 32, 1024]
    (cond_latents, text ids) -> TtXttsGenerator (KV-cache greedy) -> codes + latents [1, T, 1024]
    ref audio -> TtXttsHifiDecoder (speaker encoder mel + ResNet -> g; latents + g -> waveform)

Everything runs on device EXCEPT two host touchpoints, both outside the tensor
compute path: the BPE text tokenizer (not a tensor op) and the conditioning
80-mel spectrogram (``xtts_conditioning.wav_to_mel``, fed in as ``cond_mel``) —
the latter is the one remaining op to move on device for a strictly no-host
pipeline (the speaker-encoder mel frontend is already on device).

The GPT runs in bf16 and its latents are cast to fp32 ROW_MAJOR at the handoff to
the (fp32) HiFi-GAN decoder.
"""

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.xtts.reference.xtts_conditioning import chunk_cond_mel
from models.experimental.xtts.reference.xtts_gpt_generate import MAX_AUDIO_TOKENS
from models.experimental.xtts.tt.xtts_conditioning import TtXttsConditioning
from models.experimental.xtts.tt.xtts_full_decoder import TtXttsHifiDecoder
from models.experimental.xtts.tt.xtts_generator import TtXttsGenerator
from models.experimental.xtts.tt.xtts_gpt_model import TtXttsGptModel


class TtXtts(LightweightModule):
    """Full XTTS-v2 on device. ``ref_decoder_full`` is a reference
    ``XttsHifiDecoderFull`` used only to source the decoder / speaker-encoder /
    mel-frontend weights."""

    def __init__(self, device, state_dict, ref_decoder_full):
        super().__init__()
        self.device = device
        self.conditioning = TtXttsConditioning(state_dict, device)
        self.gpt = TtXttsGptModel(state_dict, device)
        self.generator = TtXttsGenerator(self.gpt)
        self.decoder = TtXttsHifiDecoder(device, ref_decoder_full)

    def _cond_latents(self, cond_mel):  # torch [1, 80, s] -> ttnn [1, 32, 1024]
        # coqui get_gpt_cond_latents: chunk the mel into gpt_cond_chunk_len windows, encode
        # each (get_style_emb -> [1, 1024, 32]), and average the style embeddings. A mel
        # shorter than one chunk yields a single window == the previous single-pass behaviour.
        parts = [self.conditioning(m) for m in chunk_cond_mel(cond_mel)]  # each [1, 1024, 32]
        if len(parts) == 1:
            style = parts[0]
        else:
            acc = parts[0]
            for p in parts[1:]:
                acc = ttnn.add(acc, p)
            style = ttnn.multiply(acc, 1.0 / len(parts))
        return ttnn.permute(style, (0, 2, 1))  # [1, 1024, 32] -> [1, 32, 1024]

    def _decode_wav(self, latents_tt, ref_wav_spk):
        # bf16 GPT latents -> fp32 ROW_MAJOR for the fp32 HiFi-GAN decoder.
        latents = ttnn.to_layout(ttnn.typecast(latents_tt, ttnn.float32), ttnn.ROW_MAJOR_LAYOUT)
        return self.decoder(latents, ref_wav_spk)  # [1, T_out, 1]

    def inference(
        self,
        text_ids,
        cond_mel,
        ref_wav_spk,
        max_new_tokens=MAX_AUDIO_TOKENS,
        force_codes=None,
        temperature=0.0,
        top_k=0,
        repetition_penalty=1.0,
        top_p=1.0,
        min_new_tokens=0,
    ):
        """``text_ids``: ``[START]/[STOP]``-wrapped torch ids. ``cond_mel``: torch
        80-mel ``[1, 80, s]``. ``ref_wav_spk``: ttnn 16 kHz audio ``[1, L, 1]``
        ROW_MAJOR. ``force_codes`` (optional) teacher-forces a fixed code sequence.
        ``temperature``/``top_k``/``repetition_penalty`` enable on-device sampling
        (``temperature <= 0`` = greedy). Returns ``(waveform [1, T_out, 1], codes [1, T])``."""
        cond_latents = self._cond_latents(cond_mel)
        if force_codes is not None:
            _, latents_tt = self.generator.latents_for_codes(text_ids, cond_latents, force_codes)
            codes = torch.tensor([force_codes], dtype=torch.long)
        else:
            codes, latents_tt = self.generator.generate(
                text_ids,
                cond_latents,
                max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                top_p=top_p,
                min_new_tokens=min_new_tokens,
            )
        wav = self._decode_wav(latents_tt, ref_wav_spk)
        return wav, codes
