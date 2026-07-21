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

    def inference_fully_traced(
        self,
        text_ids,
        cond_mel,
        ref_wav_spk,
        max_seq,
        max_new_tokens=MAX_AUDIO_TOKENS,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
        min_new_tokens=0,
    ):
        """Full model via THREE chained ttnn traces — every on-device op runs inside a trace:
          1. SETUP  : conditioning + speaker encoder + prefill (seeds the persistent KV cache),
          2. DECODE : one static-KV decode step, captured once and replayed per token,
          3. VOCODER: HiFi-GAN on the generated latents.
        Only the host tokenizer / conditioning-mel / per-token sampling stay eager. Assumes a
        single conditioning chunk (mel <= one chunk); returns ``(waveform [1, T_out, 1], codes)``.
        NOTE: all host->device writes are done BEFORE any capture — writes are fatal inside a trace."""
        dev = self.device
        gpt = self.gpt

        # Pre-place every host input on device up front (no host->device write inside a capture).
        mel_dev = self.conditioning.mel_to_device(cond_mel)
        text_dev = gpt.text_ids_to_device(text_ids)
        gpt.alloc_static_kv(max_seq)  # persistent zero caches, seeded by the setup trace

        def _setup():  # conditioning -> cond_latents ; speaker -> g ; prefill -> seed caches
            cl = ttnn.permute(self.conditioning.forward_dev(mel_dev), (0, 2, 1))  # [1, 32, 1024]
            g = self.decoder.speaker_embedding(ref_wav_spk)  # [1, 1, 512]
            return g, gpt.prefill_dev(text_dev, cl)

        # Warmup (compile kernels + populate the mel-frontend index cache) so the captured region
        # has no host->device writes, then capture the SETUP trace and execute it once.
        _setup()
        ttnn.synchronize_device(dev)
        stid = ttnn.begin_trace_capture(dev, cq_id=0)
        g, prompt_len = _setup()  # g + seeded caches are the trace's persistent outputs
        ttnn.end_trace_capture(dev, stid, cq_id=0)
        ttnn.synchronize_device(dev)
        ttnn.execute_trace(dev, stid, blocking=True)
        ttnn.release_trace(dev, stid)

        # DECODE: captured decode-STEP trace replayed per token, with the SAME sampling as the demo
        # (host-side rep/temp/top-k/top-p, self-terminating at STOP) so the audio matches the demo.
        # (The fully-on-device counter-PRNG variant, generate_ondevice_traced, is available but its
        # deterministic PRNG + fixed step budget degrade quality — kept for the "no host in loop" case.)
        codes, latents = self.generator.generate_on_static_kv(
            prompt_len,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            min_new_tokens=min_new_tokens,
        )

        # VOCODER trace on the generated (fixed-length) latents + the speaker embedding g.
        lat_in = ttnn.to_layout(ttnn.typecast(latents, ttnn.float32), ttnn.ROW_MAJOR_LAYOUT)
        voc = self.decoder.decoder
        _ = voc(ttnn.clone(lat_in), g)  # warmup / compile
        ttnn.synchronize_device(dev)
        vtid = ttnn.begin_trace_capture(dev, cq_id=0)
        wav_dev = voc(ttnn.clone(lat_in), g)
        ttnn.end_trace_capture(dev, vtid, cq_id=0)
        ttnn.synchronize_device(dev)
        ttnn.execute_trace(dev, vtid, blocking=True)
        ttnn.release_trace(dev, vtid)
        return wav_dev, codes
