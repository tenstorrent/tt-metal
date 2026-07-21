# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN XTTS-v2 end-to-end inference: text + reference audio -> waveform.

Composes the ported modules into the full on-device model (mirrors
``reference/xtts_inference.py``):

    cond mel  -> TtXttsConditioning ---> cond_latents [1, 32, 1024]
    (cond_latents, text ids) -> TtXttsGenerator (KV-cache greedy) -> codes + latents [1, T, 1024]
    ref audio -> TtXttsHifiDecoder (speaker encoder mel + ResNet -> g; latents + g -> waveform)

Everything runs on device. The only remaining host touchpoint is the BPE text
tokenizer (not a tensor op); the conditioning 80-mel spectrogram now runs on
device too (``TtConditioningMel`` — a port of ``xtts_conditioning.wav_to_mel``),
so callers pass the raw reference waveform and the mel is computed on device.

The GPT runs in bf16 and its latents are cast to fp32 ROW_MAJOR at the handoff to
the (fp32) HiFi-GAN decoder.
"""

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.xtts.reference.xtts_conditioning import chunk_wav
from models.experimental.xtts.reference.xtts_gpt_generate import MAX_AUDIO_TOKENS
from models.experimental.xtts.tt.xtts_conditioning import TtXttsConditioning
from models.experimental.xtts.tt.xtts_conv import cond_bias_trace_safe
from models.experimental.xtts.tt.xtts_full_decoder import TtXttsHifiDecoder
from models.experimental.xtts.tt.xtts_generator import TtXttsGenerator
from models.experimental.xtts.tt.xtts_gpt_model import TtXttsGptModel
from models.experimental.xtts.tt.xtts_mel import TtConditioningMel


class TtXtts(LightweightModule):
    """Full XTTS-v2 on device. ``ref_decoder_full`` is a reference
    ``XttsHifiDecoderFull`` used only to source the decoder / speaker-encoder /
    mel-frontend weights."""

    def __init__(self, device, state_dict, ref_decoder_full):
        super().__init__()
        self.device = device
        self.conditioning = TtXttsConditioning(state_dict, device)
        self.cond_mel_fe = TtConditioningMel(device, state_dict["mel_stats"].cpu())
        self.gpt = TtXttsGptModel(state_dict, device)
        self.generator = TtXttsGenerator(self.gpt)
        self.decoder = TtXttsHifiDecoder(device, ref_decoder_full)

    def _wav_chunk_to_device(self, chunk):  # torch [1, Lc] @ 22050 -> ttnn [1, Lc, 1] ROW_MAJOR fp32
        return ttnn.from_torch(
            chunk.reshape(1, -1, 1).float(), layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device, dtype=ttnn.float32
        )

    def _style_from_mel(self, mel_dev):  # device fp32 mel [1, 80, s] -> conditioning style [1, 1024, 32]
        return self.conditioning.forward_dev(ttnn.typecast(mel_dev, ttnn.bfloat16))

    def _cond_latents(self, cond_wav):  # torch [1, L] @ 22050 -> ttnn [1, 32, 1024]
        # coqui get_gpt_cond_latents: chunk the AUDIO into gpt_cond_chunk_len windows, compute the
        # 80-mel ON DEVICE per chunk, encode each (get_style_emb -> [1, 1024, 32]) and average the
        # style embeddings. A wav shorter than one chunk yields a single window (single-pass).
        parts = [self._style_from_mel(self.cond_mel_fe(self._wav_chunk_to_device(c))) for c in chunk_wav(cond_wav)]
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
        cond_wav,
        ref_wav_spk,
        max_new_tokens=MAX_AUDIO_TOKENS,
        force_codes=None,
        temperature=0.0,
        top_k=0,
        repetition_penalty=1.0,
        top_p=1.0,
        min_new_tokens=0,
    ):
        """``text_ids``: ``[START]/[STOP]``-wrapped torch ids. ``cond_wav``: torch raw
        22.05 kHz reference waveform ``[1, L]`` (the 80-mel is computed on device).
        ``ref_wav_spk``: ttnn 16 kHz audio ``[1, L, 1]`` ROW_MAJOR. ``force_codes``
        (optional) teacher-forces a fixed code sequence. ``temperature``/``top_k``/
        ``repetition_penalty`` enable on-device sampling (``temperature <= 0`` = greedy).
        Returns ``(waveform [1, T_out, 1], codes [1, T])``."""
        cond_latents = self._cond_latents(cond_wav)
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
        cond_wav,
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
        Only the host tokenizer / per-token sampling stay eager (the conditioning mel is now
        computed on device, inside the SETUP trace). Assumes a single conditioning chunk (ref wav
        <= one chunk); returns ``(waveform [1, T_out, 1], codes)``. NOTE: all host->device writes
        are done BEFORE any capture — writes are fatal inside a trace, so the raw wav is pre-placed
        and the mel-frontend index cache is warmed by the first _setup() call before capture."""
        dev = self.device
        gpt = self.gpt

        # Pre-place every host input on device up front (no host->device write inside a capture).
        wav_dev = self._wav_chunk_to_device(chunk_wav(cond_wav)[0])  # single conditioning chunk
        text_dev = gpt.text_ids_to_device(text_ids)
        gpt.alloc_static_kv(max_seq)  # persistent zero caches, seeded by the setup trace

        def _setup():  # cond mel (on device) -> cond_latents ; speaker -> g ; prefill -> seed caches
            cl = ttnn.permute(self._style_from_mel(self.cond_mel_fe(wav_dev)), (0, 2, 1))  # [1, 32, 1024]
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

        # DECODE: FULLY on-device — one captured decode-STEP trace replayed for a fixed budget, with
        # rep/temp/top-k/top-p sampling done ON DEVICE (Gumbel-max over host-pre-drawn noise) and
        # on-device token feedback + latent/code accumulation. This is the clean pre->device->post
        # shape: the noise is drawn on host up front (preprocessing), nothing crosses to host inside
        # the loop, and STOP self-termination becomes a post-loop trim. The sampler now matches the
        # host path in distribution (validated CER ~0.017), so quality no longer regresses vs demo.
        codes, latents = self.generator.generate_ondevice_traced(
            prompt_len,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # VOCODER trace on the generated (fixed-length) latents + the speaker embedding g.
        # The vocoder folds its conditioning bias into the conv bias via a host transfer (faster,
        # the GAN-decoder optimization) — fatal inside a trace, so switch those convs to the
        # equivalent trace-safe on-device add for the captured region (eager callers keep the fold).
        lat_in = ttnn.to_layout(ttnn.typecast(latents, ttnn.float32), ttnn.ROW_MAJOR_LAYOUT)
        voc = self.decoder.decoder
        with cond_bias_trace_safe():
            _ = voc(ttnn.clone(lat_in), g)  # warmup / compile
            ttnn.synchronize_device(dev)
            vtid = ttnn.begin_trace_capture(dev, cq_id=0)
            wav_dev = voc(ttnn.clone(lat_in), g)
            ttnn.end_trace_capture(dev, vtid, cq_id=0)
            ttnn.synchronize_device(dev)
            ttnn.execute_trace(dev, vtid, blocking=True)
            ttnn.release_trace(dev, vtid)
        return wav_dev, codes
