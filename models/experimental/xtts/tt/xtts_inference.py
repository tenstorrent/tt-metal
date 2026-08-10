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

import time

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.xtts.reference.xtts_conditioning import chunk_wav
from models.experimental.xtts.reference.xtts_gpt_generate import MAX_AUDIO_TOKENS
from models.experimental.xtts.tt.xtts_conditioning import TtXttsConditioning
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

    def _wav_chunk_to_device(self, chunk):  # torch [1, Lc] @ 22050 -> ttnn [1, Lc] ROW_MAJOR fp32
        # [1, Lc], NOT [1, Lc, 1]: a rank-3 trailing-1 shape gives ROW_MAJOR 4-byte pages, and the
        # mel frontend's reshape out of that costs 31.8 ms. See xtts_mel._flat_signal.
        return ttnn.from_torch(
            chunk.reshape(1, -1).float(), layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device, dtype=ttnn.float32
        )

    def _style_from_mel(self, mel_dev):  # device fp32 mel [1, 80, s] -> conditioning style [1, 1024, 32]
        return self.conditioning.forward_dev(ttnn.typecast(mel_dev, ttnn.bfloat16))

    def _style_window(self, wav_dev):  # ttnn [1, Lc] -> ttnn [1, 1024, 32] in DRAM
        """One conditioning window's style embedding, parked in DRAM.

        DRAM is not incidental. The mel frontend's ROW_MAJOR reshape sizes its circular buffers by
        the whole padded signal (one page), which is ~1.48 MB of the 1.5 MB L1 — see the L1 note in
        ``xtts_mel``. A style embedding left in L1 across a window boundary therefore kills the NEXT
        window's mel with "statically allocated circular buffers clash with L1 buffers" (measured:
        an 8-window 30 s reference, an L1 buffer at 1350272 against a CB region ending at 1555664).
        64 KB round-tripped to DRAM per window is the price of conditioning on long audio at all."""
        style = self._style_from_mel(self.cond_mel_fe(wav_dev))  # [1, 1024, 32], L1
        out = ttnn.to_memory_config(style, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(style)
        return out

    def _style_mean(self, wav_devs):  # list of ttnn [1, Lc] -> ttnn [1, 32, 1024]
        """coqui ``get_gpt_cond_latents``: mel + ``get_style_emb`` per conditioning window, then
        AVERAGE the ``[1, 1024, 32]`` style embeddings. One window = single pass, no averaging.

        TRACE-SAFE, which is why it takes already-placed device wavs rather than the host waveform:
        the whole thing runs inside the SETUP trace, where a host->device write would be fatal. The
        windows are summed as they are computed rather than stacked, so only the DRAM accumulator
        and the current window's style are live (a 30 s reference is 8 windows)."""
        dram = ttnn.DRAM_MEMORY_CONFIG
        acc = self._style_window(wav_devs[0])
        for w in wav_devs[1:]:
            part = self._style_window(w)
            nxt = ttnn.add(acc, part, memory_config=dram)
            ttnn.deallocate(part)
            ttnn.deallocate(acc)
            acc = nxt
        if len(wav_devs) > 1:
            mean = ttnn.multiply(acc, 1.0 / len(wav_devs), memory_config=dram)
            ttnn.deallocate(acc)
            acc = mean
        # Back to L1 for the prefill, which is where the single-window path always handed it over.
        out = ttnn.permute(acc, (0, 2, 1), memory_config=ttnn.L1_MEMORY_CONFIG)  # [1, 1024, 32] -> [1, 32, 1024]
        ttnn.deallocate(acc)
        return out

    def _cond_latents(self, cond_wav):  # torch [1, L] @ 22050 -> ttnn [1, 32, 1024]
        return self._style_mean([self._wav_chunk_to_device(c) for c in chunk_wav(cond_wav)])

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
        computed on device, inside the SETUP trace). Returns ``(waveform [1, T_out, 1], codes,
        perf)`` where ``perf`` has ``replay_s`` (execute_trace only — final inference) and
        ``compile_s`` (warmup + capture).
        NOTE: all host->device writes are done BEFORE any capture — writes are fatal inside a
        trace, so the raw wav is pre-placed and the mel-frontend / conditioning constant caches
        (framer reversal matrices, per-length matmul program configs) are warmed by the first
        _setup() call before capture."""
        dev = self.device
        gpt = self.gpt
        t_all0 = time.perf_counter()

        # Pre-place every host input on device up front (no host->device write inside a capture).
        # ALL gpt_cond_chunk_len windows of the reference audio, not just the first: coqui averages
        # the style embedding over up to gpt_cond_len (30 s), and a long reference is what that is
        # for. The windows differ in length only in the trailing one, so the capture sees at most
        # two distinct conditioning shapes.
        wav_devs = [self._wav_chunk_to_device(c) for c in chunk_wav(cond_wav)]
        text_dev = gpt.text_ids_to_device(text_ids)
        gpt.alloc_static_kv(max_seq)  # persistent zero caches, seeded by the setup trace

        def _setup():  # cond mel (on device) -> cond_latents ; speaker -> g ; prefill -> seed caches
            cl = self._style_mean(wav_devs)  # [1, 32, 1024], averaged over the windows
            g = self.decoder.speaker_embedding(ref_wav_spk)  # [1, 1, 512]
            return g, gpt.prefill_on_device(text_dev, cl)

        # Warmup (compile kernels + populate the mel-frontend index cache) so the captured region
        # has no host->device writes, then capture the SETUP trace and execute it once.
        _setup()
        ttnn.synchronize_device(dev)
        stid = ttnn.begin_trace_capture(dev, cq_id=0)
        g, prompt_len = _setup()  # g + seeded caches are the trace's persistent outputs
        ttnn.end_trace_capture(dev, stid, cq_id=0)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        ttnn.execute_trace(dev, stid, blocking=True)
        setup_replay_s = time.perf_counter() - t0
        ttnn.release_trace(dev, stid)

        # DECODE: FULLY on-device — one captured decode-STEP trace replayed for a fixed budget, with
        # rep/temp/top-k/top-p sampling done ON DEVICE (Gumbel-max over host-pre-drawn noise) and
        # on-device token feedback + latent/code accumulation. This is the clean pre->device->post
        # shape: the noise is drawn on host up front (preprocessing), nothing crosses to host inside
        # the loop, and STOP self-termination becomes a post-loop trim. The sampler now matches the
        # host path in distribution (validated CER ~0.017), so quality no longer regresses vs demo.
        codes, latents, decode_replay_s = self.generator.generate_ondevice_traced(
            prompt_len,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            min_new_tokens=min_new_tokens,
        )

        # VOCODER trace on the generated (fixed-length) latents + the speaker embedding g.
        # The vocoder folds its conditioning bias into the conv bias, which used to need a host
        # transfer per call — fatal inside a trace, so the captured region ran on the equivalent
        # trace-safe post-conv device add instead. It no longer does: the folded bias is a function
        # of g and the input signature only, so TtConv1d keeps the PREPARED one and the warmup call
        # below primes it. The captured region then does no host transfer at all, and the fast fold
        # is what gets traced — worth the trace-safe add's ~82us/pass, and it makes the replay
        # byte-identical to eager (the post-conv add was not: it diverges in the last bits).
        lat_in = ttnn.to_layout(ttnn.typecast(latents, ttnn.float32), ttnn.ROW_MAJOR_LAYOUT)
        voc = self.decoder.decoder
        _ = voc(ttnn.clone(lat_in), g)  # warmup / compile; also primes the conditioning + bias caches
        ttnn.synchronize_device(dev)
        vtid = ttnn.begin_trace_capture(dev, cq_id=0)
        wav_dev = voc(ttnn.clone(lat_in), g)
        ttnn.end_trace_capture(dev, vtid, cq_id=0)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        ttnn.execute_trace(dev, vtid, blocking=True)
        vocoder_replay_s = time.perf_counter() - t0
        ttnn.release_trace(dev, vtid)
        # The vocoder memoises everything derived from g and never evicts on its own, because a live
        # trace reads those exact addresses. ``g`` here is the SETUP trace's output — a fresh tensor
        # every call — so without this a looping caller accumulates one conditioning set per
        # utterance, not per speaker. The trace that used them is released on the line above, which
        # makes this the safe point to drop them.
        voc.generator.release_conditioning()
        replay_s = setup_replay_s + decode_replay_s + vocoder_replay_s
        compile_s = max(0.0, time.perf_counter() - t_all0 - replay_s)
        perf = {
            "replay_s": replay_s,
            "compile_s": compile_s,
            "setup_replay_s": setup_replay_s,
            "decode_replay_s": decode_replay_s,
            "vocoder_replay_s": vocoder_replay_s,
        }
        return wav_dev, codes, perf
