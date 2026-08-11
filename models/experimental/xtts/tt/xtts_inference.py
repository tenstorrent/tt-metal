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

The GPT runs in bf16 and its latents are handed to the (all-bf16) HiFi-GAN decoder
as they are — bf16 TILE, no cast or relayout at the boundary.
"""

import time

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.xtts.reference.xtts_conditioning import chunk_wav
from models.experimental.xtts.reference.xtts_gpt_block import HIDDEN_SIZE
from models.experimental.xtts.reference.xtts_gpt_generate import MAX_AUDIO_TOKENS
from models.experimental.xtts.reference.xtts_hifi_decoder import LATENT_SCALE, SR_SCALE
from models.experimental.xtts.reference.xtts_hifigan import COND_CHANNELS
from models.experimental.xtts.tt.xtts_conditioning import TtXttsConditioning
from models.experimental.xtts.tt.xtts_full_decoder import TtXttsHifiDecoder
from models.experimental.xtts.tt.xtts_generator import TtTracedDecoder, TtXttsGenerator
from models.experimental.xtts.tt.xtts_gpt_model import TtXttsGptModel
from models.experimental.xtts.tt.xtts_mel import TtConditioningMel


def _interp_len(frames):
    """Latent frames -> generator input steps: the decoder's two ``F.interpolate`` calls, each of
    which floors ``len * scale`` (``XttsHifiDecoderReference.forward``)."""
    return int(int(frames * LATENT_SCALE) * SR_SCALE)


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
        # Hand the GPT's latents over as they are: bf16 TILE. They used to be typecast to fp32 and
        # untilized to ROW_MAJOR here, for a decoder that was fp32 at the time; the decoder is all
        # bf16 now and its upsampler tilizes its input as its first op, so that pair round-tripped
        # straight back to where it started. Dropping both is bit-exact (A/B'd maxdiff 0.0).
        return self.decoder(latents_tt, ref_wav_spk)  # [1, T_out, 1]

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
        lat_in = latents  # bf16 TILE, handed straight over — see _decode_wav
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

    def traced_session(self, cond_wav, ref_wav_spk, text_len, max_seq, max_new_tokens, **sampling):
        """Capture the three traces ONCE for a fixed text length, then synthesise many texts off
        them. See :class:`TtXttsTracedSession`."""
        return TtXttsTracedSession(self, cond_wav, ref_wav_spk, text_len, max_seq, max_new_tokens, **sampling)


class TtXttsTracedSession:
    """The same three traces as :meth:`TtXtts.inference_fully_traced`, but captured ONCE and
    replayed for utterance after utterance.

    ``inference_fully_traced`` captures and releases per call, so synthesising text that has been
    split into N chunks compiles the whole model N times — by far the dominant cost (~1 minute per
    chunk against ~1.5 s of replay). Here warmup + capture happen in ``__init__`` and every chunk is
    just :meth:`run`: write the chunk's text ids into the persistent input buffer, replay SETUP
    (which re-seeds the KV cache from the new text), replay the decode step per token, replay the
    VOCODER. No compile, no allocation, no re-upload of weights between chunks.

    What that costs in flexibility — a trace is a fixed program over fixed buffers, so:

    * Every chunk must have the SAME padded text length (``text_len``), since ``prompt_len`` and
      the KV geometry are baked into the capture. The caller pads all chunks to one length.
    * The vocoder runs on a FIXED ``max_new_tokens``-frame latent buffer, zero-padded past the codes
      actually generated, and the waveform is trimmed back afterwards (:meth:`_samples_for` gives
      the exact length). Note this makes ``max_new_tokens`` a real cost: the vocoder always pays for
      the full budget, and its circular buffers grow with it.
    * Stale KV past ``prompt_len`` from the previous chunk is harmless — decode attention masks
      everything beyond the current position (``forward_decode``'s ``add_mask``), and prefill
      overwrites 0..prompt_len-1 on every SETUP replay.

    All three traces are held simultaneously (they are only released in :meth:`close`), so the
    device needs a trace region big enough for all of them at once, not just the largest.
    """

    def __init__(
        self,
        tt,
        cond_wav,
        ref_wav_spk,
        text_len,
        max_seq,
        max_new_tokens,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
        min_new_tokens=0,
    ):
        t0 = time.perf_counter()
        self.tt = tt
        self.device = dev = tt.device
        self.N = int(max_new_tokens)
        gpt = tt.gpt
        voc = tt.decoder.decoder

        # ---- 1. ALLOCATE every persistent buffer of all three stages, BEFORE any capture. ----
        # tt-metal stops tracking a trace's memory the moment it is captured, so a buffer allocated
        # afterwards can be handed an address that trace uses as scratch — and is corrupted the
        # first time the trace runs (metal warns "Allocating device buffers is unsafe due to the
        # existence of an active trace"). With one trace at a time that is survivable; with three
        # live traces it is a device hang. Allocating up front keeps every buffer visible to the
        # allocator while each capture picks its scratch, so nothing overlaps.
        # Host inputs are pre-placed here for the second reason too: a host->device write inside a
        # capture is fatal. The reference audio is the same for every chunk; only the text ids
        # change, and they are rewritten IN PLACE into this buffer before each SETUP replay.
        self.wav_devs = [tt._wav_chunk_to_device(c) for c in chunk_wav(cond_wav)]
        self.text_dev = gpt.text_ids_to_device(torch.zeros(1, text_len, dtype=torch.long))
        self.ref_wav_spk = ref_wav_spk  # held so the buffer the SETUP trace reads outlives the caller
        gpt.alloc_static_kv(max_seq)
        prompt_len = 32 + text_len  # cond perceiver latents + text; asserted against prefill below
        self.decoder = TtTracedDecoder(
            gpt,
            prompt_len,
            self.N,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            min_new_tokens=min_new_tokens,
            capture=False,  # buffers only; warmup + capture are sequenced below
        )
        # The vocoder runs on a FIXED-length latent buffer (see the class docstring).
        self.voc_in = ttnn.from_torch(
            torch.zeros(1, self.N, HIDDEN_SIZE), layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.float32
        )
        # The session's ONE speaker embedding, in DRAM. ``_setup`` rewrites it in place (eagerly at
        # warmup, then on every SETUP replay) and the vocoder reads it; keeping a single object is
        # what makes the vocoder's fast cond-bias fold trace-legal — see step 2.
        self.g = ttnn.from_torch(
            torch.zeros(1, 1, COND_CHANNELS), layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.float32
        )

        def _setup():
            cl = tt._style_mean(self.wav_devs)  # [1, 32, 1024], averaged over the windows
            g = tt.decoder.speaker_embedding(ref_wav_spk)  # [1, 1, 512]
            # Park g in DRAM. Left where the speaker encoder puts it, this handful of KB sits in the
            # L1 region the vocoder wants for its circular buffers, and the VOCODER capture dies
            # with "circular buffers ... clash with L1 buffers" (measured: an L1 buffer at 1122560
            # against a CB region ending at 1123264 — 704 bytes of overlap). Same reason the
            # per-window style embeddings go to DRAM in ``_style_window``.
            g_dram = ttnn.to_memory_config(g, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(g)
            # ...then into the ONE persistent g (see ``self.g``). Unconditional, so the eager warmup
            # and the captured replay run the IDENTICAL op graph — a branch here would leave this
            # copy out of the program cache and the capture dies on "Cannot load new binaries during
            # trace capture".
            ttnn.copy(g_dram, self.g)
            ttnn.deallocate(g_dram)
            return gpt.prefill_on_device(self.text_dev, cl)

        # ---- 2. WARM UP (compile) all three stages eagerly, still before any capture. ----
        # The vocoder warmup folds against ``self.g`` and the vocoder capture reads it, while the
        # SETUP capture rewrites it IN PLACE — one object throughout.
        #
        # That single-object discipline is what lets the vocoder capture use the FAST cond-bias fold,
        # which is bit-identical to eager. The fold is keyed on ``(id(cond_bias), input signature)``
        # (xtts_conv.TtConv1d._folded_bias, via TtHifiganGenerator._cond's id(g) memo), so a fresh g
        # per stage would miss the cache and run the fold's ``from_device`` INSIDE the capture — fatal,
        # and it deadlocks teardown. This used to be worked around with ``cond_bias_trace_safe()``
        # around both regions, which is trace-legal but ~82us/pass slower and NOT bit-exact (it adds
        # post-conv in the stage's bf16, where the fold combines in fp32). Warming against the very
        # buffer the trace will read gets the fast path legally instead.
        #
        # The cached folded bias is a function of g's VALUES, so the warmup must see the real ones:
        # it does, because this eager ``_setup()`` actually computes g (a capture would only record
        # the ops, leaving the buffer undefined). Every later replay recomputes the same values into
        # the same buffer — the reference audio is fixed for the whole session — so it never goes stale.
        warm_prompt_len = _setup()  # also populates the mel-frontend index cache
        assert warm_prompt_len == prompt_len, f"prefill gave prompt_len {warm_prompt_len}, expected {prompt_len}"
        self.decoder.warmup()  # needs the KV cache seeded, which the prefill above just did
        _ = voc(ttnn.clone(self.voc_in), self.g)  # folds + caches the cond bias against self.g
        ttnn.synchronize_device(dev)

        # ---- 3. CAPTURE the three traces back to back. ----
        self.setup_tid = ttnn.begin_trace_capture(dev, cq_id=0)
        _setup()  # the seeded caches + self.g are the trace's persistent outputs
        ttnn.end_trace_capture(dev, self.setup_tid, cq_id=0)
        ttnn.synchronize_device(dev)
        self.decoder.capture()
        self.voc_tid = ttnn.begin_trace_capture(dev, cq_id=0)
        self.wav_dev = voc(ttnn.clone(self.voc_in), self.g)
        ttnn.end_trace_capture(dev, self.voc_tid, cq_id=0)
        ttnn.synchronize_device(dev)
        # Samples per generator step, read off the capture rather than assumed.
        self.upsample = self.wav_dev.shape[-2] // _interp_len(self.N)
        self.compile_s = time.perf_counter() - t0

    def _samples_for(self, frames):
        """Exact waveform length for ``frames`` latent frames. NOT ``frames * (total / N)``: the two
        linear interpolates each floor their output length, so the mapping has a sub-frame remainder
        (192 frames -> 213760 samples is 1113.33 per frame, and 149 frames is 165888, not 165837)."""
        return _interp_len(frames) * self.upsample

    def run(self, text_ids):
        """Synthesise one text off the captured traces. ``text_ids`` is ``[1, text_len]`` (the
        padded length the session was built for). Returns ``(wav [T] torch float, codes, perf)``,
        the waveform already trimmed to the codes actually generated."""
        dev = self.device
        assert text_ids.shape[1] == self.text_dev.shape[1], (
            f"session captured for {self.text_dev.shape[1]} text tokens, got {text_ids.shape[1]} — "
            "every chunk must be padded to the same length"
        )
        # New text into the SAME buffer the SETUP trace reads, then re-seed the KV cache with it.
        ttnn.copy(self.tt.gpt.text_ids_to_device(text_ids), self.text_dev)
        t0 = time.perf_counter()
        ttnn.execute_trace(dev, self.setup_tid, blocking=True)
        setup_replay_s = time.perf_counter() - t0

        self.decoder.reset(redraw_noise=True)
        codes, lat_host, decode_replay_s = self.decoder.run()

        # Zero-pad this chunk's latents up to the captured vocoder length and replay the vocoder.
        frames = lat_host.shape[1]
        padded = torch.zeros(1, self.N, HIDDEN_SIZE, dtype=torch.float32)
        padded[:, :frames, :] = lat_host.float()
        ttnn.copy(
            ttnn.from_torch(padded, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.float32),
            self.voc_in,
        )
        t0 = time.perf_counter()
        ttnn.execute_trace(dev, self.voc_tid, blocking=True)
        vocoder_replay_s = time.perf_counter() - t0

        wav = ttnn.to_torch(self.wav_dev).float().reshape(-1)[: self._samples_for(frames)]
        replay_s = setup_replay_s + decode_replay_s + vocoder_replay_s
        perf = {
            "replay_s": replay_s,
            "compile_s": 0.0,  # paid once, in __init__ (self.compile_s)
            "setup_replay_s": setup_replay_s,
            "decode_replay_s": decode_replay_s,
            "vocoder_replay_s": vocoder_replay_s,
        }
        return wav, codes, perf

    def close(self):
        for tid in (self.setup_tid, self.voc_tid):
            ttnn.release_trace(self.device, tid)
        self.decoder.release()
