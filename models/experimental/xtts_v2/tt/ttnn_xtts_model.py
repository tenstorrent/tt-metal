# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Top-level XTTS-v2 text-to-speech pipeline on TT (single class, serving-shaped API).

Ties the validated blocks into one persistent, multi-request object (mirroring the
`models/demos/z_image_turbo` class surface: __init__ builds everything, `warmup()` compiles
+ captures, then the request-path methods are fast and reusable):

    XttsV2(mesh_device=None, ckpt_path=None)   # opens a (1,1) mesh if not given one
      .warmup()                                # compile all programs + capture the traces
      .compute_voice(ref_audio, sr) -> Voice   # reference clip -> conditioning latents (Blocks 1+2)
      .generate(text, voice, ...)  -> [1,1,N]  # text -> 24 kHz waveform (Block 3 GPT + Block 4 vocoder)
      .close()

Front-end (tokenizer, mel/STFT, prompt assembly) is the coqui-free `frontend.py` — pure
torch on host, validated bit-exact against coqui captures. Sampling (mel_head + repetition
penalty + top-k/top-p/multinomial) is on host with a VECTORIZED repetition penalty — a
per-token Python loop over the `seen` set is a known quadratic perf trap (grows linearly
with `seen` per step; measured ~240x slower by step 586).

Device layout notes carried over from the block bringups:
  * `l1_small_size=65536` is REQUIRED: the fp32 HiFi-GAN convs' halo config OOMs in the
    default/32K L1_SMALL (BUG-2/3 in the block docs).
  * `trace_region_size=160_000_000` holds the traced 30-layer GPT decode step plus one vocoder
    trace per VOC_BUCKETS shape; the traces scale with their frame count.
  * Blocks 1 (cond+perceiver) and 4 (vocoder) run fp32 activations (PCC); Blocks 2/3 bf16.

Trace lifecycle — WHY the decode trace is captured once (option (a)):
  Capturing the decode trace after a single request's prefill serves that one request; for
  serving, consecutive `generate()` calls must all replay a valid trace. The known
  hazard is that device buffers allocated AFTER capture can land on the trace's baked
  intermediate-buffer addresses; anything that must PERSIST across an `execute_trace`
  would then be scribbled over. This class therefore allocates every persistent device
  tensor (weights, KV caches, the trace's stable in/out slots, `_pos`) BEFORE capture, and
  everything allocated afterwards (per-request prefill activations, Block-1/2/4
  activations, per-step host->device embeddings) is transient: dead before the next
  trace replay reads or writes anything. `warmup()` captures ONE decode trace at the
  model-cap max_seq (32 cond + 404 text + 1 START + 605 audio = 1042, rounded up internally
  per BUG-1), and each request then just does reset_caches + prefill + traced steps. The
  vocoder traces (one per VOC_BUCKETS shape) obey the same rule plus one of their own: their
  persistent state — stable slots AND ttnn's cached conv weights — must exist before ANY capture,
  or a replay lands on another trace's buffers. Hence the _alloc_vocoder / _capture_vocoder split
  around the decode capture.
  Verified empirically during bringup: two consecutive
  generates produce sane audio, and a teacher-forced golden replay through the SAME
  captured trace AFTER those generates still matches `golden/gpt/latents.pt` at
  PCC > 0.999 — i.e. prefill/vocoder allocations and post-capture program compiles did
  not corrupt the trace. Option (b) (re-capture per request) was not needed.
"""

import os
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from loguru import logger

import ttnn

from models.experimental.xtts_v2.frontend import (
    PromptTables,
    XttsTokenizer,
    assemble_prompt,
    conditioning_mels,
    speaker_logmel,
)
from models.experimental.xtts_v2.reference.xtts_gpt_ref import (
    START_AUDIO_TOKEN,
    STOP_AUDIO_TOKEN,
    load_full_state,
    load_gen_head,
    resolve_ckpt,
)
from models.experimental.xtts_v2.tt.ttnn_xtts_cond import (
    LATENTS,
    TTNNConditioningEncoder,
    TTNNPerceiver,
    preprocess_encoder_parameters,
    preprocess_perceiver_parameters,
)
from models.experimental.xtts_v2.tt.ttnn_xtts_gpt import preprocess_gpt_parameters
from models.experimental.xtts_v2.tt.ttnn_xtts_gpt_decode import TTNNGPTTracedDecoder
from models.experimental.xtts_v2.tt.ttnn_xtts_hifigan import (
    TTNNHifiganGenerator,
    preprocess_hifigan_parameters,
)
from models.experimental.xtts_v2.tt.ttnn_xtts_speaker import (
    TTNNSpeakerEncoder,
    preprocess_speaker_parameters,
)

# coqui Xtts.inference sampling defaults (do_sample=True path)
TEMPERATURE, TOP_K, TOP_P = 0.75, 50, 0.85
REPETITION_PENALTY = 10.0
# model caps: mel_pos is [608,1024] -> up to 605 audio codes; text_pos is [404,1024]
GPT_MAX_AUDIO = 605
MAX_PREFIX = LATENTS + 404  # 32 cond latents + start/text/stop (<= 404 embedded text rows)
# HifiDecoder.forward constants: gpt code stride 1024 samples @ 22.05 kHz, vocoder hop 256,
# and the 22050 -> 24000 output resample — both applied as host linear interpolates on z.
AR_COMP, HOP, ISR, OSR = 1024, 256, 22050, 24000
FADE_MS = 5  # output fade; a request cut mid-word would otherwise end on a step
# The model sometimes keeps emitting codes after the sentence instead of STOP, and they vocode to
# silence. A natural tail is a fraction of RUN_ON_S; a run-on is several times it, so trimming back
# to TAIL_MS only touches the pathological case.
RUN_ON_S, TAIL_MS = 0.5, 150
OUTPUT_SR = OSR
# The vocoder runs at one of a FIXED set of frame counts (the cap: 605 codes -> 2420 frames
# @22.05kHz -> 2634 @24kHz); z is padded up to its bucket (_voc_pad) and the waveform trimmed.
# WHY a fixed set: ttnn conv/conv_transpose sliding-window ("halo") config tensors are pinned
# in L1_SMALL for the lifetime of their program-cache entry, so every DISTINCT input length
# permanently consumes L1_SMALL — with per-utterance lengths the second generate() OOMs
# ("Not enough space to allocate ... L1_SMALL buffer"). Buckets bound the compiled shapes and
# keep short utterances off the cap, whose cost per frame is also the worst.
VOC_L = (GPT_MAX_AUDIO * AR_COMP // HOP) * OSR // ISR  # 2634
# A request pays for its whole bucket, so the gap to the bucket below is its worst-case waste.
# Steps alternate 1.5x / 1.33x: even spacing leaves the bottom of the range doubling, which is
# exactly where the vocoder is the largest share of a request.
# 329, 494, 658, 988, 1317, 1976, 2634.
VOC_BUCKETS = tuple(round(VOC_L * k / 16) for k in (2, 3, 4, 6, 8, 12, 16))


@dataclass
class Voice:
    """A cloned voice: everything `generate()` needs from the reference clip."""

    gpt_cond_latent: torch.Tensor  # [1, 32, 1024] — Block 1 (cond encoder + Perceiver) output
    speaker_embedding: torch.Tensor  # [1, 512, 1]  — Block 2 (ResNet d-vector) output


def _voc_bucket(L):
    """Smallest VOC_BUCKETS frame count that fits L."""
    assert L <= VOC_L, f"vocoder input {L} frames exceeds the fixed cap {VOC_L}"
    return next(b for b in VOC_BUCKETS if b >= L)


def _voc_input(gpt_latents):
    """GPT latents [1,T,1024] -> vocoder input z [1,1024,L]: code stride -> hop, then 22.05->24 kHz."""
    z = F.interpolate(gpt_latents.transpose(1, 2), scale_factor=AR_COMP / HOP, mode="linear")
    return F.interpolate(z, scale_factor=OSR / ISR, mode="linear")


def _voc_pad(z, Lb):
    """Pad z [1,1024,L] up to Lb frames by HOLDING the last frame.

    conv_pre has a bias, so zero frames are a step change at the boundary, and the generator answers
    with a burst that reaches back into the audio that is kept. Repeating the last frame keeps the
    boundary smooth; the padding itself is discarded either way."""
    return torch.cat([z, z[..., -1:].expand(-1, -1, Lb - z.shape[-1])], -1)


def _trailing_silence(wav):
    """Seconds of near-silence at the end of a waveform.

    RMS over short windows rather than raw samples: a lone nonzero sample is the noise floor, not
    speech. The loudness bar is relative to the clip's own peak as well as absolute, because
    speakers differ in level and a fixed bar alone would call a quiet speaker's tail silence."""
    n = int(OUTPUT_SR * 0.02)
    flat = wav.reshape(-1)
    if flat.numel() < n:
        return 0.0  # shorter than one window; the empty-audio contract is handled by the caller
    frames = flat[: flat.numel() // n * n].reshape(-1, n)
    rms = frames.pow(2).mean(1).sqrt()
    loud = (rms > max(0.01, 0.02 * rms.max().item())).nonzero()
    if not len(loud):
        return len(frames) * n / OUTPUT_SR  # silent throughout, which is a fault worth seeing
    return (len(frames) - 1 - loud[-1].item()) * n / OUTPUT_SR


def _trim_run_on(wav):
    """Cut a run-on tail back to a natural length -> (wav, seconds removed)."""
    silence = _trailing_silence(wav)
    if silence <= RUN_ON_S:
        return wav, 0.0
    keep = wav.shape[-1] - int((silence - TAIL_MS / 1000) * OUTPUT_SR)
    return wav[..., :keep], (wav.shape[-1] - keep) / OUTPUT_SR


def _fade_out(wav, ms=FADE_MS):
    """Ramp the last `ms` of a waveform to zero, so a request never ends on a step.

    One cut short by max_new_tokens ends mid-word at full amplitude, which is a click; one that
    ends by itself is already near silence there and this is inaudible."""
    n = min(int(ms * OUTPUT_SR / 1000), wav.shape[-1])
    if n == 0:
        return wav
    out = wav.clone()
    out[..., -n:] *= torch.linspace(1, 0, n) ** 0.5  # equal-power, so the level falls evenly
    return out


def _sample_token(latent, seen, gen, mh_w, mh_b, penalty=REPETITION_PENALTY):
    """coqui's decode strategy on one latent [1,1,1024]: mel_head -> repetition penalty ->
    temperature -> top-k -> top-p -> multinomial draw with the request's own RNG.

    The repetition penalty is VECTORIZED (one gather/scatter over the `seen` set), because a
    per-token Python loop is O(len(seen)) per step and so quadratic over an utterance — enough to
    dominate the device time. Indexing once is bit-identical: each index is touched exactly once."""
    logits = (latent @ mh_w.t() + mh_b)[0, 0].clone().float()  # [1026]
    if seen:
        idx = torch.tensor(sorted(seen))
        v = logits[idx]
        logits[idx] = torch.where(v > 0, v / penalty, v * penalty)
    logits = logits / TEMPERATURE
    if TOP_K and TOP_K < logits.numel():
        kth = torch.topk(logits, TOP_K).values[-1]
        logits[logits < kth] = float("-inf")
    if TOP_P < 1.0:
        sl, si = torch.sort(logits, descending=True)
        drop = torch.softmax(sl, dim=-1).cumsum(dim=-1) > TOP_P
        drop[1:] = drop[:-1].clone()
        drop[0] = False  # always keep the top-1
        logits[si[drop]] = float("-inf")
    return int(torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=gen))


class XttsV2:
    """XTTS-v2 text-to-speech on a Tenstorrent device (see module docstring).

    Usage:
        tts = XttsV2()                       # or XttsV2(mesh_device=..., ckpt_path=...)
        tts.warmup()                         # once; compiles + captures the traces
        voice = tts.compute_voice(ref, sr)   # once per speaker
        wav = tts.generate("Hello!", voice)  # [1,1,N] float @ 24 kHz; repeatable
        tts.close()

    Trace lifecycle: option (a) — one trace captured at warmup at the model-cap max_seq;
    per request only reset_caches + prefill + traced steps. All persistent device state is
    allocated before capture; see the module docstring for the hazard analysis and the
    empirical verification that consecutive generates keep the trace intact."""

    def __init__(self, mesh_device=None, ckpt_path=None):
        t0 = time.time()
        self.ckpt_path = resolve_ckpt(ckpt_path)

        if mesh_device is None:
            # l1_small >= 64K is REQUIRED for the fp32 vocoder convs (BUG-2/3). We open with
            # 256K because conv halo-config tensors are PINNED in L1_SMALL per compiled shape
            # (see VOC_BUCKETS): the vocoder is held to 4 shapes, and compute_voice legitimately
            # sees a new speaker-encoder conv shape per distinct reference-clip length, and
            # each pins a few KB. The trace region holds the decode step + one vocoder trace
            # per bucket.
            self.mesh_device = ttnn.open_mesh_device(
                ttnn.MeshShape(1, 1), l1_small_size=262144, trace_region_size=160_000_000
            )
            self._owns_device = True
        else:
            self.mesh_device = mesh_device
            self._owns_device = False
        self.mesh_device.enable_program_cache()

        # --- host front-end (coqui-free, validated bit-exact vs coqui) ---
        vocab = os.path.join(os.path.dirname(self.ckpt_path), "vocab.json")
        self.tokenizer = XttsTokenizer(vocab)
        self.tables = PromptTables(self.ckpt_path)
        # generation head stays on host: mel_head (latent->logits), mel_emb+mel_pos (embed codes)
        self.heads = load_gen_head(self.ckpt_path)

        # --- device blocks (weights preprocessed once; load_full_state is lru_cached) ---
        dev = self.mesh_device
        self._spk_params = preprocess_speaker_parameters(dev, self.ckpt_path)
        self.speaker_encoder = TTNNSpeakerEncoder(dev, self._spk_params)
        # Block 1 runs fp32 (PCC); the encoder instance is per-chunk (it bakes T-dependent
        # masks), so keep only the preprocessed weights here.
        self._enc_params = preprocess_encoder_parameters(dev, self.ckpt_path, dtype=ttnn.float32)
        self._perc_params = preprocess_perceiver_parameters(dev, self.ckpt_path, dtype=ttnn.float32)
        self.vocoder = TTNNHifiganGenerator(dev, preprocess_hifigan_parameters(dev, self.ckpt_path))
        # The traced decoder is sized to the MODEL cap so one captured trace serves every
        # request: prefix (<=436) + START_AUDIO + up to 605 codes = 1042; the decoder rounds
        # the KV cache up to an even tile count internally (BUG-1).
        self.decoder = TTNNGPTTracedDecoder(
            dev,
            preprocess_gpt_parameters(dev, self.ckpt_path, dtype=ttnn.bfloat16),
            max_seq=MAX_PREFIX + 1 + GPT_MAX_AUDIO,
        )
        # Everything above has copied what it needs out of the lru_cached checkpoint dict,
        # and the request path (warmup/compute_voice/generate) never reloads it — so drop
        # the ~1.9 GB pin now (single-instance serving path). A multi-instance
        # data-parallel build that calls the preprocess_* functions once per chip
        # legitimately wants the cache DURING its build loop; it is unaffected because
        # clearing happens per-XttsV2 instance, after that instance's own build.
        load_full_state.cache_clear()
        self._warm = False
        self._voc_slots = {}  # bucket length -> (z_in, g_in), allocated before any capture
        self._voc_traces = {}  # bucket length -> (trace_id, z_in, g_in, out), captured at warmup
        self.last_timings = {}
        self.last_generation = {}  # codes + latents of the last request; see generate()
        logger.info(f"[XttsV2] built (ckpt={self.ckpt_path}) in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------ warmup
    def warmup(self):
        """Compile every program the request path uses, then capture the traces.

        Order matters (same discipline as z_image_turbo): programs compiled after capture
        risk landing where the trace keeps intermediates, so compile Blocks 1/2 (a full 6 s
        conditioning chunk — the shape every long reference hits), the vocoder's slots and
        programs, one prefill, and the decode step BEFORE `capture()`. The vocoder TRACES are
        captured after it — see _alloc_vocoder / _capture_vocoder, which own that split.
        (Per-request prefill at a NEW prompt length still compiles a few programs post-capture
        — covered by the teacher-forced trace check in the bringup validation.)"""
        t0 = time.time()
        # 1) Blocks 1+2 on a dummy full-length (6 s) reference clip.
        g = torch.Generator().manual_seed(0)
        dummy = torch.randn(6 * 22050, generator=g) * 0.1
        voice = self.compute_voice(dummy, 22050)
        logger.info(f"[XttsV2] warmup: Blocks 1+2 compiled in {time.time() - t0:.1f}s")

        # 2) Vocoder slots + programs at every bucket shape (see _alloc_vocoder: before capture).
        t1 = time.time()
        self._alloc_vocoder(voice.speaker_embedding)
        logger.info(f"[XttsV2] warmup: vocoder compiled at {len(VOC_BUCKETS)} buckets in {time.time() - t1:.1f}s")

        # 3) One eager prefill (compiles fill_cache/SDPA/etc.), then capture the step trace.
        t1 = time.time()
        prefix = assemble_prompt(
            self.tokenizer.encode("Warm up the decoder.", "en"), voice.gpt_cond_latent, self.tables
        )
        self.decoder.reset_caches()
        self.decoder.prefill(prefix.contiguous())
        self.decoder.capture()
        self._warm = True
        logger.info(
            f"[XttsV2] warmup: GPT prefill + trace captured in {time.time() - t1:.1f}s (max_seq={self.decoder.max_seq})"
        )

        # 4) Vocoder traces, from the slots step 2 allocated (see _capture_vocoder: after capture).
        t1 = time.time()
        self._capture_vocoder()
        logger.info(f"[XttsV2] warmup: vocoder traced at {len(VOC_BUCKETS)} buckets in {time.time() - t1:.1f}s")

        # 5) Tiny end-to-end generate: replays the fresh trace over the whole request path.
        t1 = time.time()
        self.generate("Warm up the pipeline.", voice, seed=0, max_new_tokens=24)
        logger.info(f"[XttsV2] warmup: tiny generate in {time.time() - t1:.1f}s")
        logger.info(f"[XttsV2] warmup total {time.time() - t0:.1f}s")

    # ------------------------------------------------------------ voice cloning
    def compute_voice(self, ref_audio, sr) -> Voice:
        """Reference clip (torch [N] / [1,N] float waveform, sample rate sr) -> Voice.

        Block 2 (speaker d-vector) runs once on the whole clip's logmel. Block 1 runs once
        PER <=6 s CHUNK and the 32 conditioning latents are MEANED across chunks — exactly
        coqui's get_gpt_cond_latents (one style embedding per chunk, averaged), which is what
        lifts the old "reference must be <=6 s" caveat (GAP-3): long references now
        contribute all their chunks instead of only the first."""
        t0 = time.time()
        dev = self.mesh_device
        audio = torch.as_tensor(ref_audio).float().reshape(1, -1)

        # Block 2: ResNet speaker encoder on the 16 kHz logmel.
        logmel = speaker_logmel(audio, sr)  # [1,64,T]
        logmel_tt = ttnn.from_torch(logmel, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        emb = self.speaker_encoder(logmel_tt)
        if isinstance(emb, tuple):
            emb = emb[0]
        spk = ttnn.to_torch(emb).to(torch.float32).reshape(1, 512, 1)

        # Block 1: conditioning encoder + Perceiver, once per 6 s chunk, latents averaged.
        latents = []
        for mel in conditioning_mels(audio, sr, self.tables.mel_stats):  # each [1,80,T]
            T = mel.shape[2]
            S = ((T + 31) // 32) * 32
            mel_f = F.pad(mel.permute(0, 2, 1).contiguous(), (0, 0, 0, S - T))  # [1,S,80]
            enc = TTNNConditioningEncoder(dev, self._enc_params, t_real=T, s_pad=S)
            perc = TTNNPerceiver(dev, self._perc_params)
            km = torch.zeros(1, 1, 1, LATENTS + S)
            km[:, :, :, LATENTS + T :] = -1e9
            km_tt = ttnn.from_torch(km, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
            frames = enc(ttnn.from_torch(mel_f, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev))
            latents.append(ttnn.to_torch(perc(frames, km_tt)).to(torch.float32))  # [1,32,1024]
        cond = torch.stack(latents, dim=0).mean(dim=0)

        self.last_timings["compute_voice_s"] = time.time() - t0
        return Voice(gpt_cond_latent=cond, speaker_embedding=spk)

    # --------------------------------------------------------------- generation
    def _alloc_vocoder(self, speaker_embedding):
        """Allocate the vocoder's stable in-slots and compile every VOC_BUCKETS shape.

        Runs BEFORE the decode capture: this eager pass also prepares the conv weights, which the
        generator caches and keeps. Persistent buffers allocated after a capture can land on that
        trace's baked addresses, and its replays then overwrite them."""
        dev = self.mesh_device
        for Lb in VOC_BUCKETS:
            z_in = ttnn.from_torch(
                torch.zeros(1, 1, Lb, 1024), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev
            )
            g_in = ttnn.from_torch(
                speaker_embedding.reshape(1, 512), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev
            )
            self.vocoder(z_in, g_in)  # a capture cannot compile, so every shape runs eagerly first
            self._voc_slots[Lb] = (z_in, g_in)
        ttnn.synchronize_device(dev)

    def _capture_vocoder(self):
        """Capture one replayable trace per bucket from the slots _alloc_vocoder made.

        Runs AFTER the decode capture — capturing first leaves it emitting NaN logits. Only each
        trace's own output is allocated here, and its replay rewrites that before anything reads
        it."""
        dev = self.mesh_device
        for Lb, (z_in, g_in) in self._voc_slots.items():
            trace_id = ttnn.begin_trace_capture(dev, cq_id=0)
            out = self.vocoder(z_in, g_in)
            ttnn.end_trace_capture(dev, trace_id, cq_id=0)
            ttnn.synchronize_device(dev)
            self._voc_traces[Lb] = (trace_id, z_in, g_in, out)

    def _vocode(self, z, speaker_embedding):
        """z torch [1,1024,L] + d-vector -> waveform torch [1,1,L*HOP], padded to z's bucket."""
        dev = self.mesh_device
        L, Lb = z.shape[-1], _voc_bucket(z.shape[-1])
        z_nhwc = _voc_pad(z, Lb).permute(0, 2, 1).reshape(1, 1, Lb, 1024)
        g_2d = speaker_embedding.reshape(1, 512)
        trace = self._voc_traces.get(Lb)
        if trace is None:  # not warmed up (or a shape warmup did not cover): run eager
            z_tt = ttnn.from_torch(z_nhwc, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
            g_tt = ttnn.from_torch(g_2d, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
            out = self.vocoder(z_tt, g_tt)
        else:
            trace_id, z_in, g_in, out = trace
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(z_nhwc, dtype=ttnn.float32), z_in)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(g_2d, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), g_in)
            ttnn.execute_trace(dev, trace_id, cq_id=0, blocking=False)
        wav = ttnn.to_torch(out).to(torch.float32).reshape(1, 1, -1)
        return wav[:, :, : L * HOP]  # trim the zero-padded tail

    def generate(self, text, voice: Voice, language="en", seed=None, max_new_tokens=GPT_MAX_AUDIO):
        """text + Voice -> waveform torch [1,1,N] float @ 24 kHz.

        GPT: eager batched prefill of [cond, start_text, text, stop_text], then the traced
        single-token decode with host-side coqui sampling (temperature 0.75, top-k 50,
        top-p 0.85, repetition penalty 10.0, per-request torch.Generator). Vocoder: the two
        host interpolates (code stride -> hop, 22.05 -> 24 kHz) then HiFi-GAN on device.

        Empty-audio contract: if the very FIRST sampled code is STOP (a legitimate, rare,
        seed-dependent outcome — coqui's HF generate returns zero codes for it), or
        max_new_tokens is 0, there is nothing to vocode and this returns an EMPTY waveform
        torch.zeros(1, 1, 0) rather than raising; callers should check wav.shape[-1].

        Publishes `last_generation` = {codes, latents}: the sampled codes and the latents they
        were predicted from, so a gate can ask the CPU reference what those codes imply."""
        if not self._warm:
            raise RuntimeError("call warmup() before generate()")
        dev = self.mesh_device
        dec = self.decoder
        heads = self.heads
        mel_emb, mel_pos = heads["mel_emb"], heads["mel_pos"]
        mh_w, mh_b = heads["mel_head_w"], heads["mel_head_b"]
        gen = torch.Generator()
        gen.manual_seed(seed if seed is not None else torch.seed() % (2**63))
        max_new = min(int(max_new_tokens), GPT_MAX_AUDIO)

        # --- prompt prefix + prefill ---
        t0 = time.time()
        ids = self.tokenizer.encode(text, language)
        prefix = assemble_prompt(ids, voice.gpt_cond_latent, self.tables)  # [1,P,1024]
        P = prefix.shape[1]
        # prefill right-pads P to a tile count it can seed correctly, which also buckets its
        # program variants. Decode keeps the TRUE P and sdpa_decode only attends 0..cur_pos, so
        # the pad rows' K/V are never read (see TTNNGPTTracedDecoder.prefill).
        dec.reset_caches()
        dec.prefill(prefix.contiguous())
        t_prefill = time.time() - t0

        def step(emb_1x1, pos):
            emb_dev = ttnn.from_torch(
                emb_1x1.contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                mesh_mapper=dec.mesh_mapper,
            )
            return ttnn.to_torch(dec.step_device(emb_dev, pos)).float()  # [1,1,1024]

        # --- traced decode + host sampling ---
        # Vocoder-frame convention (VERIFIED vs coqui 2026-08-11): coqui's return_latent
        # forward feeds the vocoder the transformer outputs at positions [START, code_0, ...,
        # code_{N-2}] — the latent each code was PREDICTED from, starting with the
        # START-position output. (Sliding-window match against coqui's captured latents:
        # cosine 1.000 at the START offset, 0.73 one position later.) So an accepted code's
        # frame is the `last` latent it was sampled FROM; a code's own output only becomes a
        # frame if the NEXT code is accepted. The earlier off-by-one (collecting each code's
        # own output) dropped the true first frame and audibly garbled the first word.
        # STOP as the FIRST sample means zero codes -> empty-audio contract (docstring);
        # the stop token is never fed back.
        t0 = time.time()
        last = step((mel_emb[START_AUDIO_TOKEN] + mel_pos[0]).view(1, 1, -1), P)
        # HF-parity detail: coqui's generate() runs the repetition penalty over its input_ids,
        # which is a FAKE prefix of fill_value=1 tokens plus START — so coqui permanently
        # suppresses code 1 (and START) from the first sample onward. Replicate it.
        seen = {1, START_AUDIO_TOKEN}
        # Two reasons to stop, reported apart: STOP finished the sentence, the budget cut it off.
        # STOP is checked first, so a model that would have stopped anyway is not called truncated.
        codes, vlat, truncated = [], [], False
        while True:
            nxt = _sample_token(last, seen, gen, mh_w, mh_b)
            if nxt == STOP_AUDIO_TOKEN:
                break
            if len(codes) >= max_new:
                truncated = True
                break
            codes.append(nxt)
            vlat.append(last)  # `last` predicted nxt -> it is nxt's vocoder frame
            seen.add(nxt)
            # feed the accepted code; its output predicts (and may become the frame of) the next
            last = step((mel_emb[nxt] + mel_pos[len(codes)]).view(1, 1, -1), P + len(codes))
        t_decode = time.time() - t0
        if not vlat:  # zero codes (first sample was STOP) or max_new_tokens == 0
            self.last_timings.update(
                {
                    "prefill_s": t_prefill,
                    "prefix_tokens": P,
                    "decode_s": t_decode,
                    "decode_ms_per_token": 1000.0 * t_decode,  # the lone START step
                    "codes": 0,
                    "truncated": truncated,
                    "run_on_s": 0.0,
                    "vocoder_s": 0.0,
                    "wav_samples": 0,
                }
            )
            self.last_generation = {"codes": codes, "latents": None}
            return torch.zeros(1, 1, 0)
        gpt_latents = torch.cat(vlat, dim=1)  # [1,T,1024]
        self.last_generation = {"codes": codes, "latents": gpt_latents}

        # --- vocoder: two host interpolates (HifiDecoder.forward) + HiFi-GAN on device. ---
        t0 = time.time()
        wav, run_on = _trim_run_on(self._vocode(_voc_input(gpt_latents), voice.speaker_embedding))
        wav = _fade_out(wav)
        t_voc = time.time() - t0

        self.last_timings.update(
            {
                "prefill_s": t_prefill,
                "prefix_tokens": P,
                "decode_s": t_decode,
                "decode_ms_per_token": 1000.0 * t_decode / max(len(vlat) + 1, 1),
                "codes": len(codes),
                "truncated": truncated,
                "run_on_s": run_on,
                "vocoder_s": t_voc,
                "wav_samples": wav.shape[-1],
            }
        )
        return wav

    # -------------------------------------------------------------------- close
    def close(self):
        """Release the traces and close the device (only if this instance opened it)."""
        for trace_id, *_ in self._voc_traces.values():
            try:
                ttnn.release_trace(self.mesh_device, trace_id)
            except Exception:
                pass  # older ttnn builds release traces on device close
        self._voc_traces.clear()
        self._voc_slots.clear()  # frees the stable in-slots those traces replayed from
        if getattr(self.decoder, "trace_id", None) is not None:
            try:
                ttnn.release_trace(self.mesh_device, self.decoder.trace_id)
            except Exception:
                pass  # older ttnn builds release traces on device close
            self.decoder.trace_id = None
        if self._owns_device:
            ttnn.close_mesh_device(self.mesh_device)
            self.mesh_device = None
