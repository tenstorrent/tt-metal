# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""XTTS-v2 **reference** demo: text + reference audio -> spoken WAV, entirely on CPU.

The host-only twin of ``xtts_demo.py``. It runs the same pipeline through the
pure-PyTorch reference modules under ``models/experimental/xtts/reference/`` —
no ``ttnn``, no device is opened, nothing here touches Tenstorrent hardware:

    ref audio -> 80-mel -> conditioning encoder + perceiver -> cond latents [1, 32, 1024]
    text      -> BPE ids -> [START]/[STOP] wrapped
    (cond latents, text ids) -> GPT autoregressive decode -> codes + latents [1, T, 1024]
    ref audio -> 64-mel @ 16 kHz -> speaker encoder -> g [1, 512, 1]
    (latents, g) -> linear upsample + HiFi-GAN -> waveform @ 24 kHz

Use it as the ground truth to A/B the device demo against (same text, same
reference audio, same sampling knobs), or to hear XTTS-v2 on a machine with no
accelerator attached.

Decoding uses a KV cache, which is a pure speed change: the cached hidden states
are the same numbers the reference's full-recompute loop
(:func:`reference.xtts_gpt_generate.greedy_generate`) produces, to float error.
Pass ``--no-kv-cache`` to run the naive recompute loop instead (much slower;
useful to prove the two agree).

Sampling mirrors ``tt/xtts_sampler.py`` — repetition penalty, then temperature,
then top-k, then nucleus over the top-k window — so a device run and a CPU run
with the same seed are sampling from the same shaped distribution (they still
diverge token-wise: the two RNGs differ).

Usage:
    source python_env/bin/activate
    export PYTHONPATH=$(pwd)
    python models/experimental/xtts/demo/xtts_reference_demo.py \
        --text "Hello from Tenstorrent." --ref-audio LJ025-0076.wav

    # greedy (deterministic — the correctness anchor the ttnn port is checked against):
    python models/experimental/xtts/demo/xtts_reference_demo.py --temperature 0
"""

import argparse
import math
import os
import re
import time

import numpy as np
import torch
from loguru import logger
from transformers.cache_utils import DynamicCache

from models.experimental.xtts.config import (
    COQUI_CLIP_RE,
    DEMO,
    REFERENCE_DEMO,
    SENTENCE_FINAL_PUNCT_RE,
    SENTENCE_SPLIT_RE,
)
from models.experimental.xtts.reference.xtts_conditioning import (
    GPT_COND_CHUNK_SEC,
    GPT_COND_LEN_SEC,
    MEL_SR,
    chunk_wav,
    load_coqui_test_audio,
    load_reference_audio,
    wav_to_mel,
)
from models.experimental.xtts.reference.xtts_gpt_block import load_xtts_state_dict
from models.experimental.xtts.reference.xtts_gpt_generate import START_AUDIO_TOKEN, STOP_AUDIO_TOKEN, wrap_text_ids
from models.experimental.xtts.reference.xtts_hifi_decoder import OUTPUT_SAMPLE_RATE
from models.experimental.xtts.reference.xtts_inference import XttsReference
from models.experimental.xtts.reference.xtts_mel import SAMPLE_RATE as SPK_SR
from models.experimental.xtts.reference.xtts_text_embedding import preprocess_text

_COQUI_CLIP_RE = re.compile(COQUI_CLIP_RE)  # coqui-ai/TTS tests/data/ljspeech/wavs

# Single-pass budgets and every other default come from config.REFERENCE_DEMO.
MAX_TEXT_IDS = REFERENCE_DEMO.max_text_ids
MAX_PASS_CODES = REFERENCE_DEMO.max_pass_codes
CODES_PER_ID = REFERENCE_DEMO.codes_per_id


def _load_audio_22k(ref_audio, max_seconds):
    """Load reference audio as [1, samples] at 22.05 kHz from path or Coqui clips."""
    if os.path.exists(ref_audio):
        import soundfile as sf
        from scipy.signal import resample_poly

        audio, sr = sf.read(ref_audio, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != MEL_SR:
            g = math.gcd(MEL_SR, sr)
            audio = resample_poly(audio, MEL_SR // g, sr // g)
        audio = audio[: MEL_SR * max_seconds]
        return torch.from_numpy(audio.astype("float32")).unsqueeze(0)
    clips = ref_audio.split("+")
    if all(_COQUI_CLIP_RE.match(c) for c in clips):
        return load_coqui_test_audio(samples=clips, max_seconds=max_seconds)
    return load_reference_audio(sample=ref_audio, max_seconds=max_seconds)


def _split_into_chunks(text, lang):
    """Split text into sentence groups that fit one CPU synthesis pass."""

    def ids_of(t):
        """Count wrapped text token ids for a string."""
        return preprocess_text(re.sub(SENTENCE_FINAL_PUNCT_RE, "", t), lang=lang).shape[-1]

    whole = text.strip()
    if ids_of(whole) <= MAX_TEXT_IDS and ids_of(whole) * CODES_PER_ID <= MAX_PASS_CODES:
        return [whole]

    parts = [p.strip() for p in re.split(SENTENCE_SPLIT_RE, whole) if p.strip()]
    for p in parts:
        if ids_of(p) > MAX_TEXT_IDS or ids_of(p) * CODES_PER_ID > MAX_PASS_CODES:
            logger.warning(
                f"sentence is {len(p.split())} words (~{int(ids_of(p) * CODES_PER_ID)} audio codes) — over the "
                f"single-pass budget ({MAX_PASS_CODES} codes / {MAX_TEXT_IDS} text ids). A sentence is never "
                f"split, so this pass will be cut short. Shorten it or add punctuation."
            )
    chunks, cur = [], []
    for part in parts:
        cand = " ".join(cur + [part])
        n = ids_of(cand)
        if cur and (n > MAX_TEXT_IDS or n * CODES_PER_ID > MAX_PASS_CODES):
            chunks.append(" ".join(cur))
            cur = [part]
        else:
            cur.append(part)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _postprocess(wav_np, cfg=REFERENCE_DEMO.audio_post):
    """Apply raised-cosine fade and silence padding to fix abrupt onset/offset."""
    fade_n = min(int(cfg.fade_seconds * OUTPUT_SAMPLE_RATE), wav_np.shape[0] // 2)
    if fade_n > 0:
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, fade_n, dtype=wav_np.dtype)))
        wav_np[:fade_n] *= ramp
        wav_np[-fade_n:] *= ramp[::-1]
    pad = np.zeros(int(cfg.pad_seconds * OUTPUT_SAMPLE_RATE), dtype=wav_np.dtype)
    return np.concatenate([pad, wav_np, pad])


# ---------------------------------------------------------------------------
# sampling — the host mirror of tt/xtts_sampler.py
# ---------------------------------------------------------------------------
def _sample(logits, seen, *, temperature, top_k, top_p, rep, suppress=()):
    """Sample one token id from logits with repetition penalty, top-k, and top-p."""
    L = logits.detach().float().reshape(-1).clone()

    if rep != 1.0 and seen:
        idx = torch.tensor(sorted(seen), dtype=torch.long)
        v = L[idx]
        L[idx] = torch.where(v > 0, v / rep, v * rep)

    if temperature > 0.0 and temperature != 1.0:
        L = L / temperature

    if top_k and 0 < top_k < L.shape[0]:
        vals, _ = torch.topk(L, top_k)  # descending
        thr = vals[-1]
        if 0.0 < top_p < 1.0:
            # Keep the shortest prefix of the sorted window whose cumulative probability first
            # exceeds top_p (the token that crosses is kept); its smallest logit is the nucleus
            # threshold, combined with the top-k threshold by max().
            probs = torch.softmax(vals, dim=-1)
            keep = (torch.cumsum(probs, dim=-1) - probs) < top_p
            thr = torch.maximum(vals[keep].min(), thr)
        L = torch.where(L >= thr, L, torch.full_like(L, -float("inf")))

    for token in suppress:  # applied AFTER shaping so temperature cannot rescale it back
        L[token] = -float("inf")

    if temperature <= 0.0:
        return int(L.argmax().item())
    return int(torch.multinomial(torch.softmax(L, dim=-1), 1).item())


# ---------------------------------------------------------------------------
# GPT decode
# ---------------------------------------------------------------------------
def _gpt_forward_cached(gpt, hidden, cache, mask):
    """Run cached GPT blocks plus ln_f and final_norm over hidden states."""
    for block in gpt.stack.h:
        out = block(hidden, past_key_values=cache, attention_mask=mask, use_cache=True)
        hidden = out[0] if isinstance(out, tuple) else out
    return gpt.final_norm(gpt.stack.ln_f(hidden))


@torch.no_grad()
def generate_cached(gpt, text_ids, cond_latents, *, max_new_tokens, min_new_tokens, sampler_kw):
    """Autoregressive decode with a KV cache, returning codes and latents."""
    text_len = text_ids.shape[1]
    text_emb = gpt.text_embedding(text_ids) + gpt.text_pos_embedding(torch.arange(text_len))
    start = torch.full((1, 1), START_AUDIO_TOKEN, dtype=torch.long)
    mel_emb = gpt.mel_embedding(start) + gpt.mel_pos_embedding(torch.tensor([0]))

    # PREFILL: [cond latents | text | start_audio] under a causal mask, seeding the cache.
    prompt = torch.cat([cond_latents, text_emb, mel_emb], dim=1)
    n = prompt.shape[1]
    causal = torch.triu(torch.full((n, n), torch.finfo(prompt.dtype).min), diagonal=1).view(1, 1, n, n)
    cache = DynamicCache()
    hidden = _gpt_forward_cached(gpt, prompt, cache, causal)
    logits = gpt.mel_head(hidden[:, -1])[0]  # [V] — the distribution over c_0

    codes, latents, seen = [], [], set()
    kv_len = n
    while True:
        suppress = (STOP_AUDIO_TOKEN,) if len(codes) < min_new_tokens else ()
        token = _sample(logits, seen, suppress=suppress, **sampler_kw)
        if token == STOP_AUDIO_TOKEN:
            break
        codes.append(token)
        seen.add(token)

        # DECODE step: feed the code we just sampled; its output state is that code's latent.
        pos = len(codes)  # mel position of c_i is i + 1
        emb = gpt.mel_embedding(torch.tensor([[token]])) + gpt.mel_pos_embedding(torch.tensor([pos]))
        mask = torch.zeros(1, 1, 1, kv_len + 1, dtype=emb.dtype)  # a decode step attends to all of it
        hidden = _gpt_forward_cached(gpt, emb, cache, mask)
        kv_len += 1
        latents.append(hidden)
        if len(codes) >= max_new_tokens:
            break
        logits = gpt.mel_head(hidden[:, -1])[0]

    if not codes:
        raise RuntimeError("the model emitted STOP immediately — no audio codes were generated")
    return torch.tensor(codes, dtype=torch.long).reshape(1, -1), torch.cat(latents, dim=1)


@torch.no_grad()
def generate_recompute(gpt, text_ids, cond_latents, *, max_new_tokens, min_new_tokens, sampler_kw):
    """Naive recomputing decode loop without a KV cache."""
    mel_ids = torch.full((1, 1), START_AUDIO_TOKEN, dtype=torch.long)
    codes, seen = [], set()
    while len(codes) < max_new_tokens:
        _, mel_logits = gpt(text_ids, mel_ids, cond_latents=cond_latents)
        suppress = (STOP_AUDIO_TOKEN,) if len(codes) < min_new_tokens else ()
        token = _sample(mel_logits[0, -1], seen, suppress=suppress, **sampler_kw)
        if token == STOP_AUDIO_TOKEN:
            break
        codes.append(token)
        seen.add(token)
        mel_ids = torch.cat([mel_ids, torch.tensor([[token]], dtype=torch.long)], dim=1)

    if not codes:
        raise RuntimeError("the model emitted STOP immediately — no audio codes were generated")
    latents = gpt(text_ids, mel_ids, cond_latents=cond_latents, return_latent=True)[:, 1:]  # drop start
    return mel_ids[:, 1:], latents


# ---------------------------------------------------------------------------
# pipeline stages
# ---------------------------------------------------------------------------
@torch.no_grad()
def cond_latents_from_wav(reference, wav, mel_stats):
    """Average per-chunk GPT conditioning latents from reference wav."""
    parts = [reference.conditioning(wav_to_mel(w, mel_stats)) for w in chunk_wav(wav)]  # each [1, 1024, 32]
    style = torch.stack(parts, dim=0).mean(dim=0) if len(parts) > 1 else parts[0]
    return style.transpose(1, 2)


def _audio_duration_s(wav_np, sample_rate=OUTPUT_SAMPLE_RATE):
    """Return audio duration in seconds from sample count and sample rate."""
    return float(np.asarray(wav_np).reshape(-1).shape[0]) / float(sample_rate)


def _log_perf_metrics(*, wall_s, audio_s, stages, n_codes):
    """CPU speed summary. RTF = wall/audio (< 1 is faster than real time — not expected on CPU)."""
    rtf = wall_s / audio_s if audio_s > 0 else float("inf")
    logger.info("---------- XTTS reference (CPU) performance ----------")
    for name, dt in stages.items():
        logger.info(f"  {name:<22}: {dt:.3f} s")
    logger.info(f"  {'audio codes':<22}: {n_codes} ({stages['gpt decode'] / max(1, n_codes) * 1e3:.1f} ms/code)")
    logger.info(f"Wall time / latency     : {wall_s:.3f} s  (conditioning + decode + vocoder)")
    logger.info(f"Audio duration          : {audio_s:.3f} s  ({OUTPUT_SAMPLE_RATE} Hz)")
    logger.info(f"RTF (Real-Time Factor)  : {rtf:.3f}  (wall/audio; <1 = faster than real-time)")
    logger.info("------------------------------------------------------")


@torch.no_grad()
def synthesize(reference, wrapped, cond_latents, g, args):
    """One pass: text ids -> codes + latents -> waveform. Returns ``(wav_np, codes, timings)``."""
    sampler_kw = dict(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        rep=args.repetition_penalty,
    )
    generate = generate_recompute if args.no_kv_cache else generate_cached
    # Decode is launch-bound (tiny GEMMs); fewer threads is faster. Vocoder/conditioning keep the
    # global thread count.
    prev_threads = torch.get_num_threads()
    t0 = time.time()
    try:
        if args.decode_threads > 0:
            torch.set_num_threads(args.decode_threads)
        codes, latents = generate(
            reference.gpt,
            wrapped,
            cond_latents,
            max_new_tokens=args.max_tokens,
            min_new_tokens=args.min_tokens_resolved,
            sampler_kw=sampler_kw,
        )
    finally:
        torch.set_num_threads(prev_threads)
    t_gpt = time.time() - t0

    t0 = time.time()
    wav = reference.decoder_full.decoder(latents, g)  # [1, 1, T_out]
    t_voc = time.time() - t0

    wav_np = _postprocess(wav.reshape(-1).float().numpy())
    return wav_np, codes, {"gpt decode": t_gpt, "vocoder": t_voc}


def main():
    """CLI entrypoint for the host-only XTTS reference demo."""
    ap = argparse.ArgumentParser(description="XTTS-v2 CPU reference text-to-speech demo")
    cfg, gen = REFERENCE_DEMO, REFERENCE_DEMO.generation
    ap.add_argument("--text", default=cfg.text)
    ap.add_argument(
        "--ref-audio",
        default=cfg.ref_audio,
        help="voice to clone: local WAV path, coqui-ai/TTS test clip name (LJ001-0001.wav, or "
        "'LJ001-0001.wav+LJ001-0003.wav+...' to concatenate clips into a longer reference), "
        "or HF sample name (en_sample.wav).",
    )
    ap.add_argument("--output", default=cfg.output)
    ap.add_argument("--lang", default=cfg.language)
    ap.add_argument(
        "--max-tokens", type=int, default=gen.max_tokens, help="cap on generated audio codes (STOP ends it earlier)"
    )
    ap.add_argument(
        "--min-tokens",
        type=int,
        default=gen.min_tokens,
        help="STOP-suppression floor in audio codes. 0 (default) = disabled, matches HF. "
        "-1 = auto (~2x the wrapped text length). Raise it if a LONG prompt is only partly spoken.",
    )
    ap.add_argument("--temperature", type=float, default=gen.temperature, help="0 = greedy/deterministic")
    ap.add_argument("--top-k", type=int, default=gen.top_k)
    ap.add_argument("--top-p", type=float, default=gen.top_p)
    ap.add_argument("--repetition-penalty", type=float, default=gen.repetition_penalty)
    ap.add_argument("--seed", type=int, default=gen.seed, help="torch seed for reproducible sampling")
    ap.add_argument("--ref-seconds", type=int, default=cfg.ref_seconds, help="conditioning window (gpt_cond_len)")
    ap.add_argument(
        "--spk-seconds",
        type=int,
        default=cfg.spk_seconds,
        help="speaker-embedding window. Defaults to the whole reference (coqui max_ref_length); "
        f"pass {DEMO.spk_seconds} to match the on-device demo, which keeps a margin under L1.",
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=cfg.threads,
        help="torch CPU threads for the big-tensor stages (0 = leave torch's default, i.e. one per "
        "core — which oversubscribes and thrashes on a shared host)",
    )
    ap.add_argument(
        "--decode-threads",
        type=int,
        default=cfg.decode_threads,
        help="torch CPU threads for the autoregressive loop only. A single-token step is many tiny "
        "GEMMs, so it is launch-bound: fewer threads is much faster (0 = use --threads)",
    )
    ap.add_argument("--no-kv-cache", action="store_true", help="naive full-recompute decode (slow; for checking)")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    logger.info(
        f"running on CPU (torch {torch.__version__}, {torch.get_num_threads()} threads, "
        f"{args.decode_threads or torch.get_num_threads()} in the decode loop)"
    )

    from scipy.signal import resample_poly

    t_wall0 = time.time()
    logger.info("loading XTTS-v2 weights ...")
    t0 = time.time()
    sd = load_xtts_state_dict()
    reference = XttsReference(sd)  # conditioning + GPT + (speaker encoder | mel | HiFi-GAN)
    t_load = time.time() - t0

    # Inputs: reference audio at 22.05 kHz (conditioning) and 16 kHz (speaker encoder), plus text.
    wav = _load_audio_22k(args.ref_audio, args.ref_seconds)
    g_ratio = math.gcd(SPK_SR, MEL_SR)
    spk_src = wav[0].numpy()[: MEL_SR * args.spk_seconds]
    spk_wav = torch.from_numpy(resample_poly(spk_src, SPK_SR // g_ratio, MEL_SR // g_ratio).astype("float32"))
    spk_wav = spk_wav.unsqueeze(0)
    cond_windows = chunk_wav(wav)
    logger.info(
        f"reference audio: {wav.shape[-1] / MEL_SR:.2f}s loaded | conditioning on "
        f"{sum(c.shape[-1] for c in cond_windows) / MEL_SR:.2f}s as {len(cond_windows)} window(s) "
        f"(gpt_cond_len {GPT_COND_LEN_SEC}s, chunk {GPT_COND_CHUNK_SEC}s) | speaker embedding on "
        f"{spk_wav.shape[-1] / SPK_SR:.2f}s"
    )

    # Strip trailing sentence-final punctuation: the final "." is its own token (id 9) and the model
    # tends to VERBALIZE it as "dot" at the tail. Internal commas (prosody) are kept.
    chunk_texts = _split_into_chunks(args.text, args.lang)
    chunks = [
        (clean, wrap_text_ids(preprocess_text(clean, lang=args.lang)))
        for clean in (re.sub(SENTENCE_FINAL_PUNCT_RE, "", ct.strip()) for ct in chunk_texts)
    ]
    if len(chunks) == 1:
        logger.info(f"text fits ONE pass ({chunks[0][1].shape[1]} tokens): {chunks[0][0]!r}")
    else:
        logger.info(f"text exceeds the single-pass budget -> {len(chunks)} passes:")
        for i, (clean, w) in enumerate(chunks):
            logger.info(f"  [{i + 1}/{len(chunks)}] {w.shape[1]:3d} tokens  {clean!r}")

    # STOP-suppression floor. Auto (-1) scales with the longest chunk (~2x its wrapped length),
    # clamped under the code budget; 0 disables it (HF default, right for short prompts).
    longest = max(w.shape[1] for _, w in chunks)
    floor = int(gen.min_tokens_auto_factor * longest) if args.min_tokens < 0 else args.min_tokens
    args.min_tokens_resolved = max(0, min(floor, args.max_tokens - 1))
    logger.info(f"min audio codes before STOP allowed: {args.min_tokens_resolved} (0 = disabled)")

    mode = (
        "greedy"
        if args.temperature <= 0
        else f"sampled (temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}, rep={args.repetition_penalty})"
    )
    logger.info(
        f"generating on CPU [{mode}], up to {args.max_tokens} codes per pass"
        f"{' [no KV cache]' if args.no_kv_cache else ''} ..."
    )

    t0 = time.time()
    cond_latents = cond_latents_from_wav(reference, wav, sd["mel_stats"].cpu())  # [1, 32, 1024]
    spk_g = reference.decoder_full.speaker_embedding(spk_wav)  # [1, 512, 1]
    t_cond = time.time() - t0

    gap = np.zeros(int(cfg.audio_post.chunk_gap_seconds * OUTPUT_SAMPLE_RATE), dtype="float32")  # between passes
    pieces, code_parts = [], []
    stages = {"conditioning + speaker": t_cond, "gpt decode": 0.0, "vocoder": 0.0}
    t_gen0 = time.time()
    for i, (clean, wrapped) in enumerate(chunks):
        wav_i, codes_i, t_i = synthesize(reference, wrapped, cond_latents, spk_g, args)
        for k, v in t_i.items():
            stages[k] += v
        n_i = codes_i.shape[1]
        logger.info(
            f"  pass {i + 1}/{len(chunks)}: {n_i} codes ({'stop' if n_i < args.max_tokens else 'max'}), "
            f"audio {_audio_duration_s(wav_i):.2f}s, decode {t_i['gpt decode']:.2f}s, "
            f"vocoder {t_i['vocoder']:.2f}s"
        )
        pieces.append(wav_i.astype("float32"))
        code_parts.append(codes_i)
    wav_out = (
        pieces[0]
        if len(pieces) == 1
        else np.concatenate([p for k, pc in enumerate(pieces) for p in ((gap, pc) if k else (pc,))])
    )
    codes = torch.cat(code_parts, dim=1)
    gen_s = time.time() - t_gen0

    import soundfile as sf

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    sf.write(args.output, wav_out, OUTPUT_SAMPLE_RATE)
    logger.info(f"wrote reference (CPU) audio -> {os.path.abspath(args.output)}")

    _log_perf_metrics(
        wall_s=gen_s,
        audio_s=_audio_duration_s(wav_out),
        stages=stages,
        n_codes=codes.shape[1],
    )
    logger.info(f"  weight load + build   : {t_load:.2f} s  (excluded from wall/RTF)")
    logger.info(f"  end-to-end            : {time.time() - t_wall0:.2f} s  (incl. weight load + audio prep + write)")


if __name__ == "__main__":
    main()
