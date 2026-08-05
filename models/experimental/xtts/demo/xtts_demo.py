# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""XTTS-v2 on-device demo: text + reference audio -> spoken WAV.

Runs the whole model on a Tenstorrent device (audio conditioning -> GPT KV-cache
autoregressive decode -> HiFi-GAN vocoder) and writes the generated 24 kHz audio
to a WAV file you can play.

Decoding is **greedy** (deterministic, the validated path). Real XTTS samples
(temperature / top-k / top-p) for more natural prosody, so expect flatter, and at
longer lengths possibly repetitive, output from greedy — that is an on-device
sampling feature, not a bug in this pipeline.

Everything runs on device except the BPE tokenizer and the conditioning 80-mel
(both host, outside the tensor-compute path).

Usage:
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd); export PYTHONPATH=$(pwd)
    python models/experimental/xtts/demo/xtts_demo.py \\
        --text "Hello from Tenstorrent." --max-tokens 200

    # bring your own reference voice + write the torch reference too, for A/B:
    python models/experimental/xtts/demo/xtts_demo.py \\
        --ref-audio /path/to/voice.wav --write-torch-ref --output out.wav
"""

import argparse
import math
import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.experimental.xtts.reference.xtts_conditioning import MEL_SR, load_reference_audio, wav_to_mel
from models.experimental.xtts.reference.xtts_gpt_block import load_xtts_state_dict
from models.experimental.xtts.reference.xtts_gpt_generate import STOP_TEXT_TOKEN, wrap_text_ids
from models.experimental.xtts.reference.xtts_hifi_decoder import OUTPUT_SAMPLE_RATE
from models.experimental.xtts.reference.xtts_inference import XttsReference
from models.experimental.xtts.reference.xtts_mel import SAMPLE_RATE as SPK_SR
from models.experimental.xtts.reference.xtts_text_embedding import preprocess_text
from models.experimental.xtts.tt.xtts_inference import TtXtts

TILE = 32


def _load_audio_22k(ref_audio, max_seconds):
    """Reference audio as ``[1, samples]`` @ 22.05 kHz — a local WAV path if it
    exists, else an HF ``coqui/XTTS-v2`` sample name (e.g. ``en_sample.wav``)."""
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
    return load_reference_audio(sample=ref_audio, max_seconds=max_seconds)


MAX_TEXT_IDS = 352  # keep the padded text under MAX_TEXT_POS (404) with headroom
# TWO budgets, because the wall behaves differently in the two cases. It is an ALLOCATION COLLISION
# ("Statically allocated circular buffers ... clash with L1 buffers"), not a clean size limit, so it
# degrades as device open/close cycles accumulate in a process. Measured on p150 by running the demo:
#
#   ONE pass, fresh device (words -> codes):  20->175 PASS | 23->177,182,184 PASS | 22->192 PASS
#                                             25->203 PASS | 24->207 PASS  <-- highest seen to pass
#                                             27-> FAIL cb-clash | 29-> FAIL
#   Nth pass, same process:                   a chunk estimated at ~204 codes FAILED as the 5th cycle,
#                                             while 207 passed as the 1st -> per-chunk headroom SHRINKS
#                                             with chunk count, so the chunk budget must be lower.
# Note the same text varies +/-4% run to run (23 words gave 177/182/184), so leave margin.
MAX_SINGLE_PASS_CODES = 205  # above this, split into chunks
MAX_CHUNK_CODES = 165  # per chunk once splitting; lower than the single-pass budget on purpose
CODES_PER_ID = 156 / 71.0  # measured: 71 text ids -> 156 audio codes


def _split_into_chunks(text, lang):
    """Return the sentence groups to synthesise. ONE group (single pass) whenever the whole text fits.

    XTTS generates one utterance per pass, bounded by the text position embedding (MAX_TEXT_POS 404),
    the audio-code budget, and the single-shot vocoder's circular buffers. If the whole text fits in a
    pass it is returned unsplit — that is the fast path and what the default text is sized for.

    Only when it does not fit is it split at sentence boundaries, and then against the SMALLER
    MAX_CHUNK_CODES, because per-chunk headroom shrinks as device cycles accumulate (see the budgets).
    A sentence is never split, so a single sentence over ~25 words cannot be made to fit — the caller
    is warned rather than silently crashing.
    """

    def ids_of(t):
        return preprocess_text(re.sub(r"[.!?]+\s*$", "", t), lang=lang).shape[-1]

    whole = text.strip()
    if ids_of(whole) <= MAX_TEXT_IDS and ids_of(whole) * CODES_PER_ID <= MAX_SINGLE_PASS_CODES:
        return [whole]  # single pass

    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", whole) if p.strip()]
    for p in parts:
        if ids_of(p) * CODES_PER_ID > MAX_SINGLE_PASS_CODES:
            logger.warning(
                f"sentence is {len(p.split())} words (~{int(ids_of(p) * CODES_PER_ID)} audio codes) — over the "
                f"~{int(MAX_SINGLE_PASS_CODES / CODES_PER_ID / 3.7)}-word single-pass limit. A sentence is never "
                f"split, so this pass may fail with an L1 circular-buffer clash. Shorten it or add punctuation."
            )
    chunks, cur = [], []
    for part in parts:
        cand = " ".join(cur + [part])
        n = ids_of(cand)
        if cur and (n > MAX_TEXT_IDS or n * CODES_PER_ID > MAX_CHUNK_CODES):
            chunks.append(" ".join(cur))
            cur = [part]
        else:
            cur.append(part)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _postprocess(wav_np):
    """Fix the abrupt ("crimped") onset: short raised-cosine fade in/out + leading/trailing
    silence (the vocoder starts at the first content code with no natural lead-in)."""
    fade_n = min(int(0.015 * OUTPUT_SAMPLE_RATE), wav_np.shape[0] // 2)  # ~15 ms
    if fade_n > 0:
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, fade_n, dtype=wav_np.dtype)))
        wav_np[:fade_n] *= ramp
        wav_np[-fade_n:] *= ramp[::-1]
    pad = np.zeros(int(0.06 * OUTPUT_SAMPLE_RATE), dtype=wav_np.dtype)  # ~60 ms lead-in/out
    return np.concatenate([pad, wav_np, pad])


_ASR = {}


def _score_take(wav_np, text, codes):
    """Rank a candidate take: lower is better. Primary = CER (Whisper-base.en transcription
    vs the input text — directly measures "does the audio say the text"). Falls back to a
    code-diversity heuristic (1 - unique/total, which penalises collapsed/droning takes) if
    the Whisper/jiwer backends are unavailable. Returns (score, detail_str)."""
    try:
        import jiwer
        from scipy.signal import resample_poly
        from transformers import pipeline

        if "asr" not in _ASR:
            _ASR["asr"] = pipeline("automatic-speech-recognition", model="openai/whisper-base.en", device="cpu")
        g = math.gcd(16000, OUTPUT_SAMPLE_RATE)
        wav16 = resample_poly(wav_np.astype("float32"), 16000 // g, OUTPUT_SAMPLE_RATE // g)
        hyp = _ASR["asr"]({"array": wav16.astype("float32"), "sampling_rate": 16000})["text"].strip()
        norm = lambda s: re.sub(r"[^a-z ]", "", s.lower()).strip()  # noqa: E731
        cer = jiwer.cer(norm(text), norm(hyp))
        return cer, f"CER {cer:.3f} :: {hyp!r}"
    except Exception as e:  # backend missing / offline — degrade gracefully
        flat = codes.reshape(-1).tolist()
        diversity = len(set(flat)) / max(1, len(flat))
        logger.warning(f"CER scoring unavailable ({type(e).__name__}: {e}); using code-diversity fallback")
        return 1.0 - diversity, f"diversity {diversity:.3f} (fallback; no CER)"


def _audio_duration_s(wav_np, sample_rate=OUTPUT_SAMPLE_RATE) -> float:
    return float(np.asarray(wav_np).reshape(-1).shape[0]) / float(sample_rate)


def _log_perf_metrics(
    *,
    wall_s: float,
    audio_s: float,
    ttfc_s: float | None,
    label: str = "",
    compile_s: float | None = None,
):
    """Log Coqui/HF-style TTS speed metrics.

    * Wall time / latency — **trace replay only** (setup + decode + vocoder ``execute_trace``).
      Warmup and capture are excluded; they are compile / data collection for the trace.
    * RTF — ``replay_time / audio_duration``; RTF < 1 ⇒ faster than real-time.
    * Time-to-first-chunk — first chunk's replay time (equals wall when single-pass).
    * Compile (optional) — warmup + capture + other non-replay work inside the traced path.
    """
    rtf = wall_s / audio_s if audio_s > 0 else float("inf")
    prefix = f"{label} " if label else ""
    logger.info(f"{prefix}---------- XTTS performance ----------")
    logger.info(f"{prefix}Wall time / latency     : {wall_s:.3f} s  (trace replay only)")
    if compile_s is not None:
        logger.info(f"{prefix}Compile / capture       : {compile_s:.3f} s  (excluded from wall/RTF)")
    logger.info(f"{prefix}Audio duration          : {audio_s:.3f} s  ({OUTPUT_SAMPLE_RATE} Hz)")
    logger.info(f"{prefix}RTF (Real-Time Factor)  : {rtf:.3f}  (replay/audio; <1 = faster than real-time)")
    if ttfc_s is None:
        logger.info(f"{prefix}Time-to-first-chunk     : n/a")
    elif abs(ttfc_s - wall_s) < 1e-6:
        logger.info(
            f"{prefix}Time-to-first-chunk     : {ttfc_s:.3f} s  " f"(non-streaming: first audio == full clip ready)"
        )
    else:
        logger.info(
            f"{prefix}Time-to-first-chunk     : {ttfc_s:.3f} s  " f"(first chunk/pass ready; remaining audio follows)"
        )
    logger.info(f"{prefix}---------------------------------------")


def _generate_one(tt, wrapped, cond_wav, spk_wav_tt, args):
    """One full device generation + vocode + onset post-processing.

    Returns ``(wav_np, codes, replay_s, compile_s)``. ``replay_s`` is final inference
    (``execute_trace`` only); ``compile_s`` is warmup + capture (excluded from wall/RTF).
    """
    # FULL-MODEL TRACE: the entire model runs inside ttnn traces — SETUP trace (on-device
    # conditioning mel + speaker encoder + prefill that seeds the KV cache), the DECODE-step trace
    # captured once and replayed per token, and the VOCODER trace. Needs the fixed KV-cache length:
    # prompt (cond perceiver latents = 32 + wrapped/padded text) + the decode budget.
    prompt_len = 32 + wrapped.shape[1]
    max_seq = -(-(prompt_len + args.max_tokens + 2) // TILE) * TILE
    wav_dev, codes, perf = tt.inference_fully_traced(
        wrapped,
        cond_wav,
        spk_wav_tt,
        max_seq,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        min_new_tokens=args.min_tokens_resolved,
    )
    wav_np = ttnn.to_torch(wav_dev).float().reshape(-1).numpy()  # [T_out]
    return _postprocess(wav_np), codes, float(perf["replay_s"]), float(perf["compile_s"])


def _take_on_device(sd, ref_decoder_full, wrapped, cond_wav, spk_wav, args, seed_offset):
    """Open a FRESH device, build the model, run one generation, close the device. The fp32
    HiFi-GAN vocoder exhausts L1_SMALL when several full generations share one device, so
    best-of-N isolates each take on its own device (same reason the tests use a per-test
    device fixture). Returns ``(wav_np, codes, replay_s, compile_s)``."""
    device = ttnn.open_device(device_id=0, l1_small_size=65536)
    try:
        tt = TtXtts(device, sd, ref_decoder_full)
        spk_wav_tt = ttnn.from_torch(
            spk_wav.reshape(1, -1, 1).float(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.float32
        )
        if args.seed is not None:
            # distinct-but-reproducible-ish seed per take (ttnn sampling isn't bit-exact across
            # runs regardless, so takes differ even without this).
            ttnn.manual_seed(args.seed + seed_offset, device=device)
        return _generate_one(tt, wrapped, cond_wav, spk_wav_tt, args)
    finally:
        ttnn.close_device(device)


def main():
    ap = argparse.ArgumentParser(description="XTTS-v2 on-device text-to-speech demo")
    ap.add_argument(
        "--text",
        # "can already" (not "can now"): "can now" is a /n/#/n/ nasal collision the vocoder
        # merges into "cannow/cannot" — "already" starts with a vowel and transcribes cleanly (CER 0.008).
        default="Voice synthesis has come a long way, and modern systems can already generate natural sounding speech with remarkable accuracy. Hey how are you doing?.",
    )
    ap.add_argument(
        "--ref-audio",
        default="reference.wav",
        help="local WAV path or HF sample name (e.g. en_sample.wav) that sets the voice.",
    )
    ap.add_argument(
        "--min-tokens",
        type=int,
        default=0,
        help="STOP-suppression floor in audio codes. 0 (default) = disabled, matches HF. "
        "-1 = auto (~2x the wrapped text length). Raise it if a LONG prompt is only partly "
        "spoken; leave it at 0 for short prompts, where a floor makes the model ramble.",
    )
    args = ap.parse_args()

    # Only --text / --ref-audio / --min-tokens are exposed; everything else is fixed to the tuned
    # XTTS-v2 defaults so the demo runs full-model-traced with no other knobs.
    args.lang = "en"
    args.ref_seconds = 30  # conditioning window (coqui gpt_cond_len)
    args.spk_seconds = 8  # speaker-embedding window (device mel frontend caps long audio)
    # Cap on audio codes. This is NOT a "stop earlier if you can" budget: the traced decode loop
    # replays a fixed max_tokens steps and treats STOP as a post-loop trim (a captured trace cannot
    # branch), so every step above what the text actually needs is ~9.4 ms of pure waste — 400 cost
    # ~2 s per pass while real single-pass generations land at 160-210 codes. 240 sits above
    # MAX_SINGLE_PASS_CODES (205) with margin for the overshoot the sampler is entitled to, and it
    # also shrinks the KV cache (max_seq scales with it), so nothing that fits the pass is truncated.
    args.max_tokens = 240
    # args.min_tokens comes from --min-tokens. It stays 0 by default because a floor is only right
    # for *long* prompts. Greedy, on this reference text (96 wrapped tokens, needs ~196 codes):
    # 0 stops at 152 codes and transcribes at CER 0.151, cut after "remarkable accuracy"; -1
    # (floor 192) reaches 181 codes at CER 0.000. On a short prompt it inverts -- "Hello from
    # Tenstorrent." goes CER 0.273 -> 0.591, the floor forcing 12 extra codes of invented tail.
    args.num_outputs = 1  # single take (coqui num_gpt_outputs=1)
    args.temperature = 0.65  # sampling temperature (0 = greedy); 0.65 = cleanest single take
    args.top_k = 50
    args.top_p = 0.85  # nucleus cutoff (XTTS uses 0.85)
    args.repetition_penalty = 5.0  # XTTS uses 5.0
    args.seed = None
    args.output = "generated/xtts_demo/xtts_demo.wav"
    args.write_torch_ref = False

    from scipy.signal import resample_poly

    t_wall0 = time.time()  # end-to-end wall clock (weights + audio prep + generation + write)
    logger.info("loading XTTS-v2 weights ...")
    sd = load_xtts_state_dict()

    # Inputs: reference audio (22.05 kHz for conditioning, 16 kHz for the speaker encoder) + text.
    # The 80-mel is now computed ON DEVICE (TtConditioningMel), so the device path takes the raw wav.
    wav = _load_audio_22k(args.ref_audio, args.ref_seconds)
    g = math.gcd(SPK_SR, MEL_SR)
    # Speaker path is capped independently — the device mel frontend can't reshape very long audio.
    spk_src = wav[0].numpy()[: MEL_SR * args.spk_seconds]
    spk_wav = torch.from_numpy(resample_poly(spk_src, SPK_SR // g, MEL_SR // g).astype("float32")).unsqueeze(0)

    # Strip trailing sentence-final punctuation: the final "." is its own token (id 9) and the
    # model tends to VERBALIZE it as "dot" at the tail. Internal commas (prosody) are kept.
    # Text that fits ONE pass stays a single pass (the fast path); only text over the single-pass
    # budget is split at sentence boundaries into several passes that are joined back together.
    chunk_texts = _split_into_chunks(args.text, args.lang)
    chunks = []
    for ct in chunk_texts:
        clean = re.sub(r"[.!?]+\s*$", "", ct.strip())
        w = wrap_text_ids(preprocess_text(clean, lang=args.lang))
        pad = (-w.shape[1]) % TILE
        if pad:
            w = F.pad(w, (0, pad), value=STOP_TEXT_TOKEN)
        chunks.append((clean, w))
    if len(chunks) == 1:
        logger.info(f"text fits ONE pass ({chunks[0][1].shape[1]} tokens): {chunks[0][0]!r}")
    else:
        logger.info(f"text exceeds the single-pass budget -> CHUNKED into {len(chunks)} passes:")
        for i, (clean, w) in enumerate(chunks):
            logger.info(f"  [{i + 1}/{len(chunks)}] {w.shape[1]:3d} tokens  {clean!r}")

    # Resolve the STOP-suppression floor, PER CHUNK. Auto (-1) scales with that chunk's text (~2x its
    # wrapped length), clamped below max-tokens, so a longer prompt is protected from stopping short
    # while a short one isn't forced to ramble. 0 disables (HF default).
    def resolve_min_tokens(n_tok):
        floor = int(2.0 * n_tok) if args.min_tokens < 0 else args.min_tokens
        return max(0, min(floor, args.max_tokens - 1))

    logger.info(
        f"min audio codes before STOP allowed: {[resolve_min_tokens(w.shape[1]) for _, w in chunks]} "
        f"(0 = disabled, per chunk)"
    )

    reference = XttsReference(sd)  # supplies decoder/speaker/mel weights (and optional A/B wav)

    mode = (
        "greedy"
        if args.temperature <= 0
        else f"sampled (temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}, rep={args.repetition_penalty})"
    )
    n = max(1, args.num_outputs)
    logger.info(f"generating on device [{mode}], up to {args.max_tokens} codes, {n} take(s) ...")

    # Each take runs on its own freshly-opened device (see _take_on_device); best-of-N keeps
    # the lowest-CER take. A single take (default) is just one open/generate/close.
    wav_tt, codes, best_score, best_detail = None, None, None, None
    best_wall_s, best_ttfc_s, best_compile_s = None, None, None
    gap = np.zeros(int(0.12 * OUTPUT_SAMPLE_RATE), dtype="float32")  # ~120 ms between chunks
    for i in range(n):
        pieces, code_parts, dt, compile_s = [], [], 0.0, 0.0
        t_take0 = time.time()
        ttfc_s = None  # replay time until first audio piece is available
        for j, (clean, w) in enumerate(chunks):
            args.min_tokens_resolved = resolve_min_tokens(w.shape[1])
            wav_j, codes_j, dt_j, compile_j = _take_on_device(sd, reference.decoder_full, w, wav, spk_wav, args, i)
            dt += dt_j
            compile_s += compile_j
            if ttfc_s is None:
                ttfc_s = dt_j  # first chunk/pass replay → analogue of time-to-first-chunk
            pieces.append(wav_j.astype("float32"))
            code_parts.append(codes_j)
            nc = codes_j.shape[1]
            audio_j_s = _audio_duration_s(wav_j)
            rtf_j = dt_j / audio_j_s if audio_j_s > 0 else float("inf")
            if len(chunks) > 1:
                logger.info(
                    f"  take {i + 1}/{n} chunk {j + 1}/{len(chunks)}: {nc} codes "
                    f"({'stop' if nc < args.max_tokens else 'max'}), "
                    f"audio {audio_j_s:.2f}s, replay {dt_j:.3f}s, RTF {rtf_j:.3f}"
                )
        latency = time.time() - t_take0
        # single pass -> the waveform IS the output, no join
        wav_i = (
            pieces[0]
            if len(pieces) == 1
            else np.concatenate([p for k, pc in enumerate(pieces) for p in ((gap, pc) if k else (pc,))])
        )
        codes_i = torch.cat(code_parts, dim=1)
        n_codes = codes_i.shape[1]
        stopped = n_codes < args.max_tokens
        audio_s = _audio_duration_s(wav_i)
        score, detail = _score_take(wav_i, args.text, codes_i) if n > 1 else (0.0, "")
        logger.info(
            f"take {i + 1}/{n}: {n_codes} codes ({'stop' if stopped else 'max'}), "
            f"audio {audio_s:.2f}s, replay {dt:.3f}s, wall {latency:.1f}s" + (f" | {detail}" if detail else "")
        )
        _log_perf_metrics(wall_s=dt, audio_s=audio_s, ttfc_s=ttfc_s, label=f"take {i + 1}/{n}", compile_s=compile_s)
        if best_score is None or score < best_score:
            wav_tt, codes, best_score, best_detail = wav_i, codes_i, score, detail
            best_wall_s, best_ttfc_s, best_compile_s = dt, ttfc_s, compile_s
    if n > 1:
        logger.info(f"selected best of {n} -> {best_detail}")
        _log_perf_metrics(
            wall_s=best_wall_s,
            audio_s=_audio_duration_s(wav_tt),
            ttfc_s=best_ttfc_s,
            label="best",
            compile_s=best_compile_s,
        )

    import soundfile as sf

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    sf.write(args.output, wav_tt, OUTPUT_SAMPLE_RATE)
    logger.info(f"wrote device audio -> {os.path.abspath(args.output)}")
    # Final summary for the selected take (same numbers as the last/best log block).
    _log_perf_metrics(
        wall_s=best_wall_s,
        audio_s=_audio_duration_s(wav_tt),
        ttfc_s=best_ttfc_s,
        label="final",
        compile_s=best_compile_s,
    )

    e2e = time.time() - t_wall0
    logger.info(f"  end-to-end          : {e2e:.2f} s  (incl. weight load + audio prep + write)")

    if args.write_torch_ref:
        # A/B on the SAME codes the best device take produced (teacher-forced), so the two WAVs
        # are the same utterance — not an independent greedy run (which would collapse). Runs on
        # host (CPU torch), so no device is needed here (torch reference uses the host wav_to_mel).
        # Single-pass A/B only: with several chunks the codes are the concatenation of all passes,
        # so they no longer correspond to one text prompt.
        if len(chunks) > 1:
            logger.warning("--write-torch-ref uses chunk 1's text against all chunks' codes; A/B is approximate")
        cond_mel = wav_to_mel(wav, sd["mel_stats"].cpu())  # host 80-mel [1, 80, s] for the torch reference
        wav_ref = reference.wav_from_codes(chunks[0][1], cond_mel, spk_wav, codes[0].tolist())
        ref_path = args.output.replace(".wav", "_reference.wav")
        sf.write(ref_path, wav_ref.reshape(-1).numpy(), OUTPUT_SAMPLE_RATE)
        logger.info(f"wrote torch reference audio (same codes) -> {os.path.abspath(ref_path)}")


if __name__ == "__main__":
    main()
