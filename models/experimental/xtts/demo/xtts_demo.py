# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""XTTS-v2 on-device demo: text + reference audio -> spoken WAV.

Runs the whole model on a Tenstorrent device (audio conditioning -> GPT KV-cache
autoregressive decode -> HiFi-GAN vocoder) and writes the generated 24 kHz audio
to a WAV file you can play.

Decoding SAMPLES on device (temperature 0.65 / top-k 50 / top-p 0.85 / repetition
penalty 5.0 — the tuned XTTS-v2 settings), which is what gives natural prosody and
lets a take self-terminate at STOP instead of droning to the token cap.

Only ``--text`` / ``--ref-audio`` / ``--min-tokens`` are exposed; every other knob is
fixed to those tuned defaults in ``main()``, so the demo always runs full-model-traced.

Everything runs on device except the BPE tokenizer, the reference-audio load/resample,
and the fade-in/out on the finished waveform — all host, outside the tensor-compute
path. Both mels (conditioning 80-mel and speaker-encoder 64-mel) run on device, inside
the traced setup.

**No configuration lives in this file.** Every number, default and budget comes from
:mod:`models.experimental.xtts.config` — the sampling knobs (``GENERATION``), the pass/chunk
code budgets (``CHUNKING``), the onset cleanup (``AUDIO_POST``), the best-of-N scorer
(``SCORING``) and the demo's own defaults (``DEMO``). Change behaviour there, not here.

Usage:
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd); export PYTHONPATH=$(pwd)
    python models/experimental/xtts/demo/xtts_demo.py --text "Hello from Tenstorrent."

    # bring your own reference voice:
    python models/experimental/xtts/demo/xtts_demo.py \
        --ref-audio /path/to/voice.wav --text "Hello from Tenstorrent."

The WAV lands in ``generated/xtts_demo/``. To also write the CPU torch reference on the
SAME codes (an A/B of the device vocoder), set ``args.write_torch_ref = True`` in ``main()``.
"""

import argparse
import math
import os
import re
import time
from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.experimental.xtts.config import (
    AUDIO_POST,
    CHUNKING,
    CLAUSE_SPLIT_RE,
    COQUI_CLIP_RE,
    DEMO,
    GENERATION,
    NUM_LATENTS,
    OUTPUT_SAMPLE_RATE,
    SCORING,
    SENTENCE_FINAL_PUNCT_RE,
    SENTENCE_SPLIT_RE,
    TILE,
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
from models.experimental.xtts.reference.xtts_gpt_generate import STOP_TEXT_TOKEN, wrap_text_ids
from models.experimental.xtts.reference.xtts_inference import XttsReference
from models.experimental.xtts.reference.xtts_mel import SAMPLE_RATE as SPK_SR
from models.experimental.xtts.reference.xtts_text_embedding import preprocess_text
from models.experimental.xtts.tt.xtts_inference import TtXtts

_COQUI_CLIP_RE = re.compile(COQUI_CLIP_RE)


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
    """Split text into groups that each fit one synthesis pass."""

    def ids_of(t):
        """Count wrapped text token ids for a string."""
        return preprocess_text(re.sub(SENTENCE_FINAL_PUNCT_RE, "", t), lang=lang).shape[-1]

    def est_codes(t):
        """Estimate the audio codes a string needs."""
        return ids_of(t) * CHUNKING.codes_per_id

    def pack(atoms, max_codes):
        """Greedily group atoms into runs that fit the code budget."""
        out, cur = [], []
        for a in atoms:
            cand = " ".join(cur + [a])
            if cur and (ids_of(cand) > CHUNKING.max_text_ids or est_codes(cand) > max_codes):
                out.append(" ".join(cur))
                cur = [a]
            else:
                cur.append(a)
        if cur:
            out.append(" ".join(cur))
        return out

    def atoms_of(sentence, max_codes):
        """Break a too-long sentence into pieces that can fit a pass.

        A pass decodes at most a fixed number of codes, so a sentence over that budget is never
        finished: the decode runs to the cap and the tail comes out as drone or noise. Break it
        at internal punctuation first (a seam where a speaker pauses anyway) and fall back to
        word wrapping for a clause that is still too long — the same approach as upstream coqui's
        ``split_sentence``, which hard-wraps any sentence over its length limit.
        """
        if est_codes(sentence) <= max_codes:
            return [sentence]
        atoms = []
        for clause in (c.strip() for c in re.split(CLAUSE_SPLIT_RE, sentence)):
            if not clause:
                continue
            atoms.extend(clause.split() if est_codes(clause) > max_codes else [clause])
        return atoms

    whole = text.strip()
    if (
        ids_of(whole) <= CHUNKING.max_text_ids
        and ids_of(whole) * CHUNKING.codes_per_id <= CHUNKING.max_single_pass_codes
    ):
        return [whole]  # single pass

    budget = CHUNKING.max_chunk_codes
    sentences = [p.strip() for p in re.split(SENTENCE_SPLIT_RE, whole) if p.strip()]
    chunks = pack([a for s in sentences for a in atoms_of(s, budget)], budget)
    for c in chunks:
        if est_codes(c) > budget:  # a single word over the budget; nothing left to split on
            logger.warning(f"chunk needs ~{int(est_codes(c))} codes, over the {budget}-code budget: {c!r}")
    return chunks


def _postprocess(wav_np, cfg=AUDIO_POST):
    """Apply raised-cosine fade and silence padding to fix abrupt onset/offset."""
    fade_n = min(int(cfg.fade_seconds * OUTPUT_SAMPLE_RATE), wav_np.shape[0] // 2)
    if fade_n > 0:
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, fade_n, dtype=wav_np.dtype)))
        wav_np[:fade_n] *= ramp
        wav_np[-fade_n:] *= ramp[::-1]
    pad = np.zeros(int(cfg.pad_seconds * OUTPUT_SAMPLE_RATE), dtype=wav_np.dtype)
    return np.concatenate([pad, wav_np, pad])


_ASR = {}


def _score_take(wav_np, text, codes, cfg=SCORING):
    """Rank a take by CER (primary) with a code-diversity fallback."""
    try:
        import jiwer
        from scipy.signal import resample_poly
        from transformers import pipeline

        if "asr" not in _ASR:
            _ASR["asr"] = pipeline("automatic-speech-recognition", model=cfg.asr_model_id, device=cfg.asr_device)
        g = math.gcd(cfg.asr_sample_rate, OUTPUT_SAMPLE_RATE)
        wav_asr = resample_poly(wav_np.astype("float32"), cfg.asr_sample_rate // g, OUTPUT_SAMPLE_RATE // g)
        hyp = _ASR["asr"]({"array": wav_asr.astype("float32"), "sampling_rate": cfg.asr_sample_rate})["text"].strip()
        norm = lambda s: re.sub(r"[^a-z ]", "", s.lower()).strip()  # noqa: E731
        cer = jiwer.cer(norm(text), norm(hyp))
        return cer, f"CER {cer:.3f} :: {hyp!r}"
    except Exception as e:  # backend missing / offline — degrade gracefully
        flat = codes.reshape(-1).tolist()
        diversity = len(set(flat)) / max(1, len(flat))
        logger.warning(f"CER scoring unavailable ({type(e).__name__}: {e}); using code-diversity fallback")
        return 1.0 - diversity, f"diversity {diversity:.3f} (fallback; no CER)"


def _audio_duration_s(wav_np, sample_rate=OUTPUT_SAMPLE_RATE) -> float:
    """Return audio duration in seconds from sample count and sample rate."""
    return float(np.asarray(wav_np).reshape(-1).shape[0]) / float(sample_rate)


def _log_perf_metrics(
    *,
    wall_s: float,
    audio_s: float,
    ttfc_s: float | None,
    label: str = "",
    compile_s: float | None = None,
):
    """Log wall-time, RTF, and related TTS speed metrics for a take."""
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


def _generate_one(tt, wrapped, real_len, cond_wav, spk_wav_tt, cfg):
    """Run one device generation, vocode, and onset post-processing."""
    # FULL-MODEL TRACE: the entire model runs inside ttnn traces — SETUP trace (on-device
    # conditioning mel + speaker encoder + prefill that seeds the KV cache), the DECODE-step trace
    # captured once and replayed per token, and the VOCODER trace. Needs the fixed KV-cache length:
    # prompt (cond perceiver latents + wrapped/padded text) + the decode budget.
    gen = cfg.generation
    prompt_len = NUM_LATENTS + wrapped.shape[1]
    max_seq = -(-(prompt_len + gen.max_tokens + 2) // TILE) * TILE
    wav_dev, codes, perf = tt.inference_fully_traced(
        wrapped,
        cond_wav,
        spk_wav_tt,
        max_seq,
        max_new_tokens=gen.max_tokens,
        temperature=gen.temperature,
        top_k=gen.top_k,
        top_p=gen.top_p,
        repetition_penalty=gen.repetition_penalty,
        min_new_tokens=gen.min_tokens,
        text_real_len=real_len,
    )
    wav_np = ttnn.to_torch(wav_dev).float().reshape(-1).numpy()  # [T_out]
    logger.info(
        f"  replay split: setup {perf['setup_replay_s']:.3f}s | decode {perf['decode_replay_s']:.3f}s "
        f"({codes.shape[1]} codes) | vocoder {perf['vocoder_replay_s']:.3f}s"
    )
    if not perf["stopped"]:
        logger.warning(
            f"pass ran out of its {gen.max_tokens}-code budget without reaching STOP — the tail may be "
            "unfinished or noisy. Shorten the text or raise GenerationConfig.max_tokens."
        )
    return (
        _postprocess(wav_np, cfg.audio_post),
        codes,
        float(perf["replay_s"]),
        bool(perf["stopped"]),
        float(perf["compile_s"]),
    )


def _generate_chunked(tt, chunks, cond_wav, spk_wav_tt, cfg):
    """Synthesize all chunks of a take from one warmup and trace capture."""
    gen = cfg.generation
    budget = cfg.chunking.chunk_max_tokens
    retries = cfg.chunking.chunk_retries
    text_len = chunks[0][1].shape[1]
    prompt_len = NUM_LATENTS + text_len
    max_seq = -(-(prompt_len + budget + 2) // TILE) * TILE
    session = tt.traced_session(
        cond_wav,
        spk_wav_tt,
        text_len,
        max_seq,
        budget,
        temperature=gen.temperature,
        top_k=gen.top_k,
        top_p=gen.top_p,
        repetition_penalty=gen.repetition_penalty,
        min_new_tokens=gen.min_tokens,
    )
    try:
        logger.info(
            f"captured setup/decode/vocoder traces ONCE in {session.compile_s:.1f}s "
            f"({text_len} text tokens, {budget}-code budget) — chunks now replay only"
        )
        out = []
        for j, (_, w, real) in enumerate(chunks):
            wav_t, codes_j, perf = session.run(w, real)
            replay_s = float(perf["replay_s"])
            # A pass that never reaches STOP ends mid-word and trails off into drone or noise.
            # Sampling is stochastic, so simply drawing again usually lands a clean take; only
            # text that genuinely does not fit the budget exhausts the retries.
            for attempt in range(retries):
                if perf["stopped"]:
                    break
                logger.warning(
                    f"  chunk {j + 1}/{len(chunks)} hit the {budget}-code cap without STOP "
                    f"(noisy tail) — retry {attempt + 1}/{retries}"
                )
                wav_t, codes_j, perf = session.run(w, real)
                replay_s += float(perf["replay_s"])
            if not perf["stopped"]:
                logger.warning(
                    f"  chunk {j + 1}/{len(chunks)} still hit the cap after {retries} retries; its tail may "
                    "be noisy. It likely needs more codes than the per-chunk budget allows."
                )
            logger.info(
                f"  chunk {j + 1}/{len(chunks)} replay split: setup {perf['setup_replay_s']:.3f}s | "
                f"decode {perf['decode_replay_s']:.3f}s ({codes_j.shape[1]} codes) | "
                f"vocoder {perf['vocoder_replay_s']:.3f}s"
            )
            wav_np = _postprocess(np.ascontiguousarray(wav_t.numpy(), dtype="float32"), cfg.audio_post)
            out.append((wav_np, codes_j, replay_s, bool(perf["stopped"])))
        return out, session.compile_s
    finally:
        session.close()


def _take_on_device(sd, ref_decoder_full, chunks, cond_wav, spk_wav, cfg, seed_offset):
    """Open a fresh device, synthesize the full take, then close the device."""
    # Both branches capture setup/decode/vocoder traces, so the trace region is unconditional.
    device = ttnn.open_device(
        device_id=cfg.device_id, l1_small_size=cfg.l1_small_size, trace_region_size=cfg.session_trace_region
    )
    try:
        tt = TtXtts(device, sd, ref_decoder_full)
        spk_wav_tt = ttnn.from_torch(
            spk_wav.reshape(1, -1).float(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.float32
        )
        if cfg.generation.seed is not None:
            # distinct-but-reproducible-ish seed per take (ttnn sampling isn't bit-exact across
            # runs regardless, so takes differ even without this).
            ttnn.manual_seed(cfg.generation.seed + seed_offset, device=device)
        if len(chunks) == 1:
            wav_np, codes, dt, stopped, compile_s = _generate_one(
                tt, chunks[0][1], chunks[0][2], cond_wav, spk_wav_tt, cfg
            )
            return [(wav_np, codes, dt, stopped)], compile_s
        return _generate_chunked(tt, chunks, cond_wav, spk_wav_tt, cfg)
    finally:
        ttnn.close_device(device)


def main():
    """CLI entrypoint for the TTNN XTTS demo."""
    ap = argparse.ArgumentParser(description="XTTS-v2 on-device text-to-speech demo")
    ap.add_argument("--text", default=DEMO.text)
    ap.add_argument(
        "--ref-audio",
        default=DEMO.ref_audio,
        help="voice to clone: local WAV path, coqui-ai/TTS test clip name (LJ001-0001.wav, or "
        "'LJ001-0001.wav+LJ001-0003.wav+...' to concatenate clips into a longer reference), "
        "or HF sample name (en_sample.wav).",
    )
    ap.add_argument(
        "--min-tokens",
        type=int,
        default=GENERATION.min_tokens,
        help="STOP-suppression floor in audio codes. 0 (default) = disabled, matches HF. "
        "-1 = auto (~2x the wrapped text length). Raise it if a LONG prompt is only partly "
        "spoken; leave it at 0 for short prompts, where a floor makes the model ramble.",
    )
    ap.add_argument("--output", default=DEMO.output, help="where to write the generated WAV")
    ap.add_argument(
        "--write-torch-ref",
        action="store_true",
        default=DEMO.write_torch_ref,
        help="also write the CPU-torch reference audio for the same codes, for A/B",
    )
    args = ap.parse_args()

    # Only --text / --ref-audio / --min-tokens / --output / --write-torch-ref are exposed; every
    # other setting comes from config.DEMO, so the demo runs full-model-traced with no other knobs.
    cfg = replace(
        DEMO,
        text=args.text,
        ref_audio=args.ref_audio,
        output=args.output,
        write_torch_ref=args.write_torch_ref,
        generation=replace(DEMO.generation, min_tokens=args.min_tokens),
    )

    from scipy.signal import resample_poly

    t_wall0 = time.time()  # end-to-end wall clock (weights + audio prep + generation + write)
    logger.info("loading XTTS-v2 weights ...")
    sd = load_xtts_state_dict()

    # Inputs: reference audio (22.05 kHz for conditioning, 16 kHz for the speaker encoder) + text.
    # The 80-mel is computed ON DEVICE (TtConditioningMel), so the device path takes the raw wav.
    wav = _load_audio_22k(cfg.ref_audio, cfg.ref_seconds)
    g = math.gcd(SPK_SR, MEL_SR)
    # Speaker path is capped independently — see DemoConfig.spk_seconds.
    spk_src = wav[0].numpy()[: MEL_SR * cfg.spk_seconds]
    spk_wav = torch.from_numpy(resample_poly(spk_src, SPK_SR // g, MEL_SR // g).astype("float32")).unsqueeze(0)
    # The GPT is conditioned on the WHOLE reference (coqui get_gpt_cond_latents): every
    # gpt_cond_chunk_len window up to gpt_cond_len is encoded and the style embeddings averaged.
    cond_windows = chunk_wav(wav)
    logger.info(
        f"reference audio: {wav.shape[-1] / MEL_SR:.2f}s loaded | conditioning on "
        f"{sum(c.shape[-1] for c in cond_windows) / MEL_SR:.2f}s as {len(cond_windows)} window(s) "
        f"(gpt_cond_len {GPT_COND_LEN_SEC}s, chunk {GPT_COND_CHUNK_SEC}s) | speaker embedding on "
        f"{spk_wav.shape[-1] / SPK_SR:.2f}s"
    )

    # Text that fits ONE pass stays a single pass (the fast path); only text over the single-pass
    # budget is split — at sentence boundaries, then at commas inside a sentence too long for a
    # pass — into several passes that are joined back together.
    chunk_texts = _split_into_chunks(cfg.text, cfg.language)
    wrapped_chunks = [
        (clean, wrap_text_ids(preprocess_text(clean, lang=cfg.language)))
        for clean in (re.sub(SENTENCE_FINAL_PUNCT_RE, "", ct.strip()) for ct in chunk_texts)
    ]
    # Pad to a COMMON length (not just each to its own tile): chunks are replayed off one capture,
    # whose prompt_len — and so the whole KV geometry — is fixed at capture time. The padding is
    # STOP_TEXT_TOKEN, which the model already reads as end-of-text, so a chunk padded past its own
    # length behaves exactly as it did when padded only to its own tile.
    pad_to = -(-max(w.shape[1] for _, w in wrapped_chunks) // TILE) * TILE
    # The real length rides along: the padding is masked out of decode attention, so a short
    # chunk stops on time instead of padding its way to the code cap.
    chunks = [
        (clean, F.pad(w, (0, pad_to - w.shape[1]), value=STOP_TEXT_TOKEN), w.shape[1]) for clean, w in wrapped_chunks
    ]
    if len(chunks) == 1:
        logger.info(f"text fits ONE pass ({chunks[0][2]} tokens): {chunks[0][0]!r}")
    else:
        logger.info(f"text exceeds the single-pass budget -> CHUNKED into {len(chunks)} passes:")
        for i, (clean, _, real) in enumerate(chunks):
            logger.info(f"  [{i + 1}/{len(chunks)}] {real:3d} tokens  {clean!r}")

    # Resolve the STOP-suppression floor. Auto (negative) scales with the text, clamped below the
    # code budget, so a longer prompt is protected from stopping short while a short one isn't
    # forced to ramble. 0 disables (HF default). One value for the whole take: the chunks share a
    # padded length, and a chunked take bakes the floor into its capture.
    budget = cfg.generation.max_tokens if len(chunks) == 1 else cfg.chunking.chunk_max_tokens
    floor = (
        int(cfg.generation.min_tokens_auto_factor * pad_to)
        if cfg.generation.min_tokens < 0
        else cfg.generation.min_tokens
    )
    cfg = replace(cfg, generation=replace(cfg.generation, min_tokens=max(0, min(floor, budget - 1))))
    logger.info(f"min audio codes before STOP allowed: {cfg.generation.min_tokens} (0 = disabled)")

    reference = XttsReference(sd)  # supplies decoder/speaker/mel weights (and optional A/B wav)

    gen = cfg.generation
    mode = (
        "greedy"
        if gen.temperature <= 0
        else f"sampled (temp={gen.temperature}, top_k={gen.top_k}, top_p={gen.top_p}, rep={gen.repetition_penalty})"
    )
    n = max(1, gen.num_outputs)
    logger.info(f"generating on device [{mode}], up to {budget} codes per pass, {n} take(s) ...")

    # Each take runs on its own freshly-opened device (see _take_on_device); best-of-N keeps
    # the lowest-CER take. A single take (default) is just one open/generate/close.
    wav_tt, codes, best_score, best_detail = None, None, None, None
    best_wall_s, best_ttfc_s, best_compile_s = None, None, None
    gap = np.zeros(int(cfg.audio_post.chunk_gap_seconds * OUTPUT_SAMPLE_RATE), dtype="float32")
    for i in range(n):
        t_take0 = time.time()
        # The whole take on ONE device: a single pass generates once, a chunked take captures once
        # and replays every chunk off that capture.
        results, compile_s = _take_on_device(sd, reference.decoder_full, chunks, wav, spk_wav, cfg, i)
        pieces, code_parts, stops = [], [], []
        dt = 0.0
        ttfc_s = results[0][2]  # first chunk/pass replay → analogue of time-to-first-chunk
        for j, (wav_j, codes_j, dt_j, stopped_j) in enumerate(results):
            dt += dt_j
            pieces.append(wav_j.astype("float32"))
            code_parts.append(codes_j)
            stops.append(stopped_j)
            nc = codes_j.shape[1]
            audio_j_s = _audio_duration_s(wav_j)
            rtf_j = dt_j / audio_j_s if audio_j_s > 0 else float("inf")
            if len(chunks) > 1:
                logger.info(
                    f"  take {i + 1}/{n} chunk {j + 1}/{len(chunks)}: {nc} codes "
                    f"({'stop' if stopped_j else 'max'}), "
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
        stopped = all(stops)
        audio_s = _audio_duration_s(wav_i)
        score, detail = _score_take(wav_i, cfg.text, codes_i, cfg.scoring) if n > 1 else (0.0, "")
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

    os.makedirs(os.path.dirname(os.path.abspath(cfg.output)), exist_ok=True)
    sf.write(cfg.output, wav_tt, OUTPUT_SAMPLE_RATE)
    logger.info(f"wrote device audio -> {os.path.abspath(cfg.output)}")
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

    if cfg.write_torch_ref:
        # A/B on the SAME codes the best device take produced (teacher-forced), so the two WAVs
        # are the same utterance — not an independent greedy run (which would collapse). Runs on
        # host (CPU torch), so no device is needed here (torch reference uses the host wav_to_mel).
        # Single-pass A/B only: with several chunks the codes are the concatenation of all passes,
        # so they no longer correspond to one text prompt.
        if len(chunks) > 1:
            logger.warning("--write-torch-ref uses chunk 1's text against all chunks' codes; A/B is approximate")
        cond_mel = wav_to_mel(wav, sd["mel_stats"].cpu())  # host 80-mel [1, 80, s] for the torch reference
        wav_ref = reference.wav_from_codes(chunks[0][1], cond_mel, spk_wav, codes[0].tolist())
        ref_path = cfg.output.replace(".wav", "_reference.wav")
        sf.write(ref_path, wav_ref.reshape(-1).numpy(), OUTPUT_SAMPLE_RATE)
        logger.info(f"wrote torch reference audio (same codes) -> {os.path.abspath(ref_path)}")


if __name__ == "__main__":
    main()
