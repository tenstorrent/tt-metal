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


def _generate_one(tt, wrapped, cond_wav, spk_wav_tt, args):
    """One full device generation + vocode + onset post-processing. Returns (wav_np, codes, dt)."""
    t0 = time.time()
    wav_dev, codes = tt.inference(
        wrapped,
        cond_wav,
        spk_wav_tt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        top_p=args.top_p,
        min_new_tokens=args.min_tokens_resolved,
    )
    wav_np = ttnn.to_torch(wav_dev).float().reshape(-1).numpy()  # [T_out]
    return _postprocess(wav_np), codes, time.time() - t0


def _take_on_device(sd, ref_decoder_full, wrapped, cond_wav, spk_wav, args, seed_offset):
    """Open a FRESH device, build the model, run one generation, close the device. The fp32
    HiFi-GAN vocoder exhausts L1_SMALL when several full generations share one device, so
    best-of-N isolates each take on its own device (same reason the tests use a per-test
    device fixture). Returns ``(wav_np, codes, dt)``."""
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
        default="Voice synthesis has come a long way, and modern systems can already generate natural sounding speech with remarkable accuracy.",
    )
    ap.add_argument("--lang", default="en")
    ap.add_argument(
        "--ref-audio",
        default="reference.wav",
        help="local WAV path or HF sample name. reference.wav scored best of the repo clips "
        "(UTMOS 4.28 / CER 0.025 / SECS 0.70); en_sample.wav is the portable HF fallback.",
    )
    ap.add_argument(
        "--ref-seconds",
        type=int,
        default=30,
        help="reference audio used for conditioning (coqui gpt_cond_len=30). Split into 4s chunks "
        "and averaged; a longer clean clip improves voice cloning. Short clips use a single window.",
    )
    ap.add_argument("--max-tokens", type=int, default=400, help="cap on audio codes (sampling usually stops earlier)")
    ap.add_argument(
        "--min-tokens",
        type=int,
        default=0,
        help="floor on generated audio codes — STOP is suppressed below it. Default 0 = disabled "
        "(matches HF; the [SPACE]-token fix removes the early-stop that this used to mask). Set "
        "-1 for auto (~2x wrapped text len) or an explicit count only if a take still stops short.",
    )
    ap.add_argument(
        "--num-outputs",
        type=int,
        default=1,
        help="number of takes to generate; keeps the best (lowest CER vs the text, or code-diversity "
        "if Whisper is unavailable). Default 1 matches HF/coqui num_gpt_outputs=1; set >1 to tame "
        "run-to-run variance at Nx time cost.",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.65,
        help="sampling temperature; 0 = greedy. 0.65 gives the most reliably-clean SINGLE take "
        "(lower CER, no tail garble) with num-outputs=1; raise toward coqui's 0.75 for more "
        "expressive prosody if you use best-of-N (--num-outputs>1) to reject the occasional bad draw.",
    )
    ap.add_argument("--top-k", type=int, default=50, help="top-k sampling cutoff")
    ap.add_argument(
        "--top-p",
        type=float,
        default=0.85,
        help="nucleus (top-p) cutoff; XTTS uses 0.85. 1.0 disables it. Improves text alignment/intelligibility.",
    )
    ap.add_argument(
        "--spk-seconds",
        type=int,
        default=8,
        help="reference audio for the speaker-embedding path (separate from --ref-seconds): the "
        "on-device speaker mel frontend reshapes samples in one shot and can't take very long "
        "audio; the d-vector doesn't need >~8s. Kept small while conditioning uses the full clip.",
    )
    ap.add_argument("--repetition-penalty", type=float, default=5.0, help="repetition penalty (XTTS uses 5.0)")
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed for on-device sampling (ttnn.manual_seed) so a run is reproducible; omit for random",
    )
    ap.add_argument("--output", default="generated/xtts_demo/xtts_demo.wav")
    ap.add_argument(
        "--write-torch-ref",
        action="store_true",
        help="also write the torch reference WAV for A/B (this does NOT set the voice; use --ref-audio for that)",
    )
    args = ap.parse_args()

    from scipy.signal import resample_poly

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
    clean_text = re.sub(r"[.!?]+\s*$", "", args.text.strip())
    wrapped = wrap_text_ids(preprocess_text(clean_text, lang=args.lang))
    pad = (-wrapped.shape[1]) % TILE
    if pad:
        wrapped = F.pad(wrapped, (0, pad), value=STOP_TEXT_TOKEN)
    logger.info(f"text: {clean_text!r} -> {wrapped.shape[1]} tokens (wrapped/padded)")

    # Resolve the STOP-suppression floor. Auto (-1) scales with the text (~2x the wrapped length),
    # clamped below max-tokens, so a longer prompt is protected from stopping short while a short one
    # isn't forced to ramble. 0 disables (HF default).
    if args.min_tokens < 0:
        args.min_tokens_resolved = min(int(2.0 * wrapped.shape[1]), args.max_tokens - 1)
    else:
        args.min_tokens_resolved = min(args.min_tokens, args.max_tokens - 1)
    logger.info(f"min audio codes before STOP allowed: {args.min_tokens_resolved} (0 = disabled)")

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
    for i in range(n):
        wav_i, codes_i, dt = _take_on_device(sd, reference.decoder_full, wrapped, wav, spk_wav, args, i)
        n_codes = codes_i.shape[1]
        stopped = n_codes < args.max_tokens
        score, detail = _score_take(wav_i, args.text, codes_i) if n > 1 else (0.0, "")
        logger.info(
            f"take {i + 1}/{n}: {n_codes} codes ({'stop' if stopped else 'max'}), "
            f"{wav_i.shape[0] / OUTPUT_SAMPLE_RATE:.2f}s, {dt:.1f}s" + (f" | {detail}" if detail else "")
        )
        if best_score is None or score < best_score:
            wav_tt, codes, best_score, best_detail = wav_i, codes_i, score, detail
    if n > 1:
        logger.info(f"selected best of {n} -> {best_detail}")

    import soundfile as sf

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    sf.write(args.output, wav_tt, OUTPUT_SAMPLE_RATE)
    logger.info(f"wrote device audio -> {os.path.abspath(args.output)}")

    if args.write_torch_ref:
        # A/B on the SAME codes the best device take produced (teacher-forced), so the two WAVs
        # are the same utterance — not an independent greedy run (which would collapse). Runs on
        # host (CPU torch), so no device is needed here (torch reference uses the host wav_to_mel).
        cond_mel = wav_to_mel(wav, sd["mel_stats"].cpu())  # host 80-mel [1, 80, s] for the torch reference
        wav_ref = reference.wav_from_codes(wrapped, cond_mel, spk_wav, codes[0].tolist())
        ref_path = args.output.replace(".wav", "_reference.wav")
        sf.write(ref_path, wav_ref.reshape(-1).numpy(), OUTPUT_SAMPLE_RATE)
        logger.info(f"wrote torch reference audio (same codes) -> {os.path.abspath(ref_path)}")


if __name__ == "__main__":
    main()
