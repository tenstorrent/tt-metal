# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""XTTS-v2 on-device demo: text + reference audio -> spoken WAV.

Runs the whole model on a Tenstorrent device (audio conditioning -> GPT KV-cache
autoregressive decode -> HiFi-GAN vocoder) and writes the generated 24 kHz audio
to a WAV file you can play.

Decoding **samples** by default (temperature 0.75 / top-k 50 / top-p 0.85 / rep 5.0,
coqui XTTS's natural settings), which self-terminates via the stop token and gives
natural prosody. Pass ``--temperature 0`` for deterministic greedy decode (the
validated PCC path), but note greedy tends to collapse/repeat at length instead of
stopping. The run logs how many codes were generated and whether it hit the stop
token — if the audio is only a word or two, generation stopped early (check the code
count) or ``--max-tokens`` is too low, not a vocoder truncation.

Pass ``--eval`` to also print CER (Whisper-large-v3), UTMOS, and ECAPA2 SECS on the
generated audio.

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
import time

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


def main():
    ap = argparse.ArgumentParser(description="XTTS-v2 on-device text-to-speech demo")
    ap.add_argument(
        "--text",
        default="voice synthesis has come a long way in recent years. modern systems can now generate natural sounding speech from text with remarkable accuracy. the key challenge is capturing the unique characteristics of a speaker's voice, including their tone, rhythm, pitch, and emotional expression.",
    )
    ap.add_argument("--lang", default="en")
    ap.add_argument("--ref-audio", default="en_sample.wav", help="local WAV path or HF sample name")
    ap.add_argument("--ref-seconds", type=int, default=6, help="reference audio length used for conditioning")
    ap.add_argument("--max-tokens", type=int, default=400, help="cap on audio codes (sampling usually stops earlier)")
    ap.add_argument("--temperature", type=float, default=0.75, help="sampling temperature; 0 = greedy")
    ap.add_argument("--top-k", type=int, default=50, help="top-k sampling cutoff")
    ap.add_argument("--top-p", type=float, default=0.85, help="top-p/nucleus cutoff (XTTS uses 0.85; 1.0 = off)")
    ap.add_argument("--repetition-penalty", type=float, default=5.0, help="repetition penalty (XTTS uses 5.0)")
    ap.add_argument("--output", default="generated/xtts_demo/xtts_demo.wav")
    ap.add_argument(
        "--eval",
        action="store_true",
        help="after generating, report objective metrics (CER via Whisper-large-v3, UTMOS, ECAPA2 SECS) "
        "on the device audio; downloads those models on first use",
    )
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
    wav = _load_audio_22k(args.ref_audio, args.ref_seconds)
    cond_mel = wav_to_mel(wav, sd["mel_stats"].cpu())  # host 80-mel [1, 80, s]
    g = math.gcd(SPK_SR, MEL_SR)
    spk_wav = torch.from_numpy(resample_poly(wav[0].numpy(), SPK_SR // g, MEL_SR // g).astype("float32")).unsqueeze(0)

    wrapped = wrap_text_ids(preprocess_text(args.text, lang=args.lang))
    pad = (-wrapped.shape[1]) % TILE
    if pad:
        wrapped = F.pad(wrapped, (0, pad), value=STOP_TEXT_TOKEN)
    logger.info(f"text: {args.text!r} -> {wrapped.shape[1]} tokens (wrapped/padded)")

    reference = XttsReference(sd)  # supplies decoder/speaker/mel weights (and optional A/B wav)

    # 64 KB L1_SMALL: the HiFi-GAN conv1d scratch grows with sequence length, and long
    # utterances (hundreds of codes) overflow the 32 KB used by the short-sequence tests.
    device = ttnn.open_device(device_id=0, l1_small_size=65536)
    try:
        tt = TtXtts(device, sd, reference.decoder_full)
        spk_wav_tt = ttnn.from_torch(
            spk_wav.reshape(1, -1, 1).float(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.float32
        )

        mode = (
            "greedy"
            if args.temperature <= 0
            else f"sampled (temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}, rep={args.repetition_penalty})"
        )
        logger.info(f"generating on device [{mode}], up to {args.max_tokens} codes ...")
        t0 = time.time()
        wav_tt_dev, codes = tt.inference(
            wrapped,
            cond_mel,
            spk_wav_tt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        wav_tt = ttnn.to_torch(wav_tt_dev).float().reshape(-1).numpy()  # [T_out]
        dt = time.time() - t0

        n_codes = codes.shape[1]
        stopped = n_codes < args.max_tokens
        logger.info(
            f"generated {n_codes} codes ({'hit stop token' if stopped else 'reached max'}) "
            f"-> {wav_tt.shape[0]} samples ({wav_tt.shape[0] / OUTPUT_SAMPLE_RATE:.2f}s audio) in {dt:.1f}s"
        )

        import soundfile as sf

        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        sf.write(args.output, wav_tt, OUTPUT_SAMPLE_RATE)
        logger.info(f"wrote device audio -> {os.path.abspath(args.output)}")

        if args.write_torch_ref:
            # A/B on the SAME codes the device produced (teacher-forced), so the two WAVs
            # are the same utterance — not an independent greedy run (which would collapse).
            wav_ref = reference.wav_from_codes(wrapped, cond_mel, spk_wav, codes[0].tolist())
            ref_path = args.output.replace(".wav", "_reference.wav")
            sf.write(ref_path, wav_ref.reshape(-1).numpy(), OUTPUT_SAMPLE_RATE)
            logger.info(f"wrote torch reference audio (same codes) -> {os.path.abspath(ref_path)}")

        if args.eval:
            # Objective quality metrics on the device audio (best-effort: each backend
            # is downloaded on first use; a missing one logs a skip).
            spk_np = spk_wav[0].numpy()  # 16 kHz reference-speaker audio (SECS target)
            logger.info("================ XTTS objective eval metrics ================")
            try:
                from models.experimental.xtts.eval.xtts_eval import compute_cer

                cer, hyp = compute_cer(wav_tt, OUTPUT_SAMPLE_RATE, args.text)
                logger.info(f"CER   (Whisper-large-v3, lower=better)       : {cer:.4f}")
                logger.info(f"        whisper transcript: {hyp!r}")
            except Exception as e:
                logger.warning(f"CER   skipped ({type(e).__name__}: {e})")
            try:
                from models.experimental.xtts.eval.xtts_eval import compute_utmos

                logger.info(
                    f"UTMOS (naturalness MOS 1-5, higher=better)   : {compute_utmos(wav_tt, OUTPUT_SAMPLE_RATE):.4f}"
                )
            except Exception as e:
                logger.warning(f"UTMOS skipped ({type(e).__name__}: {e})")
            try:
                from models.experimental.xtts.eval.xtts_eval import compute_secs

                secs = compute_secs(wav_tt, OUTPUT_SAMPLE_RATE, spk_np, SPK_SR)
                logger.info(f"SECS  (ECAPA2 speaker cos-sim, higher=better) : {secs:.4f}")
            except Exception as e:
                logger.warning(f"SECS  skipped ({type(e).__name__}: {e})")
            logger.info("============================================================")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
