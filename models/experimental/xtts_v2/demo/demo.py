#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
XTTS-v2 demo — zero-shot voice-cloning text-to-speech on Tenstorrent hardware.

Hardware: single Wormhole N150 chip.
Models:   conditioning encoder + Perceiver, speaker encoder, GPT (Metal-Traced decode)
          and HiFi-GAN vocoder all on TTNN; tokenizer/mel front-end + sampling on host.

Usage:
    python -m models.experimental.xtts_v2.demo.demo "Text to speak." --ref /path/ref.wav
    python -m models.experimental.xtts_v2.demo.demo "Text to speak." --ref ref.wav \\
        --out out.wav --seed 0

The reference clip (--ref) is the voice to clone: a mono .wav/.flac/.ogg at any sample
rate (multi-channel is downmixed). Output is a 24 kHz wav.

Checkpoint resolution: --ckpt > $XTTS_CKPT > download coqui/XTTS-v2 from the HF hub.
"""

import argparse
import os
import time

from models.experimental.xtts_v2.frontend import load_reference_audio
from models.experimental.xtts_v2.tt.ttnn_xtts_model import OUTPUT_SR, XttsV2

# ── Public API ─────────────────────────────────────────────────────────────────


def run(text, ref, out="out.wav", seed=None, language="en", ckpt=None):
    """Speak `text` in the voice of the `ref` clip; write a 24 kHz wav to `out`."""
    import soundfile as sf

    ref_wav, ref_sr = load_reference_audio(ref)
    print(f"Reference: {ref} ({ref_wav.shape[-1] / ref_sr:.1f} s @ {ref_sr} Hz)")

    tts = XttsV2(ckpt_path=ckpt)
    try:
        bar = "=" * 72
        print(bar)
        print("Warming up (compile all programs + capture the Metal Traces) ...")
        print(bar)
        t0 = time.time()
        tts.warmup()
        t_warmup = time.time() - t0

        t0 = time.time()
        voice = tts.compute_voice(ref_wav, ref_sr)
        t_voice = time.time() - t0

        print(f"\nText: {text!r}  (seed={seed}, language={language})")
        t0 = time.time()
        wav = tts.generate(text, voice, language=language, seed=seed)
        t_gen = time.time() - t0

        if wav.shape[-1] == 0:  # generate()'s empty-audio contract: first sampled code was STOP
            print("\nThe model produced no audio for this text/seed (STOP sampled first); try another --seed.")
            return None

        out = os.path.abspath(out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        sf.write(out, wav[0, 0].numpy(), OUTPUT_SR)

        tm = tts.last_timings
        dur = wav.shape[-1] / OUTPUT_SR
        print(f"\n  → {out}  ({dur:.2f} s of audio)")
        print("\nTimings:")
        print(f"  warmup           {t_warmup:8.2f} s   (one-time: compile + trace capture)")
        print(f"  compute_voice    {t_voice:8.2f} s   (once per speaker)")
        print(f"  prefill          {tm['prefill_s'] * 1000:8.1f} ms  (prompt = {tm['prefix_tokens']} tokens)")
        print(
            f"  decode           {tm['decode_s']:8.2f} s   "
            f"({tm['codes']} codes, {tm['decode_ms_per_token']:.1f} ms/token)"
        )
        print(f"  vocoder          {tm['vocoder_s']:8.2f} s   (traced HiFi-GAN at its length bucket)")
        print(f"  generate total   {t_gen:8.2f} s   ({dur / t_gen:.1f}x real-time)")

        return out
    finally:
        tts.close()


# ── CLI ────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="XTTS-v2 — voice-cloning text-to-speech on Wormhole N150",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python -m models.experimental.xtts_v2.demo.demo "Hello there." --ref my_voice.wav
  python -m models.experimental.xtts_v2.demo.demo "Hello there." --ref ref.wav --out hi.wav --seed 0
""",
    )
    parser.add_argument("text", help="Text to speak (English)")
    parser.add_argument("--ref", required=True, help="Reference voice clip (.wav/.flac/.ogg)")
    parser.add_argument("--out", default="out.wav", help="Output wav path (default: out.wav)")
    parser.add_argument("--seed", type=int, default=None, help="Sampling seed (default: random)")
    parser.add_argument(
        "--language", default="en", help="Language code (default: en; see frontend.SUPPORTED_LANGUAGES)"
    )
    parser.add_argument("--ckpt", default=None, help="XTTS-v2 model.pth (default: $XTTS_CKPT, else HF hub)")
    args = parser.parse_args()

    run(
        text=args.text,
        ref=args.ref,
        out=args.out,
        seed=args.seed,
        language=args.language,
        ckpt=args.ckpt,
    )


if __name__ == "__main__":
    main()
