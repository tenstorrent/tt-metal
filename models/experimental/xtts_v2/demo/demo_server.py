#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
XTTS-v2 interactive server — REPL for speaking typed text in a cloned voice.

Loads + warms the model once (GPT decode is Metal-Traced), computes the voice from the
reference clip once, then every line of input generates a numbered 24 kHz wav.

Usage:
    python -m models.experimental.xtts_v2.demo.demo_server --ref /path/ref.wav
    python -m models.experimental.xtts_v2.demo.demo_server --ref ref.wav --seed 42

REPL commands (besides plain text to speak):
    \\seed N     set the base sampling seed (utterance i uses seed N+i)
    \\ref PATH   switch to a new reference clip (recomputes the voice)
    \\quit       exit (Ctrl-D / Ctrl-C also work)
"""

import argparse
import os
import time

from models.experimental.xtts_v2.frontend import load_reference_audio
from models.experimental.xtts_v2.tt.ttnn_xtts_model import OUTPUT_SR, XttsV2

DEFAULT_SEED = 42
OUTPUT_DIR = "outputs"


def main():
    parser = argparse.ArgumentParser(
        description="XTTS-v2 interactive server with Metal Trace",
    )
    parser.add_argument("--ref", required=True, help="Reference voice clip (.wav/.flac/.ogg)")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Base sampling seed (default: {DEFAULT_SEED}); utterance i uses seed + i so repeats vary",
    )
    parser.add_argument(
        "--language", default="en", help="Language code (default: en; see frontend.SUPPORTED_LANGUAGES)"
    )
    parser.add_argument("--ckpt", default=None, help="XTTS-v2 model.pth (default: $XTTS_CKPT, else HF hub)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help=f"Wav output directory (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        import readline  # noqa: F401  -- enables history/editing inside input()
    except ImportError:
        pass  # readline unavailable; continue without line-editing support

    import soundfile as sf

    # 1) Load + warmup (compile all programs + capture the traces), then the voice.
    tts = XttsV2(ckpt_path=args.ckpt)
    try:
        bar = "=" * 72
        print(bar)
        print("Warming up (first run is slow: compile all programs + trace captures)")
        print(bar)
        t0 = time.time()
        tts.warmup()
        print(f"  WARMUP: {time.time() - t0:.1f} s")

        ref_wav, ref_sr = load_reference_audio(args.ref)
        t0 = time.time()
        voice = tts.compute_voice(ref_wav, ref_sr)
        print(f"  VOICE:  {time.time() - t0:.1f} s  ({args.ref})\n")

        # 2) REPL loop.
        print(bar)
        print("Ready. Type text and press ENTER to speak it.")
        print("Commands: \\seed N   \\ref PATH   \\quit   (Ctrl-D / Ctrl-C also exit)")
        print(bar)

        seed = args.seed
        idx = 1
        while True:
            try:
                line = input(f"\ntext [{idx}]> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye")
                break

            if not line:
                continue

            if line.startswith("\\"):
                cmd, _, arg = line.partition(" ")
                if cmd == "\\quit":
                    print("bye")
                    break
                elif cmd == "\\seed":
                    try:
                        seed = int(arg)
                        print(f"  base seed = {seed}")
                    except ValueError:
                        print(f"  ERROR: \\seed needs an integer, got {arg!r}")
                elif cmd == "\\ref":
                    try:
                        ref_wav, ref_sr = load_reference_audio(arg.strip())
                        t0 = time.time()
                        voice = tts.compute_voice(ref_wav, ref_sr)
                        print(f"  VOICE: {time.time() - t0:.1f} s  ({arg.strip()})")
                    except Exception as e:
                        print(f"  ERROR: {e}")
                else:
                    print(f"  unknown command {cmd!r} (try \\seed N, \\ref PATH, \\quit)")
                continue

            try:
                path = os.path.join(args.output_dir, f"out_{idx}.wav")
                t_start = time.time()
                wav = tts.generate(line, voice, language=args.language, seed=seed + idx)
                if wav.shape[-1] == 0:  # generate()'s empty-audio contract: STOP sampled first
                    print("  no audio produced for this text/seed (STOP sampled first); retry or change \\seed")
                    idx += 1  # keep the seed advancing so a plain retry draws differently
                    continue
                sf.write(path, wav[0, 0].numpy(), OUTPUT_SR)
                elapsed = time.time() - t_start
                tm = tts.last_timings
                dur = wav.shape[-1] / OUTPUT_SR
                print(
                    f"  END-TO-END: {elapsed:.2f} s  |  {dur:.2f} s audio ({dur / elapsed:.1f}x RT), "
                    f"{tm['codes']} codes @ {tm['decode_ms_per_token']:.1f} ms/token  |  {os.path.abspath(path)}"
                )
                idx += 1
            except KeyboardInterrupt:
                print("\n  Interrupted during generation; shutting down.")
                break
            except Exception as e:
                print(f"\n  ERROR: {e}\n")
    finally:
        tts.close()


if __name__ == "__main__":
    main()
