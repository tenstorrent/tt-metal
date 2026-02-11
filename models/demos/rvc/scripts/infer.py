#!/usr/bin/env python3
import argparse
import os
import sys

import soundfile as sf

from rvc.vc.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RVC Nano inference from a wav file.")
    parser.add_argument("-i", "--input", required=True, help="Input audio path (wav).")
    parser.add_argument("-o", "--output", required=True, help="Output audio path (wav).")
    parser.add_argument("--speaker-id", type=int, default=0, help="Speaker ID (default: 0).")
    parser.add_argument("--f0-method", default="pm", choices=["pm", "crepe", "rmvpe"], help="F0 method.")
    parser.add_argument("--f0-up-key", type=int, default=0, help="Pitch shift in semitones.")
    parser.add_argument("--index-rate", type=float, default=0.75, help="Index rate (unused if no index).")
    parser.add_argument("--resample-sr", type=int, default=0, help="Target sample rate (0 keeps model rate).")
    parser.add_argument("--rms-mix-rate", type=float, default=0.25, help="RMS mix rate.")
    parser.add_argument("--protect", type=float, default=0.33, help="Protect rate.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not os.getenv("RVC_CONFIGS_DIR"):
        print("Error: RVC_CONFIGS_DIR is not set.", file=sys.stderr)
        return 2
    if not os.getenv("RVC_ASSETS_DIR"):
        print("Error: RVC_ASSETS_DIR is not set.", file=sys.stderr)
        return 2

    pipe = Pipeline(if_f0=True, version="v1", num="48k")
    audio = pipe.infer(
        args.input,
        speaker_id=args.speaker_id,
        f0_up_key=args.f0_up_key,
        f0_method=args.f0_method,
        index_rate=args.index_rate,
        resample_sr=args.resample_sr,
        rms_mix_rate=args.rms_mix_rate,
        protect=args.protect,
    )
    sf.write(args.output, audio, pipe.tgt_sr, subtype="PCM_16")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
