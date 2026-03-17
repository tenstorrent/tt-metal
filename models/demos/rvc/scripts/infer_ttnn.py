# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os

import soundfile as sf

import ttnn
from models.demos.rvc.tt_impl.vc.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RVC Nano TTNN inference from a wav file.")
    parser.add_argument("-i", "--input", required=True, help="Input audio path (wav).")
    parser.add_argument("-o", "--output", required=True, help="Output audio path (wav).")
    parser.add_argument("--speaker-id", type=int, default=0, help="Speaker ID (default: 0).")
    parser.add_argument("--f0-method", default="pm", choices=["pm"], help="F0 method.")
    parser.add_argument("--f0-up-key", type=int, default=0, help="Pitch shift in semitones.")
    parser.add_argument("--index-rate", type=float, default=0.75, help="Index rate (unused if no index).")
    parser.add_argument("--resample-sr", type=int, default=0, help="Target sample rate (0 keeps model rate).")
    parser.add_argument("--rms-mix-rate", type=float, default=0.25, help="RMS mix rate.")
    parser.add_argument("--protect", type=float, default=0.33, help="Protect rate.")
    parser.add_argument("--device-id", type=int, default=0, help="TT device id.")
    parser.add_argument("--l1-small-size", type=int, default=65384, help="CreateDevice l1_small_size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.getenv("RVC_CONFIGS_DIR"):
        raise RuntimeError("RVC_CONFIGS_DIR is not set.")
    if not os.getenv("RVC_ASSETS_DIR"):
        raise RuntimeError("RVC_ASSETS_DIR is not set.")

    device = None
    try:
        device = ttnn.CreateDevice(device_id=args.device_id, l1_small_size=args.l1_small_size)
        pipe = Pipeline(tt_device=device, if_f0=True, version="v1", num="48k")
        import time

        for _ in range(2):
            start_time = time.time()
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
            end_time = time.time()
            print(f"Inference took {end_time - start_time:.2f} seconds.")
        sf.write(args.output, audio, pipe.tgt_sr, subtype="PCM_16")
    finally:
        if device is not None:
            ttnn.close_device(device)


if __name__ == "__main__":
    main()
