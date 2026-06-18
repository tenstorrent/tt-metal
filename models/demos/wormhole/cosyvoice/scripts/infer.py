#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone inference script for CosyVoice on TT hardware.

This script provides a more detailed inference interface compared to demo.py,
with support for batch processing, timing, and output in various formats.

Usage:
    # Single inference
    python models/demos/wormhole/cosyvoice/scripts/infer.py \\
        --text "Hello world"

    # Batch inference from file
    python models/demos/wormhole/cosyvoice/scripts/infer.py \\
        --input_file texts.txt --output_dir outputs/

    # With pretrained weights
    python models/demos/wormhole/cosyvoice/scripts/infer.py \\
        --text "Hello" --weights /path/to/cosyvoice.pt
"""

import argparse
import json
import os
import time

import torch
from loguru import logger

import ttnn


def parse_args():
    parser = argparse.ArgumentParser(description="CosyVoice inference")
    parser.add_argument("--text", type=str, default=None, help="Input text")
    parser.add_argument("--input_file", type=str, default=None, help="Input text file (one per line)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--mode", type=str, default="sft", choices=["sft", "zero_shot", "cross_lingual", "instruct"])
    parser.add_argument("--weights", type=str, default=None, help="Path to pretrained weights")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--device", type=int, default=0, help="TT device ID")
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize device
    device = ttnn.open_device(device_id=args.device)
    logger.info(f"Using device: {device}")

    try:
        from models.demos.wormhole.cosyvoice.tt.pipeline import TtCosyVoicePipeline
        from models.demos.wormhole.cosyvoice.tt.model_config import CosyVoiceModelConfig
        from models.demos.wormhole.cosyvoice.demo.demo import _load_or_create_state_dict

        config = CosyVoiceModelConfig()
        state_dict = _load_or_create_state_dict(config)

        if args.weights and os.path.exists(args.weights):
            state_dict = torch.load(args.weights, map_location="cpu", weights_only=True)
            logger.info(f"Loaded weights from {args.weights}")

        pipeline = TtCosyVoicePipeline(device, config, state_dict)

        # Get input texts
        texts = []
        if args.text:
            texts.append(args.text)
        if args.input_file:
            with open(args.input_file, "r") as f:
                texts.extend([line.strip() for line in f if line.strip()])

        if not texts:
            logger.error("No input text provided")
            return

        # Run inference
        os.makedirs(args.output_dir, exist_ok=True)
        results = []

        for i, text in enumerate(texts):
            logger.info(f"[{i+1}/{len(texts)}] Generating: {text[:50]}...")

            start = time.time()
            audio = pipeline.tts(text=text, mode=args.mode, language=args.language)
            elapsed = time.time() - start

            # Save audio
            import scipy.io.wavfile as wav
            output_path = os.path.join(args.output_dir, f"output_{i:04d}.wav")
            audio_int16 = (audio.squeeze().clamp(-1, 1) * 32767).short()
            wav.write(output_path, 24000, audio_int16.numpy())

            results.append({
                "index": i,
                "text": text,
                "audio_path": output_path,
                "duration_s": audio.shape[-1] / 24000,
                "inference_time_s": round(elapsed, 2),
                "rtf": round(elapsed / (audio.shape[-1] / 24000), 3),
            })

            logger.info(f"  -> Saved to {output_path} ({results[-1]['duration_s']:.1f}s audio in {elapsed:.1f}s)")

        # Save results summary
        summary_path = os.path.join(args.output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Summary saved to {summary_path}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
