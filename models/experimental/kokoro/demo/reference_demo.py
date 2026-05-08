#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Kokoro-82M PyTorch reference demo: generates waveform audio using repo-owned modules + HF weights.

This uses upstream `KPipeline(model=False)` for G2P/voicepack (no extra model weight load),
matching other demos/tests in this directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from loguru import logger


def main() -> int:
    parser = argparse.ArgumentParser(description="Kokoro-82M PyTorch reference demo")
    parser.add_argument("--text", type=str, default="Hello from Kokoro PyTorch reference demo.")
    parser.add_argument("--voice", type=str, default="af_heart")
    parser.add_argument("--lang-code", type=str, default="a")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=("cpu", "cuda"),
        help="Torch device for reference model. Default: auto.",
    )
    parser.add_argument("--output", type=str, default="kokoro_reference.wav")
    parser.add_argument("--disable-complex", action="store_true", help="Disable complex ops in decoder (if needed).")
    args = parser.parse_args()

    try:
        from kokoro import KPipeline
    except ImportError:
        logger.error('Install upstream kokoro: pip install "kokoro>=0.9.2"')
        return 2

    try:
        import soundfile as sf
    except ImportError:
        sf = None

    from models.demos.kokoro.reference import KokoroConfig
    from models.demos.kokoro.reference.kokoro_full_model import load_full_reference_from_huggingface

    torch_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using torch device={torch_device}")

    pipe = KPipeline(lang_code=args.lang_code, model=False)
    results = list(pipe(args.text, voice=args.voice, speed=args.speed))
    if not results:
        logger.error("Pipeline produced no chunks (check espeak-ng / text).")
        return 3

    phonemes = results[0].phonemes
    if not phonemes:
        logger.error("First chunk has empty phonemes.")
        return 3

    pack = pipe.load_voice(args.voice)
    ref_s = pack[len(phonemes) - 1].to(torch_device)
    if ref_s.dim() == 1:
        ref_s = ref_s.unsqueeze(0)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Instantiation prints the module (see kokoro_full_model.py loader).
    model = load_full_reference_from_huggingface(device=torch_device, disable_complex=args.disable_complex)

    torch.manual_seed(0)
    out = model(phonemes=phonemes, ref_s=ref_s, speed=args.speed)
    audio = out.audio.detach().cpu().numpy()

    sr = KokoroConfig.sample_rate_hz
    if sf is None:
        logger.info(f"Generated audio samples={audio.shape[-1]} sr={sr} (install soundfile to write wav).")
        return 0

    sf.write(str(out_path), audio, sr)
    logger.info(f"Wrote {out_path.resolve()} samples={audio.shape[-1]} sr={sr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
