#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Kokoro-82M (reference) CPU/GPU bring-up.

This is a *reference* (PyTorch) bring-up only: it validates that the upstream
Kokoro pipeline can run in this repo's environment and produces audio output.

Upstream references:
- https://huggingface.co/hexgrad/Kokoro-82M
- https://github.com/hexgrad/kokoro
"""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch
from loguru import logger


def main() -> int:
    parser = argparse.ArgumentParser(description="Kokoro-82M reference bring-up (CPU/CUDA via PyTorch)")
    parser.add_argument("--text", type=str, default="Hello from Tenstorrent. This is Kokoro.", help="Text to speak.")
    parser.add_argument("--voice", type=str, default="af_heart", help="Voice name (see VOICES.md upstream).")
    parser.add_argument("--lang-code", type=str, default="a", help="Language code (e.g. 'a' en-us, 'b' en-gb).")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed multiplier.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=("cpu", "cuda"),
        help="Force torch device. Default: auto-detect.",
    )
    parser.add_argument("--output", type=str, default="kokoro_reference.wav", help="Output WAV path.")
    args = parser.parse_args()

    from models.demos.kokoro.reference import KokoroPipelineReference, load_reference_model

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device={device} (torch.cuda.is_available()={torch.cuda.is_available()})")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kmodel = load_reference_model(repo_id="hexgrad/Kokoro-82M", device=device)
    logger.info(f"Loaded repo_id={kmodel.repo_id} params={kmodel.param_count():,}")
    pipeline = KokoroPipelineReference(lang_code=args.lang_code, repo_id=kmodel.repo_id, model=kmodel, device=device)

    # Upstream API yields (gs, ps, audio) chunks. Concatenate audio if multiple chunks.
    audios: list[torch.Tensor] = []
    last_ps = None
    for i, chunk in enumerate(pipeline.generate(args.text, voice=args.voice, speed=args.speed)):
        last_ps = chunk.phonemes
        if chunk.audio is None:
            continue
        logger.info(
            f"Chunk {i}: gs_len={len(chunk.graphemes)} ps_len={len(chunk.phonemes)} samples={chunk.audio.numel()}"
        )
        audios.append(chunk.audio.detach().cpu().flatten())

    if not audios:
        logger.error("No audio produced (empty generator output).")
        return 3

    audio_cat = torch.cat(audios, dim=0).numpy()

    # Kokoro sample rate is 24kHz per upstream docs.
    sf.write(str(out_path), audio_cat, 24000)
    logger.info(f"Wrote: {out_path.resolve()}")
    if last_ps is not None:
        logger.info(f"Last phonemes (ps): {last_ps[:120]}{'...' if len(last_ps) > 120 else ''}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
