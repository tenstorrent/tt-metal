#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Kokoro-82M on Tenstorrent: PL-BERT `bert_encoder` linear runs on device; other blocks on PyTorch.

Phonemes and voice embeddings use upstream `KPipeline(model=False)` (no second full weight load
for G2P / voicepack), matching `tests/test_reference_vs_official.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch
from loguru import logger

import ttnn


def main() -> int:
    parser = argparse.ArgumentParser(description="Kokoro-82M TT bring-up: hybrid PL-BERT projection on device")
    parser.add_argument(
        "--text", type=str, default="Hello from Tenstorrent. This is Kokoro on TT.", help="Text to speak."
    )
    parser.add_argument("--voice", type=str, default="af_heart", help="Voice name (upstream VOICES).")
    parser.add_argument("--lang-code", type=str, default="a", help="Language code (e.g. 'a' en-us).")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed multiplier.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=("cpu", "cuda"),
        help="Torch device for predictor/decoder. Default: auto.",
    )
    parser.add_argument("--output", type=str, default="kokoro_tt.wav", help="Output WAV path.")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Hugging Face repo id (default: KokoroConfig.repo_id).",
    )
    args = parser.parse_args()

    try:
        from kokoro import KPipeline
    except ImportError:
        logger.error('Install upstream kokoro: pip install "kokoro>=0.9.2"')
        return 2

    from models.demos.kokoro.reference import KokoroConfig
    from models.demos.kokoro.tt.kokoro_tt_hybrid_model import KokoroTtHybridFull

    torch_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    repo_id = args.repo_id or KokoroConfig.repo_id

    logger.info(f"Torch blocks device={torch_device}, repo_id={repo_id}")

    pipe = KPipeline(lang_code=args.lang_code, model=False)
    results = list(pipe(args.text, voice=args.voice, speed=args.speed))
    if not results:
        logger.error("Pipeline produced no chunks (check espeak-ng / text).")
        return 3

    pack = pipe.load_voice(args.voice)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mesh_shape = ttnn.MeshShape(1, 1)
    logger.info(f"Opening mesh device {mesh_shape}")
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape)
    try:
        model = KokoroTtHybridFull(mesh_device, repo_id=repo_id, torch_device=torch_device)
        wave_chunks: list[torch.Tensor] = []
        for chunk_idx, result in enumerate(results):
            phonemes = result.phonemes
            if not phonemes:
                logger.warning(f"Skipping empty phonemes chunk index={chunk_idx}")
                continue
            ref_s = pack[len(phonemes) - 1].to(torch_device)
            if ref_s.dim() == 1:
                ref_s = ref_s.unsqueeze(0)
            out = model(phonemes=phonemes, ref_s=ref_s, speed=args.speed)
            wave_chunks.append(out.audio.detach().cpu().flatten())
            logger.info(f"Chunk {chunk_idx}: phoneme_len={len(phonemes)} samples={wave_chunks[-1].numel()}")

        if not wave_chunks:
            logger.error("No audio produced from pipeline chunks.")
            return 3

        audio = torch.cat(wave_chunks, dim=0).numpy()
    finally:
        ttnn.close_mesh_device(mesh_device)

    sr = KokoroConfig.sample_rate_hz
    sf.write(str(out_path), audio, sr)
    logger.info(f"Wrote {out_path.resolve()} samples={audio.shape[-1]} sr={sr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
