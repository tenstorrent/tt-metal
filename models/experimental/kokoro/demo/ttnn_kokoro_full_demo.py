# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Kokoro-82M full TTNN demo (PL-BERT + predictor + experimental ISTFTNet vocoder on device)."""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch
from loguru import logger

import ttnn


def main() -> int:
    parser = argparse.ArgumentParser(description="Kokoro-82M full TTNN (experimental) demo")
    parser.add_argument("--text", type=str, default="Hello from Tenstorrent Kokoro full TTNN.")
    parser.add_argument("--voice", type=str, default="af_heart")
    parser.add_argument("--lang-code", type=str, default="a")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="kokoro_experimental_ttnn.wav")
    args = parser.parse_args()

    try:
        from kokoro import KPipeline
    except ImportError:
        logger.error('Install upstream kokoro: pip install "kokoro>=0.9.2"')
        return 2

    from models.experimental.kokoro.reference import KokoroConfig
    from models.experimental.kokoro.tt.ttnn_kokoro_full_pipeline import KokoroFullTtnn

    pipe = KPipeline(lang_code=args.lang_code, model=False)
    results = list(pipe(args.text, voice=args.voice, speed=args.speed))
    if not results:
        logger.error("Pipeline produced no chunks.")
        return 3
    pack = pipe.load_voice(args.voice)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=24576)
    try:
        model = KokoroFullTtnn(device, repo_id=KokoroConfig.repo_id, disable_complex=True)
        wave_chunks: list[torch.Tensor] = []
        torch.manual_seed(0)
        for chunk_idx, result in enumerate(results):
            phonemes = result.phonemes
            if not phonemes:
                logger.warning(f"Skipping empty phonemes chunk index={chunk_idx}")
                continue
            ref_s = pack[len(phonemes) - 1].to("cpu")
            if ref_s.dim() == 1:
                ref_s = ref_s.unsqueeze(0)
            out = model(phonemes=phonemes, ref_s=ref_s, speed=args.speed, deterministic=True)
            wave_chunks.append(out.audio.detach().float().flatten())
            logger.info(f"Chunk {chunk_idx}: phoneme_len={len(phonemes)} samples={wave_chunks[-1].numel()}")

        if not wave_chunks:
            logger.error("No audio produced from pipeline chunks.")
            return 3
        audio = torch.cat(wave_chunks, dim=0).numpy()
    finally:
        ttnn.close_mesh_device(device)

    sf.write(str(out_path), audio, KokoroConfig.sample_rate_hz)
    logger.info(f"Wrote {out_path.resolve()} samples={audio.shape[-1]} sr={KokoroConfig.sample_rate_hz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
