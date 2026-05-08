# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""
Kokoro-82M full TTNN demo: generates waveform audio.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch
from loguru import logger

import ttnn


def main() -> int:
    parser = argparse.ArgumentParser(description="Kokoro-82M full TTNN demo")
    parser.add_argument("--text", type=str, default="Hello from Tenstorrent Kokoro full TTNN demo.")
    parser.add_argument("--voice", type=str, default="af_heart")
    parser.add_argument("--lang-code", type=str, default="a")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="kokoro_ttnn.wav")
    args = parser.parse_args()

    try:
        from kokoro import KPipeline
    except ImportError:
        logger.error('Install upstream kokoro: pip install "kokoro>=0.9.2"')
        return 2

    from models.demos.kokoro.reference import KokoroConfig
    from models.demos.kokoro.tt.kokoro_tt_full_model import KokoroTtFull

    pipe = KPipeline(lang_code=args.lang_code, model=False)
    results = list(pipe(args.text, voice=args.voice, speed=args.speed))
    if not results:
        logger.error("Pipeline produced no chunks.")
        return 3
    phonemes = results[0].phonemes
    pack = pipe.load_voice(args.voice)
    ref_s = pack[len(phonemes) - 1].to("cpu")
    if ref_s.dim() == 1:
        ref_s = ref_s.unsqueeze(0)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        model = KokoroTtFull(mesh_device, repo_id=KokoroConfig.repo_id)
        torch.manual_seed(0)
        out = model(phonemes=phonemes, ref_s=ref_s, speed=args.speed)
        audio = out.audio.detach().cpu().numpy()
    finally:
        ttnn.close_mesh_device(mesh_device)

    sf.write(str(out_path), audio, KokoroConfig.sample_rate_hz)
    logger.info(f"Wrote {out_path.resolve()} samples={audio.shape[-1]} sr={KokoroConfig.sample_rate_hz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
