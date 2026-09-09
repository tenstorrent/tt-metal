#!/usr/bin/env python
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generate one song on the chip and record the qualitative statistics (stage 06).

Writes ``generated/<name>.wav`` (float32 stereo 44.1 kHz), ``generated/<name>.codes.pt`` (the ``[F, 8]`` frame codes)
and ``doc/pipeline/qualitative/<name>.json`` (audio statistics from ``tt/audio_metrics.py``, code repetition
statistics, timings, DRAM usage). The wav itself is never committed (``generated/`` is gitignored).

    source ~/mm3-bringup/common.sh && cd $MM3_WT
    with_hw_lock timeout 3600 $MM3_PY $MM3_MODEL_DIR/scripts/generate_song.py --preset golden --seed 7 --duration 60
    with_hw_lock timeout 3600 $MM3_PY $MM3_MODEL_DIR/scripts/generate_song.py --preset techno --seed 7 --duration 60
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import soundfile as sf
import torch
from loguru import logger

import ttnn
from models.autoports.minimaxai_minimax_music3.tt.audio_metrics import audio_stats, code_stats, summarize_for_log

MODEL_DIR = Path(__file__).resolve().parents[1]
GENERATED = MODEL_DIR / "generated"
QUAL_DIR = MODEL_DIR / "doc" / "pipeline" / "qualitative"
os.environ.setdefault("TT_DIT_CACHE_DIR", str(GENERATED / "tt_dit_cache"))

PRESETS = {
    "golden": (
        "Genre: acoustic pop. BPM: 96. Key: C major. Warm and intimate, building gently into the chorus. "
        "Vocals: soft female lead, close and breathy, light stacked harmonies in the chorus. "
        "Arrangement: fingerpicked guitar and soft piano; brushed drums and upright bass enter in the chorus.",
        "[verse]\nMorning light filtering through the pine\nEvery quiet street is yours and mine\n[chorus]\nSoftly the world begins to breathe",
    ),
    "techno": (
        "Genre: melodic techno. BPM: 126. Key: F minor. Dark, driving and hypnotic, with a long build into a heavy drop. "
        "Vocals: processed male vocal chops, sparse. Arrangement: punchy four-on-the-floor kick, rolling sub bass, "
        "arpeggiated analog synths, sidechained pads, white-noise risers.",
        "[intro]\nSteel and neon\n[verse]\nWe move like shadows under strobe light\nEvery heartbeat locked to the night\n[drop]\nLet the bass take over",
    ),
    "short": (
        "Genre: acoustic pop. BPM: 96. Key: C major. A short intimate vocal phrase over one guitar.",
        "[verse]\nMorning light through the pine",
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS), default="golden")
    ap.add_argument("--prompt")
    ap.add_argument("--lyrics")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--name")
    ap.add_argument("--threads", type=int, default=max(8, (os.cpu_count() or 8) - 4))
    ap.add_argument("--trace-region", type=int, default=int(os.environ.get("MM3_TRACE_REGION_SIZE", 90_000_000)))
    args = ap.parse_args()
    prompt, lyrics = PRESETS[args.preset]
    prompt = args.prompt or prompt
    lyrics = args.lyrics or lyrics
    name = args.name or f"{args.preset}_seed{args.seed}_{int(args.duration)}s"
    torch.set_num_threads(args.threads)

    from models.autoports.minimaxai_minimax_music3.tt.pipeline import MiniMaxMusic3Pipeline

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=args.trace_region)
    mesh.enable_program_cache()
    try:
        pipe = MiniMaxMusic3Pipeline.load(mesh)
        out = pipe.generate(
            prompt, lyrics, audio_duration=args.duration, seed=args.seed, num_inference_steps=args.steps
        )
        GENERATED.mkdir(exist_ok=True)
        QUAL_DIR.mkdir(parents=True, exist_ok=True)
        wav_path = GENERATED / f"{name}.wav"
        sf.write(wav_path, out["audio"].T, out["sampling_rate"], subtype="FLOAT")
        torch.save(
            {"codes": out["codes"], "frame0_codes": out["frame0_codes"], "seed": out["seed"]},
            GENERATED / f"{name}.codes.pt",
        )
        stats = audio_stats(out["audio"], out["sampling_rate"])
        codes = code_stats(out["codes"])
        record = {
            "name": name,
            "preset": args.preset,
            "prompt": prompt,
            "lyrics": lyrics,
            "seed": out["seed"],
            "audio_duration_requested": args.duration,
            "num_inference_steps": args.steps,
            "frames": out["frames"],
            "stopped_by": out["stopped_by"],
            "prompt_len": out["prompt_len"],
            "chunk_starts": out["chunk_starts"],
            "latent_lengths": [int(t.shape[-1]) for t in out["latents"]],
            "audio_seconds": out["audio"].shape[-1] / out["sampling_rate"],
            "timings": out["timings"],
            "audio_stats": stats,
            "code_stats": codes,
            "load_log": pipe.load_log,
            "wav": str(wav_path.relative_to(MODEL_DIR)),
            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "loadavg_1m": os.getloadavg()[0],
        }
        (QUAL_DIR / f"{name}.json").write_text(json.dumps(record, indent=2, default=float) + "\n")
        logger.info(f"audio: {summarize_for_log(stats)}")
        logger.info(f"codes: {summarize_for_log(codes)}")
        logger.info(f"timings: {json.dumps(out['timings'], default=float)}")
        logger.info(f"wrote {wav_path} and {QUAL_DIR / (name + '.json')}")
        pipe.release()
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
