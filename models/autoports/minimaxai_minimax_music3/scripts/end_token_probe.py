# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Free-running device probe: how long does a short lyric run before the end token, and how does the end token's
probability evolve? Writes ``generated/end_token_probe_<seed>.json``.

    source ~/mm3-bringup/common.sh && cd $MM3_WT && \
    with_hw_lock timeout 3600 $MM3_PY $MM3_MODEL_DIR/scripts/end_token_probe.py --max-frames 9000 --seed 7
"""
import argparse
import json
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.autoports.minimaxai_minimax_music3.reference import hf_llm as R
from models.autoports.minimaxai_minimax_music3.tt.ar_generator import ARGenerator
from models.autoports.minimaxai_minimax_music3.tt.depth_decoder import DepthDecoder
from models.autoports.minimaxai_minimax_music3.tt.llm import MusicLLM

CAPTION = "Genre: acoustic pop. BPM: 96. Key: C major. A short intimate vocal phrase over one guitar."
LYRICS = "[verse]\nMorning light through the pine"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=9000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--caption", default=CAPTION)
    ap.add_argument("--lyrics", default=LYRICS)
    args = ap.parse_args()
    out_dir = Path(__file__).resolve().parents[1] / "generated"
    out_dir.mkdir(exist_ok=True)

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=90_000_000)
    mesh.enable_program_cache()
    try:
        llm = MusicLLM(mesh)
        depth = DepthDecoder.from_pretrained(mesh, R.weights_dir())
        ar = ARGenerator(llm, depth)
        try:
            t0 = time.perf_counter()
            out = ar.generate(
                args.caption, args.lyrics, max_frames=args.max_frames, seed=args.seed, collect_end_token_stats=True
            )
            wall = time.perf_counter() - t0
        finally:
            ar.release()
            llm.release()
    finally:
        ttnn.close_mesh_device(mesh)
    stats = out["end_token_stats"]
    ranks = [s["rank_conditional"] for s in stats]
    probs = [s["prob_sampling"] for s in stats]
    logger.info(
        f"seed {args.seed}: stopped_by={out['stopped_by']} after {out['frames']} frames ({out['frames'] / 25:.1f} s) in {wall:.0f}s; "
        f"end-token rank in conditional row: min {min(ranks)}, median {sorted(ranks)[len(ranks) // 2]}; "
        f"max sampling prob {max(probs):.4f}; frames with prob > 0: {sum(p > 0 for p in probs)}"
    )
    hist = torch.bincount(out["codes"][:, 0], minlength=16384)
    json.dump(
        {
            "caption": args.caption,
            "lyrics": args.lyrics,
            "seed": args.seed,
            "max_frames": args.max_frames,
            "stopped_by": out["stopped_by"],
            "frames": out["frames"],
            "wall_seconds": wall,
            "distinct_semantic_codes": int((hist > 0).sum()),
            "most_common_semantic_code": [int(hist.argmax()), float(hist.max()) / out["frames"]],
            "end_token_stats": stats,
            "timings": {k: v for k, v in out["timings"].items() if k != "per_frame"},
        },
        open(out_dir / f"end_token_probe_{args.seed}.json", "w"),
        indent=1,
    )


if __name__ == "__main__":
    main()
