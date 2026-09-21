# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Score saved, uncompressed sampled frames; CLIP is alignment, not video fidelity."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from PIL import Image
import torch

from models.tt_dit.tests.dataset_eval.clip_encoder import CLIPEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("--cache-dir", required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    encoder = CLIPEncoder(clip_version="ViT-B/32", pretrained="openai", cache_dir=args.cache_dir)
    rows = []
    config = None
    for variant in ("stock", "D", "C", "B", "E", "F", "G"):
        root = args.suite / variant
        manifest = json.loads((root / "manifest.json").read_text())
        assert manifest["status"] == "completed"
        current = {
            k: manifest[k]
            for k in ("checkpoint_revision", "width", "height", "frames", "steps", "seed", "guidance", "prompts")
        }
        assert config is None or current == config
        config = current
        for result in manifest["results"]:
            prompt_id = result["prompt_id"]
            scores = []
            for frame in result["sampled_frames"]:
                path = root / frame["file"]
                assert hashlib.sha256(path.read_bytes()).hexdigest() == frame["sha256"]
                with Image.open(path) as image:
                    scores.append(
                        float(encoder.get_clip_score(manifest["prompts"][prompt_id], image.convert("RGB")).item())
                    )
            row = dict(
                variant=variant,
                prompt_id=prompt_id,
                mean_clip=statistics.mean(scores),
                frame_clip=scores,
                pipeline_seconds=result["pipeline_seconds"],
            )
            rows.append(row)
            print("WAN_CLIP", json.dumps(row), flush=True)
    with (args.suite / "clip-scores.json").open("x") as handle:
        json.dump(
            dict(
                metric="OpenAI CLIP ViT-B/32 raw cosine, eight sampled frames; not temporal quality",
                config=config,
                rows=rows,
            ),
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
