# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Paired raw CLIP cosine scores using tt-dit's existing evaluator."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics

from PIL import Image
import torch

from models.tt_dit.tests.dataset_eval.clip_encoder import CLIPEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", type=Path, nargs="*")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not args.prepare_only and (not args.manifests or args.output is None):
        parser.error("Scoring requires manifests and --output")
    if args.output is not None and args.output.exists():
        parser.error("Use a fresh output path")
    torch.set_num_threads(8)
    clip = CLIPEncoder(clip_version="ViT-B/32", pretrained="openai", cache_dir=str(args.cache_dir))
    if args.prepare_only:
        print("CLIP ViT-B/32 OpenAI weights ready", flush=True)
        return
    rows = []
    suite = None
    seen = set()
    for path in args.manifests:
        manifest = json.loads(path.read_text())
        if manifest["status"] != "completed":
            raise ValueError(f"Incomplete evaluation: {path}")
        config = {
            k: manifest[k]
            for k in (
                "checkpoint",
                "width",
                "height",
                "steps",
                "guidance",
                "prompts",
                "seeds",
                "vae_sdpa_chunks",
                "embedding_cache",
                "conditioning",
                "model_repair",
            )
        }
        if suite is not None and config != suite:
            raise ValueError(f"Unpaired evaluation configuration or embeddings: {path}")
        suite = config
        for result in manifest["results"]:
            identity = (manifest["variant"], result["prompt_id"], result["seed"])
            if identity in seen:
                raise ValueError(f"Duplicate image identity: {identity}")
            seen.add(identity)
            image_path = path.parent / result["image"]
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            if digest != result["image_sha256"]:
                raise ValueError(f"Image hash changed: {image_path}")
            prompt = manifest["prompts"][result["prompt_id"]]
            with Image.open(image_path) as image:
                score = float(clip.get_clip_score(prompt, image.convert("RGB")).item())
            row = dict(
                variant=manifest["variant"],
                prompt_id=result["prompt_id"],
                seed=result["seed"],
                prompt=prompt,
                image=str(image_path),
                image_sha256=digest,
                clip_cosine=score,
                manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    baseline = {(r["prompt"], r["seed"]): r["clip_cosine"] for r in rows if r["variant"] == "D"}
    for row in rows:
        ref = baseline.get((row["prompt"], row["seed"]))
        row["paired_delta_vs_D"] = None if ref is None else row["clip_cosine"] - ref
    report = dict(
        metric="Raw normalized image/text cosine; not scaled CLIPScore, not image-fidelity measurement",
        model="ViT-B/32",
        pretrained="openai",
        open_clip_version=importlib.metadata.version("open_clip_torch"),
        rows=rows,
        suite=suite,
        means={
            variant: statistics.mean(r["clip_cosine"] for r in rows if r["variant"] == variant)
            for variant in sorted({r["variant"] for r in rows})
        },
    )
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
