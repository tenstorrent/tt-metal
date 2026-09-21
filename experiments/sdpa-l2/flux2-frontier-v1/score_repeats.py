# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Score and display repeated stock generations, not alternative attention recipes."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from PIL import Image, ImageDraw, ImageFont
import torch

from models.tt_dit.tests.dataset_eval.clip_encoder import CLIPEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--suite-manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--decoded-subdir", default="decoded")
    args = parser.parse_args()
    folder = args.source / args.decoded_subdir
    manifest = json.loads((folder / "manifest.json").read_text())
    report = json.loads((args.source / "report.json").read_text())
    prompt = json.loads(args.suite_manifest.read_text())["prompts"][0]
    assert manifest["status"] == report["status"] == "completed"
    assert len(manifest["rows"]) == 6
    torch.set_num_threads(8)
    encoder = CLIPEncoder(clip_version="ViT-B/32", pretrained="openai", cache_dir=str(args.cache_dir))
    sheet = Image.new("RGB", (3 * 384, 70 + 2 * 445), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    draw.text((12, 8), "Stock repeatability control | identical prompt / seed 0 | 50 steps", fill="black", font=font)
    draw.text((12, 35), prompt, fill="black", font=font)
    rows = []
    for slot, record in enumerate(manifest["rows"]):
        path = folder / record["image"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["image_sha256"]
        with Image.open(path) as source:
            image = source.convert("RGB")
            score = float(encoder.get_clip_score(prompt, image).item())
            x, y = slot % 3 * 384, 70 + slot // 3 * 445
            sheet.paste(image.resize((384, 384), Image.Resampling.LANCZOS), (x, y))
        original = next(r for r in report["rows"] if r["index"] == record["index"])
        row = dict(**record, clip_cosine=score, latent_l2_vs_first_pct=original["l2_vs_first_pct"])
        rows.append(row)
        draw.text((x + 8, y + 390), f"{record['index']} {record['mode']} | CLIP {score:.4f}", fill="black", font=font)
        draw.text((x + 8, y + 417), f"Latent L2 vs first: {original['l2_vs_first_pct']:.2f}%", fill="black", font=font)
    result = dict(
        prompt=prompt,
        rows=rows,
        mean=statistics.mean(r["clip_cosine"] for r in rows),
        scope="Same stock denoiser repeated; " + manifest["decoder"],
    )
    with (folder / "clip-scores.json").open("x") as handle:
        json.dump(result, handle, indent=2)
    sheet.save(folder / "comparison.png")


if __name__ == "__main__":
    main()
