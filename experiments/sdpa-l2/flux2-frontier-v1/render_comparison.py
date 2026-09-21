# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Render actual generated images, never illustrative or regenerated substitutes."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from PIL import Image, ImageDraw, ImageFont
import torch

ORDER = ["stock", "D", "C", "B", "A", "E", "F", "G"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifests = {}
    for path in args.manifests:
        data = json.loads(path.read_text())
        assert data["status"] == "completed"
        assert data["variant"] not in manifests
        manifests[data["variant"]] = (path, data)
    scores = json.loads(args.scores.read_text())
    score_map = {(r["variant"], r["prompt_id"], r["seed"]): r["clip_cosine"] for r in scores["rows"]}
    variants = [v for v in ORDER if v in manifests]
    exploratory = any(
        data.get("evaluation_mode") == "exploratory_repeatability_not_required" for _, data in manifests.values()
    )
    base = manifests["D"][1]
    for _, data in manifests.values():
        for field in (
            "steps",
            "width",
            "height",
            "guidance",
            "prompts",
            "seeds",
            "checkpoint",
            "embedding_cache",
            "conditioning",
            "model_repair",
        ):
            assert data[field] == base[field], f"Unpaired field {field}"
    font = ImageFont.load_default(size=19)
    small = ImageFont.load_default(size=16)
    results = []
    for prompt_id, prompt in enumerate(base["prompts"]):
        for seed in base["seeds"]:
            width, cell, label, top = 4 * 384, 384, 58, 100 if exploratory else 70
            sheet = Image.new("RGB", (width, top + ((len(variants) + 3) // 4) * (cell + label)), "white")
            draw = ImageDraw.Draw(sheet)
            draw.text(
                (12, 8),
                f"FLUX.2-dev | prompt {prompt_id}, seed {seed} | {base['steps']} steps | 1024x1024",
                fill="black",
                font=font,
            )
            # Wrap without truncating the prompt.
            import textwrap

            draw.multiline_text((12, 33), "\n".join(textwrap.wrap(prompt, 160)), fill="black", font=small)
            if exploratory:
                draw.text(
                    (12, 74),
                    "Exploratory: replay checks are reported separately; latent differences are not isolated attention error.",
                    fill="#922222",
                    font=small,
                )
            reference = None
            for v in ["D"] + [x for x in variants if x != "D"]:
                path, data = manifests[v]
                record = next(r for r in data["results"] if r["prompt_id"] == prompt_id and r["seed"] == seed)
                latent_path = path.parent / record["latents"]
                assert hashlib.sha256(latent_path.read_bytes()).hexdigest() == record["latents_sha256"]
                latents = torch.load(latent_path, weights_only=True).double()
                assert bool(torch.isfinite(latents).all())
                if v == "D":
                    reference = latents
                l2 = 100 * float(torch.linalg.vector_norm(latents - reference) / torch.linalg.vector_norm(reference))
                pcc = float(torch.corrcoef(torch.stack([latents.flatten(), reference.flatten()]))[0, 1])
                results.append(
                    dict(variant=v, prompt_id=prompt_id, seed=seed, latent_l2_vs_D_pct=l2, latent_pcc_vs_D=pcc)
                )
            for index, v in enumerate(variants):
                path, data = manifests[v]
                record = next(r for r in data["results"] if r["prompt_id"] == prompt_id and r["seed"] == seed)
                image_path = path.parent / record["image"]
                assert hashlib.sha256(image_path.read_bytes()).hexdigest() == record["image_sha256"]
                x, y = (index % 4) * cell, top + (index // 4) * (cell + label)
                with Image.open(image_path) as im:
                    sheet.paste(im.convert("RGB").resize((cell, cell), Image.Resampling.LANCZOS), (x, y))
                draw.text(
                    (x + 8, y + cell + 5), f"{v} | CLIP {score_map[v, prompt_id, seed]:.4f}", fill="black", font=font
                )
                row = next(r for r in results if (r["variant"], r["prompt_id"], r["seed"]) == (v, prompt_id, seed))
                draw.text(
                    (x + 8, y + cell + 31),
                    f"Latent L2 vs D: {row['latent_l2_vs_D_pct']:.2f}%",
                    fill="black",
                    font=small,
                )
            sheet.save(args.output / f"prompt{prompt_id}-seed{seed}.png")
    performance = {}
    for v, (_, data) in manifests.items():
        steps = [t * 1000 for result in data["results"] for t in result["denoising_step_seconds"]]
        performance[v] = dict(
            median_denoising_step_ms=statistics.median(steps),
            min_step_ms=min(steps),
            max_step_ms=max(steps),
            block_bench=data.get("block_bench", []),
            schedule=data["attention_schedule"],
            traced=data["traced"],
        )
    (args.output / "comparison.json").write_text(
        json.dumps(dict(latent_comparisons=results, performance=performance), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
