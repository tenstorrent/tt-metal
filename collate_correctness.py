#!/usr/bin/env python3
"""Compute CLIP score + PSNR(vs baseline) for the Wan2.1 correctness sweep.

Reads PNGs saved by the instrumented sweep tests under outputs/correctness/<run_tag>/<BxHxW>.png.
- CLIP score: open_clip ViT-B-32 (openai), cosine(image, PROMPT) x 100. No reference.
- PSNR (dB): vs the baseline run for the same shape. Baseline run_tag default below.

Writes wan2_1_correctness_metrics.md and prints a summary, flagging low-CLIP / low-PSNR
outliers (candidate "bad image" configs).
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROMPT = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
ROOT = Path("outputs/correctness")
BASELINE = "4x4_sp0tp1_untraced"

# Heuristic thresholds for flagging (tuned after baseline is collected).
CLIP_FLAG = 22.0  # CLIP score below this => suspicious (blurry/misaligned)
PSNR_FLAG = 20.0  # PSNR vs baseline below this => meaningfully diverged


def psnr(ref: np.ndarray, test: np.ndarray) -> float | None:
    if ref.shape != test.shape:
        return None
    mse = np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(255.0) - 10.0 * np.log10(mse)


def main():
    if not ROOT.exists():
        print(f"no {ROOT}; run the instrumented sweep with WAN_RUN_TAG set first")
        sys.exit(1)

    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tok = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    with torch.no_grad():
        text_feat = model.encode_text(tok([PROMPT]))
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    def clip_score(img_path: Path) -> float:
        img = Image.open(img_path).convert("RGB")
        with torch.no_grad():
            feat = model.encode_image(preprocess(img).unsqueeze(0))
            feat = feat / feat.norm(dim=-1, keepdim=True)
            return float((feat @ text_feat.T).item() * 100.0)

    run_tags = sorted(d.name for d in ROOT.iterdir() if d.is_dir())
    # Load baseline images for PSNR reference.
    base_imgs = {}
    if BASELINE in run_tags:
        for p in (ROOT / BASELINE).glob("*.png"):
            base_imgs[p.stem] = np.array(Image.open(p).convert("RGB"))

    lines = ["# Wan2.1 correctness metrics", ""]
    lines.append(f"CLIP = open_clip ViT-B-32 cosine(image, prompt)x100. PSNR vs baseline `{BASELINE}`.")
    lines.append(f"Flags: CLIP < {CLIP_FLAG} or PSNR < {PSNR_FLAG} dB.")
    lines.append("")

    summary = {}
    flagged = []
    for tag in run_tags:
        shapes = sorted(p.stem for p in (ROOT / tag).glob("*.png"))
        lines.append(f"## {tag}")
        lines.append("")
        lines.append("| shape | CLIP | PSNR vs base (dB) | flag |")
        lines.append("|---|---|---|---|")
        clips = []
        for shp in shapes:
            ip = ROOT / tag / f"{shp}.png"
            c = clip_score(ip)
            clips.append(c)
            test_img = np.array(Image.open(ip).convert("RGB"))
            p = psnr(base_imgs[shp], test_img) if (shp in base_imgs and tag != BASELINE) else None
            pstr = "—" if p is None else ("inf" if p == float("inf") else f"{p:.1f}")
            flag = ""
            if c < CLIP_FLAG:
                flag += "LOW_CLIP "
            if p is not None and p != float("inf") and p < PSNR_FLAG:
                flag += "LOW_PSNR"
            if flag:
                flagged.append((tag, shp, c, pstr, flag.strip()))
            lines.append(f"| {shp} | {c:.2f} | {pstr} | {flag.strip()} |")
        mean_clip = sum(clips) / len(clips) if clips else 0.0
        summary[tag] = mean_clip
        lines.append("")
        lines.append(f"**mean CLIP: {mean_clip:.2f}** ({len(clips)} shapes)")
        lines.append("")

    lines.append("## Summary (mean CLIP per run)")
    lines.append("")
    lines.append("| run_tag | mean CLIP |")
    lines.append("|---|---|")
    for tag in run_tags:
        lines.append(f"| {tag} | {summary[tag]:.2f} |")
    lines.append("")
    if flagged:
        lines.append("## FLAGGED (candidate bad images)")
        lines.append("")
        lines.append("| run_tag | shape | CLIP | PSNR | flag |")
        lines.append("|---|---|---|---|---|")
        for tag, shp, c, pstr, fl in flagged:
            lines.append(f"| {tag} | {shp} | {c:.2f} | {pstr} | {fl} |")
    else:
        lines.append("No flagged images.")

    out = Path("wan2_1_correctness_metrics.md")
    out.write_text("\n".join(lines))
    print(f"Wrote {out}\n")
    print("\n".join(lines[-(len(run_tags) + 12) :]))


if __name__ == "__main__":
    main()
