"""W1 verdict host scorer — CLIP + frame extraction from the two lossless frame dumps.

Mirrors test_pipeline_ltx_distilled.check_output_with_clip: sample 8 evenly-spaced frames,
score each against DEFAULT_LTX_PROMPT with the same CLIPEncoder, report min/max/mean. Reads the
torch.save'd (F,H,W,C) uint8 dumps directly (no mp4 codec loss), so numbers run a touch above the
mp4-readback gate — the A/B delta is what convicts or clears W1. Also writes a mid-frame PNG per arm
for a visual guitar check (the guitar is ground truth; CLIP only corroborates).
"""
import sys

import numpy as np
import torch
from PIL import Image

from models.tt_dit.tests.dataset_eval.clip_encoder import CLIPEncoder
from models.tt_dit.utils.ltx import DEFAULT_LTX_PROMPT

ARMS = {"w1on (DEDUP=1, shipping)": sys.argv[1], "w1off (DEDUP=0)": sys.argv[2]}
print(f"PROMPT: {DEFAULT_LTX_PROMPT!r}\n")

enc = CLIPEncoder()
for label, path in ARMS.items():
    frames = torch.load(path)  # (F, H, W, C) uint8
    frames = np.asarray(frames)
    F = frames.shape[0]
    idxs = np.linspace(0, F - 1, min(8, F), dtype=int)
    scores = []
    for i in idxs:
        pil = Image.fromarray(frames[int(i)].astype(np.uint8))
        scores.append(enc.get_clip_score(DEFAULT_LTX_PROMPT, pil).item() * 100.0)
    mid = int(idxs[len(idxs) // 2])
    out_png = path + "_mid.png"
    Image.fromarray(frames[mid].astype(np.uint8)).save(out_png)
    print(f"{label}")
    print(f"  shape={frames.shape} sampled={list(idxs)}")
    print(f"  CLIP min={min(scores):.2f} max={max(scores):.2f} mean={sum(scores)/len(scores):.2f}")
    print(f"  per-frame={[round(s, 2) for s in scores]}")
    print(f"  mid-frame[{mid}] PNG -> {out_png}\n")
