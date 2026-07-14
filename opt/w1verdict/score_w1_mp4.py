"""W1 verdict host scorer — CLIP + frame dump from the gen #0 CAPTURE mp4s.

The frames_w1{on,off} torch dumps are the degenerate traced-REPLAY pass (all frames identical,
byte-identical across arms) — scoring them is invalid. This scorer reads the gen #0 CAPTURE mp4
(the genuine render) via imageio, samples 8 evenly-spaced frames, scores each against
DEFAULT_LTX_PROMPT with the same CLIPEncoder the test uses, and writes 3 PNGs/arm (early/mid/late)
so a human can LOOK for the guitar. The guitar is ground truth; CLIP only corroborates.
"""
import sys

import imageio.v3 as iio
import numpy as np
from PIL import Image

from models.tt_dit.tests.dataset_eval.clip_encoder import CLIPEncoder
from models.tt_dit.utils.ltx import DEFAULT_LTX_PROMPT

ARMS = {"w1on_DEDUP1_shipping": sys.argv[1], "w1off_DEDUP0": sys.argv[2]}
print(f"PROMPT: {DEFAULT_LTX_PROMPT!r}\n")

enc = CLIPEncoder()
for label, path in ARMS.items():
    frames = np.asarray(iio.imread(path, plugin="pyav"))  # (F, H, W, C) uint8
    F = frames.shape[0]
    idxs = np.linspace(0, F - 1, min(8, F), dtype=int)
    scores = []
    for i in idxs:
        pil = Image.fromarray(frames[int(i)].astype(np.uint8))
        scores.append(enc.get_clip_score(DEFAULT_LTX_PROMPT, pil).item() * 100.0)
    # dump early / mid / late for a visual guitar check
    for tag, i in (("early", idxs[1]), ("mid", idxs[len(idxs) // 2]), ("late", idxs[-2])):
        out_png = f"{path}.{tag}.png"
        Image.fromarray(frames[int(i)].astype(np.uint8)).save(out_png)
    print(f"{label}")
    print(f"  shape={frames.shape} sampled={list(idxs)}")
    print(f"  CLIP min={min(scores):.2f} max={max(scores):.2f} mean={sum(scores)/len(scores):.2f}")
    print(f"  per-frame={[round(s, 2) for s in scores]}")
    print(f"  PNGs -> {path}.early/mid/late.png\n")
