# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bench only (not for merge): relative L2 (%) of each recipe's dumped latents against FAST's, per step.

    python compare_recipe_latents.py LATENT_ROOT   # LATENT_ROOT/<dur>s_<recipe>_<kv>/stepNNN.pt
"""

import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
runs = sorted(p for p in root.iterdir() if p.is_dir())
for base in [r for r in runs if "_FAST_" in r.name]:
    dur = base.name.split("_")[0]
    for other in [r for r in runs if r.name.startswith(dur + "_") and r is not base]:
        cells = []
        for step in sorted(base.glob("step*.pt")):
            peer = other / step.name
            if not peer.exists():
                continue
            a, b = torch.load(step), torch.load(peer)
            rel = {k: (100 * (b[k] - a[k]).norm() / a[k].norm()).item() for k in ("video", "audio")}
            cells.append(f"{step.stem}: video {rel['video']:.2f}% audio {rel['audio']:.2f}%")
        print(f"{other.name} vs {base.name}: " + " | ".join(cells))
