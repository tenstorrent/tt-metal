# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Worker for Q-Align. Runs under ~/ext_vqa_modern_env (torch 2.14, numpy 2.x) --
pyiqa's Q-Align needs transformers~=5.x, which conflicts with both ~/vbench_env's
transformers==4.33.2 pin and ~/ext_vqa_legacy_env's torch~=1.13 pin; see
ext_vqa_score.py's docstring for env setup.

Invoked by ext_vqa_score.py as a subprocess; not meant to be run standalone except
for debugging in isolation.
"""

from __future__ import annotations

import argparse
import json
import sys


def score_qalign(video_path: str, device: str, num_frames: int = 8) -> dict:
    """pyiqa's `qalign` arch is image-only (raises NotImplementedError on input_='video');
    there is no registered video variant. Proxy: average the image-mode quality score over
    `num_frames` evenly-spaced frames -- not the native Q-Align video model."""
    import decord
    import numpy as np
    import pyiqa
    import torch

    metric = pyiqa.create_metric("qalign", device=device)

    vr = decord.VideoReader(video_path)
    idxs = np.linspace(0, len(vr) - 1, num=min(num_frames, len(vr)), dtype=int)

    scores = []
    for i in idxs:
        frame = torch.from_numpy(vr[int(i)].asnumpy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        with torch.no_grad():
            scores.append(float(metric(frame)))

    return {"per_frame_scores": scores, "mean_score": float(np.mean(scores)), "num_frames_sampled": len(scores)}


DISPATCH = {"qalign": score_qalign}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    results = {}
    for name in (n.strip() for n in args.metrics.split(",")):
        if not name:
            continue
        try:
            results[name] = DISPATCH[name](args.video, args.device)
        except Exception as exc:  # noqa: BLE001 - report the failure, don't crash the whole worker
            results[name] = {"error": repr(exc)}

    print(json.dumps(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
