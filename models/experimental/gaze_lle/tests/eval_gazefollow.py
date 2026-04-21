# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""GazeFollow test-set evaluation for torch reference vs TT-NN.

Metric formulas follow ``gazelle/utils.py`` from fkryan/gazelle verbatim:

  * AUC: mark every annotator gaze point on a (H, W) target map, compute
    roc_auc_score between the bilinearly-upsampled predicted heatmap and the
    binary target map.
  * Avg L2: Euclidean distance between argmax(heatmap) and the mean of the
    annotator gaze points (normalized coords).
  * Min L2: minimum distance to any single annotator point.

Paper numbers for ``gazelle_dinov2_vitb14_inout`` (from fkryan/gazelle README):
    AUC 0.956, Avg L2 0.151, Min L2 0.099

Runs as ``python -m models.experimental.gaze_lle.tests.eval_gazefollow``.
"""

from __future__ import annotations

import argparse
import io
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

WEIGHTS_DIR = Path("/home/ttuser/experiments/gaze-lle/weights")
DATASET_PATH = Path("/home/ttuser/experiments/gaze-lle/data/gazefollow/test.parquet")


def load_models(device=None):
    from models.experimental.gaze_lle.reference.torch_gaze_lle import build_gaze_lle
    from models.experimental.gaze_lle.reference.load_pretrained import load_pretrained
    from models.experimental.gaze_lle.tt.tt_gaze_lle import TtGazeLLE

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    ref = build_gaze_lle("vitb14", inout=True).eval()
    load_pretrained(ref, verbose=True)

    tt_model = None
    if device is not None:
        tt_model = TtGazeLLE(ref, device, inout=True)
    return ref, tt_model


_TRANSFORM = T.Compose([
    T.Resize((448, 448)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def iterate_samples(df, max_samples=None):
    """Yield (image_tensor, head_bbox, gazes_list_xy, orig_h, orig_w, in_out_gt_or_None)."""
    n = len(df) if max_samples is None else min(max_samples, len(df))
    for i in range(n):
        row = df.iloc[i]
        img_bytes = row["image"]["bytes"]
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        W, H = pil.size
        img_tensor = _TRANSFORM(pil).unsqueeze(0)
        gazes = row["gazes"]
        # All gazes in a row share the same head_bbox (verified on test set).
        hb = gazes[0]["head_bbox"]
        head_bbox = (float(hb["xmin"]), float(hb["ymin"]), float(hb["xmax"]), float(hb["ymax"]))
        gaze_xy = [(float(g["gaze"]["x"]), float(g["gaze"]["y"])) for g in gazes]
        # in_out is either None or 0/1 per-annotator; majority for the frame.
        inout_labels = [g.get("in_out") for g in gazes if g.get("in_out") is not None]
        gt_in = None
        if inout_labels:
            gt_in = 1 if sum(inout_labels) > len(inout_labels) / 2 else 0
        yield img_tensor, head_bbox, gaze_xy, H, W, gt_in


def gazefollow_auc(heatmap: torch.Tensor, gazes_xy, H: int, W: int) -> float:
    """Mirror of gazelle/utils.py gazefollow_auc."""
    target = np.zeros((H, W), dtype=np.uint8)
    for gx, gy in gazes_xy:
        if gx >= 0 and gy >= 0:
            x = min(int(gx * W), W - 1)
            y = min(int(gy * H), H - 1)
            target[y, x] = 1
    resized = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0).float(),
                            (H, W), mode="bilinear", align_corners=False).squeeze()
    try:
        return float(roc_auc_score(target.flatten(), resized.cpu().flatten().numpy()))
    except ValueError:
        return float("nan")


def gazefollow_l2(heatmap: torch.Tensor, gazes_xy):
    """Mirror of gazelle/utils.py gazefollow_l2."""
    h, w = heatmap.shape[-2:]
    flat_idx = int(heatmap.flatten().argmax().item())
    py, px = flat_idx // w, flat_idx % w
    px = px / float(w)
    py = py / float(h)
    gx = np.array([g[0] for g in gazes_xy])
    gy = np.array([g[1] for g in gazes_xy])
    avg_l2 = float(np.sqrt((px - gx.mean()) ** 2 + (py - gy.mean()) ** 2))
    all_l2s = np.sqrt((px - gx) ** 2 + (py - gy) ** 2)
    min_l2 = float(all_l2s.min())
    return avg_l2, min_l2


def evaluate_torch(ref, df, max_samples, desc="torch"):
    aucs, avg_l2s, min_l2s = [], [], []
    with torch.no_grad():
        for img_t, bbox, gaze_xy, H, W, _gt_in in tqdm(
            iterate_samples(df, max_samples), total=(max_samples or len(df)), desc=desc
        ):
            out = ref(img_t, [bbox])
            hm = out["heatmap"][0]
            aucs.append(gazefollow_auc(hm, gaze_xy, H, W))
            a, m = gazefollow_l2(hm, gaze_xy)
            avg_l2s.append(a); min_l2s.append(m)
    return np.nanmean(aucs), float(np.mean(avg_l2s)), float(np.mean(min_l2s))


def evaluate_tt(tt_model, df, max_samples, desc="tt"):
    aucs, avg_l2s, min_l2s = [], [], []
    for img_t, bbox, gaze_xy, H, W, _gt_in in tqdm(
        iterate_samples(df, max_samples), total=(max_samples or len(df)), desc=desc
    ):
        out = tt_model(img_t, [bbox])
        hm = out["heatmap"][0]
        aucs.append(gazefollow_auc(hm, gaze_xy, H, W))
        a, m = gazefollow_l2(hm, gaze_xy)
        avg_l2s.append(a); min_l2s.append(m)
    return np.nanmean(aucs), float(np.mean(avg_l2s)), float(np.mean(min_l2s))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None, help="Cap #test images (default: all 4782)")
    parser.add_argument("--skip-torch", action="store_true")
    parser.add_argument("--skip-tt", action="store_true")
    parser.add_argument("--device-id", type=int, default=int(os.environ.get("GAZE_LLE_DEVICE", "0")))
    args = parser.parse_args()

    df = pd.read_parquet(DATASET_PATH)
    n_total = len(df)
    n_eval = args.max_samples or n_total
    print(f"GazeFollow test set: {n_total} images; evaluating {n_eval}")

    import ttnn
    device = None
    if not args.skip_tt:
        device = ttnn.open_device(device_id=args.device_id)

    try:
        ref, tt_model = load_models(device)

        report = {}
        if not args.skip_torch:
            t0 = time.perf_counter()
            torch_res = evaluate_torch(ref, df, args.max_samples)
            t_torch = time.perf_counter() - t0
            report["torch"] = (*torch_res, t_torch)

        if not args.skip_tt and tt_model is not None:
            # warm-up pass so program cache is populated before timing begins
            first_img, first_bbox, *_ = next(iterate_samples(df, 1))
            _ = tt_model(first_img, [first_bbox])
            t0 = time.perf_counter()
            tt_res = evaluate_tt(tt_model, df, args.max_samples)
            t_tt = time.perf_counter() - t0
            report["tt"] = (*tt_res, t_tt)
    finally:
        if device is not None:
            ttnn.close_device(device)

    print()
    print("=" * 72)
    print(f"  {'impl':8s} {'AUC':>8s} {'Avg L2':>9s} {'Min L2':>9s} {'wall_s':>9s}")
    print(f"  {'paper':8s} {0.956:>8.4f} {0.151:>9.4f} {0.099:>9.4f} {'-':>9s}")
    for name, vals in report.items():
        auc, avg_l2, min_l2, wall = vals
        print(f"  {name:8s} {auc:>8.4f} {avg_l2:>9.4f} {min_l2:>9.4f} {wall:>9.2f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
