# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Generate media/source_{N}.png + media/target_{N}.png demo artifacts.

Four multi-person demos from the GazeFollow test set. Each pair follows the
canonical Gaze-LLE inference pipeline from the official Colab:

    1. RetinaFace.detect_faces(image)  →  head bboxes
    2. TtGazeLLE(image, bboxes)        →  per-person heatmap + in/out score

No hand-picked bboxes — every bbox in every demo comes from a face detector.
One TT-NN forward per image drives N predictions (backbone runs once,
decoder tail runs N times over the shared scene features).

Run from repo root with::

    PYTHONPATH=$PWD:$TT_METAL_HOME:$TT_METAL_HOME/ttnn \
    TT_VISIBLE_DEVICES=<n> \
    python -m models.experimental.gaze_lle.demo.make_demo

Requires ``retina-face`` and ``tf-keras`` (``pip install retina-face tf-keras``).
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

# Four GazeFollow test rows pre-verified to have ≥2 RetinaFace face detections
# at ≥0.9 confidence each (office scene, mother + child, soccer, baseball plate).
_SAMPLE_INDICES = [800, 1000, 2400, 2600]
_MULTI_DETECTOR_MIN_CONFIDENCE = 0.9

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PARQUET = Path(os.environ.get("TT_GAZE_LLE_DATA", _REPO_ROOT / "data")) / "gazefollow" / "test.parquet"
_MEDIA = Path(__file__).resolve().parent / "media"


def visualize_multi(img_pil: Image.Image, bboxes, heatmaps_64: torch.Tensor, inout_scores, out_path: Path) -> None:
    """Draw all N head bboxes + N yellow gaze arrows + per-head peak on one image.

    heatmaps_64: (N, 64, 64) torch tensor after sigmoid.
    """
    W, H = img_pil.size
    N = heatmaps_64.shape[0]

    # Accumulate the max over N heatmaps for a single overlay (keeps the image
    # readable when you have multiple heads).
    combined = heatmaps_64.max(dim=0).values.float().cpu().numpy()
    combined_pil = Image.fromarray((combined * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    combined_arr = np.asarray(combined_pil) / 255.0

    colors = ["lime", "cyan", "magenta", "orange", "yellow"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10 * H / W))
    ax.imshow(img_pil)
    ax.imshow(combined_arr, cmap="jet", alpha=0.40)

    for i in range(N):
        hm_pil = Image.fromarray((heatmaps_64[i].float().cpu().numpy() * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
        hm_arr = np.asarray(hm_pil) / 255.0
        py, px = np.unravel_index(hm_arr.argmax(), hm_arr.shape)

        bcol = colors[i % len(colors)]
        xmin, ymin, xmax, ymax = bboxes[i]
        ax.add_patch(plt.Rectangle(
            (xmin * W, ymin * H), (xmax - xmin) * W, (ymax - ymin) * H,
            fill=False, edgecolor=bcol, linewidth=3,
        ))
        ax.scatter([px], [py], c=bcol, s=220, marker="x", linewidths=4)

        cx = (xmin + xmax) / 2 * W
        cy = (ymin + ymax) / 2 * H
        ax.annotate(
            "", xy=(px, py), xytext=(cx, cy),
            arrowprops=dict(arrowstyle="->", color=bcol, lw=4,
                             shrinkA=8, shrinkB=8, mutation_scale=22),
        )

    title_bits = [f"head{i}: inout={float(inout_scores[i]):.2f}" for i in range(N)]
    ax.set_title("  |  ".join(title_bits), fontsize=12)
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", dpi=110)
    plt.close(fig)


def visualize(img_pil: Image.Image, bbox, heatmap_64: torch.Tensor, inout_score: float, out_path: Path) -> None:
    W, H = img_pil.size
    hm = heatmap_64.float().cpu().numpy()
    hm_pil = Image.fromarray((hm * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    hm_arr = np.asarray(hm_pil) / 255.0

    fig, ax = plt.subplots(1, 1, figsize=(10, 10 * H / W))
    ax.imshow(img_pil)
    ax.imshow(hm_arr, cmap="jet", alpha=0.45)

    xmin, ymin, xmax, ymax = bbox
    ax.add_patch(plt.Rectangle(
        (xmin * W, ymin * H), (xmax - xmin) * W, (ymax - ymin) * H,
        fill=False, edgecolor="lime", linewidth=3,
    ))

    py, px = np.unravel_index(hm_arr.argmax(), hm_arr.shape)
    ax.scatter([px], [py], c="red", s=220, marker="x", linewidths=4)

    cx = (xmin + xmax) / 2 * W
    cy = (ymin + ymax) / 2 * H
    ax.annotate(
        "", xy=(px, py), xytext=(cx, cy),
        arrowprops=dict(arrowstyle="->", color="yellow", lw=4,
                         shrinkA=8, shrinkB=8, mutation_scale=22),
    )

    ax.set_title(f"gaze target (red ×), gaze direction (yellow arrow),  inout={inout_score:.3f}", fontsize=13)
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", dpi=110)
    plt.close(fig)


def _retinaface_detect(pil_image: Image.Image, min_confidence: float) -> list:
    """Return a list of normalized (xmin, ymin, xmax, ymax) bboxes using RetinaFace.

    This mirrors the "detect faces" cell in fkryan/gazelle's Colab demo:

        from retinaface import RetinaFace
        resp = RetinaFace.detect_faces(np.array(image))
        bboxes = [resp[k]['facial_area'] for k in resp.keys()]

    Returned boxes are sorted left-to-right for stable ordering.
    """
    from retinaface import RetinaFace  # imported lazily — optional dep
    W, H = pil_image.size
    resp = RetinaFace.detect_faces(np.array(pil_image))
    if not isinstance(resp, dict):
        return []
    out = []
    for key, meta in resp.items():
        score = float(meta.get("score", 0.0))
        if score < min_confidence:
            continue
        x0, y0, x1, y1 = [int(v) for v in meta["facial_area"]]
        out.append((x0 / W, y0 / H, x1 / W, y1 / H))
    out.sort(key=lambda b: b[0])
    return out


def _load_samples_from_parquet(path: Path, indices):
    df = pd.read_parquet(path)
    out = []
    for idx in indices:
        row = df.iloc[idx]
        pil = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        hb = row["gazes"][0]["head_bbox"]
        bbox = (float(hb["xmin"]), float(hb["ymin"]), float(hb["xmax"]), float(hb["ymax"]))
        out.append((pil, bbox))
    return out


def main() -> None:
    import ttnn

    from models.experimental.gaze_lle.reference.load_pretrained import load_pretrained
    from models.experimental.gaze_lle.reference.torch_gaze_lle import build_gaze_lle
    from models.experimental.gaze_lle.tt.tt_gaze_lle import TtGazeLLE

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    ref = build_gaze_lle("vitb14", inout=True).eval()
    load_pretrained(ref, verbose=False)

    _MEDIA.mkdir(parents=True, exist_ok=True)
    samples = _load_samples_from_parquet(_PARQUET, _SAMPLE_INDICES)

    device_id = int(os.environ.get("GAZE_LLE_DEVICE", "0"))
    d = ttnn.open_device(device_id=device_id)
    try:
        tt_model = TtGazeLLE(ref, d, inout=True)

        tf = T.Compose([
            T.Resize((448, 448)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Need at least one forward to populate the program cache before timing
        # / correctness matters; the first sample's detection also warms
        # RetinaFace's weight load.
        warm_pil = samples[0][0]
        warm_bboxes = _retinaface_detect(warm_pil, min_confidence=_MULTI_DETECTOR_MIN_CONFIDENCE)
        if warm_bboxes:
            _ = tt_model(tf(warm_pil).unsqueeze(0), warm_bboxes)

        for i, (img_pil, _gt_bbox) in enumerate(samples, start=1):
            src_path = _MEDIA / f"source_{i}.png"
            img_pil.save(src_path)

            bboxes = _retinaface_detect(img_pil, min_confidence=_MULTI_DETECTOR_MIN_CONFIDENCE)
            if not bboxes:
                print(f"  {src_path.name}  ({img_pil.size[0]}x{img_pil.size[1]})  RetinaFace found 0 faces — skipping")
                continue

            out = tt_model(tf(img_pil).unsqueeze(0), bboxes)
            heatmaps = out["heatmap"]                               # (N, 64, 64)
            inouts = out["inout"]                                   # (N,)

            tgt_path = _MEDIA / f"target_{i}.png"
            visualize_multi(img_pil, bboxes, heatmaps, inouts, tgt_path)
            inout_str = "[" + ", ".join(f"{float(v):.2f}" for v in inouts) + "]"
            print(
                f"  {src_path.name}  ({img_pil.size[0]}x{img_pil.size[1]})  "
                f"N={len(bboxes)} faces  inouts={inout_str}  →  {tgt_path.name}"
            )
    finally:
        ttnn.close_device(d)


if __name__ == "__main__":
    main()
