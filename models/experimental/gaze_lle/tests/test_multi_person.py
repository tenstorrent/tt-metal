# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Multi-person verification for TtGazeLLE.

``TtGazeLLE.__call__(image, [bbox_a, bbox_b, ...])`` runs the DINOv2 backbone
and the gaze projection ONCE per image and then runs the decoder per head.
This test verifies that the per-head outputs match N independent
single-person torch forward passes on the same image (PCC ≥ 0.99 per head,
peak-pixel distance ≤ 5 px in the 64x64 heatmap).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torchvision.transforms as T
from PIL import Image

from models.experimental.gaze_lle.reference.load_pretrained import load_pretrained
from models.experimental.gaze_lle.reference.torch_gaze_lle import build_gaze_lle
from models.experimental.gaze_lle.tt.tt_gaze_lle import TtGazeLLE

DATA_DIR = Path(os.environ.get("TT_GAZE_LLE_DATA", "./data"))


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item() + 1e-12
    return float((a @ b).item() / denom)


def _peak(heatmap_64: torch.Tensor):
    h, w = heatmap_64.shape[-2:]
    idx = int(heatmap_64.flatten().argmax().item())
    return idx // w, idx % w


_MULTI_CASES = [
    # (image_filename, minimum detected faces). Head bboxes are obtained at runtime
    # via RetinaFace — matching the official Gaze-LLE Colab pipeline — rather
    # than hand-picked, so we compare against a principled detector output.
    ("the_office.png", 2),
]


# With real face-detector bboxes, per-head TT matches torch-reference single-
# head forwards at the same ~0.99 PCC seen in the pretrained test.
_PCC_THRESHOLD = 0.99
_PEAK_DIST_THRESHOLD_PX = 5.0


def _detect_faces(pil_image):
    """RetinaFace → list of normalized (xmin, ymin, xmax, ymax). Raises if retinaface isn't installed."""
    import numpy as np  # imported lazily
    from retinaface import RetinaFace
    W, H = pil_image.size
    resp = RetinaFace.detect_faces(np.array(pil_image))
    if not isinstance(resp, dict):
        return []
    boxes = []
    for meta in resp.values():
        x0, y0, x1, y1 = [int(v) for v in meta["facial_area"]]
        boxes.append((x0 / W, y0 / H, x1 / W, y1 / H))
    boxes.sort(key=lambda b: b[0])
    return boxes


@pytest.mark.parametrize("image_name,min_faces", _MULTI_CASES)
def test_multi_person_matches_n_single(device, image_name, min_faces, capsys):
    image_path = DATA_DIR / image_name
    if not image_path.exists():
        pytest.skip(f"missing {image_path}")

    pil = Image.open(image_path).convert("RGB")
    try:
        bboxes = _detect_faces(pil)
    except ImportError:
        pytest.skip("retina-face not installed — `pip install retina-face tf-keras`")
    if len(bboxes) < min_faces:
        pytest.skip(f"RetinaFace found {len(bboxes)} face(s) in {image_name}; need ≥ {min_faces}")

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    ref = build_gaze_lle("vitb14", inout=True).eval()
    load_pretrained(ref, verbose=False)
    tt_model = TtGazeLLE(ref, device, inout=True)

    tf = T.Compose([
        T.Resize((448, 448)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = tf(pil).unsqueeze(0)

    # Torch: run single-person forward once per bbox (the reference model's
    # existing API). Each call runs the full torch forward, so this is the
    # ground-truth for what the per-head TT outputs should approximate.
    torch_heatmaps = []
    torch_inouts = []
    with torch.no_grad():
        for bb in bboxes:
            out = ref(img, [bb])
            torch_heatmaps.append(out["heatmap"][0])
            if out["inout"] is not None:
                torch_inouts.append(float(out["inout"][0]))

    # TT: ONE forward with N head bboxes, producing (N, 64, 64) + (N,).
    _ = tt_model(img, [bboxes[0]])  # warm-up: populate program cache
    tt_out = tt_model(img, list(bboxes))
    tt_heatmaps = tt_out["heatmap"]  # (N, 64, 64)
    tt_inouts = tt_out["inout"]      # (N,)

    assert tt_heatmaps.shape[0] == len(bboxes), f"expected {len(bboxes)} heatmaps, got {tt_heatmaps.shape}"
    assert tt_inouts.shape[0] == len(bboxes), f"expected {len(bboxes)} inout scalars, got {tt_inouts.shape}"

    failures = []
    with capsys.disabled():
        print()
        print(f"=== {image_name}, N={len(bboxes)} heads ===")
        print(f"  {'head':>4s} {'pcc':>7s} {'peak_torch':>11s} {'peak_tt':>9s} {'inout_t':>8s} {'inout_tt':>9s}")
    for i, bb in enumerate(bboxes):
        pcc = _pcc(torch_heatmaps[i], tt_heatmaps[i])
        tp = _peak(torch_heatmaps[i])
        pp = _peak(tt_heatmaps[i])
        dist = ((tp[0] - pp[0]) ** 2 + (tp[1] - pp[1]) ** 2) ** 0.5
        io_t = torch_inouts[i] if i < len(torch_inouts) else float("nan")
        io_tt = float(tt_inouts[i])
        with capsys.disabled():
            print(f"  {i:>4d} {pcc:>7.4f} ({tp[0]:>2d},{tp[1]:>2d})    ({pp[0]:>2d},{pp[1]:>2d}) {io_t:>8.4f} {io_tt:>9.4f}")
        if pcc < _PCC_THRESHOLD:
            failures.append(f"head {i}: PCC {pcc:.4f} < {_PCC_THRESHOLD}")
        if dist > _PEAK_DIST_THRESHOLD_PX:
            failures.append(f"head {i}: peak distance {dist:.2f} px > {_PEAK_DIST_THRESHOLD_PX}")

    assert not failures, "multi-person mismatches:\n  " + "\n  ".join(failures)
