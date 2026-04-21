# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Pretrained-weight evaluation: torch reference vs TT-NN on real images.

Loads the official Gaze-LLE checkpoint (gazelle_dinov2_vitb14_inout.pt) and
DINOv2 backbone (dinov2_vitb14_pretrain.pth) into the reference model, then
constructs :class:`TtGazeLLE` from it. Runs both paths on real images and
asserts heatmap PCC ≥ 0.99 (i.e. the tt port reproduces the pretrained model
within bf16/bfp8/LoFi numerics on real data, not just random weights).

Heatmap and peak-gaze predictions are also dumped to artifacts for visual
inspection.

Assumes these files exist (download scripts elsewhere):
    /home/ttuser/experiments/gaze-lle/weights/dinov2_vitb14_pretrain.pth
    /home/ttuser/experiments/gaze-lle/weights/gazelle_dinov2_vitb14_inout.pt
    /home/ttuser/experiments/gaze-lle/data/the_office.png
    /home/ttuser/experiments/gaze-lle/data/succession.png
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

WEIGHTS_DIR = Path("/home/ttuser/experiments/gaze-lle/weights")
DATA_DIR = Path("/home/ttuser/experiments/gaze-lle/data")
ARTIFACTS_DIR = Path("/home/ttuser/experiments/gaze-lle/eval_artifacts")


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item() + 1e-12
    return float((a @ b).item() / denom)


def _load_and_transform(path: Path, size: int = 448):
    """PIL → (1, 3, size, size) ImageNet-normalized torch tensor."""
    tf = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(path).convert("RGB")
    return tf(img).unsqueeze(0), img.size  # (W, H) of original


def _save_heatmap_overlay(original_size, image_path: Path, heatmap: torch.Tensor,
                           bbox, out_path: Path):
    """Save the heatmap overlayed on the image with the bbox drawn."""
    import numpy as np
    from PIL import ImageDraw

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return  # Matplotlib is optional; skip overlay if unavailable.

    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    hm_np = heatmap.detach().float().cpu().numpy()  # (64, 64)
    # Upsample to original size for visualization.
    from PIL import Image as PImage
    hm_pil = PImage.fromarray((hm_np * 255).astype(np.uint8)).resize((W, H), PImage.BILINEAR)
    hm_np = np.asarray(hm_pil) / 255.0

    fig, ax = plt.subplots(1, 1, figsize=(8, 8 * H / W))
    ax.imshow(img)
    ax.imshow(hm_np, cmap="jet", alpha=0.45)
    xmin, ymin, xmax, ymax = bbox
    ax.add_patch(plt.Rectangle(
        (xmin * W, ymin * H), (xmax - xmin) * W, (ymax - ymin) * H,
        fill=False, edgecolor="lime", linewidth=2,
    ))
    # Peak
    py, px = np.unravel_index(hm_np.argmax(), hm_np.shape)
    ax.scatter([px], [py], c="red", s=80, marker="x")
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


_TEST_CASES = [
    # (image_filename, head_bbox_normalized, description)
    # Bounding boxes are rough hand-picked approximations of face locations.
    ("the_office.png", (0.38, 0.10, 0.53, 0.43), "the_office.png — central-left face"),
    ("succession.png", (0.40, 0.15, 0.60, 0.60), "succession.png — central face"),
]


def _require_weights():
    dino = WEIGHTS_DIR / "dinov2_vitb14_pretrain.pth"
    gaze = WEIGHTS_DIR / "gazelle_dinov2_vitb14_inout.pt"
    if not dino.exists() or not gaze.exists():
        pytest.skip(f"pretrained weights not found in {WEIGHTS_DIR}")
    return dino, gaze


@pytest.fixture(scope="module")
def pretrained_ref_model():
    """Build ref model + load pretrained weights (shared across parametrizations)."""
    _require_weights()
    torch.manual_seed(0)
    model = build_gaze_lle("vitb14", inout=True).eval()
    load_pretrained(model, verbose=True)
    return model


@pytest.mark.parametrize("image_name,bbox,description", _TEST_CASES)
def test_pretrained_torch_vs_tt(device, pretrained_ref_model, image_name, bbox, description, capsys):
    """Torch-pretrained reference vs TT-NN inference on a real image + bbox."""
    image_path = DATA_DIR / image_name
    if not image_path.exists():
        pytest.skip(f"test image missing: {image_path}")

    img_tensor, orig_size = _load_and_transform(image_path)
    bboxes = [bbox]

    with torch.no_grad():
        torch_out = pretrained_ref_model(img_tensor, bboxes)

    tt_model = TtGazeLLE(pretrained_ref_model, device, inout=True)
    _ = tt_model(img_tensor, bboxes)  # warm up (populate program cache)
    tt_out = tt_model(img_tensor, bboxes)

    # PCC on the sigmoid'd heatmap.
    pcc = _pcc(torch_out["heatmap"], tt_out["heatmap"])

    torch_peak = _peak_coord(torch_out["heatmap"][0])
    tt_peak = _peak_coord(tt_out["heatmap"][0])
    torch_inout = float(torch_out["inout"][0]) if torch_out["inout"] is not None else None
    tt_inout = float(tt_out["inout"][0]) if tt_out["inout"] is not None else None

    with capsys.disabled():
        print()
        print(f"=== {description} ===")
        print(f"  heatmap PCC         : {pcc:.4f}")
        print(f"  torch peak (row,col): {torch_peak}")
        print(f"  tt    peak (row,col): {tt_peak}")
        print(f"  torch inout score   : {torch_inout:.4f}" if torch_inout is not None else "")
        print(f"  tt    inout score   : {tt_inout:.4f}" if tt_inout is not None else "")

    # Save artifacts for visual inspection.
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    _save_heatmap_overlay(orig_size, image_path, torch_out["heatmap"][0], bbox,
                           ARTIFACTS_DIR / f"{stem}_torch.png")
    _save_heatmap_overlay(orig_size, image_path, tt_out["heatmap"][0], bbox,
                           ARTIFACTS_DIR / f"{stem}_tt.png")

    assert pcc >= 0.99, f"torch↔TT heatmap PCC {pcc:.4f} < 0.99 for {description}"
    # Peak locations should be close: allow up to 5 px in the 64x64 heatmap (~8%).
    peak_dist = ((torch_peak[0] - tt_peak[0]) ** 2 + (torch_peak[1] - tt_peak[1]) ** 2) ** 0.5
    assert peak_dist < 5.0, f"peak locations too far apart: {peak_dist:.2f} px for {description}"


def _peak_coord(heatmap_64: torch.Tensor):
    """Return (row, col) of the argmax pixel in a (64, 64) heatmap."""
    h, w = heatmap_64.shape
    idx = int(heatmap_64.flatten().argmax().item())
    return idx // w, idx % w
