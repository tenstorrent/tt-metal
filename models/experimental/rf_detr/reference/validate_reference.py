# ------------------------------------------------------------------------
# RF-DETR-base reference validation / acceptance gate.
#
# Builds + strictly loads the model, preprocesses the canonical COCO image
# (two cats + two remotes on a pink couch), runs the forward pass, post-
# processes detections, prints the top-12, asserts the expected objects are
# detected, and saves reference tensors for the tt-nn port.
#
# Usage:
#   export TT_METAL_HOME=/home/ttuser/experiments/rf-detr/tt-metal
#   export ARCH_NAME=blackhole
#   export PYTHONPATH=$TT_METAL_HOME/ttnn:$TT_METAL_HOME:$PYTHONPATH
#   source /home/ttuser/experiments/sam3/tt-metal/python_env/bin/activate
#   python -m models.experimental.rf_detr.reference.validate_reference
# (or run the file directly; it self-bootstraps sys.path)
# ------------------------------------------------------------------------
"""Validate the RF-DETR-base reference and dump oracle tensors."""

from __future__ import annotations

import os
import sys

import torch
from PIL import Image

# Allow running as a plain script (python validate_reference.py).
if __package__ in (None, ""):
    _EXP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _EXP not in sys.path:
        sys.path.insert(0, _EXP)
    from rf_detr.reference.weights import get_preprocessor, load_rf_detr_base
else:
    from .weights import get_preprocessor, load_rf_detr_base

IMAGE_PATH = "/home/ttuser/experiments/rf-detr/assets/cats_000000039769.jpg"
OUTPUT_PATH = "/home/ttuser/experiments/rf-detr/reference_outputs.pt"
CONF_THRESHOLD = 0.5


def main() -> None:
    torch.manual_seed(0)

    # --- build + strict load ---
    model, cfg = load_rf_detr_base()
    id2label = cfg.id2label

    # --- preprocess canonical image ---
    image = Image.open(IMAGE_PATH).convert("RGB")
    orig_w, orig_h = image.size  # (640, 480)
    pre = get_preprocessor(cfg)
    pixel_values = pre(image)
    assert tuple(pixel_values.shape) == (1, 3, cfg.image_resolution, cfg.image_resolution)

    # --- forward ---
    out = model(pixel_values, collect_intermediates=True)
    logits = out.logits  # (1, 300, 91)
    pred_boxes = out.pred_boxes  # (1, 300, 4) cxcywh normalized
    assert tuple(logits.shape) == (1, cfg.num_queries, cfg.num_labels)
    assert tuple(pred_boxes.shape) == (1, cfg.num_queries, 4)

    # --- post-process: sigmoid scores, top queries, cxcywh -> xyxy scaled ---
    prob = logits.sigmoid()[0]  # (300, 91)
    scores, labels = prob.max(-1)
    boxes = pred_boxes[0]

    def to_xyxy(box):
        cx, cy, w, h = box.tolist()
        return [
            (cx - w / 2) * orig_w,
            (cy - h / 2) * orig_h,
            (cx + w / 2) * orig_w,
            (cy + h / 2) * orig_h,
        ]

    topv, topi = scores.topk(12)
    print("\nTop-12 detections (label score [x0,y0,x1,y1] scaled to %dx%d):" % (orig_w, orig_h))
    for s, qi in zip(topv.tolist(), topi.tolist()):
        lab = labels[qi].item()
        x0, y0, x1, y1 = to_xyxy(boxes[qi])
        name = id2label.get(lab, str(lab))
        print(f"  {name:14s} {s:.3f} [{x0:7.1f},{y0:7.1f},{x1:7.1f},{y1:7.1f}]")

    # --- acceptance assertions: >=2 cat and >=1 remote among conf>0.5 ---
    high_conf = scores > CONF_THRESHOLD
    detected = [(id2label.get(labels[i].item(), ""), scores[i].item())
                for i in range(scores.shape[0]) if high_conf[i]]
    n_cat = sum(1 for name, _ in detected if name == "cat")
    n_remote = sum(1 for name, _ in detected if name == "remote")
    print(f"\nHigh-confidence (>{CONF_THRESHOLD}) detections: {len(detected)}")
    print(f"  cat: {n_cat}, remote: {n_remote}")
    assert n_cat >= 2, f"expected >=2 cats, got {n_cat}"
    assert n_remote >= 1, f"expected >=1 remote, got {n_remote}"

    # --- save reference oracle tensors ---
    ref = {
        "pixel_values": pixel_values.detach().cpu(),
        "logits": logits.detach().cpu(),
        "pred_boxes": pred_boxes.detach().cpu(),
        "backbone_feature_maps": [f.detach().cpu() for f in out.backbone_feature_maps],
        "projector_out": out.projector_out.detach().cpu(),
        "decoder_hidden_states": [h.detach().cpu() for h in out.decoder_hidden_states],
        "init_reference_points": out.init_reference_points.detach().cpu(),
    }
    torch.save(ref, OUTPUT_PATH)
    print(f"\nSaved reference tensors -> {OUTPUT_PATH}")
    print("  pixel_values:", tuple(ref["pixel_values"].shape))
    print("  logits:", tuple(ref["logits"].shape))
    print("  pred_boxes:", tuple(ref["pred_boxes"].shape))
    print("  backbone_feature_maps:", [tuple(f.shape) for f in ref["backbone_feature_maps"]])
    print("  projector_out:", tuple(ref["projector_out"].shape))
    print("  decoder_hidden_states:", [tuple(h.shape) for h in ref["decoder_hidden_states"]])

    print("\nREFERENCE OK")


if __name__ == "__main__":
    main()
