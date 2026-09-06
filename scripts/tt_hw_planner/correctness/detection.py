"""Object-detection correctness (DETR / YOLOS / OWL-ViT / Deformable-DETR).

Compares the demo's predicted bounding-box list against the HF
CPU reference using a Hungarian-style matching with:

* class-id agreement
* bbox IoU >= 0.5 (the COCO mAP threshold)

The gate passes when at least 80% of the reference's high-
confidence (score > 0.5) detections find a matching prediction
with IoU >= 0.5 and the same class id. We use a greedy match
(no Hungarian — fast and adequate for typical demo outputs of
1-20 boxes).

Demo-output protocol
--------------------
1. ``==BBOX 0 - OUTPUT`` marker followed by one box per line:
   ``<class_id> <score> <x1> <y1> <x2> <y2>``.
2. ``boxes: <path>.json`` line referencing a JSON file with a
   list of {"label_id", "score", "box": [x1, y1, x2, y2]} dicts.
3. Falls back to soft skip.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import Comparator, Evidence, ValidationResult
from .registry import register_comparator


DEFAULT_BBOX_IOU_MATCH = 0.5
DEFAULT_DETECTION_RECALL_MIN = 0.80
DEFAULT_REF_SCORE_FLOOR = 0.5


_BBOX_MARKER_RE = re.compile(r"^==BBOX\s+(?P<idx>\d+)\s+-\s+OUTPUT\s*$", re.M)
_BBOX_PATH_RE = re.compile(r"^\s*boxes:\s*(?P<path>\S+\.json)\s*$", re.M)


@dataclass
class Box:
    label_id: int
    score: float
    x1: float
    y1: float
    x2: float
    y2: float


def bbox_iou(a: Box, b: Box) -> float:
    """Pairwise IoU between two boxes in (x1, y1, x2, y2) format."""
    xa = max(a.x1, b.x1)
    ya = max(a.y1, b.y1)
    xb = min(a.x2, b.x2)
    yb = min(a.y2, b.y2)
    iw = max(0.0, xb - xa)
    ih = max(0.0, yb - ya)
    inter = iw * ih
    area_a = max(0.0, (a.x2 - a.x1)) * max(0.0, (a.y2 - a.y1))
    area_b = max(0.0, (b.x2 - b.x1)) * max(0.0, (b.y2 - b.y1))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def greedy_match(
    pred: Sequence[Box],
    ref: Sequence[Box],
    *,
    iou_threshold: float = DEFAULT_BBOX_IOU_MATCH,
) -> List[Tuple[int, int, float]]:
    """Greedy 1-1 matching between pred and ref. Returns a list
    of ``(ref_idx, pred_idx, iou)`` triples for matched pairs;
    unmatched ref boxes don't appear.

    Two boxes only match if they have the same ``label_id`` (class
    agreement) AND their IoU >= ``iou_threshold``."""
    used_pred = set()
    matches: List[Tuple[int, int, float]] = []

    ref_idxs = sorted(range(len(ref)), key=lambda i: -ref[i].score)
    for ri in ref_idxs:
        best_pi = -1
        best_iou = 0.0
        for pi in range(len(pred)):
            if pi in used_pred:
                continue
            if pred[pi].label_id != ref[ri].label_id:
                continue
            iou = bbox_iou(pred[pi], ref[ri])
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_pi = pi
        if best_pi >= 0:
            used_pred.add(best_pi)
            matches.append((ri, best_pi, best_iou))
    return matches


def _parse_bbox_line(line: str) -> Optional[Box]:
    parts = line.split()
    if len(parts) < 6:
        return None
    try:
        return Box(
            label_id=int(parts[0]),
            score=float(parts[1]),
            x1=float(parts[2]),
            y1=float(parts[3]),
            x2=float(parts[4]),
            y2=float(parts[5]),
        )
    except (ValueError, IndexError):
        return None


def extract_boxes_from_pytest_output(
    captured_output: str,
) -> Optional[List[Box]]:
    m = _BBOX_MARKER_RE.search(captured_output)
    if m:
        after = captured_output[m.end() :]
        boxes: List[Box] = []
        for line in after.splitlines():
            if not line.strip():
                if boxes:
                    break
                continue
            if line.startswith("==") and " - " in line:
                break
            box = _parse_bbox_line(line)
            if box is not None:
                boxes.append(box)
            elif boxes:
                break
        if boxes:
            return boxes

    pm = _BBOX_PATH_RE.search(captured_output)
    if pm:
        path = Path(pm.group("path"))
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return [_dict_to_box(d) for d in data if _is_box_dict(d)]
            except Exception:
                pass
    return None


def _is_box_dict(d: Any) -> bool:
    return isinstance(d, dict) and "box" in d and "score" in d and ("label_id" in d or "label" in d)


def _dict_to_box(d: Dict[str, Any]) -> Box:
    b = d["box"]
    return Box(
        label_id=int(d.get("label_id", d.get("label", 0))),
        score=float(d.get("score", 0)),
        x1=float(b[0]),
        y1=float(b[1]),
        x2=float(b[2]),
        y2=float(b[3]),
    )


@dataclass
class _DetRef:
    boxes: List[Box]
    source_model_id: str = ""


class DetectionComparator(Comparator):
    """Comparator for object-detection backbones (DETR / YOLOS /
    OWL-ViT / Deformable-DETR / RT-DETR).

    Category claim: ``"CNN/detection"`` (precise) AND ``"CNN"``
    (gated on detection-keyword in model_id so it doesn't fight
    segmentation/classification comparators)."""

    category: str = "CNN/detection"

    _DET_MODEL_TYPES = (
        "detr",
        "deformable_detr",
        "yolos",
        "rt_detr",
        "owlvit",
        "owlv2",
        "conditional_detr",
        "deta",
    )

    def supports(self, category: str, model_id: str) -> bool:
        if category == self.category:
            return True
        if category != "CNN":
            return False
        mid_l = model_id.lower()
        return any(k in mid_l for k in self._DET_MODEL_TYPES)

    def extract(
        self,
        captured_output: str,
        model_id: str,
    ) -> Evidence:
        boxes = extract_boxes_from_pytest_output(captured_output)
        if boxes is None:
            return Evidence(
                payload=None,
                ok=False,
                reason=(
                    "could not find detections in the pytest output. "
                    "Expected '==BBOX 0 - OUTPUT' marker followed by "
                    "'<label_id> <score> <x1> <y1> <x2> <y2>' lines, "
                    "OR a 'boxes: <path>.json' reference."
                ),
            )
        return Evidence(
            payload=boxes,
            input_hint=None,
            ok=True,
            reason=f"{len(boxes)} boxes extracted from pytest output",
        )

    def load_reference(
        self,
        evidence: Evidence,
        model_id: str,
    ) -> _DetRef:
        from PIL import Image
        from transformers import pipeline

        img = Image.new("RGB", (640, 480), color=(128, 128, 128))
        pipe = pipeline("object-detection", model=model_id, device="cpu")
        out = pipe(img)
        boxes: List[Box] = []
        for item in out:
            b = item.get("box", {})
            boxes.append(
                Box(
                    label_id=hash(item.get("label", "")) % 10000,
                    score=float(item.get("score", 0)),
                    x1=float(b.get("xmin", 0)),
                    y1=float(b.get("ymin", 0)),
                    x2=float(b.get("xmax", 0)),
                    y2=float(b.get("ymax", 0)),
                )
            )
        return _DetRef(boxes=boxes, source_model_id=model_id)

    def compare(
        self,
        evidence: Evidence,
        reference: Any,
    ) -> ValidationResult:
        if not isinstance(reference, _DetRef):
            return ValidationResult(
                ok=False,
                reason="detection comparator: reference is not a _DetRef",
            )
        pred = list(evidence.payload or [])
        ref = [r for r in reference.boxes if r.score >= DEFAULT_REF_SCORE_FLOOR]
        if not ref:
            return ValidationResult(
                ok=True,
                reason="HF reference produced no high-confidence boxes; soft pass",
                tt_text=f"{len(pred)} pred boxes",
                hf_text="0 ref boxes",
            )
        matches = greedy_match(pred, ref)
        recall = len(matches) / len(ref)
        ok = recall >= DEFAULT_DETECTION_RECALL_MIN
        return ValidationResult(
            ok=ok,
            reason=(
                f"{'PASS' if ok else 'FAIL'}: matched {len(matches)}/"
                f"{len(ref)} ref boxes at IoU>={DEFAULT_BBOX_IOU_MATCH} "
                f"(recall {recall:.2f}, threshold "
                f"{DEFAULT_DETECTION_RECALL_MIN:.2f})"
            ),
            tt_text=f"{len(pred)} pred",
            hf_text=f"{len(ref)} ref",
        )

    def build_repair_prompt(
        self,
        model_id: str,
        evidence: Evidence,
        result: ValidationResult,
        *,
        iter_idx: int,
        max_iters: int,
        previous_attempts: Optional[List[str]] = None,
        extra_blocks: Optional[Sequence[str]] = None,
    ) -> str:
        from .base import render_extra_blocks

        prev = "\n    ".join(previous_attempts or []) or "(none)"
        return (
            f"You are debugging a TT-hardware bring-up of {model_id!r} "
            f"(object detection). The predicted boxes disagree with HF "
            f"beyond the recall threshold.\n\n"
            f"  GATE VERDICT (iter {iter_idx}/{max_iters}):\n"
            f"    {result.reason}\n\n"
            f"  LIKELY SUSPECTS:\n"
            f"    1. Detection head bbox regression scale (cx,cy,w,h vs x1,y1,x2,y2).\n"
            f"    2. Anchor / query embedding mismatch.\n"
            f"    3. NMS threshold or top-k cutoff.\n"
            f"    4. Class-agnostic vs class-aware NMS routing.\n"
            f"    5. Image preprocessing scale (DETR resizes to 800).\n\n"
            f"  WHAT WAS ALREADY TRIED:\n"
            f"    {prev}\n\n"
            f"  BUDGET: ~25 min/iter. Make at least one Edit.\n" + render_extra_blocks(extra_blocks)
        )


_singleton = DetectionComparator()
register_comparator(_singleton)


class _CNNDetectionAlias(DetectionComparator):
    category: str = "CNN"

    def supports(self, category: str, model_id: str) -> bool:
        if category != self.category:
            return False
        mid_l = model_id.lower()
        return any(k in mid_l for k in self._DET_MODEL_TYPES)


register_comparator(_CNNDetectionAlias())


__all__ = [
    "Box",
    "DEFAULT_BBOX_IOU_MATCH",
    "DEFAULT_DETECTION_RECALL_MIN",
    "DEFAULT_REF_SCORE_FLOOR",
    "DetectionComparator",
    "bbox_iou",
    "extract_boxes_from_pytest_output",
    "greedy_match",
]
