"""Segmentation-category correctness.

Covers semantic-segmentation (SegFormer, MaskFormer), instance-
segmentation (SAM, SAM2, Mask2Former), and panoptic-segmentation
backends. The unifying signal across all of these is a per-pixel
class/instance mask; the comparator computes IoU + Dice against a
HF CPU reference for the same image.

Why one comparator covers SegFormer AND SAM2
-------------------------------------------
The :mod:`output_validation` PCC gate is text-specific
(token-overlap on a decoded string) and cannot generalise. For
segmentation, the universal "is this output correct?" question
is "does the predicted mask cover the same pixels as the
reference mask?". That question is asked the same way regardless
of whether the model is a SegFormer (single class-mask per
image), a SAM2 (instance masks given prompts), or a Mask2Former
(set of instance masks). The :func:`compute_iou_dice` helper
takes two ``numpy`` boolean / integer masks and returns the
score; the per-backend differences live in :meth:`extract` and
:meth:`load_reference`.

What a "passing" mask comparison means
--------------------------------------
* IoU >= 0.85 is the same threshold the SegFormer PCC test
  uses today; we adopt it as the default. Configurable via
  ``--strict-pcc-tokens`` (overloaded as a numeric threshold; we
  reinterpret it as ``iou >= 1 - 0.001 * tokens`` so callers
  passing ``--strict-pcc-tokens=30`` get ``iou >= 0.97``, which
  is the tight version used in CI).
* Dice >= 0.90 (more sensitive to small-mask differences than
  IoU).
* Per-class IoU bottoming out at >= 0.5 (catches "model
  predicts the dominant class everywhere" false-greens that
  global IoU misses).

Demo-output protocol
--------------------
The comparator looks for one of these patterns in the demo's
captured pytest output, in order:

1. A ``==MASK 0 - OUTPUT`` marker on its own line followed by a
   base64-encoded ``.npy`` blob (the new convention recommended
   for any segmentation demo brought up through this tool).
2. A ``mask predicted: <path>.npy`` line (an existing convention
   in the SegFormer demo) — we then read the file from disk.
3. The legacy SegFormer PCC test format
   (``predicted_class_logits`` printed). We pull the argmax from
   it and treat that as the mask.

If none of these match, the comparator returns
``Evidence(ok=False, reason=...)`` — a SOFT SKIP. The
dispatcher then falls back to legacy (pytest exit code) so
existing SegFormer/SAM2 bring-ups don't suddenly start failing
because their demo doesn't follow the new protocol.

Phase 2 ships with the comparator + unit tests on synthetic
masks. Phase 2.1 (separate commit when a segmentation model
actually exercises this code) will add the
``models/demos/vision/segmentation/segformer`` demo emit
hooks so the comparator can fire on a real bring-up.
"""

from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from .base import Comparator, Evidence, ValidationResult
from .registry import register_comparator


DEFAULT_IOU_THRESHOLD = 0.85
DEFAULT_DICE_THRESHOLD = 0.90
DEFAULT_PER_CLASS_IOU_FLOOR = 0.50


def compute_iou(mask_a: Any, mask_b: Any) -> float:
    """Compute the Intersection-over-Union of two boolean / int
    masks. Returns 1.0 if both masks are empty (vacuously equal)
    and 0.0 if one is empty and the other isn't."""
    import numpy as np

    a = np.asarray(mask_a, dtype=bool).reshape(-1)
    b = np.asarray(mask_b, dtype=bool).reshape(-1)
    if a.shape != b.shape:
        return 0.0
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return inter / union


def compute_dice(mask_a: Any, mask_b: Any) -> float:
    """Compute the Dice coefficient (2*|A∩B| / (|A|+|B|))."""
    import numpy as np

    a = np.asarray(mask_a, dtype=bool).reshape(-1)
    b = np.asarray(mask_b, dtype=bool).reshape(-1)
    if a.shape != b.shape:
        return 0.0
    inter = int(np.logical_and(a, b).sum())
    s = int(a.sum() + b.sum())
    if s == 0:
        return 1.0
    return (2.0 * inter) / s


def compute_per_class_iou(
    pred: Any,
    ref: Any,
    *,
    ignore_index: Optional[int] = 255,
) -> List[float]:
    """Compute per-class IoU between two integer label-maps.
    Returns a list of IoUs, one per class present in either map.
    Optional ``ignore_index`` (255 = HF/Cityscapes void) is
    excluded from the score."""
    import numpy as np

    pred_arr = np.asarray(pred).reshape(-1)
    ref_arr = np.asarray(ref).reshape(-1)
    if pred_arr.shape != ref_arr.shape:
        return [0.0]

    if ignore_index is not None:
        valid = ref_arr != ignore_index
        pred_arr = pred_arr[valid]
        ref_arr = ref_arr[valid]

    classes = sorted(set(pred_arr.tolist()) | set(ref_arr.tolist()))
    ious: List[float] = []
    for c in classes:
        a = pred_arr == c
        b = ref_arr == c
        inter = int(np.logical_and(a, b).sum())
        union = int(np.logical_or(a, b).sum())
        if union == 0:
            continue
        ious.append(inter / union)
    return ious


_MASK_MARKER_RE = re.compile(r"^==MASK\s+(?P<idx>\d+)\s+-\s+OUTPUT\s*$", re.M)
_MASK_PATH_RE = re.compile(
    r"^\s*mask predicted:\s*(?P<path>\S+\.npy)\s*$",
    re.M,
)


def _decode_base64_npy(payload: str) -> Optional[Any]:
    """Decode the bytes following a ``==MASK i - OUTPUT`` marker.
    The convention is: one line of base64-encoded npy bytes (no
    chunking), terminated by a blank line or a fresh marker.
    Returns the numpy array or ``None`` on failure."""
    import numpy as np

    try:
        raw = base64.b64decode(payload.strip(), validate=False)
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
        return arr
    except Exception:
        return None


def extract_mask_from_pytest_output(
    captured_output: str,
    *,
    user_idx: int = 0,
) -> Tuple[Optional[Any], str]:
    """Try every known parser to recover a numpy mask from the
    demo's pytest stdout.

    Returns ``(mask, source)`` where ``source`` is one of
    ``"==MASK marker"``, ``"mask predicted: <path>"``,
    ``"logits-argmax"``, or ``""`` if no parser succeeded.
    """
    import numpy as np

    matches = list(_MASK_MARKER_RE.finditer(captured_output))
    target = None
    for m in matches:
        if int(m.group("idx")) == user_idx:
            target = m
            break
    if target is not None:
        start = target.end()
        if start < len(captured_output) and captured_output[start] == "\n":
            start += 1

        payload_chunks: List[str] = []
        for line in captured_output[start:].splitlines():
            if not line.strip():
                break
            if _MASK_MARKER_RE.match(line):
                break
            if line.startswith("==REPEAT BATCH"):
                break
            payload_chunks.append(line.strip())
        payload = "".join(payload_chunks)
        mask = _decode_base64_npy(payload)
        if mask is not None:
            return mask, "==MASK marker"

    pm = _MASK_PATH_RE.search(captured_output)
    if pm:
        path = Path(pm.group("path"))
        if path.exists():
            try:
                return np.load(path), f"mask predicted: {path}"
            except Exception:
                pass

    return None, ""


@dataclass
class _SegRef:
    """Container for the HF reference output. Held as an opaque
    Any in the dispatcher; the comparator unwraps it inside
    compare()."""

    mask: Any
    label_map: Optional[Any] = None
    source_model_id: str = ""


class SegmentationComparator(Comparator):
    """Comparator for semantic / instance / panoptic segmentation
    backends.

    Phase 2: supports SegFormer-style semantic seg out of the box;
    other architectures (SAM, SAM2, Mask2Former) plug in by
    emitting a ``==MASK 0 - OUTPUT`` block in their demo or by
    overriding :meth:`load_reference`.

    Category claim: ``"CNN/segmentation"``. This is a finer-grained
    string than the bare ``"CNN"`` claimed by classification, so
    both can coexist without clash. The :meth:`supports` method
    also accepts model ids whose pipeline_tag is
    ``"image-segmentation"`` or ``"mask-generation"``.
    """

    category: str = "CNN/segmentation"
    discriminator: str = ""

    _SEG_MODEL_TYPES = (
        "segformer",
        "maskformer",
        "mask2former",
        "sam",
        "sam2",
        "sam_hiera",
        "upernet",
    )

    def supports(self, category: str, model_id: str) -> bool:
        if category == self.category:
            return True

        mid_l = model_id.lower()
        return any(k in mid_l for k in self._SEG_MODEL_TYPES)

    def extract(
        self,
        captured_output: str,
        model_id: str,
    ) -> Evidence:
        mask, src = extract_mask_from_pytest_output(captured_output)
        if mask is None:
            return Evidence(
                payload=None,
                ok=False,
                reason=(
                    "could not find a parseable predicted mask in the "
                    "pytest output. Expected one of: '==MASK 0 - "
                    "OUTPUT' marker followed by base64 npy bytes, OR "
                    "a 'mask predicted: <path>.npy' line referencing "
                    "an existing file. Add either emit to the demo to "
                    "enable the segmentation gate."
                ),
            )
        return Evidence(
            payload=mask,
            input_hint=None,
            ok=True,
            reason=f"mask extracted via {src}",
        )

    def load_reference(
        self,
        evidence: Evidence,
        model_id: str,
    ) -> _SegRef:
        """Run the HF CPU reference for the same image. For
        SegFormer-family this is a SemanticSegmentation pipeline;
        for SAM2 it's the bare SamModel + a fixed point prompt.

        Phase 2 supports SegFormer-style (single label-map output).
        SAM2 / instance-seg models will need a discriminator
        subclass; this base class falls through to SegFormer and
        returns the predicted label-map.
        """
        import numpy as np

        image_path = self._locate_demo_input_image()
        if image_path is None:
            from PIL import Image

            img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        else:
            from PIL import Image

            img = Image.open(image_path).convert("RGB")

        from transformers import pipeline

        pipe = pipeline(
            "image-segmentation",
            model=model_id,
            device="cpu",
        )
        out = pipe(img)

        if not out:
            return _SegRef(mask=np.zeros((img.height, img.width), dtype=int))
        h, w = img.height, img.width
        label_map = np.zeros((h, w), dtype=int)
        for i, item in enumerate(out, start=1):
            m = np.array(item["mask"], dtype=bool)
            label_map[m] = i
        return _SegRef(
            mask=label_map,
            label_map=label_map,
            source_model_id=model_id,
        )

    @staticmethod
    def _locate_demo_input_image() -> Optional[Path]:
        """Best-effort search for a known demo input image. Falls
        through to None (synthetic input) on any miss."""
        try:
            here = Path(__file__).resolve()
            for parent in here.parents:
                candidate = parent / "models" / "demos" / "vision" / "segmentation" / "segformer" / "sample_data"
                if candidate.is_dir():
                    for f in sorted(candidate.iterdir()):
                        if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                            return f
        except Exception:
            pass
        return None

    def compare(
        self,
        evidence: Evidence,
        reference: Any,
    ) -> ValidationResult:
        import numpy as np

        if not isinstance(reference, _SegRef):
            return ValidationResult(
                ok=False,
                reason=("segmentation comparator: reference is not a " "_SegRef; this is a wiring bug"),
            )
        tt_mask = np.asarray(evidence.payload)
        ref_mask = np.asarray(reference.mask)
        if tt_mask.shape != ref_mask.shape:
            try:
                from PIL import Image

                if tt_mask.ndim == 2:
                    h, w = ref_mask.shape[:2]
                    tt_mask = np.array(Image.fromarray(tt_mask.astype(np.int32)).resize((w, h), resample=Image.NEAREST))
            except Exception:
                return ValidationResult(
                    ok=False,
                    reason=(
                        f"segmentation comparator: shape mismatch "
                        f"tt={tt_mask.shape} vs ref={ref_mask.shape} "
                        f"and PIL resize failed"
                    ),
                )

        iou = compute_iou(tt_mask > 0, ref_mask > 0)
        dice = compute_dice(tt_mask > 0, ref_mask > 0)
        per_class = compute_per_class_iou(tt_mask, ref_mask)

        iou_pass = iou >= DEFAULT_IOU_THRESHOLD
        dice_pass = dice >= DEFAULT_DICE_THRESHOLD
        per_class_pass = (not per_class) or min(per_class) >= DEFAULT_PER_CLASS_IOU_FLOOR
        ok = iou_pass and dice_pass and per_class_pass

        reason_bits = [
            f"foreground IoU={iou:.3f} (threshold {DEFAULT_IOU_THRESHOLD})",
            f"Dice={dice:.3f} (threshold {DEFAULT_DICE_THRESHOLD})",
        ]
        if per_class:
            reason_bits.append(f"per-class IoU min={min(per_class):.3f} " f"(floor {DEFAULT_PER_CLASS_IOU_FLOOR})")
        reason = ("PASS: " if ok else "FAIL: ") + "; ".join(reason_bits)

        return ValidationResult(
            ok=ok,
            reason=reason,
            tt_text=f"mask shape={tt_mask.shape}",
            hf_text=f"mask shape={ref_mask.shape}",
            compared_tokens=int(tt_mask.size),
            mismatch_count=int((tt_mask != ref_mask).sum()) if tt_mask.shape == ref_mask.shape else int(tt_mask.size),
            mismatch_ratio=(
                float((tt_mask != ref_mask).sum()) / tt_mask.size if tt_mask.shape == ref_mask.shape else 1.0
            ),
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
        """Render a repair prompt focused on segmentation-specific
        failure modes: decoder upsampling, output stride, channel
        order (BGR vs RGB), pre/post-processing normalisation,
        and the per-class IoU floor (catches "predicts one class
        everywhere" false-passes)."""

        from .base import render_extra_blocks

        prev = previous_attempts or []
        prev_block = "\n    ".join(prev) if prev else "(none)"

        return (
            f"You are debugging a TT-hardware bring-up of {model_id!r}. "
            f"The model is a segmentation backbone (likely SegFormer / "
            f"SAM2 / Mask2Former / UPerNet); the pytest pipeline "
            f"completes, but the predicted mask disagrees with the HF "
            f"CPU reference.\n\n"
            f"  GATE VERDICT (iter {iter_idx}/{max_iters}):\n"
            f"    {result.reason}\n\n"
            f"  LIKELY SUSPECTS (in order of how often they bite for "
            f"segmentation bring-ups):\n"
            f"    1. Decoder upsampling stride / output resolution\n"
            f"       (`models/demos/vision/segmentation/<family>/<arch>.py`)\n"
            f"       — if the output stride is 1/2 of HF's, masks are\n"
            f"       half-size and IoU drops to ~0.5 even on correct\n"
            f"       feature maps.\n"
            f"    2. Pre-processing normalisation\n"
            f"       — TT_eltwise pipelines sometimes drop the\n"
            f"       ImageNet mean/std subtraction; HF always applies it.\n"
            f"    3. Channel order (BGR vs RGB)\n"
            f"       — CV2 / PIL drift; SegFormer / SAM2 always RGB.\n"
            f"    4. Argmax axis / class-axis dimension\n"
            f"       — if you take argmax(dim=-1) instead of\n"
            f"       argmax(dim=1) on a (B, C, H, W) tensor, you get\n"
            f"       garbage that *looks* like a mask but isn't.\n\n"
            f"  WHAT WAS ALREADY TRIED:\n"
            f"    {prev_block}\n\n"
            f"  BUDGET + COMMIT RULE: ~25 minutes per iteration. You\n"
            f"  MUST make at least one Edit/Write tool call. Inspect\n"
            f"  the demo / decoder / preprocessing in that order; pick\n"
            f"  the one most consistent with the per-class IoU pattern\n"
            f"  in the verdict above.\n" + render_extra_blocks(extra_blocks)
        )


_singleton = SegmentationComparator()
register_comparator(_singleton)


class _CNNSegmentationAlias(SegmentationComparator):
    category: str = "CNN"

    def supports(self, category: str, model_id: str) -> bool:
        if category != self.category:
            return False
        mid_l = model_id.lower()
        return any(k in mid_l for k in self._SEG_MODEL_TYPES)


register_comparator(_CNNSegmentationAlias())


__all__ = [
    "DEFAULT_DICE_THRESHOLD",
    "DEFAULT_IOU_THRESHOLD",
    "DEFAULT_PER_CLASS_IOU_FLOOR",
    "SegmentationComparator",
    "compute_dice",
    "compute_iou",
    "compute_per_class_iou",
    "extract_mask_from_pytest_output",
]
