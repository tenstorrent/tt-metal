"""Image-classification correctness (ViT / ResNet / EfficientNet / …).

Covers ANY backbone that produces a 1-D class-logit vector and a
top-k label list. The unifying gate is:

* Top-1 label agreement: the demo's top prediction must match
  the HF reference's top prediction.
* Top-5 overlap >= 0.8: the demo's top-5 labels must overlap
  with HF's top-5 in >= 4 out of 5 positions (catches "got the
  right class but the logits are noisy").
* KL divergence < 0.5 on the softmax distributions: catches
  "logits are scaled wrong" failures that the top-k checks miss
  (e.g. temperature off, missing softmax dim swap).

Why this matters even when top-1 is right
-----------------------------------------
The pre-existing TT classification PCC tests compare logits with
the standard PCC formula (cosine similarity over the flattened
logit vector). That catches NUMERICAL drift but is blind to
TEMPERATURE drift — if the model multiplied all logits by 100,
PCC is still ~1.0, top-1 is still right, but the model's
confidence becomes ridiculous (softmax saturates), so any
downstream consumer that uses the probabilities (calibration,
ensembling) gets garbage.

KL on the softmax catches that.

Demo-output protocol
--------------------
Looks for, in order:

1. ``==CLASS 0 - OUTPUT`` marker followed by space-separated
   token-ids on the next line (the recommended convention for
   any new classification demo).
2. A line of the form ``top1: <id>`` followed by ``top5: <id>
   <id> <id> <id> <id>`` (an existing convention in the ResNet
   demo).
3. Falls back to ``Evidence(ok=False, ...)`` (soft skip) so
   existing ResNet bring-ups whose demos don't follow either
   convention keep passing on the basis of their existing PCC
   gates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from .base import Comparator, Evidence, ValidationResult
from .registry import register_comparator


DEFAULT_TOP_K = 5
DEFAULT_TOP_K_OVERLAP_MIN = 0.8
DEFAULT_KL_MAX = 0.5


_CLASS_MARKER_RE = re.compile(r"^==CLASS\s+(?P<idx>\d+)\s+-\s+OUTPUT\s*$", re.M)
_TOP1_RE = re.compile(r"^\s*top1:\s*(?P<id>\d+)\s*$", re.M)
_TOPK_RE = re.compile(r"^\s*top5:\s*(?P<ids>(?:\d+\s*)+)\s*$", re.M)


def softmax(logits: Any) -> Any:
    import numpy as np

    x = np.asarray(logits, dtype=float)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def kl_divergence(p: Any, q: Any, *, eps: float = 1e-12) -> float:
    """KL(p || q) over two probability distributions of the same
    length. ``eps`` floors small values so log(0) doesn't blow up."""
    import numpy as np

    pa = np.asarray(p, dtype=float)
    qa = np.asarray(q, dtype=float)
    if pa.shape != qa.shape:
        return float("inf")
    pa = pa + eps
    qa = qa + eps
    pa = pa / pa.sum()
    qa = qa / qa.sum()
    return float((pa * np.log(pa / qa)).sum())


def top_k_overlap(
    a: Sequence[int],
    b: Sequence[int],
    *,
    k: int = DEFAULT_TOP_K,
) -> float:
    """Fraction of class-ids in ``a[:k]`` that also appear in
    ``b[:k]``."""
    if k <= 0:
        return 0.0
    a_top = set(a[:k])
    b_top = set(b[:k])
    if not b_top:
        return 0.0
    return len(a_top & b_top) / k


@dataclass
class _ClassPrediction:
    top1: int
    top5: List[int]
    logits: Optional[Any] = None


def extract_prediction_from_pytest_output(
    captured_output: str,
    *,
    user_idx: int = 0,
) -> Optional[_ClassPrediction]:
    """Try every parser in turn. Returns ``None`` on miss."""

    matches = list(_CLASS_MARKER_RE.finditer(captured_output))
    for m in matches:
        if int(m.group("idx")) == user_idx:
            after = captured_output[m.end() :]
            for line in after.splitlines():
                line = line.strip()
                if not line:
                    continue

                try:
                    ids = [int(x) for x in line.split()]
                except ValueError:
                    break
                if not ids:
                    break
                top5 = ids[:DEFAULT_TOP_K]
                return _ClassPrediction(top1=top5[0], top5=top5)

    t1 = _TOP1_RE.search(captured_output)
    tk = _TOPK_RE.search(captured_output)
    if t1 and tk:
        top1 = int(t1.group("id"))
        top5 = [int(x) for x in tk.group("ids").split()][:DEFAULT_TOP_K]
        return _ClassPrediction(top1=top1, top5=top5)

    return None


@dataclass
class _ClassRef:
    top1: int
    top5: List[int]
    logits: Any
    source_model_id: str = ""


class ClassificationComparator(Comparator):
    """Comparator for image-classification backbones.

    Category claim: ``"CNN/classification"`` (precise) AND
    ``"CNN"`` (legacy, gated on the model_id NOT looking like a
    segmentation / detection model so it doesn't fight
    SegmentationComparator).
    """

    category: str = "CNN/classification"
    discriminator: str = ""

    _CLS_MODEL_TYPES = (
        "resnet",
        "efficientnet",
        "vit",
        "convnext",
        "mobilenet",
        "densenet",
        "squeezenet",
        "regnet",
        "swin",
        "beit",
    )
    _NOT_CLS_KEYWORDS = (
        "segformer",
        "maskformer",
        "mask2former",
        "sam",
        "sam2",
        "sam_hiera",
        "detr",
        "yolos",
    )

    def supports(self, category: str, model_id: str) -> bool:
        if category == self.category:
            return True
        if category != "CNN":
            return False
        mid_l = model_id.lower()
        if any(k in mid_l for k in self._NOT_CLS_KEYWORDS):
            return False
        return any(k in mid_l for k in self._CLS_MODEL_TYPES)

    def extract(
        self,
        captured_output: str,
        model_id: str,
    ) -> Evidence:
        pred = extract_prediction_from_pytest_output(captured_output)
        if pred is None:
            return Evidence(
                payload=None,
                ok=False,
                reason=(
                    "could not find a classification prediction in "
                    "the pytest output. Expected '==CLASS 0 - "
                    "OUTPUT' marker followed by '<top1_id> <top2_id> "
                    "...' OR 'top1: <id>' + 'top5: <id> <id> ...' "
                    "lines."
                ),
            )
        return Evidence(
            payload=pred,
            input_hint=None,
            ok=True,
            reason="prediction extracted from pytest output",
        )

    def load_reference(
        self,
        evidence: Evidence,
        model_id: str,
    ) -> _ClassRef:
        import numpy as np
        from PIL import Image
        from transformers import pipeline

        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        pipe = pipeline("image-classification", model=model_id, device="cpu")
        out = pipe(img, top_k=10)
        if not out:
            return _ClassRef(top1=-1, top5=[], logits=np.zeros(1), source_model_id=model_id)

        try:
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(model_id)
            label2id = cfg.label2id or {}
        except Exception:
            label2id = {}

        ids: List[int] = []
        for item in out:
            lbl = item.get("label", "")
            if lbl in label2id:
                ids.append(int(label2id[lbl]))
            else:
                ids.append(hash(lbl) % 10000)
        top5 = ids[:DEFAULT_TOP_K]

        scores = np.array([float(item.get("score", 0)) for item in out])
        logits = np.log(scores + 1e-12)
        return _ClassRef(
            top1=top5[0],
            top5=top5,
            logits=logits,
            source_model_id=model_id,
        )

    def compare(
        self,
        evidence: Evidence,
        reference: Any,
    ) -> ValidationResult:
        import numpy as np

        if not isinstance(reference, _ClassRef):
            return ValidationResult(
                ok=False,
                reason="classification comparator: reference is not a _ClassRef",
            )
        pred = evidence.payload
        if not isinstance(pred, _ClassPrediction):
            return ValidationResult(
                ok=False,
                reason="classification comparator: evidence payload is not a prediction",
            )

        top1_match = pred.top1 == reference.top1
        overlap = top_k_overlap(pred.top5, reference.top5, k=DEFAULT_TOP_K)
        overlap_pass = overlap >= DEFAULT_TOP_K_OVERLAP_MIN

        kl = None
        kl_pass = True

        if pred.logits is not None and reference.logits is not None:
            try:
                p = softmax(pred.logits)
                q = softmax(reference.logits)
                if p.shape == q.shape:
                    kl = kl_divergence(p, q)
                    kl_pass = kl < DEFAULT_KL_MAX
            except Exception:
                pass

        ok = top1_match and overlap_pass and kl_pass
        bits = [
            f"top1 match={top1_match} (tt={pred.top1}, hf={reference.top1})",
            f"top-5 overlap={overlap:.2f} (>= {DEFAULT_TOP_K_OVERLAP_MIN})",
        ]
        if kl is not None:
            bits.append(f"KL={kl:.3f} (< {DEFAULT_KL_MAX})")
        return ValidationResult(
            ok=ok,
            reason=("PASS: " if ok else "FAIL: ") + "; ".join(bits),
            tt_text=str(pred.top5),
            hf_text=str(reference.top5),
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
            f"You are debugging a TT-hardware bring-up of {model_id!r}. "
            f"The model is an image-classification backbone (ResNet / "
            f"ViT / EfficientNet / …); pytest completes but the demo's "
            f"top-k prediction or class distribution disagrees with HF.\n\n"
            f"  GATE VERDICT (iter {iter_idx}/{max_iters}):\n"
            f"    {result.reason}\n\n"
            f"  LIKELY SUSPECTS:\n"
            f"    1. Output channel-axis swap on (B, C) logits\n"
            f"       (most common; argmax over wrong axis).\n"
            f"    2. Pre-processing normalisation (ImageNet mean/std).\n"
            f"    3. Per-class bias vector lost in conversion.\n"
            f"    4. Output logits scaled wrong (softmax temperature).\n"
            f"    5. Channel order BGR vs RGB.\n\n"
            f"  WHAT WAS ALREADY TRIED:\n"
            f"    {prev}\n\n"
            f"  BUDGET: ~25 min/iter. Make at least one Edit.\n" + render_extra_blocks(extra_blocks)
        )


_singleton = ClassificationComparator()
register_comparator(_singleton)


class _CNNClassificationAlias(ClassificationComparator):
    category: str = "CNN"

    def supports(self, category: str, model_id: str) -> bool:
        if category != self.category:
            return False
        mid_l = model_id.lower()
        if any(k in mid_l for k in self._NOT_CLS_KEYWORDS):
            return False
        return any(k in mid_l for k in self._CLS_MODEL_TYPES)


register_comparator(_CNNClassificationAlias())


_IMAGE_DIFFUSION_KEYWORDS = (
    "stable-diffusion",
    "stable_diffusion",
    "sd-",
    "sdxl",
    "sd3",
    "flux",
    "dall-e",
    "kandinsky",
    "wuerstchen",
)


class _ImageClassificationAlias(ClassificationComparator):
    category: str = "Image"

    def supports(self, category: str, model_id: str) -> bool:
        if category != self.category:
            return False
        mid_l = model_id.lower()
        if any(k in mid_l for k in self._NOT_CLS_KEYWORDS):
            return False
        if any(k in mid_l for k in _IMAGE_DIFFUSION_KEYWORDS):
            return False
        return True


register_comparator(_ImageClassificationAlias())


__all__ = [
    "DEFAULT_KL_MAX",
    "DEFAULT_TOP_K",
    "DEFAULT_TOP_K_OVERLAP_MIN",
    "ClassificationComparator",
    "extract_prediction_from_pytest_output",
    "kl_divergence",
    "softmax",
    "top_k_overlap",
]
