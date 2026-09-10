"""Category-agnostic correctness gating for ``tt_hw_planner``.

The ``tt_hw_planner`` tool brings up models on TT hardware across many
HuggingFace categories: LLM, VLM, STT, TTS, vision-classification,
vision-segmentation, diffusion, embeddings, NLP, video. Until 2026-05,
only the LLM/VLM fast paths had any "is the runtime output actually
correct?" check (the token-overlap PCC gate in
:mod:`output_validation`). Every other category was rubber-stamped
as a successful bring-up the moment pytest exited 0, which (a) misses
late-decode collapse even on LLMs (see medgemma-4b-it), and (b)
completely fails to detect "right runtime, wrong answer" outcomes
on segmentation / classification / audio / etc.

This package is the home for a category-keyed correctness layer.
Each category contributes a :class:`Comparator` (declared in
:mod:`.base`) registered in :mod:`.registry`. The planner calls
:func:`run_gate` (the dispatcher in :mod:`.engine`) with the category
and the captured pytest output; the dispatcher routes to the
appropriate comparator.

Phased rollout (see audit dated 2026-05-23):

* **Phase 0** (this commit). Package skeleton, ``Comparator`` ABC,
  registry, ``text.TextComparator`` thin-wrapping the existing
  :mod:`output_validation` logic, ``--pcc-engine`` CLI flag (default
  ``legacy``). **Behavior change: none.** The flag exists, both
  engines route to the same legacy code, the planner ignores
  category-keyed comparators. We ship Phase 0 first so that every
  later phase can be reverted with a single flag flip.
* **Phase 1**. The ``evidence`` engine grows real machinery:
  :mod:`.evidence`, :mod:`.hypothesis`, :mod:`.diagnose`,
  :mod:`.planner`. Still LLM/VLM only. Widens the gate from 32 →
  128 tokens, adds a mid-sequence collapse detector, makes prior
  iterations' diffs visible to the next iteration.
* **Phase 2+**. New comparators: ``segmentation`` (SAM2),
  ``classification`` (ViT/ResNet), ``audio_asr`` (Whisper),
  ``embedding`` (sentence-BERT), ``diffusion`` (SD), ``detection``
  (DETR). Each phase adds one category at a time, opt-in via
  ``--pcc-categories=<list>``.

Public surface (kept narrow so the rest of the tool depends on a
small API):

* :class:`Comparator` — the ABC every category implements.
* :class:`ValidationResult` — re-exported from :mod:`output_validation`
  so existing call sites keep working.
* :func:`get_comparator` — registry lookup by category string.
* :func:`run_gate` — engine dispatcher; routes (engine, category) →
  legacy vs evidence path.
"""

from __future__ import annotations

from .base import Comparator, ValidationResult
from .registry import (
    get_comparator,
    list_categories,
    register_comparator,
)


from . import text as _text_module
from . import segmentation as _segmentation_module
from . import classification as _classification_module
from . import audio_asr as _audio_asr_module
from . import embedding as _embedding_module
from . import diffusion as _diffusion_module
from . import detection as _detection_module


def run_gate(*args, **kwargs):
    """Dispatcher entry point. Implementation lives in
    :mod:`.engine` to avoid an import cycle at package init time."""
    from .engine import run_gate as _run_gate

    return _run_gate(*args, **kwargs)


def run_evidence_gate(*args, **kwargs):
    """Convenience re-export of :func:`.engine.run_evidence_gate`
    so tests and the cli can ``from .correctness import
    run_evidence_gate`` without depending on the submodule layout."""
    from .engine import run_evidence_gate as _run_evidence_gate

    return _run_evidence_gate(*args, **kwargs)


__all__ = [
    "Comparator",
    "ValidationResult",
    "get_comparator",
    "list_categories",
    "register_comparator",
    "run_evidence_gate",
    "run_gate",
]
