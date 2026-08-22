"""Self-updating model_type classification via the installed transformers registries.

A model_type unknown to the hardcoded tables is classified from the venv's
transformers task-mappings, so new model_types are covered when transformers is
upgraded (which tracks upstream tt-metal via registry_sync) — no hand-edits. Only
a fallback: known types keep their curated category.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _f():
    return importlib.import_module("scripts.tt_hw_planner.probe")._category_from_transformers_registry


def test_real_transformers_classifies_unhardcoded_types():
    f = _f()
    assert f("mistral") == "LLM"
    assert f("deit") == "CNN"
    assert f("llava_next") == "VLM"


def test_precedence_vlm_stt_cnn_over_llm(monkeypatch):
    f = _f()
    ma = importlib.import_module("transformers.models.auto.modeling_auto")
    monkeypatch.setattr(ma, "MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES", {"dual": "X"}, raising=False)
    monkeypatch.setattr(ma, "MODEL_FOR_CAUSAL_LM_MAPPING_NAMES", {"dual": "Y", "onlylm": "Z"}, raising=False)
    monkeypatch.setattr(ma, "MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES", {"vlmtype": "V"}, raising=False)

    assert f("dual") == "CNN"
    assert f("onlylm") == "LLM"
    assert f("vlmtype") == "VLM"
    assert f("nope") is None


def test_unknown_type_returns_none():
    f = _f()
    assert f("definitely_not_a_real_model_type_xyz") is None
