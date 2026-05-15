from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "tools/ci/m6_bug_escape_enrichment.py"
    spec = importlib.util.spec_from_file_location("m6_bug_escape_enrichment", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_classify_escape_type_detects_lower_to_higher_escape() -> None:
    mod = _load_module()
    escape_type, confidence = mod.classify_escape_type(failure_layer="models", fix_layer="llk")
    assert escape_type == "layer_escape_lower_to_higher"
    assert confidence == "high"


def test_infer_layer_from_text_prefers_specific_signals() -> None:
    mod = _load_module()
    assert mod.infer_layer_from_text("TT_FATAL in tt_metal/llrt/something.cpp") == "metalium"
    assert mod.infer_layer_from_text("flaky model pipeline error") == "models"
    assert mod.infer_layer_from_text("something unknown") == "unknown"
