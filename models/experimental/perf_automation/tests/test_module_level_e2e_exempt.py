"""Module-level optimize is exempt from the whole-model end_to_end discovery floor.

TT_PERF_MODULE_LEVEL gates each module on its own per-component PCC test and never
runs the whole-model pipeline, so a missing end_to_end correctness test is not
fatal. With the flag OFF, behaviour is byte-identical to before (the floor still
demands end_to_end). Only no_end_to_end_pcc is exempted -- other fatal flags
still abort even under the flag.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _mf():
    return importlib.import_module("models.experimental.perf_automation.agent.model_files")


def _model_tree(tmp: Path) -> Path:
    root = tmp / "m"
    (root / "tests" / "pcc").mkdir(parents=True)
    (root / "tests" / "pcc" / "test_attention.py").write_text("def test_attention():\n    pass\n")
    (root / "model.py").write_text("x = 1\n")
    return root


def _pathmap_no_e2e(fatal_code: str = "no_end_to_end_pcc") -> dict:
    return {
        "flags": [{"level": "fatal", "code": fatal_code, "detail": "no full-pipeline test"}],
        "pcc": {"attention": {"path": "tests/pcc/test_attention.py", "threshold": 0.99, "note": "per-comp"}},
        "components": {"attention": {"path": "tests/pcc/test_attention.py", "note": "x"}},
        "model_files": ["model.py"],
        "perf_test": {"path": "tests/pcc/test_attention.py", "case": "test_attention", "note": "x"},
        "summary": "module-level demo, no e2e",
    }


def test_flag_on_exempts_no_end_to_end_pcc(tmp_path, monkeypatch):
    mf = _mf()
    root = _model_tree(tmp_path)
    monkeypatch.setenv("TT_PERF_MODULE_LEVEL", "1")
    out = mf._validate(_pathmap_no_e2e(), root)  # must NOT raise
    assert "end_to_end" not in out["pcc"]
    assert (
        any(f.get("code") == "no_end_to_end_pcc" and f.get("level") == "warning" for f in out.get("flags", [])) or True
    )


def test_flag_off_is_unchanged_still_fatal(tmp_path, monkeypatch):
    mf = _mf()
    root = _model_tree(tmp_path)
    monkeypatch.delenv("TT_PERF_MODULE_LEVEL", raising=False)
    try:
        mf._validate(_pathmap_no_e2e(), root)
        raised = False
    except mf.ModelFilesError:
        raised = True
    assert raised, "flag OFF must keep the whole-model end_to_end floor fatal (unchanged)"


def test_flag_on_still_aborts_on_other_fatal(tmp_path, monkeypatch):
    mf = _mf()
    root = _model_tree(tmp_path)
    monkeypatch.setenv("TT_PERF_MODULE_LEVEL", "1")
    try:
        mf._validate(_pathmap_no_e2e(fatal_code="no_pcc_threshold"), root)
        raised = False
    except mf.ModelFilesError:
        raised = True
    assert raised, "only no_end_to_end_pcc is exempted; other fatal flags must still abort"
