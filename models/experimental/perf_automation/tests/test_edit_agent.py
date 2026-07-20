"""edit_file sub-agent — deterministic prompt + result validation (no key)."""

import pytest

from agent.edit_agent import EditError, _validate_edit_result, build_edit_prompt


def test_prompt_includes_lever_section_and_files():
    p = build_edit_prompt("mlp-fidelity-walk", "walk HiFi2 -> HiFi3", ["model.py", "attn.py"])
    assert "mlp-fidelity-walk" in p
    assert "walk HiFi2 -> HiFi3" in p
    assert "model.py" in p and "attn.py" in p


def test_prompt_has_lazy_fix_and_minimal_guards():
    p = build_edit_prompt("x", "y", ["m.py"]).lower()
    assert "smallest change" in p
    assert "do not delete" in p  # lazy-fix guard


def test_validate_accepts_well_formed():
    r = _validate_edit_result('{"files": ["model.py"], "summary": "bumped fidelity"}')
    assert r["files"] == ["model.py"]
    assert r["summary"] == "bumped fidelity"


def test_validate_rejects_empty_files():
    with pytest.raises(EditError):
        _validate_edit_result('{"files": [], "summary": "x"}')


def test_validate_rejects_non_json():
    with pytest.raises(EditError):
        _validate_edit_result("not json at all")
