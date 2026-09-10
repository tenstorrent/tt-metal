"""Unit tests for the cli-side wiring of the Step 1->2->3 orchestrator.

These cover the three helpers added to ``cli.py`` to bridge between
the cmd_up flow and the standalone orchestrator module:

  * ``_resolve_model_type`` — extracts HF model_type for the family key
  * ``_list_graduated_components_for_orchestrator`` — reads
    bringup_status.json and renders the orchestrator-shaped spec list
  * ``_maybe_run_e2e_orchestrator`` — gated entry point that fires
    only when ``TT_HW_PLANNER_USE_E2E_ORCHESTRATOR=1``

The orchestrator itself is tested in
``test_e2e_orchestrator.py``; these tests just confirm the wiring
correctly reads inputs and gates on the env var.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from scripts.tt_hw_planner.cli import (
    _list_graduated_components_for_orchestrator,
    _maybe_run_e2e_orchestrator,
)


@contextmanager
def _env_var(name: str, value):
    sentinel = object()
    prev = os.environ.get(name, sentinel)
    try:
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
        yield
    finally:
        if prev is sentinel:
            os.environ.pop(name, None)
        else:
            os.environ[name] = prev  # type: ignore[arg-type]


# ─── _list_graduated_components_for_orchestrator ────────────────────


def test_list_returns_empty_for_missing_demo_dir(tmp_path: Path) -> None:
    assert _list_graduated_components_for_orchestrator(tmp_path / "nope") == []


def test_list_returns_empty_for_missing_manifest(tmp_path: Path) -> None:
    assert _list_graduated_components_for_orchestrator(tmp_path) == []


def test_list_returns_empty_for_malformed_manifest(tmp_path: Path) -> None:
    (tmp_path / "bringup_status.json").write_text("{not json")
    assert _list_graduated_components_for_orchestrator(tmp_path) == []


def test_list_renders_component_specs(tmp_path: Path) -> None:
    """Reads name + stub_path + hf_reference + class_name from manifest."""
    (tmp_path / "bringup_status.json").write_text(
        json.dumps(
            {
                "components": [
                    {
                        "name": "vision_encoder",
                        "stub_path": "_stubs/ve.py",
                        "hf_reference": "model.vision",
                        "class_name": "VisionEncoder",
                        "status": "NEW",
                    },
                    {
                        "name": "mask_decoder",
                        "hf_reference": "model.decoder",
                        "class_name": "MaskDecoder",
                        "status": "ADAPT",
                    },
                ]
            }
        )
    )
    specs = _list_graduated_components_for_orchestrator(tmp_path)
    assert len(specs) == 2
    assert specs[0]["name"] == "vision_encoder"
    assert specs[0]["stub_path"] == "_stubs/ve.py"
    assert specs[1]["name"] == "mask_decoder"
    # Default stub_path when missing from manifest
    assert specs[1]["stub_path"].endswith("mask_decoder.py")


def test_list_skips_components_without_name(tmp_path: Path) -> None:
    """Defensive: a manifest entry with no name shouldn't break the
    spec build."""
    (tmp_path / "bringup_status.json").write_text(
        json.dumps(
            {
                "components": [
                    {"hf_reference": "x", "class_name": "Y"},  # no name
                    {"name": "good", "hf_reference": "z", "class_name": "W"},
                ]
            }
        )
    )
    specs = _list_graduated_components_for_orchestrator(tmp_path)
    names = [s["name"] for s in specs]
    assert "good" in names
    assert len(specs) == 1


# ─── _maybe_run_e2e_orchestrator ────────────────────────────────────


def test_maybe_returns_none_when_env_var_unset(tmp_path: Path) -> None:
    """No env var → orchestrator stays disabled, existing flow runs.
    This is the safety net during rollout."""
    with _env_var("TT_HW_PLANNER_USE_E2E_ORCHESTRATOR", None):
        assert _maybe_run_e2e_orchestrator(model_id="org/m", demo_dir=tmp_path) is None


def test_maybe_returns_none_when_env_var_not_1(tmp_path: Path) -> None:
    """Env var must be exactly '1' — '0' / 'true' / empty all skip.
    Strict opt-in matches the rest of the tool's env-var conventions."""
    for val in ("0", "true", "yes", "false", ""):
        with _env_var("TT_HW_PLANNER_USE_E2E_ORCHESTRATOR", val):
            assert _maybe_run_e2e_orchestrator(model_id="org/m", demo_dir=tmp_path) is None


def test_maybe_returns_none_for_empty_model_id(tmp_path: Path) -> None:
    """Empty model_id → orchestrator can't look up family — skip."""
    with _env_var("TT_HW_PLANNER_USE_E2E_ORCHESTRATOR", "1"):
        assert _maybe_run_e2e_orchestrator(model_id="", demo_dir=tmp_path) is None


def test_maybe_returns_none_for_no_demo_dir() -> None:
    with _env_var("TT_HW_PLANNER_USE_E2E_ORCHESTRATOR", "1"):
        assert _maybe_run_e2e_orchestrator(model_id="org/m", demo_dir=None) is None


def test_maybe_invokes_orchestrator_when_enabled(tmp_path: Path) -> None:
    """Env var set + valid inputs → orchestrator fires. Mock at the
    point cli imports it (run_e2e_bringup) so we don't exercise the
    real Step 1->2->3 flow here (that's e2e_orchestrator's own tests)."""
    from scripts.tt_hw_planner._cli_helpers.e2e_orchestrator import E2EBringupResult

    fake_result = E2EBringupResult(status="VERIFY_PASSED", family_key="sam2")
    (tmp_path / "bringup_status.json").write_text(json.dumps({"components": []}))

    with _env_var("TT_HW_PLANNER_USE_E2E_ORCHESTRATOR", "1"), patch(
        "scripts.tt_hw_planner._cli_helpers.e2e_orchestrator.run_e2e_bringup",
        return_value=fake_result,
    ), patch("scripts.tt_hw_planner.cli._resolve_model_type", return_value="sam2"):
        result = _maybe_run_e2e_orchestrator(model_id="org/m", demo_dir=tmp_path)
    assert result is fake_result


def test_maybe_returns_none_when_orchestrator_raises(tmp_path: Path, capsys) -> None:
    """If the orchestrator raises, the helper catches and returns
    None — existing cli flow continues as the safety net."""
    with _env_var("TT_HW_PLANNER_USE_E2E_ORCHESTRATOR", "1"), patch(
        "scripts.tt_hw_planner._cli_helpers.e2e_orchestrator.run_e2e_bringup",
        side_effect=RuntimeError("orchestrator boom"),
    ), patch("scripts.tt_hw_planner.cli._resolve_model_type", return_value="sam2"):
        result = _maybe_run_e2e_orchestrator(model_id="org/m", demo_dir=tmp_path)
    assert result is None
    # And the failure is surfaced to stderr so operators can see it
    err = capsys.readouterr().err
    assert "orchestrator boom" in err


def test_maybe_forwards_chain_divergence_summary(tmp_path: Path) -> None:
    """The drift summary from Item 1 must propagate into the
    orchestrator call so synthesis has full context."""
    captured = {}

    def _fake_run(**kwargs):
        captured.update(kwargs)
        from scripts.tt_hw_planner._cli_helpers.e2e_orchestrator import E2EBringupResult

        return E2EBringupResult(status="VERIFY_PASSED")

    (tmp_path / "bringup_status.json").write_text(json.dumps({"components": []}))
    with _env_var("TT_HW_PLANNER_USE_E2E_ORCHESTRATOR", "1"), patch(
        "scripts.tt_hw_planner._cli_helpers.e2e_orchestrator.run_e2e_bringup",
        side_effect=_fake_run,
    ), patch("scripts.tt_hw_planner.cli._resolve_model_type", return_value="sam2"):
        _maybe_run_e2e_orchestrator(
            model_id="org/m",
            demo_dir=tmp_path,
            chain_divergence_summary="layer_0 drift 0.5",
        )
    assert captured.get("chain_divergence_summary") == "layer_0 drift 0.5"
