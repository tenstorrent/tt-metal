"""Unit tests for the LLM verify pass (Item 2 of the 2026-06-02 audit).

The verify pass is a static code-review step that compares HF's
``forward()`` to the TT chained forward and emits a structured JSON
verdict naming what's missing. Distinct from the chain-divergence
diagnostic (Item 1) which uses runtime activation statistics.

These tests cover:

  * ``build_verify_prompt`` — pure function, schema fields rendered,
    placeholder handling for missing inputs.
  * ``parse_verify_verdict`` — schema validation, every malformed
    input degrades to None rather than raising.
  * ``resolve_tt_chained_source`` — file lookup priority (demo/demo.py
    before bare demo.py), graceful return-None on missing.
  * ``run_llm_verify_pass`` — orchestrator with ``_invoke_agent``
    mocked; covers each success and failure mode.

``resolve_hf_forward_source`` is not unit-tested in isolation because
it requires real transformers imports and a real config object —
that's covered indirectly via the orchestrator's hf_forward_src
override path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

from scripts.tt_hw_planner._cli_helpers.llm_verify import (
    LLMVerifyVerdict,
    build_verify_prompt,
    parse_verify_verdict,
    resolve_tt_chained_source,
    run_llm_verify_pass,
)


# ─── build_verify_prompt ─────────────────────────────────────────────


def test_prompt_includes_model_id_and_paths(tmp_path: Path) -> None:
    """Prompt must carry the model id and the verdict path so the LLM
    knows what to write where. Pin so a refactor doesn't drop them."""
    verdict_path = tmp_path / "verdict.json"
    p = build_verify_prompt(
        model_id="org/model",
        hf_forward_src="def forward(self, x): ...",
        tt_chained_src="def chained_forward(x): ...",
        drift_summary="layer_0 drift 0.5",
        verdict_path=verdict_path,
    )
    assert "org/model" in p
    assert str(verdict_path) in p
    assert "PASS" in p and "FAIL" in p  # verdict schema visible


def test_prompt_renders_no_chained_source_placeholder(tmp_path: Path) -> None:
    """When TT chain doesn't exist (tt_chained_src=None), prompt must
    show a placeholder so the LLM knows to escalate to synthesis."""
    p = build_verify_prompt(
        model_id="org/m",
        hf_forward_src="def forward(self, x): ...",
        tt_chained_src=None,
        drift_summary="",
        verdict_path=tmp_path / "v.json",
    )
    assert "no TT chained forward exists yet" in p


def test_prompt_renders_no_drift_placeholder(tmp_path: Path) -> None:
    """Empty drift_summary → placeholder, not blank section."""
    p = build_verify_prompt(
        model_id="org/m",
        hf_forward_src="def forward(self, x): ...",
        tt_chained_src="...",
        drift_summary="",
        verdict_path=tmp_path / "v.json",
    )
    assert "no chain-divergence diagnostic available" in p


def test_prompt_includes_schema_fields_for_diagnostic(tmp_path: Path) -> None:
    """The diagnostic schema fields must be visible to the LLM so it
    knows what keys to populate. Pinning so a doc update doesn't
    accidentally drop a key the parser later expects."""
    p = build_verify_prompt(
        model_id="org/m",
        hf_forward_src="x",
        tt_chained_src="y",
        drift_summary="",
        verdict_path=tmp_path / "v.json",
    )
    for field in (
        "missing_args",
        "missing_branches",
        "missing_intermediate_ops",
        "missing_modules",
        "summary",
    ):
        assert field in p, f"diagnostic field {field!r} missing from prompt"


def test_prompt_forbids_source_edits(tmp_path: Path) -> None:
    """Verify is READ-ONLY. The prompt must explicitly forbid the LLM
    from modifying source files. Test pins the safety contract."""
    p = build_verify_prompt(
        model_id="org/m",
        hf_forward_src="x",
        tt_chained_src="y",
        drift_summary="",
        verdict_path=tmp_path / "v.json",
    )
    assert "DO NOT modify" in p or "DO NOT make any edits" in p


# ─── parse_verify_verdict ───────────────────────────────────────────


def _write(path: Path, content: Any) -> None:
    if isinstance(content, str):
        path.write_text(content, encoding="utf-8")
    else:
        path.write_text(json.dumps(content), encoding="utf-8")


def test_parse_returns_none_when_file_missing(tmp_path: Path) -> None:
    assert parse_verify_verdict(tmp_path / "nope.json") is None


def test_parse_returns_none_for_malformed_json(tmp_path: Path) -> None:
    p = tmp_path / "v.json"
    _write(p, "{not json")
    assert parse_verify_verdict(p) is None


def test_parse_returns_none_when_top_level_not_dict(tmp_path: Path) -> None:
    """A list / scalar at the top is invalid — only dicts are accepted."""
    p = tmp_path / "v.json"
    _write(p, ["PASS", "no_diagnostic"])
    assert parse_verify_verdict(p) is None


def test_parse_returns_none_when_verdict_missing(tmp_path: Path) -> None:
    p = tmp_path / "v.json"
    _write(p, {"diagnostic": {"summary": "looks fine"}})  # no verdict key
    assert parse_verify_verdict(p) is None


def test_parse_returns_none_for_unrecognized_verdict_value(tmp_path: Path) -> None:
    """Only PASS / FAIL accepted — strict whitelist prevents the LLM
    smuggling 'MAYBE' / 'NEEDS_REVIEW' / etc. that the downstream router
    has no rules for."""
    p = tmp_path / "v.json"
    _write(p, {"verdict": "MAYBE", "diagnostic": {}})
    assert parse_verify_verdict(p) is None


def test_parse_normalizes_pass_verdict_case(tmp_path: Path) -> None:
    """LLM may emit 'pass' or 'Pass'; parser uppercases to PASS so the
    downstream ok-check is case-insensitive."""
    p = tmp_path / "v.json"
    _write(p, {"verdict": "pass", "diagnostic": {"summary": "ok"}})
    result = parse_verify_verdict(p)
    assert result is not None
    assert result.verdict == "PASS"
    assert result.ok is True


def test_parse_fail_verdict_sets_ok_false(tmp_path: Path) -> None:
    p = tmp_path / "v.json"
    _write(p, {"verdict": "FAIL", "diagnostic": {"summary": "missing reshape"}})
    result = parse_verify_verdict(p)
    assert result is not None
    assert result.verdict == "FAIL"
    assert result.ok is False
    assert result.diagnostic == {"summary": "missing reshape"}


def test_parse_preserves_raw_text_for_debugging(tmp_path: Path) -> None:
    """raw_text field carries the source bytes so a stuck-iter post-mortem
    can see exactly what the LLM said."""
    p = tmp_path / "v.json"
    body = '{"verdict": "PASS", "diagnostic": {"summary": "ok"}}'
    p.write_text(body, encoding="utf-8")
    result = parse_verify_verdict(p)
    assert result is not None
    assert result.raw_text == body


def test_parse_treats_non_dict_diagnostic_as_empty(tmp_path: Path) -> None:
    """diagnostic field present but wrong type → default to empty dict,
    don't reject the whole verdict (the verdict itself is the load-bearing
    field for routing)."""
    p = tmp_path / "v.json"
    _write(p, {"verdict": "PASS", "diagnostic": "not a dict"})
    result = parse_verify_verdict(p)
    assert result is not None
    assert result.diagnostic == {}


# ─── resolve_tt_chained_source ──────────────────────────────────────


def test_resolver_returns_none_for_missing_demo_dir(tmp_path: Path) -> None:
    """No demo_dir → None (signal to verify prompt that synthesis is needed)."""
    assert resolve_tt_chained_source(tmp_path / "does-not-exist") is None


def test_resolver_returns_none_when_no_demo_py(tmp_path: Path) -> None:
    """demo_dir exists but has no demo.py → None."""
    (tmp_path / "_stubs").mkdir()
    assert resolve_tt_chained_source(tmp_path) is None


def test_resolver_prefers_demo_subdir_over_bare(tmp_path: Path) -> None:
    """``demo/demo.py`` (per-model bring-up convention) takes priority
    over a bare ``demo.py`` at the demo_dir root. Pin priority so a
    refactor doesn't accidentally flip it."""
    bare = tmp_path / "demo.py"
    bare.write_text("# bare\n", encoding="utf-8")
    sub = tmp_path / "demo"
    sub.mkdir()
    inner = sub / "demo.py"
    inner.write_text("# inner\n", encoding="utf-8")
    src = resolve_tt_chained_source(tmp_path)
    assert src is not None
    assert "inner" in src
    assert "bare" not in src


def test_resolver_falls_back_to_bare_demo_py(tmp_path: Path) -> None:
    """No demo/ subdir but bare demo.py exists → use bare."""
    bare = tmp_path / "demo.py"
    bare.write_text("# bare\n", encoding="utf-8")
    src = resolve_tt_chained_source(tmp_path)
    assert src is not None
    assert "bare" in src


# ─── run_llm_verify_pass orchestrator ───────────────────────────────


def _stub_invoke_agent(verdict_payload: Dict[str, Any]):
    """Build a side-effect for patched _invoke_agent that writes the
    verdict file at the expected path and returns rc=0."""

    def _stub(prompt, *, expected_deliverable_files=None, **kwargs):
        if expected_deliverable_files:
            for p in expected_deliverable_files:
                Path(p).write_text(json.dumps(verdict_payload), encoding="utf-8")
        return 0

    return _stub


def test_orchestrator_returns_verdict_on_happy_path(tmp_path: Path) -> None:
    """Agent writes valid verdict file → orchestrator parses + returns."""
    payload = {"verdict": "PASS", "diagnostic": {"summary": "looks correct"}}
    with patch(
        "scripts.tt_hw_planner._cli_helpers.agent._invoke_agent",
        side_effect=_stub_invoke_agent(payload),
    ):
        result = run_llm_verify_pass(
            model_id="org/m",
            demo_dir=tmp_path,
            hf_forward_src="def forward(self): ...",
            tt_chained_src="def chained(): ...",
        )
    assert isinstance(result, LLMVerifyVerdict)
    assert result.ok is True
    assert result.diagnostic == {"summary": "looks correct"}


def test_orchestrator_returns_none_on_agent_nonzero_rc(tmp_path: Path) -> None:
    """Agent failed (timeout, OOM, etc.) → no verdict, return None."""

    def _stub(prompt, **kwargs):
        return 124  # timeout rc

    with patch(
        "scripts.tt_hw_planner._cli_helpers.agent._invoke_agent",
        side_effect=_stub,
    ):
        result = run_llm_verify_pass(
            model_id="org/m",
            demo_dir=tmp_path,
            hf_forward_src="x",
            tt_chained_src="y",
        )
    assert result is None


def test_orchestrator_returns_none_on_agent_exception(tmp_path: Path) -> None:
    """_invoke_agent raises → caught, returns None. Never propagates."""
    with patch(
        "scripts.tt_hw_planner._cli_helpers.agent._invoke_agent",
        side_effect=RuntimeError("agent boom"),
    ):
        result = run_llm_verify_pass(
            model_id="org/m",
            demo_dir=tmp_path,
            hf_forward_src="x",
            tt_chained_src="y",
        )
    assert result is None


def test_orchestrator_returns_none_when_agent_skips_writing_verdict(tmp_path: Path) -> None:
    """Agent returned rc=0 but didn't write the verdict file (e.g.
    misunderstood the prompt) → parser returns None, orchestrator
    propagates that."""

    def _stub(prompt, **kwargs):
        return 0  # rc=0 but no file written

    with patch(
        "scripts.tt_hw_planner._cli_helpers.agent._invoke_agent",
        side_effect=_stub,
    ):
        result = run_llm_verify_pass(
            model_id="org/m",
            demo_dir=tmp_path,
            hf_forward_src="x",
            tt_chained_src="y",
        )
    assert result is None


def test_orchestrator_creates_verify_subdir(tmp_path: Path) -> None:
    """The verdict file lands under demo_dir/_verify/, so the
    subdir must be created before the agent runs (otherwise the
    write would fail and the agent would return rc=0 without writing,
    silently)."""
    captured: Dict[str, Any] = {}

    def _stub(prompt, *, expected_deliverable_files=None, **kwargs):
        if expected_deliverable_files:
            captured["paths"] = list(expected_deliverable_files)
        Path(expected_deliverable_files[0]).write_text(json.dumps({"verdict": "PASS"}), encoding="utf-8")
        return 0

    with patch(
        "scripts.tt_hw_planner._cli_helpers.agent._invoke_agent",
        side_effect=_stub,
    ):
        run_llm_verify_pass(
            model_id="org/m",
            demo_dir=tmp_path,
            hf_forward_src="x",
            tt_chained_src="y",
        )
    assert (tmp_path / "_verify").is_dir()
    assert "paths" in captured
    assert any("_verify" in str(p) for p in captured["paths"])


def test_orchestrator_clears_stale_verdict_before_agent_runs(tmp_path: Path) -> None:
    """Prior run left a stale verdict file. Orchestrator must remove
    it before invoking the agent so a parse never reads stale data
    when the agent fails to write."""
    verify_dir = tmp_path / "_verify"
    verify_dir.mkdir()
    stale = verify_dir / "verdict.json"
    stale.write_text(json.dumps({"verdict": "PASS"}), encoding="utf-8")

    def _stub(prompt, **kwargs):
        # Agent fails to write — stale file should NOT carry through.
        return 1

    with patch(
        "scripts.tt_hw_planner._cli_helpers.agent._invoke_agent",
        side_effect=_stub,
    ):
        result = run_llm_verify_pass(
            model_id="org/m",
            demo_dir=tmp_path,
            hf_forward_src="x",
            tt_chained_src="y",
        )
    assert result is None
    # Stale file was deleted before the agent ran
    assert not stale.exists()


def test_orchestrator_passes_drift_summary_into_prompt(tmp_path: Path) -> None:
    """The chain-divergence diagnostic (Item 1) provides a drift_summary;
    Item 2's prompt must include it so the LLM has full context."""
    captured_prompt: Dict[str, Any] = {}

    def _stub(prompt, *, expected_deliverable_files=None, **kwargs):
        captured_prompt["text"] = prompt
        Path(expected_deliverable_files[0]).write_text(json.dumps({"verdict": "PASS"}), encoding="utf-8")
        return 0

    drift_msg = "first divergence at vision_encoder.layer_0 (mean drift 0.5)"
    with patch(
        "scripts.tt_hw_planner._cli_helpers.agent._invoke_agent",
        side_effect=_stub,
    ):
        run_llm_verify_pass(
            model_id="org/m",
            demo_dir=tmp_path,
            hf_forward_src="x",
            tt_chained_src="y",
            drift_summary=drift_msg,
        )
    assert drift_msg in captured_prompt["text"]
