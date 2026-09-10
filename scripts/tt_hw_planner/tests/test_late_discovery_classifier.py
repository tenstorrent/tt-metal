"""Unit tests for the late-discovery classifier (Item 5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from scripts.tt_hw_planner._cli_helpers.late_discovery_classifier import (
    MissingPieceClassification,
    build_classify_prompt,
    heuristic_classify,
    parse_classify_verdict,
    run_classify_pass,
)


# ─── Prompt builder ─────────────────────────────────────────────────


def test_prompt_includes_missing_piece_description(tmp_path: Path) -> None:
    p = build_classify_prompt(
        model_id="org/m",
        missing_piece_description="reshape from (HW,B,C) to (B,C,H,W)",
        hf_forward_src="def forward(self, x): ...",
        graduated_components=[],
        verdict_path=tmp_path / "v.json",
    )
    assert "reshape from (HW,B,C) to (B,C,H,W)" in p
    assert "Case A" in p and "Case B" in p and "Case C" in p


def test_prompt_lists_graduated_components(tmp_path: Path) -> None:
    """Already-graduated components must be listed so LLM doesn't
    re-suggest Case C for one we already have."""
    p = build_classify_prompt(
        model_id="org/m",
        missing_piece_description="missing something",
        hf_forward_src="",
        graduated_components=[{"name": "attn", "class_name": "Attention"}],
        verdict_path=tmp_path / "v.json",
    )
    assert "attn" in p


def test_prompt_forbids_source_edits(tmp_path: Path) -> None:
    """Classifier is read-only — pin the contract in the prompt text."""
    p = build_classify_prompt(
        model_id="org/m",
        missing_piece_description="x",
        hf_forward_src="",
        graduated_components=[],
        verdict_path=tmp_path / "v.json",
    )
    assert "DO NOT" in p


# ─── Verdict parser ─────────────────────────────────────────────────


def _write(path: Path, blob: Any) -> None:
    if isinstance(blob, str):
        path.write_text(blob)
    else:
        path.write_text(json.dumps(blob))


def test_parse_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert parse_classify_verdict(tmp_path / "nope.json") is None


def test_parse_returns_none_for_malformed_json(tmp_path: Path) -> None:
    p = tmp_path / "v.json"
    _write(p, "{not valid")
    assert parse_classify_verdict(p) is None


def test_parse_returns_none_for_missing_case(tmp_path: Path) -> None:
    p = tmp_path / "v.json"
    _write(p, {"piece_kind": "tensor_op"})
    assert parse_classify_verdict(p) is None


def test_parse_returns_none_for_invalid_case_value(tmp_path: Path) -> None:
    p = tmp_path / "v.json"
    _write(p, {"case": "Z", "piece_kind": "other"})
    assert parse_classify_verdict(p) is None


def test_parse_case_a_extracts_ttnn_call(tmp_path: Path) -> None:
    p = tmp_path / "v.json"
    _write(
        p,
        {
            "case": "A",
            "piece_kind": "tensor_op",
            "description": "reshape",
            "ttnn_call": {"op": "ttnn.reshape", "args": [[1, 2, 3]]},
            "notes": "single op",
        },
    )
    result = parse_classify_verdict(p)
    assert result is not None
    assert result.is_case_a
    assert result.ttnn_call == {"op": "ttnn.reshape", "args": [[1, 2, 3]]}
    assert result.cpu_module is None
    assert result.submodule_spec is None


def test_parse_case_b_extracts_cpu_module(tmp_path: Path) -> None:
    p = tmp_path / "v.json"
    _write(
        p,
        {
            "case": "B",
            "piece_kind": "control_flow",
            "cpu_module": "vision_encoder.fpn_extract",
            "notes": "tiny",
        },
    )
    result = parse_classify_verdict(p)
    assert result is not None
    assert result.is_case_b
    assert result.cpu_module == "vision_encoder.fpn_extract"


def test_parse_case_c_extracts_submodule_spec(tmp_path: Path) -> None:
    p = tmp_path / "v.json"
    spec = {"name": "fpn", "hf_reference": "model.fpn", "class_name": "FPN"}
    _write(
        p,
        {
            "case": "C",
            "piece_kind": "submodule",
            "submodule_spec": spec,
            "notes": "real module",
        },
    )
    result = parse_classify_verdict(p)
    assert result is not None
    assert result.is_case_c
    assert result.submodule_spec == spec


def test_parse_normalizes_lowercase_case_letter(tmp_path: Path) -> None:
    p = tmp_path / "v.json"
    _write(p, {"case": "a", "notes": "x"})
    result = parse_classify_verdict(p)
    assert result is not None
    assert result.case == "A"


def test_parse_defaults_piece_kind_when_missing(tmp_path: Path) -> None:
    p = tmp_path / "v.json"
    _write(p, {"case": "A", "notes": "x"})
    result = parse_classify_verdict(p)
    assert result is not None
    assert result.piece_kind == "other"


def test_parse_handles_wrong_type_for_optional_fields(tmp_path: Path) -> None:
    """ttnn_call etc. provided as wrong type → silently drop, don't
    reject the whole verdict."""
    p = tmp_path / "v.json"
    _write(
        p,
        {
            "case": "A",
            "ttnn_call": "not a dict",  # wrong type
            "cpu_module": 42,  # wrong type
            "submodule_spec": [],  # wrong type
        },
    )
    result = parse_classify_verdict(p)
    assert result is not None
    assert result.ttnn_call is None
    assert result.cpu_module is None
    assert result.submodule_spec is None


# ─── heuristic_classify ─────────────────────────────────────────────


def test_heuristic_matches_tensor_op_keywords() -> None:
    """Common tensor-op keywords → Case A."""
    for kw in ("reshape", "permute", "transpose", "view", "squeeze"):
        result = heuristic_classify(f"missing {kw} from B,C,H,W to ...")
        assert result is not None
        assert result.is_case_a, f"keyword {kw!r} should trigger Case A"


def test_heuristic_matches_control_flow_keywords() -> None:
    for kw in ("dict access", "conditional", "list construction"):
        result = heuristic_classify(f"chain needs {kw}")
        assert result is not None
        assert result.is_case_b, f"keyword {kw!r} should trigger Case B"


def test_heuristic_returns_none_for_no_match() -> None:
    """No keyword matches → fall through to LLM."""
    assert heuristic_classify("we need a small conv layer for feature projection") is None


def test_heuristic_returns_none_for_empty_input() -> None:
    assert heuristic_classify("") is None
    assert heuristic_classify(None) is None  # type: ignore[arg-type]


# ─── run_classify_pass orchestrator ─────────────────────────────────


def _stub_agent(payload: Dict[str, Any]):
    def _invoke(prompt, *, expected_deliverable_files, timeout_s, **kwargs):
        for p in expected_deliverable_files:
            Path(p).write_text(json.dumps(payload))
        return 0

    return _invoke


def test_run_uses_heuristic_first_when_match(tmp_path: Path) -> None:
    """If heuristic matches, don't bother with the LLM."""
    agent_called = [False]

    def _agent_should_not_fire(*a, **kw):
        agent_called[0] = True
        return 0

    result = run_classify_pass(
        model_id="org/m",
        demo_dir=tmp_path,
        missing_piece_description="missing reshape from B,C to C,B",
        agent_invoker=_agent_should_not_fire,
    )
    assert result is not None
    assert result.is_case_a
    assert agent_called[0] is False


def test_run_falls_through_to_llm_when_no_heuristic_match(tmp_path: Path) -> None:
    """No keyword match → LLM gets the call."""
    payload = {"case": "C", "submodule_spec": {"name": "fpn", "hf_reference": "x", "class_name": "X"}}
    result = run_classify_pass(
        model_id="org/m",
        demo_dir=tmp_path,
        missing_piece_description="needs a small projection module",
        agent_invoker=_stub_agent(payload),
    )
    assert result is not None
    assert result.is_case_c
    assert result.submodule_spec is not None


def test_run_returns_none_on_agent_failure(tmp_path: Path) -> None:
    """Agent returned rc!=0 → no classification → None."""

    def _failing(*a, **kw):
        return 1

    result = run_classify_pass(
        model_id="org/m",
        demo_dir=tmp_path,
        missing_piece_description="not a heuristic match",
        agent_invoker=_failing,
    )
    assert result is None


def test_run_returns_none_on_agent_exception(tmp_path: Path) -> None:
    """Agent raised → caught, returns None. Never propagates."""

    def _raising(*a, **kw):
        raise RuntimeError("agent boom")

    result = run_classify_pass(
        model_id="org/m",
        demo_dir=tmp_path,
        missing_piece_description="not a heuristic match",
        agent_invoker=_raising,
    )
    assert result is None


def test_run_returns_none_when_agent_writes_garbage(tmp_path: Path) -> None:
    """Agent rc=0 but verdict file is malformed → parser returns None,
    orchestrator propagates."""

    def _writes_garbage(prompt, *, expected_deliverable_files, **kwargs):
        Path(expected_deliverable_files[0]).write_text("{not valid")
        return 0

    result = run_classify_pass(
        model_id="org/m",
        demo_dir=tmp_path,
        missing_piece_description="not a heuristic match",
        agent_invoker=_writes_garbage,
    )
    assert result is None


def test_run_can_bypass_heuristic(tmp_path: Path) -> None:
    """use_heuristic_first=False → LLM fires even on a heuristic match."""
    payload = {"case": "C", "submodule_spec": {"name": "x", "hf_reference": "y", "class_name": "Z"}}
    result = run_classify_pass(
        model_id="org/m",
        demo_dir=tmp_path,
        missing_piece_description="missing reshape",  # heuristic would say A
        use_heuristic_first=False,
        agent_invoker=_stub_agent(payload),
    )
    # LLM said C, so we go with C
    assert result is not None
    assert result.is_case_c
