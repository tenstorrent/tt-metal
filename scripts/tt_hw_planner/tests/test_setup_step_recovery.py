"""Unit tests for the two-layer setup-step recovery harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from scripts.tt_hw_planner._cli_helpers.setup_step_recovery import (
    RecoveryAction,
    RecoveryProposal,
    apply_recovery_action,
    build_recovery_prompt,
    find_rule_proposal,
    parse_recovery_verdict,
    run_setup_step_recovery,
)


# ─── Rule registry ─────────────────────────────────────────────────


def test_rule_automodel_cascade_matches_phi3_error():
    """The exact ValueError shape from the Phi-3.5 run."""
    exc = ValueError(
        "Unrecognized configuration class "
        "<class 'transformers_modules.microsoft.Phi-3.5-mini-instruct.X.configuration_phi3.Phi3Config'> "
        "for this kind of AutoModel: AutoModel."
    )
    p = find_rule_proposal(exc, step_name="step4_autofill")
    assert p is not None
    assert p.action == RecoveryAction.CASCADE_CLASS
    assert p.args["module"] == "transformers"
    assert "AutoModelForCausalLM" in p.args["attrs"]
    assert p.source == "rule"


def test_rule_automodel_cascade_skips_unrelated_valueerror():
    exc = ValueError("totally unrelated")
    assert find_rule_proposal(exc, step_name="step4_autofill") is None


def test_rule_mkdir_missing_matches_filenotfounderror(tmp_path: Path):
    exc = FileNotFoundError(2, "No such file or directory", str(tmp_path / "_stubs" / "attention.py"))
    p = find_rule_proposal(exc, step_name="step4_autofill")
    assert p is not None
    assert p.action == RecoveryAction.MKDIR_PARENTS
    assert p.args["path"].endswith("_stubs")


def test_rule_mkdir_missing_skips_other_oserror():
    exc = OSError(13, "Permission denied", "/etc/foo")
    assert find_rule_proposal(exc, step_name="step4_autofill") is None


# ─── Parser ────────────────────────────────────────────────────────


def _write_verdict(tmp_path: Path, body: Dict[str, Any]) -> Path:
    v = tmp_path / "verdict.json"
    v.write_text(json.dumps(body))
    return v


def test_parse_valid_cascade_class(tmp_path: Path):
    v = _write_verdict(
        tmp_path,
        {
            "action": "cascade_class",
            "args": {"module": "transformers", "attrs": ["AutoModelForCausalLM"]},
            "reasoning": "trust_remote_code config needs task-specific class",
        },
    )
    p = parse_recovery_verdict(v)
    assert p is not None
    assert p.action == RecoveryAction.CASCADE_CLASS
    assert p.args["attrs"] == ["AutoModelForCausalLM"]
    assert p.source == "llm"


def test_parse_valid_mkdir_parents(tmp_path: Path):
    v = _write_verdict(tmp_path, {"action": "mkdir_parents", "args": {"path": "models/foo/_stubs"}})
    p = parse_recovery_verdict(v)
    assert p is not None
    assert p.action == RecoveryAction.MKDIR_PARENTS


def test_parse_valid_dtype_downgrade(tmp_path: Path):
    v = _write_verdict(tmp_path, {"action": "dtype_downgrade", "args": {"from": "bf16", "to": "bfp8_b"}})
    p = parse_recovery_verdict(v)
    assert p is not None
    assert p.action == RecoveryAction.DTYPE_DOWNGRADE


def test_parse_valid_overlay_drop(tmp_path: Path):
    v = _write_verdict(tmp_path, {"action": "overlay_drop", "args": {"model_id": "microsoft/Phi-3.5-mini-instruct"}})
    p = parse_recovery_verdict(v)
    assert p is not None
    assert p.action == RecoveryAction.OVERLAY_DROP


def test_parse_valid_skip_component(tmp_path: Path):
    v = _write_verdict(tmp_path, {"action": "skip_component", "args": {"component_name": "attention"}})
    p = parse_recovery_verdict(v)
    assert p is not None
    assert p.action == RecoveryAction.SKIP_COMPONENT


def test_parse_valid_re_exec(tmp_path: Path):
    v = _write_verdict(tmp_path, {"action": "re_exec", "args": {"env": {"FOO": "bar", "BAZ_QUX": "1"}}})
    p = parse_recovery_verdict(v)
    assert p is not None
    assert p.action == RecoveryAction.RE_EXEC


def test_parse_valid_cannot_recover(tmp_path: Path):
    v = _write_verdict(tmp_path, {"action": "cannot_recover", "args": {"reason": "wrong arch"}})
    p = parse_recovery_verdict(v)
    assert p is not None
    assert p.action == RecoveryAction.CANNOT_RECOVER


# ─── Parser SAFETY: reject malformed verdicts ──────────────────────


def test_parse_rejects_unknown_action(tmp_path: Path):
    v = _write_verdict(tmp_path, {"action": "evil_action", "args": {}})
    assert parse_recovery_verdict(v) is None


def test_parse_rejects_cascade_with_bad_attr_name(tmp_path: Path):
    v = _write_verdict(tmp_path, {"action": "cascade_class", "args": {"module": "transformers", "attrs": ["rm -rf /"]}})
    assert parse_recovery_verdict(v) is None


def test_parse_rejects_cascade_with_non_module_name(tmp_path: Path):
    v = _write_verdict(
        tmp_path,
        {"action": "cascade_class", "args": {"module": "transformers; evil()", "attrs": ["AutoModel"]}},
    )
    assert parse_recovery_verdict(v) is None


def test_parse_rejects_cascade_with_empty_attrs(tmp_path: Path):
    v = _write_verdict(tmp_path, {"action": "cascade_class", "args": {"module": "transformers", "attrs": []}})
    assert parse_recovery_verdict(v) is None


@pytest.mark.parametrize(
    "evil_path",
    [
        "../../etc/passwd",
        "/etc/foo",
        "models/foo;rm-rf",
        "models/$(whoami)",
        "models/`whoami`",
        "models/foo|cat",
    ],
)
def test_parse_rejects_mkdir_unsafe_path(tmp_path: Path, evil_path: str):
    v = _write_verdict(tmp_path, {"action": "mkdir_parents", "args": {"path": evil_path}})
    assert parse_recovery_verdict(v) is None, f"should reject {evil_path!r}"


def test_parse_rejects_mkdir_empty_path(tmp_path: Path):
    v = _write_verdict(tmp_path, {"action": "mkdir_parents", "args": {"path": ""}})
    assert parse_recovery_verdict(v) is None


def test_parse_rejects_dtype_downgrade_unknown_dtype(tmp_path: Path):
    v = _write_verdict(tmp_path, {"action": "dtype_downgrade", "args": {"from": "int8", "to": "bfp8_b"}})
    assert parse_recovery_verdict(v) is None


def test_parse_rejects_overlay_drop_with_shell_chars(tmp_path: Path):
    v = _write_verdict(tmp_path, {"action": "overlay_drop", "args": {"model_id": "org/name;rm-rf"}})
    assert parse_recovery_verdict(v) is None


def test_parse_rejects_re_exec_with_shell_meta_in_value(tmp_path: Path):
    v = _write_verdict(tmp_path, {"action": "re_exec", "args": {"env": {"FOO": "value; rm -rf /"}}})
    assert parse_recovery_verdict(v) is None


def test_parse_rejects_re_exec_with_lowercase_var(tmp_path: Path):
    v = _write_verdict(tmp_path, {"action": "re_exec", "args": {"env": {"foo": "bar"}}})
    assert parse_recovery_verdict(v) is None


def test_parse_rejects_skip_component_unsafe_name(tmp_path: Path):
    v = _write_verdict(tmp_path, {"action": "skip_component", "args": {"component_name": "attn; rm -rf /"}})
    assert parse_recovery_verdict(v) is None


def test_parse_rejects_missing_file(tmp_path: Path):
    assert parse_recovery_verdict(tmp_path / "nope.json") is None


def test_parse_rejects_malformed_json(tmp_path: Path):
    v = tmp_path / "verdict.json"
    v.write_text("not json")
    assert parse_recovery_verdict(v) is None


def test_parse_rejects_top_level_list(tmp_path: Path):
    v = tmp_path / "verdict.json"
    v.write_text("[]")
    assert parse_recovery_verdict(v) is None


# ─── apply_recovery_action ────────────────────────────────────────


def test_apply_mkdir_parents_creates_directory(tmp_path: Path):
    target = "newdir/nested/_stubs"
    p = RecoveryProposal(action=RecoveryAction.MKDIR_PARENTS, args={"path": target})
    ok, note = apply_recovery_action(p, repo_root=tmp_path)
    assert ok
    assert (tmp_path / target).is_dir()


def test_apply_mkdir_parents_idempotent(tmp_path: Path):
    target = tmp_path / "_stubs"
    target.mkdir()
    p = RecoveryProposal(action=RecoveryAction.MKDIR_PARENTS, args={"path": "_stubs"})
    ok, _ = apply_recovery_action(p, repo_root=tmp_path)
    assert ok


def test_apply_cascade_class_returns_ok_without_executing(tmp_path: Path):
    p = RecoveryProposal(
        action=RecoveryAction.CASCADE_CLASS,
        args={"module": "transformers", "attrs": ["AutoModelForCausalLM"]},
    )
    ok, note = apply_recovery_action(p, repo_root=tmp_path)
    assert ok
    assert "AutoModelForCausalLM" in note


def test_apply_cannot_recover_returns_ok_with_no_fix_note(tmp_path: Path):
    """CANNOT_RECOVER is a legitimate verdict the LLM may pick — it
    means 'I explicitly decline.' The orchestrator surfaces this to
    the caller via ``proposal.action``; the apply step itself returns
    ok=True because the decision is valid (just non-recoverable)."""
    p = RecoveryProposal(action=RecoveryAction.CANNOT_RECOVER, args={"reason": "no fix"})
    ok, note = apply_recovery_action(p, repo_root=tmp_path)
    assert ok
    assert "no fix" in note or "no-fix" in note


def test_apply_overlay_drop_calls_drop_scope(tmp_path: Path, monkeypatch):
    called = {}

    def fake_drop_scope(model_id: str):
        called["model_id"] = model_id
        return (3, ["a", "b", "c"])

    # Patch overlay_manager.drop_scope via the import path used in setup_step_recovery
    import scripts.tt_hw_planner.overlay_manager as om

    monkeypatch.setattr(om, "drop_scope", fake_drop_scope)
    p = RecoveryProposal(action=RecoveryAction.OVERLAY_DROP, args={"model_id": "org/name"})
    ok, note = apply_recovery_action(p, repo_root=tmp_path)
    assert ok
    assert called["model_id"] == "org/name"
    assert "3 overlay" in note


# ─── Orchestrator (rules + LLM layering) ────────────────────────────


def test_orchestrator_uses_rule_when_match(tmp_path: Path):
    exc = ValueError("Unrecognized configuration class <X> for this kind of AutoModel: AutoModel.")
    p = run_setup_step_recovery(
        exc=exc,
        step_name="step4_autofill",
        work_dir=tmp_path,
        repo_root=tmp_path,
        agent_invoker=lambda *args, **kwargs: pytest.fail("agent must not be invoked when rule matches"),
    )
    assert p is not None
    assert p.action == RecoveryAction.CASCADE_CLASS
    assert p.source == "rule"


def test_orchestrator_falls_back_to_llm_when_no_rule_matches(tmp_path: Path):
    """Trigger the LLM fallback with a synthetic non-matching exception."""

    def fake_agent(prompt, *, expected_deliverable_files, timeout_s, **_):
        verdict_path = expected_deliverable_files[0]
        verdict_path.write_text(
            json.dumps(
                {
                    "action": "cannot_recover",
                    "args": {"reason": "novel failure mode"},
                    "reasoning": "LLM determined nothing safe to do",
                }
            )
        )
        return 0

    exc = RuntimeError("totally novel failure")
    p = run_setup_step_recovery(
        exc=exc,
        step_name="step4_autofill",
        work_dir=tmp_path,
        repo_root=tmp_path,
        agent_invoker=fake_agent,
    )
    assert p is not None
    assert p.action == RecoveryAction.CANNOT_RECOVER
    assert p.source == "llm"


def test_orchestrator_returns_none_when_llm_returns_garbage(tmp_path: Path):
    def evil_agent(prompt, *, expected_deliverable_files, timeout_s, **_):
        verdict_path = expected_deliverable_files[0]
        verdict_path.write_text(json.dumps({"action": "rm_rf", "args": {}}))
        return 0

    p = run_setup_step_recovery(
        exc=RuntimeError("novel"),
        step_name="step4_autofill",
        work_dir=tmp_path,
        repo_root=tmp_path,
        agent_invoker=evil_agent,
    )
    assert p is None


def test_orchestrator_returns_none_when_llm_unreachable(tmp_path: Path):
    def broken_agent(prompt, *, expected_deliverable_files, timeout_s, **_):
        raise RuntimeError("agent crashed")

    p = run_setup_step_recovery(
        exc=RuntimeError("novel"),
        step_name="step4_autofill",
        work_dir=tmp_path,
        repo_root=tmp_path,
        agent_invoker=broken_agent,
    )
    assert p is None


def test_orchestrator_honors_reentry_guard(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TT_HW_PLANNER_SETUP_RECOVERY_ATTEMPTED", "1")
    p = run_setup_step_recovery(
        exc=ValueError("Unrecognized configuration class X for this kind of AutoModel: AutoModel."),
        step_name="step4_autofill",
        work_dir=tmp_path,
        repo_root=tmp_path,
        agent_invoker=lambda *a, **k: pytest.fail("guard should have blocked"),
    )
    assert p is None


# ─── Prompt builder ────────────────────────────────────────────────


def test_prompt_includes_step_name_and_exception(tmp_path: Path):
    prompt = build_recovery_prompt(
        step_name="step4_autofill",
        exception_class="ValueError",
        exception_message="Unrecognized configuration class for AutoModel.",
        traceback_text="File X, line 1, in foo",
        workspace_summary="(scratch)",
        verdict_path=tmp_path / "verdict.json",
    )
    assert "step4_autofill" in prompt
    assert "ValueError" in prompt
    assert "Unrecognized configuration class" in prompt
    assert "cannot_recover" in prompt
    assert "cascade_class" in prompt
    assert str(tmp_path / "verdict.json") in prompt


def test_prompt_includes_safety_instructions(tmp_path: Path):
    prompt = build_recovery_prompt(
        step_name="x",
        exception_class="X",
        exception_message="y",
        traceback_text="z",
        workspace_summary="w",
        verdict_path=tmp_path / "v.json",
    )
    assert "DO NOT" in prompt
    assert "ONLY write the verdict JSON" in prompt
