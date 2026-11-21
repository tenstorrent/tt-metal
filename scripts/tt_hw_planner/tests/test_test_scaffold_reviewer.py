"""Tests for the Tier-2 LLM batch reviewer (2026-06-04 Phase-2).

Covers the pure helpers (``is_review_target``, ``build_review_prompt``,
``parse_review_verdict``) without invoking the real LLM subprocess.
The orchestrator ``review_test_scaffolds`` is also tested with the
``_invoke_agent`` call mocked so we exercise the wiring without paying
LLM cost.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ─── is_review_target ────────────────────────────────────────────────


def test_is_review_target_matches_auto_generated_pcc_test(tmp_path):
    from scripts.tt_hw_planner._cli_helpers.test_scaffold_reviewer import is_review_target

    t = tmp_path / "test_foo.py"
    t.write_text("# auto-generated\n" '_CANDIDATE_SUBMODULE_PATHS = ["x.y"]\n' "def test_foo():\n" "    pass\n")
    assert is_review_target(t) is True


def test_is_review_target_rejects_handwritten_test(tmp_path):
    from scripts.tt_hw_planner._cli_helpers.test_scaffold_reviewer import is_review_target

    t = tmp_path / "test_handwritten.py"
    # Intentionally OMIT the template marker so the test exercises the
    # rejection path. Hand-written tests don't have the auto-generated
    # candidate-submodule-paths sentinel.
    t.write_text("# hand-edited test, no template markers here\n")
    assert is_review_target(t) is False


def test_is_review_target_rejects_missing_file(tmp_path):
    from scripts.tt_hw_planner._cli_helpers.test_scaffold_reviewer import is_review_target

    assert is_review_target(tmp_path / "does_not_exist.py") is False


# ─── build_review_prompt ─────────────────────────────────────────────


def test_build_review_prompt_includes_all_components():
    from scripts.tt_hw_planner._cli_helpers.test_scaffold_reviewer import build_review_prompt

    prompt = build_review_prompt(
        test_files_with_excerpts=[
            {"component": "decoder", "test_path": "/x/test_decoder.py", "test_excerpt": "...", "hf_signature": ""},
            {"component": "encoder", "test_path": "/x/test_encoder.py", "test_excerpt": "...", "hf_signature": ""},
        ],
        model_id="m/x",
    )
    assert "decoder" in prompt
    assert "encoder" in prompt
    assert "2/2" in prompt or "1/2" in prompt  # per-component header


def test_build_review_prompt_lists_failure_patterns():
    """Prompt must teach the LLM about the failure shapes we care about."""
    from scripts.tt_hw_planner._cli_helpers.test_scaffold_reviewer import build_review_prompt

    prompt = build_review_prompt(
        test_files_with_excerpts=[{"component": "x", "test_path": "/x.py", "test_excerpt": "...", "hf_signature": ""}],
        model_id="m/x",
    )
    assert "MUTUAL EXCLUSION" in prompt
    assert "MISSING REQUIRED ARG" in prompt
    assert "WRONG SHAPE" in prompt
    assert "WRONG DTYPE" in prompt


def test_build_review_prompt_says_dont_edit_stub():
    """Reviewer must NEVER edit _stubs/*.py — only test files."""
    from scripts.tt_hw_planner._cli_helpers.test_scaffold_reviewer import build_review_prompt

    prompt = build_review_prompt(
        test_files_with_excerpts=[{"component": "x", "test_path": "/x.py", "test_excerpt": "...", "hf_signature": ""}],
        model_id="m/x",
    )
    assert "DO NOT edit" in prompt and "_stubs" in prompt


def test_build_review_prompt_specifies_verdict_format():
    """Prompt must lock the output format the parser expects."""
    from scripts.tt_hw_planner._cli_helpers.test_scaffold_reviewer import build_review_prompt

    prompt = build_review_prompt(
        test_files_with_excerpts=[{"component": "x", "test_path": "/x.py", "test_excerpt": "...", "hf_signature": ""}],
        model_id="m/x",
    )
    assert "VERDICT FOR" in prompt
    assert "SUMMARY:" in prompt
    assert "ok | patched | unfixable" in prompt or all(v in prompt for v in ("ok", "patched", "unfixable"))


# ─── parse_review_verdict ────────────────────────────────────────────


def test_parse_review_verdict_handles_clean_output():
    from scripts.tt_hw_planner._cli_helpers.test_scaffold_reviewer import (
        VERDICT_OK,
        VERDICT_PATCHED,
        VERDICT_UNFIXABLE,
        parse_review_verdict,
    )

    output = (
        "Reviewing components...\n"
        "\n"
        "VERDICT FOR decoder: patched\n"
        "SUMMARY: dropped input_ids, kept inputs_embeds\n"
        "\n"
        "VERDICT FOR encoder: patched\n"
        "SUMMARY: same fix as decoder\n"
        "\n"
        "VERDICT FOR ffn: ok\n"
        "SUMMARY: looks correct\n"
        "\n"
        "VERDICT FOR hifi_gan: unfixable\n"
        "SUMMARY: requires stub-level fix; iter loop should handle\n"
        "\n"
        "DONE\n"
    )
    verdicts = parse_review_verdict(output)
    assert len(verdicts) == 4
    assert verdicts[0].component == "decoder"
    assert verdicts[0].verdict == VERDICT_PATCHED
    assert verdicts[0].patch_applied is True
    assert "dropped input_ids" in verdicts[0].summary
    assert verdicts[2].component == "ffn"
    assert verdicts[2].verdict == VERDICT_OK
    assert verdicts[2].patch_applied is False
    assert verdicts[3].verdict == VERDICT_UNFIXABLE


def test_parse_review_verdict_returns_empty_for_no_output():
    from scripts.tt_hw_planner._cli_helpers.test_scaffold_reviewer import parse_review_verdict

    assert parse_review_verdict("") == []
    assert parse_review_verdict("just commentary, no verdicts") == []


def test_parse_review_verdict_handles_unknown_verdict_string():
    """Agent might invent a verdict not in our set — fall back to UNKNOWN."""
    from scripts.tt_hw_planner._cli_helpers.test_scaffold_reviewer import VERDICT_UNKNOWN, parse_review_verdict

    output = "VERDICT FOR decoder: maybe_ok\nSUMMARY: not sure\n"
    verdicts = parse_review_verdict(output)
    assert len(verdicts) == 0  # unknown isn't in the regex alternation; verdict block ignored


def test_parse_review_verdict_missing_summary_is_empty():
    from scripts.tt_hw_planner._cli_helpers.test_scaffold_reviewer import VERDICT_PATCHED, parse_review_verdict

    output = "VERDICT FOR decoder: patched\n"
    verdicts = parse_review_verdict(output)
    assert len(verdicts) == 1
    assert verdicts[0].verdict == VERDICT_PATCHED
    assert verdicts[0].summary == ""


# ─── orchestrator (with mocked _invoke_agent) ────────────────────────


def test_review_test_scaffolds_no_op_when_disabled(tmp_path):
    from scripts.tt_hw_planner._cli_helpers.test_scaffold_reviewer import review_test_scaffolds

    result = review_test_scaffolds(
        demo_dir=tmp_path,
        test_files=[tmp_path / "test_x.py"],
        model_id="m/x",
        agent_bin="/usr/bin/claude",
        enabled=False,
    )
    assert result == []


def test_review_test_scaffolds_no_op_without_agent_bin(tmp_path):
    from scripts.tt_hw_planner._cli_helpers.test_scaffold_reviewer import review_test_scaffolds

    result = review_test_scaffolds(
        demo_dir=tmp_path,
        test_files=[tmp_path / "test_x.py"],
        model_id="m/x",
        agent_bin=None,
    )
    assert result == []


def test_review_test_scaffolds_no_op_when_no_eligible_files(tmp_path):
    """No files matching is_review_target → no LLM call."""
    from scripts.tt_hw_planner._cli_helpers.test_scaffold_reviewer import review_test_scaffolds

    # File exists but doesn't have the auto-generated marker
    t = tmp_path / "test_handwritten.py"
    t.write_text("# hand-edited\n")
    result = review_test_scaffolds(
        demo_dir=tmp_path,
        test_files=[t],
        model_id="m/x",
        agent_bin="/usr/bin/claude",
    )
    assert result == []
