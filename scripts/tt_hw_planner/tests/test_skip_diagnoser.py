"""Tests for the skip_diagnoser LLM agent (2026-06-03).

The diagnoser fires when a pytest SKIP matches harness-pattern reasons
and the deterministic Tier-1 fixes (discover-side path indexing +
runtime ModuleList fallback) didn't recover the SKIP. The agent reads
the test file + skip reason + HF source + captured-input manifest,
classifies the root cause, and either fixes the test in-place or
flags the component for decompose/manual.

Tests cover the pure helpers (is_harness_skip, prompt assembly,
verdict parsing) without invoking the actual LLM subprocess.
"""

from __future__ import annotations

from pathlib import Path


# ─── is_harness_skip — pattern classifier ───────────────────────────


def test_is_harness_skip_matches_modulelist_reason():
    from scripts.tt_hw_planner._cli_helpers.skip_diagnoser import is_harness_skip

    reason = (
        "HF reference forward([]) raised NotImplementedError for "
        "hifi_gan_residual_block: Module [ModuleList] is missing the "
        'required "forward" function -- the synthetic inputs from '
        "_make_arg_for() are incompatible with this submodule's expected shapes."
    )
    assert is_harness_skip(reason) is True


def test_is_harness_skip_matches_make_arg_for_failure():
    from scripts.tt_hw_planner._cli_helpers.skip_diagnoser import is_harness_skip

    reason = "synthetic inputs from _make_arg_for() failed shape inference"
    assert is_harness_skip(reason) is True


def test_is_harness_skip_does_not_match_real_stub_failure():
    from scripts.tt_hw_planner._cli_helpers.skip_diagnoser import is_harness_skip

    # A real stub-level error (ttnn op assertion) — not a harness gap.
    reason = (
        "TT_FATAL: rotary_embedding_hf head_dim must be divisible by 64; "
        "this is a real op-level constraint, not a harness issue"
    )
    assert is_harness_skip(reason) is False


def test_is_harness_skip_empty_string():
    from scripts.tt_hw_planner._cli_helpers.skip_diagnoser import is_harness_skip

    assert is_harness_skip("") is False
    assert is_harness_skip(None) is False  # type: ignore[arg-type]


# ─── prompt assembly ────────────────────────────────────────────────


def test_build_diagnoser_prompt_includes_required_sections():
    from scripts.tt_hw_planner._cli_helpers.skip_diagnoser import build_diagnoser_prompt

    prompt = build_diagnoser_prompt(
        component_name="hifi_gan_residual_block",
        skip_reason="Module [ModuleList] is missing the required forward function",
        test_file_content="_CANDIDATE_SUBMODULE_PATHS = ['vocoder.hifi_gan.resblocks']",
        hf_reference_excerpt="class HifiGanResidualBlock(nn.Module):\n    def forward(self, x): ...",
        captured_inputs_excerpt='{"input_shape": [1, 512, 100]}',
    )

    # Section markers
    assert "PYTEST SKIP REASON:" in prompt
    assert "AUTO-GENERATED TEST FILE" in prompt
    assert "HF REFERENCE FORWARD SIGNATURE" in prompt
    assert "CAPTURED INPUT MANIFEST" in prompt
    # Content embedded
    assert "hifi_gan_residual_block" in prompt
    assert "Module [ModuleList]" in prompt
    assert "_CANDIDATE_SUBMODULE_PATHS" in prompt
    # Output format instructions
    assert "VERDICT:" in prompt
    assert "fixed" in prompt and "decompose" in prompt and "manual" in prompt
    # Guard against editing the stub
    assert "DO NOT edit the stub" in prompt


def test_build_diagnoser_prompt_omits_empty_optional_sections():
    """If no HF excerpt or captured-input excerpt is supplied, those
    sections shouldn't appear (keeps prompt compact)."""
    from scripts.tt_hw_planner._cli_helpers.skip_diagnoser import build_diagnoser_prompt

    prompt = build_diagnoser_prompt(
        component_name="x",
        skip_reason="some reason",
        test_file_content="some test",
    )
    # Required sections still present
    assert "PYTEST SKIP REASON:" in prompt
    assert "AUTO-GENERATED TEST FILE" in prompt
    # Optional sections absent
    assert "HF REFERENCE FORWARD SIGNATURE" not in prompt
    assert "CAPTURED INPUT MANIFEST" not in prompt


# ─── verdict parsing ────────────────────────────────────────────────


def test_parse_verdict_fixed_with_summary():
    from scripts.tt_hw_planner._cli_helpers.skip_diagnoser import parse_diagnoser_verdict

    out = (
        "I analyzed the test. The path lands on a ModuleList.\n"
        "I edited the test to use 'resblocks[0]' instead.\n"
        "\n"
        "VERDICT: fixed\n"
        "SUMMARY: Changed candidate path from resblocks to resblocks[0]\n"
    )
    parsed = parse_diagnoser_verdict(out)
    assert parsed["verdict"] == "fixed"
    assert "resblocks" in parsed["summary"]


def test_parse_verdict_decompose():
    from scripts.tt_hw_planner._cli_helpers.skip_diagnoser import parse_diagnoser_verdict

    out = (
        "This component is a pure container.\n"
        "VERDICT: decompose\n"
        "SUMMARY: Component is a Sequential of incompatible children; decomposer should split it\n"
    )
    parsed = parse_diagnoser_verdict(out)
    assert parsed["verdict"] == "decompose"


def test_parse_verdict_manual():
    from scripts.tt_hw_planner._cli_helpers.skip_diagnoser import parse_diagnoser_verdict

    out = "Cannot auto-fix.\nVERDICT: manual\nSUMMARY: Component needs hand-authored test with custom input fixtures\n"
    parsed = parse_diagnoser_verdict(out)
    assert parsed["verdict"] == "manual"


def test_parse_verdict_unknown_when_no_verdict_line():
    from scripts.tt_hw_planner._cli_helpers.skip_diagnoser import parse_diagnoser_verdict

    out = "I tried to diagnose but got confused. No verdict here."
    parsed = parse_diagnoser_verdict(out)
    assert parsed["verdict"] == "unknown"
    assert parsed["summary"] == ""


def test_parse_verdict_picks_last_verdict_when_multiple():
    """If the agent reasoned aloud earlier and mentioned 'VERDICT' in
    a non-output line, the final block's verdict is what we trust."""
    from scripts.tt_hw_planner._cli_helpers.skip_diagnoser import parse_diagnoser_verdict

    out = (
        "Considering candidates... maybe VERDICT: unknown.\n"
        "Actually, looking more carefully...\n"
        "VERDICT: fixed\n"
        "SUMMARY: real final fix\n"
    )
    parsed = parse_diagnoser_verdict(out)
    assert parsed["verdict"] == "fixed"
    assert parsed["summary"] == "real final fix"


def test_parse_verdict_rejects_unknown_verdict_string():
    """If the agent invented a verdict not in the allowed set, fall
    back to UNKNOWN."""
    from scripts.tt_hw_planner._cli_helpers.skip_diagnoser import parse_diagnoser_verdict

    out = "VERDICT: maybe_fixed\nSUMMARY: not sure\n"
    parsed = parse_diagnoser_verdict(out)
    assert parsed["verdict"] == "unknown"


# ─── diagnose_skips_in_demo — short-circuit for non-harness SKIPs ───


def test_diagnose_skips_in_demo_short_circuits_non_harness(tmp_path, monkeypatch):
    """For SKIPs that DON'T match harness patterns, the diagnoser
    must return UNKNOWN without spawning an LLM subprocess (saves cost
    and time)."""
    from scripts.tt_hw_planner._cli_helpers import skip_diagnoser

    spawned = []

    def fake_diagnose_skip(**kwargs):
        spawned.append(kwargs["component_name"])
        return {
            "component": kwargs["component_name"],
            "verdict": "fixed",
            "summary": "shouldn't be called",
            "agent_stdout": "",
            "rc": 0,
        }

    monkeypatch.setattr(skip_diagnoser, "diagnose_skip", fake_diagnose_skip)

    results = skip_diagnoser.diagnose_skips_in_demo(
        demo_dir=tmp_path,
        skipped_components=["non_harness_skip_comp"],
        skip_reasons={"non_harness_skip_comp": "real stub bug, not a harness gap"},
        agent_bin="/usr/bin/claude",
    )
    assert len(results) == 1
    assert results[0]["verdict"] == "unknown"
    # No subprocess was spawned for the non-harness SKIP.
    assert spawned == []


def test_diagnose_skips_in_demo_invokes_for_harness_skip(tmp_path, monkeypatch):
    """For SKIPs that DO match harness patterns, diagnose_skip is called."""
    from scripts.tt_hw_planner._cli_helpers import skip_diagnoser

    spawned = []

    def fake_diagnose_skip(**kwargs):
        spawned.append(kwargs["component_name"])
        return {
            "component": kwargs["component_name"],
            "verdict": "fixed",
            "summary": "patched path",
            "agent_stdout": "VERDICT: fixed\nSUMMARY: patched path",
            "rc": 0,
        }

    monkeypatch.setattr(skip_diagnoser, "diagnose_skip", fake_diagnose_skip)

    results = skip_diagnoser.diagnose_skips_in_demo(
        demo_dir=tmp_path,
        skipped_components=["hifi_gan_residual_block"],
        skip_reasons={
            "hifi_gan_residual_block": (
                "HF reference forward([]) raised NotImplementedError: "
                "Module [ModuleList] is missing the required forward function"
            )
        },
        agent_bin="/usr/bin/claude",
    )
    assert len(results) == 1
    assert results[0]["verdict"] == "fixed"
    assert spawned == ["hifi_gan_residual_block"]
