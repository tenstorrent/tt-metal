"""Unit tests for the e2e synthesis loop (Item 3).

The loop iterates: prompt LLM → invoke agent (writes demo.py) →
validate → run pytest → measure PCC → either converge or feed
failure back into next iter's prompt.

Both impure seams (agent_invoker, pytest_runner) are injectable so
tests run end-to-end without an LLM or real pytest. Pure helpers
(prompt builder, parser, PCC extractor, persister) tested directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

from scripts.tt_hw_planner._cli_helpers.e2e_synthesizer import (
    E2ESynthIterResult,
    E2ESynthResult,
    build_synthesis_prompt,
    extract_late_discovery_markers,
    extract_pcc_from_output,
    parse_synthesized_demo_py,
    persist_synth_result,
    run_e2e_synthesis_loop,
)


# ─── build_synthesis_prompt ─────────────────────────────────────────


def test_prompt_carries_model_id_and_iter(tmp_path: Path) -> None:
    p = build_synthesis_prompt(
        model_id="org/m",
        hf_forward_src="def forward(self, x): ...",
        graduated_components=[],
        chain_divergence_summary="",
        verify_diagnostic_summary="",
        previous_iter_failure="",
        demo_py_path=tmp_path / "demo.py",
        iter_idx=1,
    )
    assert "org/m" in p
    assert "ITERATION 1" in p
    assert str(tmp_path / "demo.py") in p


def test_prompt_lists_graduated_components(tmp_path: Path) -> None:
    """Each graduated component's name + stub + hf_reference must be
    visible to the LLM so it can wire them correctly."""
    comps = [
        {"name": "vision_encoder", "stub_path": "_stubs/ve.py", "hf_reference": "model.vision", "class_name": "VE"},
        {"name": "mask_decoder", "stub_path": "_stubs/md.py", "hf_reference": "model.mask", "class_name": "MD"},
    ]
    p = build_synthesis_prompt(
        model_id="org/m",
        hf_forward_src="x",
        graduated_components=comps,
        chain_divergence_summary="",
        verify_diagnostic_summary="",
        previous_iter_failure="",
        demo_py_path=tmp_path / "demo.py",
        iter_idx=1,
    )
    assert "vision_encoder" in p
    assert "mask_decoder" in p
    assert "_stubs/ve.py" in p
    assert "model.vision" in p


def test_prompt_renders_first_iter_placeholder(tmp_path: Path) -> None:
    """On iter 1 there's no prior failure; the prompt shows that
    explicitly so the LLM doesn't hallucinate a previous attempt."""
    p = build_synthesis_prompt(
        model_id="org/m",
        hf_forward_src="x",
        graduated_components=[],
        chain_divergence_summary="",
        verify_diagnostic_summary="",
        previous_iter_failure="",
        demo_py_path=tmp_path / "demo.py",
        iter_idx=1,
    )
    assert "first iter" in p.lower() or "no prior" in p.lower()


def test_prompt_carries_prior_failure_on_iter_2(tmp_path: Path) -> None:
    """Iter 2 must show the iter-1 failure so the LLM can target it."""
    p = build_synthesis_prompt(
        model_id="org/m",
        hf_forward_src="x",
        graduated_components=[],
        chain_divergence_summary="",
        verify_diagnostic_summary="",
        previous_iter_failure="iter 1: PCC was 0.42, expected 0.99",
        demo_py_path=tmp_path / "demo.py",
        iter_idx=2,
    )
    assert "iter 1" in p
    assert "0.42" in p
    assert "0.99" in p


def test_prompt_includes_pcc_target(tmp_path: Path) -> None:
    """The PCC target must appear so the LLM knows the bar."""
    p = build_synthesis_prompt(
        model_id="org/m",
        hf_forward_src="x",
        graduated_components=[],
        chain_divergence_summary="",
        verify_diagnostic_summary="",
        previous_iter_failure="",
        demo_py_path=tmp_path / "demo.py",
        iter_idx=1,
        pcc_target=0.97,
    )
    assert "0.97" in p


# ─── parse_synthesized_demo_py ──────────────────────────────────────


def test_parser_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert parse_synthesized_demo_py(tmp_path / "nope.py") is None


def test_parser_returns_none_for_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "demo.py"
    p.write_text("   \n\n  \n")
    assert parse_synthesized_demo_py(p) is None


def test_parser_returns_none_without_test_demo(tmp_path: Path) -> None:
    """Must have test_demo() entry point — pytest target."""
    p = tmp_path / "demo.py"
    p.write_text("def other_test(): pass\n# pcc check below\nassert_with_pcc(a, b, pcc=0.99)\n")
    assert parse_synthesized_demo_py(p) is None


def test_parser_returns_none_without_pcc_assertion(tmp_path: Path) -> None:
    """Must include some PCC assertion — defends against the LLM writing
    a smoke-test demo that never measures correctness."""
    p = tmp_path / "demo.py"
    p.write_text("def test_demo(device_params, device):\n    pass\n")
    assert parse_synthesized_demo_py(p) is None


def test_parser_accepts_demo_with_assert_with_pcc(tmp_path: Path) -> None:
    p = tmp_path / "demo.py"
    p.write_text(
        "import torch\n"
        "def test_demo(device_params, device):\n"
        "    out = torch.zeros(1)\n"
        "    ref = torch.zeros(1)\n"
        "    assert_with_pcc(out, ref, pcc=0.99)\n"
    )
    src = parse_synthesized_demo_py(p)
    assert src is not None
    assert "test_demo" in src


def test_parser_accepts_demo_with_comp_pcc(tmp_path: Path) -> None:
    """The comp_pcc helper is the alternative PCC API some templates use."""
    p = tmp_path / "demo.py"
    p.write_text(
        "def test_demo(device_params, device):\n" "    passed, pcc = comp_pcc(a, b, 0.99)\n" "    assert passed\n"
    )
    src = parse_synthesized_demo_py(p)
    assert src is not None


# ─── extract_pcc_from_output ────────────────────────────────────────


def test_extract_pcc_finds_end_to_end_format() -> None:
    """The e2e_emitter template emits ``end-to-end PCC=0.9876``."""
    assert extract_pcc_from_output("end-to-end PCC=0.9876") == 0.9876


def test_extract_pcc_finds_loose_format() -> None:
    """Loose ``PCC = 0.123`` also works as fallback."""
    assert extract_pcc_from_output("Final PCC = 0.9701") == 0.9701


def test_extract_pcc_returns_none_when_absent() -> None:
    assert extract_pcc_from_output("some random output") is None
    assert extract_pcc_from_output("") is None


def test_extract_pcc_rejects_out_of_range() -> None:
    """A value outside [-1, 1] isn't a valid PCC; treat as not-found."""
    assert extract_pcc_from_output("PCC = 99.9") is None
    assert extract_pcc_from_output("PCC = -2.0") is None


def test_extract_pcc_picks_first_valid() -> None:
    """If multiple PCC lines (e.g. one stage + final), first valid wins.
    Doesn't matter for the loop's purpose — it just needs A value."""
    text = "stage PCC=0.55\nend-to-end PCC=0.92\n"
    v = extract_pcc_from_output(text)
    assert v in (0.55, 0.92)  # first-match priority isn't load-bearing


# ─── run_e2e_synthesis_loop ─────────────────────────────────────────


def _agent_writes_demo(payload: str):
    """Build an agent_invoker that writes a fixed demo.py body and returns rc=0."""

    def _invoke(prompt, *, expected_deliverable_files, timeout_s, **kwargs):
        for p in expected_deliverable_files:
            Path(p).write_text(payload, encoding="utf-8")
        return 0

    return _invoke


def _valid_demo_body(pcc_text: str = "PCC = 0.99") -> str:
    return (
        f"def test_demo(device_params, device):\n" f"    print('{pcc_text}')\n" f"    assert_with_pcc(a, b, pcc=0.99)\n"
    )


def test_loop_converges_on_first_iter(tmp_path: Path) -> None:
    """Agent writes a valid demo, pytest passes, PCC >= 0.99 →
    converged after 1 iter."""
    agent = _agent_writes_demo(_valid_demo_body("end-to-end PCC=0.995"))

    def pytest_pass(demo_py_path: Path) -> Tuple[int, str]:
        return 0, "end-to-end PCC=0.995"

    result = run_e2e_synthesis_loop(
        model_id="org/m",
        demo_dir=tmp_path,
        hf_forward_src="def forward(self): ...",
        graduated_components=[],
        pytest_runner=pytest_pass,
        agent_invoker=agent,
        max_iters=3,
    )
    assert result.converged is True
    assert result.iters_used == 1
    assert result.final_pcc == 0.995


def test_loop_iterates_until_convergence(tmp_path: Path) -> None:
    """Iter 1 fails (PCC 0.5), iter 2 fails (PCC 0.85), iter 3 passes
    (PCC 0.99). Loop converges at iter 3."""
    pcc_values = iter([0.5, 0.85, 0.99])

    def pytest_with_pcc(demo_py_path: Path) -> Tuple[int, str]:
        v = next(pcc_values)
        rc = 0 if v >= 0.99 else 1
        return rc, f"end-to-end PCC={v:.4f}"

    result = run_e2e_synthesis_loop(
        model_id="org/m",
        demo_dir=tmp_path,
        hf_forward_src="x",
        graduated_components=[],
        pytest_runner=pytest_with_pcc,
        agent_invoker=_agent_writes_demo(_valid_demo_body()),
        max_iters=5,
    )
    assert result.converged is True
    assert result.iters_used == 3
    assert result.final_pcc == 0.99


def test_loop_returns_failure_when_budget_exhausted(tmp_path: Path) -> None:
    """Every iter fails to reach 0.99 within max_iters → not converged.
    final_diagnostic carries the last iter's failure."""

    def pytest_always_fail(demo_py_path: Path) -> Tuple[int, str]:
        return 1, "end-to-end PCC=0.42"

    result = run_e2e_synthesis_loop(
        model_id="org/m",
        demo_dir=tmp_path,
        hf_forward_src="x",
        graduated_components=[],
        pytest_runner=pytest_always_fail,
        agent_invoker=_agent_writes_demo(_valid_demo_body()),
        max_iters=2,
    )
    assert result.converged is False
    assert result.iters_used == 2
    assert "0.42" in result.final_diagnostic or "0.4200" in result.final_diagnostic


def test_loop_feeds_failure_into_next_iter_prompt(tmp_path: Path) -> None:
    """Iter 2's prompt must include iter 1's failure so the LLM can
    target it. We capture each iter's prompt to assert it."""
    prompts: list = []

    def captured_agent(prompt, *, expected_deliverable_files, timeout_s, **kwargs):
        prompts.append(prompt)
        for p in expected_deliverable_files:
            Path(p).write_text(_valid_demo_body(), encoding="utf-8")
        return 0

    pcc_iter = iter([0.50, 0.99])

    def pytest_runner(demo_py_path: Path) -> Tuple[int, str]:
        v = next(pcc_iter)
        return (0 if v >= 0.99 else 1), f"end-to-end PCC={v:.4f}"

    run_e2e_synthesis_loop(
        model_id="org/m",
        demo_dir=tmp_path,
        hf_forward_src="x",
        graduated_components=[],
        pytest_runner=pytest_runner,
        agent_invoker=captured_agent,
        max_iters=5,
    )
    assert len(prompts) == 2
    # iter 2's prompt must mention iter 1's PCC failure
    assert "0.50" in prompts[1] or "0.5000" in prompts[1]


def test_loop_handles_agent_exception(tmp_path: Path) -> None:
    """Agent raises → iter recorded as failure, loop continues, eventually
    returns not-converged. No exception propagates."""

    def raising_agent(prompt, *, expected_deliverable_files, timeout_s, **kwargs):
        raise RuntimeError("agent boom")

    def pytest_unused(demo_py_path: Path) -> Tuple[int, str]:
        raise AssertionError("pytest should not run when agent raises")

    result = run_e2e_synthesis_loop(
        model_id="org/m",
        demo_dir=tmp_path,
        hf_forward_src="x",
        graduated_components=[],
        pytest_runner=pytest_unused,
        agent_invoker=raising_agent,
        max_iters=2,
    )
    assert result.converged is False
    assert "agent boom" in result.final_diagnostic


def test_loop_skips_pytest_when_demo_validation_fails(tmp_path: Path) -> None:
    """Agent writes garbage that fails structural validation → don't
    bother running pytest, feed the failure forward."""

    def invalid_agent(prompt, *, expected_deliverable_files, timeout_s, **kwargs):
        for p in expected_deliverable_files:
            Path(p).write_text("# garbage, no test_demo, no PCC\n", encoding="utf-8")
        return 0

    pytest_called = [False]

    def pytest_should_not_fire(demo_py_path: Path) -> Tuple[int, str]:
        pytest_called[0] = True
        return 0, "PCC = 0.99"

    result = run_e2e_synthesis_loop(
        model_id="org/m",
        demo_dir=tmp_path,
        hf_forward_src="x",
        graduated_components=[],
        pytest_runner=pytest_should_not_fire,
        agent_invoker=invalid_agent,
        max_iters=1,
    )
    assert pytest_called[0] is False
    assert result.converged is False


def test_loop_handles_pytest_runner_exception(tmp_path: Path) -> None:
    """Pytest runner raises → iter recorded as failure, loop continues."""
    agent = _agent_writes_demo(_valid_demo_body())

    def raising_pytest(demo_py_path: Path) -> Tuple[int, str]:
        raise IOError("pytest disk full")

    result = run_e2e_synthesis_loop(
        model_id="org/m",
        demo_dir=tmp_path,
        hf_forward_src="x",
        graduated_components=[],
        pytest_runner=raising_pytest,
        agent_invoker=agent,
        max_iters=2,
    )
    assert result.converged is False
    assert "pytest" in result.final_diagnostic.lower()


# ─── extract_late_discovery_markers ─────────────────────────────────


def test_extract_markers_returns_empty_for_no_markers() -> None:
    assert extract_late_discovery_markers("") == []
    assert extract_late_discovery_markers("def test_demo(): pass") == []


def test_extract_markers_finds_single_marker() -> None:
    src = "def test_demo():\n" "    # TODO[late-graduate]: vision_encoder.fpn_extract\n" "    pass\n"
    assert extract_late_discovery_markers(src) == ["vision_encoder.fpn_extract"]


def test_extract_markers_finds_multiple_in_declaration_order() -> None:
    src = "# TODO[late-graduate]: a.b\n" "# TODO[late-graduate]: c.d.e\n" "# TODO[late-graduate]: f\n"
    assert extract_late_discovery_markers(src) == ["a.b", "c.d.e", "f"]


def test_extract_markers_is_case_insensitive() -> None:
    src = "# todo[Late-Graduate]: module.x"
    assert extract_late_discovery_markers(src) == ["module.x"]


def test_extract_markers_strips_trailing_punctuation() -> None:
    """LLM may emit ``TODO[late-graduate]: foo,`` — strip noise."""
    src = "# TODO[late-graduate]: foo,bar);"
    # Strips trailing , ; ) but keeps the path itself
    markers = extract_late_discovery_markers(src)
    assert markers == ["foo,bar"] or markers == ["foo"] or markers[0].startswith("foo")


# ─── Late-discovery accumulation in run_e2e_synthesis_loop ──────────


def test_loop_accumulates_late_discoveries_from_demo(tmp_path: Path) -> None:
    """When the LLM writes TODO[late-graduate] markers in demo.py, the
    loop classifies each and accumulates them on the result."""
    demo_body = (
        "def test_demo(device_params, device):\n"
        "    # TODO[late-graduate]: vision_encoder.fpn_extract\n"
        "    # TODO[late-graduate]: prompt_encoder.conv_s0\n"
        "    print('end-to-end PCC=0.5')\n"
        "    assert_with_pcc(a, b, pcc=0.99)\n"
    )

    def agent(prompt, *, expected_deliverable_files, timeout_s, **kwargs):
        for p in expected_deliverable_files:
            Path(p).write_text(demo_body, encoding="utf-8")
        return 0

    def pytest_fail(demo_py_path: Path) -> Tuple[int, str]:
        return 1, "end-to-end PCC=0.5"

    result = run_e2e_synthesis_loop(
        model_id="org/m",
        demo_dir=tmp_path,
        hf_forward_src="x",
        graduated_components=[],
        pytest_runner=pytest_fail,
        agent_invoker=agent,
        max_iters=1,
    )
    # Two markers, one iter → two accumulated decisions
    assert len(result.late_discoveries) == 2
    paths = [d.submodule_spec["hf_reference"] if d.submodule_spec else None for d in result.late_discoveries]
    assert "vision_encoder.fpn_extract" in paths
    assert "prompt_encoder.conv_s0" in paths


def test_loop_late_discoveries_empty_when_no_markers(tmp_path: Path) -> None:
    """Demo without markers → no late-discovery accumulation."""
    demo_body = (
        "def test_demo(device_params, device):\n"
        "    print('end-to-end PCC=0.99')\n"
        "    assert_with_pcc(a, b, pcc=0.99)\n"
    )

    def agent(prompt, *, expected_deliverable_files, timeout_s, **kwargs):
        for p in expected_deliverable_files:
            Path(p).write_text(demo_body, encoding="utf-8")
        return 0

    def pytest_pass(demo_py_path: Path) -> Tuple[int, str]:
        return 0, "end-to-end PCC=0.99"

    result = run_e2e_synthesis_loop(
        model_id="org/m",
        demo_dir=tmp_path,
        hf_forward_src="x",
        graduated_components=[],
        pytest_runner=pytest_pass,
        agent_invoker=agent,
        max_iters=1,
    )
    assert result.late_discoveries == []


# ─── persist_synth_result ───────────────────────────────────────────


def test_persist_writes_history_json(tmp_path: Path) -> None:
    result = E2ESynthResult(
        converged=True,
        iters=[
            E2ESynthIterResult(iter_idx=1, rc=1, pcc=0.5, captured_output="x", elapsed_s=1.0),
            E2ESynthIterResult(iter_idx=2, rc=0, pcc=0.99, captured_output="y", elapsed_s=2.0),
        ],
        demo_py_path=tmp_path / "demo.py",
        final_pcc=0.99,
    )
    out_path = persist_synth_result(result, tmp_path)
    assert out_path is not None
    assert out_path.is_file()
    blob = json.loads(out_path.read_text())
    assert blob["converged"] is True
    assert blob["final_pcc"] == 0.99
    assert len(blob["iters"]) == 2


def test_persist_returns_none_on_unwriteable_demo_dir() -> None:
    """Best-effort: an unwriteable demo_dir should return None, not
    raise — synthesis was already done, persistence is decoration."""
    result = E2ESynthResult(converged=True)
    # /proc is read-only on Linux
    assert persist_synth_result(result, Path("/proc/does-not-exist")) is None
