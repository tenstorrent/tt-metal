"""Unit tests for the Step 1→2→3 e2e orchestrator (Item 8).

Every injectable seam (template_finder, template_registrar,
template_promoter, verify_runner, synthesis_runner) is mocked so the
full flow runs without any LLM/pytest/HF call. Each test exercises
one routing decision (which step fired, which short-circuit hit).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.tt_hw_planner._cli_helpers.e2e_orchestrator import (
    E2EBringupResult,
    STATUS_ERROR,
    STATUS_SYNTHESIS_CONVERGED,
    STATUS_SYNTHESIS_FAILED,
    STATUS_TEMPLATE_REUSED,
    STATUS_VERIFY_PASSED,
    run_e2e_bringup,
)


# ─── Stub builders ──────────────────────────────────────────────────


@dataclass
class _StubTemplate:
    family_key: str = "sam2"
    template_demo_source: str = "models/.../demo.py"
    promoted: bool = False


@dataclass
class _StubVerdict:
    ok: bool
    diagnostic: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _StubSynthResult:
    converged: bool
    final_pcc: Optional[float] = None
    demo_py_path: Optional[Path] = None
    final_diagnostic: str = ""
    iters: List[Any] = field(default_factory=list)


def _no_template_finder(model_type=None, repo_root=None):
    return None


def _promoted_template_finder(model_type=None, repo_root=None):
    return _StubTemplate(promoted=True)


def _unpromoted_template_finder(model_type=None, repo_root=None):
    return _StubTemplate(promoted=False)


def _verify_pass(**kwargs):
    return _StubVerdict(ok=True, diagnostic={"summary": "looks good"})


def _verify_fail(**kwargs):
    return _StubVerdict(ok=False, diagnostic={"summary": "missing reshape"})


def _verify_returns_none(**kwargs):
    return None


def _synth_converge(**kwargs):
    return _StubSynthResult(
        converged=True,
        final_pcc=0.995,
        demo_py_path=Path("/fake/demo.py"),
        iters=[1, 2],
    )


def _synth_fail(**kwargs):
    return _StubSynthResult(
        converged=False,
        final_pcc=0.42,
        demo_py_path=Path("/fake/demo.py"),
        final_diagnostic="iter budget exhausted",
        iters=[1, 2, 3],
    )


def _synth_raises(**kwargs):
    raise RuntimeError("synth boom")


def _registrar_noop(**kwargs):
    return None


def _promoter_no_promote(family_key=None):
    return None


def _promoter_promote(family_key=None):
    return _StubTemplate(promoted=True)


# ─── Step 1: template reuse routing ─────────────────────────────────


def test_promoted_template_short_circuits_to_reuse(tmp_path: Path) -> None:
    """Promoted family template → STATUS_TEMPLATE_REUSED without any
    verify or synthesis. The cheapest possible path."""
    result = run_e2e_bringup(
        model_id="org/m",
        model_type="sam2",
        demo_dir=tmp_path,
        template_finder=_promoted_template_finder,
        verify_runner=_verify_fail,  # should NOT fire
        synthesis_runner=_synth_fail,  # should NOT fire
        template_registrar=_registrar_noop,
        template_promoter=_promoter_no_promote,
    )
    assert result.status == STATUS_TEMPLATE_REUSED
    assert any("step1" in s for s in result.steps)
    assert all("step3" not in s for s in result.steps)  # no synthesis fired


def test_unpromoted_template_falls_through_to_verify(tmp_path: Path) -> None:
    """Unpromoted template → run verify (not auto-trusted)."""
    verify_called = [False]

    def verify(**kwargs):
        verify_called[0] = True
        return _verify_pass(**kwargs)

    result = run_e2e_bringup(
        model_id="org/m",
        model_type="sam2",
        demo_dir=tmp_path,
        template_finder=_unpromoted_template_finder,
        verify_runner=verify,
        synthesis_runner=_synth_fail,
        template_registrar=_registrar_noop,
        template_promoter=_promoter_no_promote,
    )
    assert verify_called[0] is True
    assert result.status == STATUS_VERIFY_PASSED


def test_no_template_falls_through_to_verify(tmp_path: Path) -> None:
    """No template at all → still try verify first (it'll say
    "needs synthesis" but the path is unified)."""
    result = run_e2e_bringup(
        model_id="org/m",
        model_type="sam2",
        demo_dir=tmp_path,
        template_finder=_no_template_finder,
        verify_runner=_verify_pass,
        synthesis_runner=_synth_fail,
        template_registrar=_registrar_noop,
        template_promoter=_promoter_no_promote,
    )
    # Verify said PASS with no template — orchestrator stamps VERIFY_PASSED.
    assert result.status == STATUS_VERIFY_PASSED


# ─── Step 2: verify routing ─────────────────────────────────────────


def test_verify_pass_returns_verify_passed_status(tmp_path: Path) -> None:
    """Verify PASS → no synthesis, status reflects the route taken."""
    synth_called = [False]

    def synth(**kwargs):
        synth_called[0] = True
        return _synth_converge(**kwargs)

    result = run_e2e_bringup(
        model_id="org/m",
        model_type="sam2",
        demo_dir=tmp_path,
        template_finder=_unpromoted_template_finder,
        verify_runner=_verify_pass,
        synthesis_runner=synth,
        template_registrar=_registrar_noop,
        template_promoter=_promoter_no_promote,
    )
    assert result.status == STATUS_VERIFY_PASSED
    assert synth_called[0] is False


def test_verify_fail_escalates_to_synthesis(tmp_path: Path) -> None:
    """Verify FAIL → synthesis fires."""
    synth_called = [False]

    def synth(**kwargs):
        synth_called[0] = True
        return _synth_converge(**kwargs)

    result = run_e2e_bringup(
        model_id="org/m",
        model_type="sam2",
        demo_dir=tmp_path,
        template_finder=_unpromoted_template_finder,
        verify_runner=_verify_fail,
        synthesis_runner=synth,
        template_registrar=_registrar_noop,
        template_promoter=_promoter_no_promote,
    )
    assert synth_called[0] is True
    assert result.status == STATUS_SYNTHESIS_CONVERGED


def test_verify_returns_none_escalates_to_synthesis(tmp_path: Path) -> None:
    """Verify returning None (couldn't run) → treat as FAIL, escalate."""
    result = run_e2e_bringup(
        model_id="org/m",
        model_type="sam2",
        demo_dir=tmp_path,
        template_finder=_no_template_finder,
        verify_runner=_verify_returns_none,
        synthesis_runner=_synth_converge,
        template_registrar=_registrar_noop,
        template_promoter=_promoter_no_promote,
    )
    assert result.status == STATUS_SYNTHESIS_CONVERGED


def test_verify_diagnostic_propagates_into_synthesis_call(tmp_path: Path) -> None:
    """The verify failure diagnostic must be forwarded to synthesis
    as ``verify_diagnostic_summary`` so the LLM has full context."""
    captured: Dict[str, Any] = {}

    def synth(**kwargs):
        captured.update(kwargs)
        return _synth_converge(**kwargs)

    run_e2e_bringup(
        model_id="org/m",
        model_type="sam2",
        demo_dir=tmp_path,
        template_finder=_no_template_finder,
        verify_runner=_verify_fail,  # diagnostic = "missing reshape"
        synthesis_runner=synth,
        template_registrar=_registrar_noop,
        template_promoter=_promoter_no_promote,
    )
    assert captured.get("verify_diagnostic_summary") == "missing reshape"


# ─── Step 3: synthesis routing ──────────────────────────────────────


def test_synthesis_converged_status(tmp_path: Path) -> None:
    result = run_e2e_bringup(
        model_id="org/m",
        model_type="sam2",
        demo_dir=tmp_path,
        template_finder=_no_template_finder,
        verify_runner=_verify_fail,
        synthesis_runner=_synth_converge,
        template_registrar=_registrar_noop,
        template_promoter=_promoter_no_promote,
    )
    assert result.status == STATUS_SYNTHESIS_CONVERGED
    assert result.demo_py_path == Path("/fake/demo.py")


def test_synthesis_failed_status_carries_diagnostic(tmp_path: Path) -> None:
    result = run_e2e_bringup(
        model_id="org/m",
        model_type="sam2",
        demo_dir=tmp_path,
        template_finder=_no_template_finder,
        verify_runner=_verify_fail,
        synthesis_runner=_synth_fail,
        template_registrar=_registrar_noop,
        template_promoter=_promoter_no_promote,
    )
    assert result.status == STATUS_SYNTHESIS_FAILED
    assert "iter budget exhausted" in result.diagnostic


def test_synthesis_raises_yields_error_status(tmp_path: Path) -> None:
    """Synthesis raising shouldn't propagate — caught with structured
    ERROR status + diagnostic."""
    result = run_e2e_bringup(
        model_id="org/m",
        model_type="sam2",
        demo_dir=tmp_path,
        template_finder=_no_template_finder,
        verify_runner=_verify_fail,
        synthesis_runner=_synth_raises,
        template_registrar=_registrar_noop,
        template_promoter=_promoter_no_promote,
    )
    assert result.status == STATUS_ERROR
    assert "synth boom" in result.diagnostic


def test_synthesis_returns_none_yields_failed_status(tmp_path: Path) -> None:
    """Synthesis returning None → SYNTHESIS_FAILED, no crash."""
    result = run_e2e_bringup(
        model_id="org/m",
        model_type="sam2",
        demo_dir=tmp_path,
        template_finder=_no_template_finder,
        verify_runner=_verify_fail,
        synthesis_runner=lambda **kw: None,
        template_registrar=_registrar_noop,
        template_promoter=_promoter_no_promote,
    )
    assert result.status == STATUS_SYNTHESIS_FAILED


# ─── Template registration + promotion on convergence ──────────────


def test_synthesis_convergence_triggers_registration(tmp_path: Path) -> None:
    """Converged synthesis must call the registrar with the right args."""
    registered: Dict[str, Any] = {}

    def registrar(**kwargs):
        registered.update(kwargs)
        return None

    run_e2e_bringup(
        model_id="org/m",
        model_type="sam2",
        demo_dir=tmp_path,
        template_finder=_no_template_finder,
        verify_runner=_verify_fail,
        synthesis_runner=_synth_converge,
        template_registrar=registrar,
        template_promoter=_promoter_no_promote,
    )
    assert registered["family_key"] == "sam2"
    assert registered["source_model_id"] == "org/m"
    assert registered["final_pcc"] == 0.995


def test_synthesis_convergence_triggers_auto_promote(tmp_path: Path) -> None:
    """Registrar succeeds → promoter called → if it returns an entry,
    result.promoted is True."""
    result = run_e2e_bringup(
        model_id="org/m",
        model_type="sam2",
        demo_dir=tmp_path,
        template_finder=_no_template_finder,
        verify_runner=_verify_fail,
        synthesis_runner=_synth_converge,
        template_registrar=_registrar_noop,
        template_promoter=_promoter_promote,  # promotion fires
    )
    assert result.promoted is True


def test_synthesis_convergence_promoter_no_promote_keeps_flag_false(tmp_path: Path) -> None:
    """If the promoter says "not eligible yet" (returns None),
    result.promoted stays False — we still converged, just didn't
    cross the multi-model threshold."""
    result = run_e2e_bringup(
        model_id="org/m",
        model_type="sam2",
        demo_dir=tmp_path,
        template_finder=_no_template_finder,
        verify_runner=_verify_fail,
        synthesis_runner=_synth_converge,
        template_registrar=_registrar_noop,
        template_promoter=_promoter_no_promote,  # threshold not reached
    )
    assert result.status == STATUS_SYNTHESIS_CONVERGED
    assert result.promoted is False


def test_registry_failure_doesnt_downgrade_convergence(tmp_path: Path) -> None:
    """Synthesis converged but registrar raised — status stays
    CONVERGED; persistence is decoration, not a gate."""

    def raising_registrar(**kwargs):
        raise IOError("disk full")

    result = run_e2e_bringup(
        model_id="org/m",
        model_type="sam2",
        demo_dir=tmp_path,
        template_finder=_no_template_finder,
        verify_runner=_verify_fail,
        synthesis_runner=_synth_converge,
        template_registrar=raising_registrar,
        template_promoter=_promoter_no_promote,
    )
    assert result.status == STATUS_SYNTHESIS_CONVERGED


# ─── Audit trail ────────────────────────────────────────────────────


def test_steps_record_routing_decisions(tmp_path: Path) -> None:
    """The steps log must chronicle each decision for post-mortem.
    Pin the format so a refactor doesn't accidentally drop a step."""
    result = run_e2e_bringup(
        model_id="org/m",
        model_type="sam2",
        demo_dir=tmp_path,
        template_finder=_no_template_finder,
        verify_runner=_verify_fail,
        synthesis_runner=_synth_converge,
        template_registrar=_registrar_noop,
        template_promoter=_promoter_no_promote,
    )
    step_text = " | ".join(result.steps)
    assert "step1" in step_text  # template-lookup happened
    assert "step2" in step_text  # verify happened
    assert "step3" in step_text  # synthesis fired


# ─── Constants ──────────────────────────────────────────────────────


def test_status_constants_are_stable_strings() -> None:
    """Pin so downstream log-scrapers / dashboards don't break on rename."""
    assert STATUS_TEMPLATE_REUSED == "TEMPLATE_REUSED"
    assert STATUS_VERIFY_PASSED == "VERIFY_PASSED"
    assert STATUS_SYNTHESIS_CONVERGED == "SYNTHESIS_CONVERGED"
    assert STATUS_SYNTHESIS_FAILED == "SYNTHESIS_FAILED"
    assert STATUS_ERROR == "ERROR"
