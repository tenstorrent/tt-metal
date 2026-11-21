"""End-to-end integration test (Items 1-8 chained together).

Each module has its own unit tests pinning the contract in isolation;
this test exercises the full Step 1→2→3 flow through the orchestrator
with mocks at the LLM / pytest / HF boundaries to confirm the pieces
compose correctly. The mock layer is at the impure seams only —
internal logic (parser, classifier, registry I/O, prompt builder)
runs for real.

Scenarios:
  * First-bringup synthesis path: no family template, verify FAIL,
    synthesis converges, template registered (unpromoted).
  * Second-bringup confirmation path: previous family template exists,
    verify PASSES, no synthesis, registry confirms + auto-promotes.
  * Third-bringup reuse path: template promoted, fast-path TEMPLATE_REUSED.
  * Demoted-template path: template was demoted, treat as no-template,
    re-synthesize.
  * Synthesis-discovers-marker path: synthesis fails with a
    late-graduate marker, late_discoveries list is populated.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

from scripts.tt_hw_planner._cli_helpers.e2e_orchestrator import (
    STATUS_SYNTHESIS_CONVERGED,
    STATUS_SYNTHESIS_FAILED,
    STATUS_TEMPLATE_REUSED,
    STATUS_VERIFY_PASSED,
    run_e2e_bringup,
)
from scripts.tt_hw_planner._cli_helpers.family_template_registry import (
    demote_template,
    find_template_for_model,
    load_registry,
    register_template,
)
from scripts.tt_hw_planner._cli_helpers.template_promotion import (
    auto_promote_after_register,
)


@contextmanager
def _registry_in_tmp(tmp_path: Path):
    """Redirect the registry to tmp_path for the test."""
    with patch(
        "scripts.tt_hw_planner._cli_helpers.family_template_registry._registry_path",
        return_value=tmp_path / "learned_chained_templates.json",
    ):
        yield


# ─── Stubs for the orchestrator's injectable seams ──────────────────


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
    late_discoveries: List[Any] = field(default_factory=list)


# ─── Scenario 1: first sibling, synthesis converges ─────────────────


def test_first_sibling_synthesis_converges_registers_unpromoted(tmp_path: Path) -> None:
    """First-of-family bringup: no template → verify can't help → synthesis
    runs and converges → template registered (1 confirmation, not yet
    promoted)."""
    converged_demo = tmp_path / "demo.py"

    def synth_converges(**kwargs):
        return _StubSynthResult(
            converged=True,
            final_pcc=0.995,
            demo_py_path=converged_demo,
            iters=[1],
        )

    with _registry_in_tmp(tmp_path):
        result = run_e2e_bringup(
            model_id="facebook/sam2-hiera-tiny",
            model_type="sam2",
            demo_dir=tmp_path,
            verify_runner=lambda **kw: _StubVerdict(ok=False, diagnostic={"summary": "no chain yet"}),
            synthesis_runner=synth_converges,
        )

        assert result.status == STATUS_SYNTHESIS_CONVERGED
        assert result.demo_py_path == converged_demo

        # Registry should now have a sam2 entry, unpromoted
        reg = load_registry()
        assert "sam2" in reg
        assert reg["sam2"].confirmed_models == ["facebook/sam2-hiera-tiny"]
        assert reg["sam2"].promoted is False  # single confirmation < threshold


# ─── Scenario 2: second sibling confirms + auto-promotes ────────────


def test_second_sibling_confirmation_promotes_template(tmp_path: Path) -> None:
    """Second sibling bringup: template exists (unpromoted from
    sibling 1). With registrar + promoter in the orchestrator chain,
    the second bringup's registration triggers auto-promote."""
    with _registry_in_tmp(tmp_path):
        # Pre-state: first sibling already registered
        register_template(
            family_key="sam2",
            template_demo_source="m1_demo.py",
            source_model_id="facebook/sam2-hiera-tiny",
            clock=lambda: 100.0,
        )

        # Second bringup: verify will say PASS (template-reuse path
        # since unpromoted falls through to verify). Orchestrator
        # marks VERIFY_PASSED and does NOT call registrar (no
        # synthesis fired). To exercise promotion, we register
        # explicitly here as the synthesis-converged equivalent.
        synth_demo = tmp_path / "m2_demo.py"

        def synth_converges(**kwargs):
            return _StubSynthResult(
                converged=True,
                final_pcc=0.99,
                demo_py_path=synth_demo,
                iters=[1],
            )

        # Force the verify to fail so synthesis fires (otherwise verify
        # PASS would short-circuit before the registrar runs).
        result = run_e2e_bringup(
            model_id="facebook/sam2-hiera-large",
            model_type="sam2",
            demo_dir=tmp_path,
            verify_runner=lambda **kw: _StubVerdict(ok=False, diagnostic={"summary": "verify says fail"}),
            synthesis_runner=synth_converges,
        )

        assert result.status == STATUS_SYNTHESIS_CONVERGED
        assert result.promoted is True

        reg = load_registry()
        assert reg["sam2"].promoted is True
        assert "facebook/sam2-hiera-large" in reg["sam2"].confirmed_models


# ─── Scenario 3: promoted template → fast-path reuse ────────────────


def test_third_sibling_uses_promoted_fast_path(tmp_path: Path) -> None:
    """Third sibling: template is promoted → orchestrator short-circuits
    to STATUS_TEMPLATE_REUSED without invoking verify or synthesis."""
    with _registry_in_tmp(tmp_path):
        # Pre-state: registry has a promoted template
        register_template(
            family_key="sam2",
            template_demo_source="x",
            source_model_id="m1",
            clock=lambda: 100.0,
        )
        register_template(
            family_key="sam2",
            template_demo_source="x",
            source_model_id="m2",
            clock=lambda: 200.0,
        )
        auto_promote_after_register(family_key="sam2", clock=lambda: 250.0)

        verify_called = [False]
        synth_called = [False]

        def verify_fail(**kw):
            verify_called[0] = True
            return _StubVerdict(ok=False)

        def synth_unused(**kw):
            synth_called[0] = True
            return _StubSynthResult(converged=False)

        result = run_e2e_bringup(
            model_id="facebook/sam2-hiera-base",
            model_type="sam2",
            demo_dir=tmp_path,
            verify_runner=verify_fail,
            synthesis_runner=synth_unused,
        )

        assert result.status == STATUS_TEMPLATE_REUSED
        assert verify_called[0] is False  # short-circuit before verify
        assert synth_called[0] is False  # and before synthesis


# ─── Scenario 4: demoted template → re-synthesize ───────────────────


def test_demoted_template_forces_synthesis(tmp_path: Path) -> None:
    """A demoted template must NOT be returned by find_template_for_model.
    The orchestrator falls through as if no template existed at all,
    so synthesis fires fresh."""
    with _registry_in_tmp(tmp_path):
        register_template(
            family_key="sam2",
            template_demo_source="x",
            source_model_id="m1",
            clock=lambda: 100.0,
        )
        register_template(
            family_key="sam2",
            template_demo_source="x",
            source_model_id="m2",
            clock=lambda: 200.0,
        )
        auto_promote_after_register(family_key="sam2", clock=lambda: 250.0)
        # Now demote — should force re-synthesis even though promoted
        demote_template(family_key="sam2", reason="HF v5 regression")

        # Verify the lookup returns None for the demoted family
        assert find_template_for_model(model_type="sam2") is None

        # Orchestrator falls through to verify + synthesis
        def synth_converges(**kw):
            return _StubSynthResult(converged=True, final_pcc=0.99, demo_py_path=tmp_path / "demo.py", iters=[1])

        result = run_e2e_bringup(
            model_id="facebook/sam2-hiera-tiny",
            model_type="sam2",
            demo_dir=tmp_path,
            verify_runner=lambda **kw: _StubVerdict(ok=False),
            synthesis_runner=synth_converges,
        )
        assert result.status == STATUS_SYNTHESIS_CONVERGED


# ─── Scenario 5: synthesis fails → no promotion ─────────────────────


def test_failed_synthesis_does_not_promote(tmp_path: Path) -> None:
    """Failed synthesis must not contaminate the registry — no false
    promotion of an unverified template."""
    with _registry_in_tmp(tmp_path):

        def synth_fails(**kw):
            return _StubSynthResult(
                converged=False,
                final_pcc=0.5,
                final_diagnostic="iter budget exhausted",
                iters=[1, 2, 3],
            )

        result = run_e2e_bringup(
            model_id="org/m",
            model_type="brand_new_arch",
            demo_dir=tmp_path,
            verify_runner=lambda **kw: _StubVerdict(ok=False),
            synthesis_runner=synth_fails,
        )

        assert result.status == STATUS_SYNTHESIS_FAILED
        assert result.promoted is False
        # No registry entry for failed runs
        reg = load_registry()
        assert "brand_new_arch" not in reg


# ─── Scenario 6: late-discovery accumulation in synthesis result ────


def test_synthesis_loop_propagates_late_discoveries(tmp_path: Path) -> None:
    """When the synthesis loop's demo.py contains TODO[late-graduate]
    markers, the result's late_discoveries list is populated. The
    orchestrator can act on these (route through classifier → late_graduate)
    even though it doesn't auto-execute that here."""
    from scripts.tt_hw_planner._cli_helpers.e2e_synthesizer import run_e2e_synthesis_loop

    demo_body = (
        "def test_demo(device_params, device):\n"
        "    # TODO[late-graduate]: vision_encoder.fpn\n"
        "    assert_with_pcc(a, b, pcc=0.99)\n"
    )

    def agent(prompt, *, expected_deliverable_files, timeout_s, **kwargs):
        for p in expected_deliverable_files:
            Path(p).write_text(demo_body)
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
    assert len(result.late_discoveries) == 1
    assert result.late_discoveries[0].submodule_spec is not None
    assert result.late_discoveries[0].submodule_spec["hf_reference"] == "vision_encoder.fpn"
