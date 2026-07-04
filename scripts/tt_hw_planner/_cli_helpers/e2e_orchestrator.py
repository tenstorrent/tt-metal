"""Step 1 → 2 → 3 e2e bring-up orchestrator (Item 8).

Wires Items 1-7 into the unified flow we designed:

  Step 1 — Template reuse (Item 6)
    Look up the chained template registered for this model's family.
    If found AND promoted (Item 7): trust it, jump straight to running.
    If found but unpromoted: still try reuse but pass through Step 2.
    If not found: skip to Step 2 / Step 3 as appropriate.

  Step 2 — LLM verify (Item 2)
    Static code review: given the candidate chained forward (from
    Step 1 OR "no chain exists yet"), does it match HF.forward()?
    Returns PASS or a structured diagnostic.

  Step 3 — LLM synthesis (Item 3)
    Only fires if Step 2 said FAIL or no template existed at all.
    Iterates with the LLM to synthesize a chained demo.py, gated on
    end-to-end PCC ≥ 0.99. Includes late-discovery routing (Items 4-5)
    for any missing pieces synthesis identifies.

On Step 3 convergence:
    Register the resulting template (Item 6) and auto-promote if
    the family has reached the multi-model threshold (Item 7).

Best-effort throughout: every step's failure routes to the next
fallback. No exception propagates out of the orchestrator; the
caller gets a structured :class:`E2EBringupResult` it can read.

The orchestrator is decoupled from cli.py via injectable callables
(``verify_runner`` / ``synthesis_runner`` / ``classifier_runner`` /
``graduate_runner``), so unit tests run the full Step 1→2→3 flow
without invoking any real LLM, pytest, or HF subprocess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ─── Result schema ──────────────────────────────────────────────────


@dataclass
class E2EBringupResult:
    """Structured outcome of one Step 1→2→3 cycle.

    Fields:

      * ``status``       — terminal label (one of TEMPLATE_REUSED,
                            VERIFY_PASSED, SYNTHESIS_CONVERGED,
                            SYNTHESIS_FAILED, ERROR)
      * ``demo_py_path`` — final chained forward source, if any
      * ``family_key``   — HF model_type the orchestrator used
      * ``promoted``     — True iff a template promotion fired this run
      * ``steps``        — chronological per-step events for post-mortem
      * ``diagnostic``   — terminal failure reason on FAILED / ERROR
    """

    status: str
    demo_py_path: Optional[Path] = None
    family_key: Optional[str] = None
    promoted: bool = False
    steps: List[str] = field(default_factory=list)
    diagnostic: str = ""


# Terminal-status constants (string-compared by callers; pin the
# values so a refactor doesn't silently rename them).
STATUS_TEMPLATE_REUSED = "TEMPLATE_REUSED"
STATUS_VERIFY_PASSED = "VERIFY_PASSED"
STATUS_SYNTHESIS_CONVERGED = "SYNTHESIS_CONVERGED"
STATUS_SYNTHESIS_FAILED = "SYNTHESIS_FAILED"
STATUS_ERROR = "ERROR"


# ─── Orchestrator ───────────────────────────────────────────────────


def run_e2e_bringup(
    *,
    model_id: str,
    model_type: str,
    demo_dir: Path,
    hf_forward_src: str = "",
    graduated_components: Optional[List[Dict[str, Any]]] = None,
    chain_divergence_summary: str = "",
    template_finder: Optional[Callable[..., Any]] = None,
    template_registrar: Optional[Callable[..., Any]] = None,
    template_promoter: Optional[Callable[..., Any]] = None,
    verify_runner: Optional[Callable[..., Any]] = None,
    synthesis_runner: Optional[Callable[..., Any]] = None,
    repo_root: Optional[Path] = None,
) -> E2EBringupResult:
    """Run the Step 1 → 2 → 3 flow.

    Every injectable seam (``template_finder``, ``template_registrar``,
    ``template_promoter``, ``verify_runner``, ``synthesis_runner``)
    defaults to the corresponding Item 1-7 helper when None. Tests
    inject mocks; real callers leave them None.

    Returns :class:`E2EBringupResult`. Never raises — every failure
    path produces a structured result with diagnostic.
    """
    if graduated_components is None:
        graduated_components = []

    result = E2EBringupResult(status=STATUS_ERROR, family_key=model_type)

    # Resolve default injectables lazily so the imports don't fire
    # for unit tests that pass mocks for every seam.
    if template_finder is None:
        from .family_template_registry import find_template_for_model

        def template_finder(model_type=model_type, repo_root=repo_root):
            return find_template_for_model(model_type=model_type, repo_root=repo_root)

    if template_registrar is None:
        from .family_template_registry import register_template

        def template_registrar(**kwargs):
            return register_template(**kwargs, repo_root=repo_root)

    if template_promoter is None:
        from .template_promotion import auto_promote_after_register

        def template_promoter(family_key):
            return auto_promote_after_register(family_key=family_key, repo_root=repo_root)

    if verify_runner is None:
        from .llm_verify import run_llm_verify_pass

        def verify_runner(**kwargs):
            return run_llm_verify_pass(**kwargs)

    if synthesis_runner is None:
        from .e2e_synthesizer import run_e2e_synthesis_loop

        def synthesis_runner(**kwargs):
            return run_e2e_synthesis_loop(**kwargs)

    # ─── Step 1: template reuse lookup ───────────────────────────────
    template_entry = template_finder(model_type=model_type, repo_root=repo_root) if model_type else None
    if template_entry is not None:
        result.steps.append(f"step1: found family template (promoted={getattr(template_entry, 'promoted', False)})")
        # Promoted template: trust it directly.
        if getattr(template_entry, "promoted", False):
            result.status = STATUS_TEMPLATE_REUSED
            result.demo_py_path = Path(getattr(template_entry, "template_demo_source", ""))
            result.steps.append("step1: template is promoted, skipping verify")
            return result
    else:
        result.steps.append("step1: no family template — falling through to verify/synthesis")

    # ─── Step 2: LLM verify ──────────────────────────────────────────
    verify_verdict = None
    try:
        verify_verdict = verify_runner(
            model_id=model_id,
            demo_dir=demo_dir,
            drift_summary=chain_divergence_summary,
            hf_forward_src=hf_forward_src or None,
        )
    except Exception as exc:
        result.steps.append(f"step2: verify raised {type(exc).__name__}: {exc}")
    if verify_verdict is not None and getattr(verify_verdict, "ok", False):
        result.status = STATUS_VERIFY_PASSED
        if template_entry is not None:
            result.demo_py_path = Path(getattr(template_entry, "template_demo_source", ""))
        result.steps.append("step2: verify PASS — skipping synthesis")
        return result
    result.steps.append(
        "step2: verify FAIL or skipped — escalating to synthesis"
        + (f" (diag={verify_verdict.diagnostic.get('summary', '?')})" if verify_verdict is not None else "")
    )

    # ─── Step 3: LLM synthesis ───────────────────────────────────────
    synth_diag = (
        verify_verdict.diagnostic.get("summary", "") if verify_verdict is not None and verify_verdict.diagnostic else ""
    )
    synth_result = None
    try:
        synth_result = synthesis_runner(
            model_id=model_id,
            demo_dir=demo_dir,
            hf_forward_src=hf_forward_src,
            graduated_components=graduated_components,
            chain_divergence_summary=chain_divergence_summary,
            verify_diagnostic_summary=synth_diag,
        )
    except Exception as exc:
        result.steps.append(f"step3: synthesis raised {type(exc).__name__}: {exc}")
        result.status = STATUS_ERROR
        result.diagnostic = f"synthesis raised {type(exc).__name__}: {exc}"
        return result

    if synth_result is None:
        result.status = STATUS_SYNTHESIS_FAILED
        result.diagnostic = "synthesis_runner returned None"
        return result

    if not getattr(synth_result, "converged", False):
        result.status = STATUS_SYNTHESIS_FAILED
        result.diagnostic = getattr(synth_result, "final_diagnostic", "synthesis did not converge")
        result.demo_py_path = getattr(synth_result, "demo_py_path", None)
        result.steps.append(f"step3: synthesis FAILED after {len(getattr(synth_result, 'iters', []))} iter(s)")
        return result

    result.status = STATUS_SYNTHESIS_CONVERGED
    result.demo_py_path = getattr(synth_result, "demo_py_path", None)
    result.steps.append(
        f"step3: synthesis converged at PCC={getattr(synth_result, 'final_pcc', None)}"
        f" after {len(getattr(synth_result, 'iters', []))} iter(s)"
    )

    # ─── Register the template + auto-promote ───────────────────────
    if model_type and result.demo_py_path is not None:
        try:
            template_registrar(
                family_key=model_type,
                template_demo_source=str(result.demo_py_path),
                source_model_id=model_id,
                final_pcc=getattr(synth_result, "final_pcc", None),
                notes="from e2e_orchestrator synthesis-converged",
            )
            result.steps.append("step3+: template registered")
            promoted = template_promoter(model_type)
            if promoted is not None:
                result.promoted = True
                result.steps.append("step3+: template auto-promoted (threshold reached)")
        except Exception as exc:
            result.steps.append(f"step3+: registry update raised {type(exc).__name__}: {exc}")
            # Don't downgrade status — synthesis converged, persistence
            # is decoration.

    return result


__all__ = [
    "E2EBringupResult",
    "STATUS_TEMPLATE_REUSED",
    "STATUS_VERIFY_PASSED",
    "STATUS_SYNTHESIS_CONVERGED",
    "STATUS_SYNTHESIS_FAILED",
    "STATUS_ERROR",
    "run_e2e_bringup",
]
