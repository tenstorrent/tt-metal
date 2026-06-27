"""E2E chained forward synthesizer (Item 3).

When the LLM verify pass (Item 2) returns FAIL or no chained forward
exists yet, this module iterates with the LLM to actually SYNTHESIZE a
chained ``demo.py`` that wires the graduated TTNN components into a
top-level forward matching HF's reference. Gated on end-to-end PCC
≥ 0.99, capped by iter budget.

The existing ``e2e_emitter.py`` produces a SKELETON ``demo.py`` with
``# TODO[e2e]`` markers — that's deterministic template substitution
and explicitly punts on the dataflow inference. This module fills in
those TODOs with LLM-synthesized chained logic, iterating against
end-to-end PCC as the gate.

Distinct from existing per-component synthesis (``llm_synth.py``):

  * Per-component synthesis writes ``_stubs/<comp>.py``, one
    component at a time, gated on per-component PCC ≥ 0.99 via
    ``tests/pcc/test_<comp>.py``.
  * E2E synthesis writes the chained ``demo.py`` that calls all
    graduated components in HF-matching order, gated on end-to-end
    PCC ≥ 0.99 via the strict gate.

Both share the same agent dispatcher (``_invoke_agent``) and the
same iter-loop shape (prompt → invoke → measure → adapt). The
e2e synthesizer is just orchestration at a different layer.

Design contract
---------------
Pure: prompt builder, parser, result dataclass — unit-testable
without LLM/HF/TTNN.

Impure: ``run_e2e_synthesis_loop`` orchestrator — invokes LLM and
runs pytest. Tested with mocked agent + mocked PCC measurement.

Never raises: every iter's failure routes back into the next iter's
prompt as diagnostic context. Budget exhaustion returns a structured
failure result the caller acts on (downgrade outcome, surface report)
rather than propagating an exception.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ─── Result schema ──────────────────────────────────────────────────


@dataclass
class E2ESynthIterResult:
    """One iter's outcome — feeds into the next iter's prompt."""

    iter_idx: int
    rc: int  # 0 = pytest passed end-to-end PCC, non-zero otherwise
    pcc: Optional[float] = None  # measured end-to-end PCC if available
    captured_output: str = ""  # pytest stdout for diagnostic-prompt
    elapsed_s: float = 0.0


@dataclass
class E2ESynthResult:
    """Overall result of the synthesis loop.

    ``converged`` is True iff at least one iter reached PCC ≥ 0.99.
    ``demo_py_path`` points at the last-written demo.py (regardless of
    convergence — the caller may still want to inspect it on FAIL).
    ``iters`` records every attempt's outcome for post-mortem.
    ``late_discoveries`` carries the structured per-marker decisions
    the discovery handler produced across all iters. Callers (e.g. the
    orchestrator) act on Case-C decisions by invoking late_graduate
    for the listed modules and re-running synthesis with them
    available.
    """

    converged: bool
    iters: List[E2ESynthIterResult] = field(default_factory=list)
    demo_py_path: Optional[Path] = None
    final_pcc: Optional[float] = None
    final_diagnostic: str = ""  # last iter's failure summary, empty on converge
    late_discoveries: List[Any] = field(default_factory=list)  # List[MissingPieceClassification]

    @property
    def iters_used(self) -> int:
        return len(self.iters)


# ─── Late-discovery markers ──────────────────────────────────────────


def extract_late_discovery_markers(demo_py_src: str) -> List[str]:
    """Pull ``TODO[late-graduate]: <module_path>`` markers from a
    synthesized demo.py.

    The synthesis prompt instructs the LLM to surface missing modules
    via this marker rather than fabricating. The synthesis loop reads
    these between iters to route each marker through the late-discovery
    classifier (Item 5) and downstream handlers (Item 4 Case C).

    Returns the list of module paths in declaration order. Empty list
    when no markers are present or input is empty. Never raises.
    """
    if not demo_py_src:
        return []
    import re

    pattern = re.compile(r"TODO\[late-graduate\]\s*:\s*(\S+)", re.IGNORECASE)
    return [m.group(1).rstrip(",;)") for m in pattern.finditer(demo_py_src)]


# ─── Prompt builder ──────────────────────────────────────────────────


_PCC_TARGET = 0.99


def build_synthesis_prompt(
    *,
    model_id: str,
    hf_forward_src: str,
    graduated_components: List[Dict[str, Any]],
    chain_divergence_summary: str,
    verify_diagnostic_summary: str,
    previous_iter_failure: str,
    demo_py_path: Path,
    iter_idx: int,
    pcc_target: float = _PCC_TARGET,
) -> str:
    """Render the LLM synthesis prompt.

    Pure function — no I/O. Carries every signal the LLM needs to
    write a correct chained forward: HF source, per-component
    graduates, the Item 1 + Item 2 diagnostics, the previous iter's
    failure (for iter ≥ 2), and the file path it must write.

    The prompt explicitly enumerates what's available so the LLM
    doesn't have to discover it: each graduated component lists its
    qualified_name, stub path, and HF reference module — exactly
    what the LLM needs to wire each into the chained forward.
    """
    components_block = (
        "\n".join(
            f"  • {c.get('name', '?')}\n"
            f"      stub:           {c.get('stub_path', '?')}\n"
            f"      hf_reference:   {c.get('hf_reference', '?')}\n"
            f"      class_name:     {c.get('class_name', '?')}"
            for c in graduated_components
        )
        or "  (none)"
    )
    drift_block = chain_divergence_summary or "(no chain-divergence diagnostic)"
    verify_block = verify_diagnostic_summary or "(no verify diagnostic)"
    prev_block = previous_iter_failure or "(first iter — no prior failure)"

    return f"""You are synthesizing the chained end-to-end forward for {model_id}.

CONTEXT
-------
Per-component PCC tests have passed: every TTNN submodule below is
individually numerically faithful to its HF reference at PCC >= {pcc_target:.2f}.

Your job: write a `demo.py` that calls each TTNN submodule in the
order HF's forward() does, wiring them together to produce HF-matching
end-to-end output (PCC >= {pcc_target:.2f}).

ITERATION {iter_idx}
-----------
HF reference forward() source
------------------------------
{hf_forward_src}

Graduated TTNN components (each on device, passing per-component PCC)
--------------------------------------------------------------------
{components_block}

Chain-divergence diagnostic (from runtime activation comparison, if available)
------------------------------------------------------------------------------
{drift_block}

LLM verify diagnostic (from static code review, if available)
--------------------------------------------------------------
{verify_block}

Previous iteration's failure (this iter must address it)
--------------------------------------------------------
{prev_block}

OUTPUT
------
Write the chained demo at: {demo_py_path}

Required structure:
  • test_demo(device_params, device) — pytest entry point.
  • Compute HF reference output on CPU.
  • Build each TTNN submodule via its stub's build() helper.
  • Call them in HF-matching order, feeding outputs forward.
  • assert_with_pcc(tt_final_output, hf_reference_output, pcc={pcc_target:.2f}).

Constraints:
  • The chained forward must pass end-to-end PCC >= {pcc_target:.2f}.
  • Use ttnn ops directly for any tensor reshapes / permutes between
    submodules — CPU bridges are LAST RESORT and must be commented as such.
  • If a submodule is missing for the chain to work, surface it in a
    `# TODO[late-graduate]: <module_path>` comment; do NOT fabricate.

You MAY edit only `{demo_py_path}`. DO NOT modify any other file.
"""


# ─── Synthesized demo.py validator ──────────────────────────────────


def parse_synthesized_demo_py(demo_py_path: Path) -> Optional[str]:
    """Best-effort validation that the LLM wrote a plausible demo.py.

    Returns the source on success, ``None`` on every failure mode
    (file missing, empty, no test_demo function found, no PCC
    assertion). Doesn't try to compile/execute — that's pytest's job
    in the next step.

    The structural checks here catch the trivial failure modes (LLM
    wrote a stub, wrote nothing, wrote markdown) without needing a
    full demo invocation.
    """
    if not demo_py_path.is_file():
        return None
    try:
        src = demo_py_path.read_text(encoding="utf-8")
    except Exception:
        return None
    if not src.strip():
        return None
    # Minimum structural check — defer real validation to pytest.
    if "def test_demo" not in src:
        return None
    if "assert_with_pcc" not in src and "comp_pcc" not in src:
        return None
    return src


# ─── PCC extraction from pytest output ──────────────────────────────


def extract_pcc_from_output(captured_output: str) -> Optional[float]:
    """Pull the measured end-to-end PCC from pytest stdout, if present.

    Looks for the two formats demo.py templates use:
      • ``end-to-end PCC=<f>`` (from e2e_emitter template)
      • ``PCC = <f>`` / ``pcc=<f>`` (loose match)

    Returns the first parseable float in [-1.0, 1.0]; None if no
    PCC value is recoverable. Used by the iter loop to gate
    convergence without re-running the demo.
    """
    if not captured_output:
        return None
    import re

    patterns = (
        r"end[-_]to[-_]end\s*PCC\s*[=:]\s*([+-]?\d+\.\d+)",
        r"\bPCC\s*[=:]\s*([+-]?\d+\.\d+)",
        r"\bpcc\s*[=:]\s*([+-]?\d+\.\d+)",
    )
    for pat in patterns:
        for match in re.finditer(pat, captured_output, flags=re.IGNORECASE):
            try:
                v = float(match.group(1))
            except (TypeError, ValueError):
                continue
            if -1.0 <= v <= 1.0:
                return v
    return None


# ─── Iter loop orchestrator ─────────────────────────────────────────


def run_e2e_synthesis_loop(
    *,
    model_id: str,
    demo_dir: Path,
    hf_forward_src: str,
    graduated_components: List[Dict[str, Any]],
    chain_divergence_summary: str = "",
    verify_diagnostic_summary: str = "",
    pytest_runner: Optional[Callable[[Path], "tuple[int, str]"]] = None,
    agent_invoker: Optional[Callable[..., int]] = None,
    max_iters: int = 5,
    pcc_target: float = _PCC_TARGET,
    agent_bin: str = "claude",
    agent_model: str = "sonnet",
    agent_timeout_s: int = 1200,
) -> E2ESynthResult:
    """Iterate with the LLM to synthesize a chained demo.py that
    passes end-to-end PCC.

    The two injectable callables (``pytest_runner``, ``agent_invoker``)
    are the seams unit tests use to drive the loop without LLM/pytest:

      * ``pytest_runner(demo_py_path) -> (rc, captured_output)`` —
        defaults to a real pytest invocation when None.
      * ``agent_invoker(prompt, *, expected_deliverable_files,
        timeout_s, **kwargs) -> rc`` — defaults to
        ``_invoke_agent`` when None.

    Loop shape (mirrors the per-component iterate loop):
      for iter in 1..max_iters:
        1. Build synthesis prompt (incl. previous failure if iter > 1)
        2. Invoke agent → writes demo.py
        3. Validate the written demo (structural)
        4. Run pytest → measure end-to-end PCC
        5. PCC >= pcc_target → return converged
        6. else → record failure, continue
      return not-converged with last_diagnostic.

    Best-effort: every per-iter failure routes back as diagnostic;
    only catastrophic (agent never produces a parseable demo across
    all iters) yields converged=False.
    """
    if pytest_runner is None:
        # Use the model-id-aware factory so HF_MODEL / PLANNER_TARGET_HF_MODEL
        # are set in the pytest subprocess env (matching _run_focused_pytest).
        pytest_runner = _make_default_pytest_runner(model_id=model_id)
    if agent_invoker is None:
        agent_invoker = _default_agent_invoker(agent_bin=agent_bin, agent_model=agent_model)

    demo_dir.mkdir(parents=True, exist_ok=True)
    demo_py_path = demo_dir / "demo.py"
    iters: List[E2ESynthIterResult] = []
    previous_failure = ""
    result = E2ESynthResult(converged=False, demo_py_path=demo_py_path)

    for iter_idx in range(1, max_iters + 1):
        prompt = build_synthesis_prompt(
            model_id=model_id,
            hf_forward_src=hf_forward_src,
            graduated_components=graduated_components,
            chain_divergence_summary=chain_divergence_summary,
            verify_diagnostic_summary=verify_diagnostic_summary,
            previous_iter_failure=previous_failure,
            demo_py_path=demo_py_path,
            iter_idx=iter_idx,
            pcc_target=pcc_target,
        )

        start = time.monotonic()
        try:
            agent_rc = agent_invoker(
                prompt,
                expected_deliverable_files=[demo_py_path],
                timeout_s=agent_timeout_s,
            )
        except Exception as exc:
            previous_failure = f"agent invocation raised {type(exc).__name__}: {exc}"
            iters.append(
                E2ESynthIterResult(
                    iter_idx=iter_idx,
                    rc=2,
                    pcc=None,
                    captured_output=previous_failure,
                    elapsed_s=time.monotonic() - start,
                )
            )
            continue
        if agent_rc != 0:
            previous_failure = f"agent exited rc={agent_rc} without writing demo.py"
            iters.append(
                E2ESynthIterResult(
                    iter_idx=iter_idx,
                    rc=agent_rc,
                    pcc=None,
                    captured_output=previous_failure,
                    elapsed_s=time.monotonic() - start,
                )
            )
            continue
        if parse_synthesized_demo_py(demo_py_path) is None:
            previous_failure = (
                f"agent wrote {demo_py_path} but it failed structural "
                f"validation (missing test_demo / missing PCC assertion / empty)"
            )
            iters.append(
                E2ESynthIterResult(
                    iter_idx=iter_idx,
                    rc=2,
                    pcc=None,
                    captured_output=previous_failure,
                    elapsed_s=time.monotonic() - start,
                )
            )
            continue

        try:
            pytest_rc, captured = pytest_runner(demo_py_path)
        except Exception as exc:
            previous_failure = f"pytest runner raised {type(exc).__name__}: {exc}"
            iters.append(
                E2ESynthIterResult(
                    iter_idx=iter_idx,
                    rc=2,
                    pcc=None,
                    captured_output=previous_failure,
                    elapsed_s=time.monotonic() - start,
                )
            )
            continue

        pcc = extract_pcc_from_output(captured)
        iters.append(
            E2ESynthIterResult(
                iter_idx=iter_idx,
                rc=pytest_rc,
                pcc=pcc,
                captured_output=captured,
                elapsed_s=time.monotonic() - start,
            )
        )

        # Parse the written demo for late-discovery markers. Each
        # ``TODO[late-graduate]: <module_path>`` the LLM left in the
        # forward gets classified (Item 5) and accumulated on the
        # result for the caller to act on (Item 4 Case C runs in
        # the orchestrator, not here, since it needs a real
        # component-iterate runner).
        try:
            demo_src = demo_py_path.read_text(encoding="utf-8") if demo_py_path.is_file() else ""
        except Exception:
            demo_src = ""
        markers = extract_late_discovery_markers(demo_src)
        if markers:
            try:
                from .late_discovery_classifier import heuristic_classify

                for marker in markers:
                    desc = f"missing module: {marker}"
                    classified = heuristic_classify(desc)
                    if classified is None:
                        # Fall through to Case C (real submodule) by
                        # default — that's the conservative route when
                        # heuristic can't decide. Caller's classifier
                        # can refine via the LLM later.
                        from .late_discovery_classifier import MissingPieceClassification

                        classified = MissingPieceClassification(
                            case="C",
                            piece_kind="submodule",
                            description=desc,
                            submodule_spec={
                                "name": marker.rsplit(".", 1)[-1] or marker,
                                "hf_reference": marker,
                                "class_name": "",
                            },
                            notes="heuristic fell through; default Case C",
                        )
                    result.late_discoveries.append(classified)
            except Exception:
                pass

        if pytest_rc == 0 and pcc is not None and pcc >= pcc_target:
            result.converged = True
            result.final_pcc = pcc
            result.iters = iters
            return result

        if pcc is not None:
            previous_failure = (
                f"iter {iter_idx} demo ran but end-to-end PCC was {pcc:.4f} "
                f"(target {pcc_target:.2f}). pytest output tail:\n"
                f"{captured[-2000:]}"
            )
        else:
            previous_failure = (
                f"iter {iter_idx} demo failed pytest rc={pytest_rc} or "
                f"could not extract PCC. pytest output tail:\n{captured[-2000:]}"
            )

    result.iters = iters
    result.final_diagnostic = previous_failure
    if iters and iters[-1].pcc is not None:
        result.final_pcc = iters[-1].pcc
    return result


# ─── Default injectables ────────────────────────────────────────────


def _default_agent_invoker(*, agent_bin: str, agent_model: str) -> Callable[..., int]:
    """Build the default agent-invoker for the synthesis loop.

    Wraps ``_invoke_agent`` from the existing agent helper module so
    the synthesis loop reuses the same stream-json + deliverable-
    verification + stall-detection infrastructure as the
    per-component iterate loop.

    Returned closure has signature (prompt, *, expected_deliverable_files,
    timeout_s, **kwargs) -> int (the agent's exit code). Tests inject
    their own callable bypassing this seam.
    """

    def _invoke(prompt: str, *, expected_deliverable_files, timeout_s: int, **_) -> int:
        from .agent import _invoke_agent

        return _invoke_agent(
            prompt,
            provider="claude",
            agent_bin=agent_bin,
            cwd=Path("."),
            model=agent_model,
            timeout_s=timeout_s,
            iter_tag="e2e_synth",
            expected_deliverable_files=list(expected_deliverable_files),
        )

    return _invoke


def _make_default_pytest_runner(
    *,
    model_id: str = "",
    timeout_s: int = 600,
) -> "Callable[[Path], tuple[int, str]]":
    """Build the default pytest-runner closure for the synthesis loop.

    Carries the env-setup that ``_run_focused_pytest`` does for the
    per-component loop:
      * HF_MODEL=<model_id>            (HF token resolution + demo
                                          template uses this to load weights)
      * PLANNER_TARGET_HF_MODEL=<id>   (tool-internal target marker)
      * PYTHONUNBUFFERED=1             (stream stdout for real-time logs)

    Without these, the demo would either fail to find the model or
    fall back to a different model_id than the one the synthesis
    targets. This factory closes over ``model_id`` so the returned
    runner has the env baked in.

    Tests inject their own runner bypassing this seam.
    """
    import os as _os
    import subprocess
    import sys

    def _runner(demo_py_path: Path) -> "tuple[int, str]":
        env = dict(_os.environ)
        if model_id:
            env["HF_MODEL"] = model_id
            env["PLANNER_TARGET_HF_MODEL"] = model_id
        env["PYTHONUNBUFFERED"] = "1"
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", f"{demo_py_path}::test_demo", "-v", "-s"],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=env,
            )
            return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
        except subprocess.TimeoutExpired as exc:
            return 124, f"(pytest timed out after {timeout_s}s: {exc})"
        except Exception as exc:
            return 2, f"(pytest invocation raised {type(exc).__name__}: {exc})"

    return _runner


def _default_pytest_runner(demo_py_path: Path) -> "tuple[int, str]":
    """Back-compat shim — defers to :func:`_make_default_pytest_runner`
    with no model_id (env vars not set). Kept for legacy callers that
    invoke the synthesizer without going through ``run_e2e_synthesis_loop``.
    """
    import subprocess
    import sys

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", f"{demo_py_path}::test_demo", "-v", "-s"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired as exc:
        return 124, f"(pytest timed out after 600s: {exc})"
    except Exception as exc:
        return 2, f"(pytest invocation raised {type(exc).__name__}: {exc})"


# ─── Persistence helper ─────────────────────────────────────────────


def persist_synth_result(result: E2ESynthResult, demo_dir: Path) -> Optional[Path]:
    """Write the synthesis loop's history to disk for post-mortem.

    Saves ``<demo_dir>/_synth/e2e_synth_history.json`` with the per-iter
    rc / pcc / elapsed plus the final diagnostic. Best-effort — returns
    None on persistence failure (never blocks the caller).
    """
    if result is None or demo_dir is None:
        return None
    out_dir = demo_dir / "_synth"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None
    out_path = out_dir / "e2e_synth_history.json"
    blob = {
        "converged": result.converged,
        "demo_py_path": str(result.demo_py_path) if result.demo_py_path else None,
        "final_pcc": result.final_pcc,
        "final_diagnostic": result.final_diagnostic,
        "iters": [
            {
                "iter_idx": r.iter_idx,
                "rc": r.rc,
                "pcc": r.pcc,
                "elapsed_s": r.elapsed_s,
                "captured_output_tail": r.captured_output[-1000:] if r.captured_output else "",
            }
            for r in result.iters
        ],
    }
    try:
        out_path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
        return out_path
    except Exception:
        return None


__all__ = [
    "E2ESynthIterResult",
    "E2ESynthResult",
    "build_synthesis_prompt",
    "extract_late_discovery_markers",
    "extract_pcc_from_output",
    "parse_synthesized_demo_py",
    "persist_synth_result",
    "run_e2e_synthesis_loop",
]
