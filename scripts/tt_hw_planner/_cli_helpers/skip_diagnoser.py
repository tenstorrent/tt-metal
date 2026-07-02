"""LLM-driven diagnosis for pytest SKIPs in the bring-up auto-iterate loop.

When an auto-generated PCC test SKIPs at the harness layer (test
scaffold can't synthesize inputs, resolved submodule path is
uncallable, etc.), the orchestrator used to silently drop the
component from the candidate pool and count it as graduated. That
caused the seamless-m4t false "all graduated rc=0" outcome.

This module wires an LLM agent that diagnoses WHY the SKIP fired and
takes corrective action:

  * Rewrite `_CANDIDATE_SUBMODULE_PATHS` in the test file to point at
    a callable submodule (e.g. add `[0]` index for ModuleList).
  * Rewrite the test's `_make_arg_for()` or its sample-kwargs builder
    to add model-specific kwargs (e.g. `position_ids` for attention).
  * Mark the component as DECOMPOSE (genuinely uncallable as one unit
    — break into sub-components).
  * Mark the component as MANUAL (needs human-authored test).

This is the Tier 2 fallback that runs after the deterministic Tier 1
fixes (discover-side path indexing + runtime ModuleList fallback in
``_resolve()``) and only when those don't recover the SKIP.

Cost: ~$0.05–0.50 per SKIP investigation (one sonnet/haiku invocation,
no iter loop). Way cheaper than the wasted iters that historically
spun on a single unrecoverable component.
"""

from __future__ import annotations

import json
import re
import subprocess as _subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


# Diagnosis verdicts emitted by the agent — caller acts on these.
VERDICT_FIXED = "fixed"  # agent rewrote the test, caller should re-run pytest
VERDICT_DECOMPOSE = "decompose"  # component needs to be broken into sub-components
VERDICT_MANUAL = "manual"  # human-authored test needed; flag in OUTCOME
VERDICT_UNKNOWN = "unknown"  # agent couldn't classify; treat conservatively as MANUAL


# Harness-pattern markers we recognize as "this SKIP is a harness bug,
# not a stub bug." Keep in sync with auto_iterate.py's per-iter
# harness-marker list.
_HARNESS_MARKERS = (
    "HF reference forward",
    "_make_arg_for()",
    "synthetic inputs from _make_arg_for",
    "incompatible with this submodule's expected shapes",
    "the synthetic inputs",
    "Module [ModuleList]",
    "Module [Sequential]",
    "missing the required",
)


def is_harness_skip(reason_text: str) -> bool:
    """Pure check: does the SKIP reason text look like a harness gap
    (test scaffold couldn't run) rather than a real stub issue?

    Used by the iter loop to decide whether to invoke the LLM
    diagnoser vs. treat the SKIP as a stub-level problem (which the
    regular iter agent would address).
    """
    if not reason_text:
        return False
    return any(marker in reason_text for marker in _HARNESS_MARKERS)


def build_diagnoser_prompt(
    *,
    component_name: str,
    skip_reason: str,
    test_file_content: str,
    hf_reference_excerpt: str = "",
    captured_inputs_excerpt: str = "",
) -> str:
    """Assemble the prompt for the SKIP-diagnoser agent.

    Sections:
      1. The SKIP reason (verbatim, from pytest)
      2. The auto-generated test file (so the agent can see and edit
         _CANDIDATE_SUBMODULE_PATHS or input-builder logic)
      3. HF reference source excerpt (so the agent knows the real
         forward signature)
      4. Captured-input manifest (so the agent knows the actual shapes
         and kwargs the model expects)
      5. Verdict instructions
    """
    sep = "─" * 70
    prompt_lines = [
        "You are diagnosing a pytest SKIP in a Tenstorrent bring-up auto-test.",
        "",
        f"COMPONENT: {component_name}",
        "",
        sep,
        "PYTEST SKIP REASON:",
        sep,
        skip_reason.strip() or "(no reason captured)",
        "",
        sep,
        "AUTO-GENERATED TEST FILE (tests/pcc/test_<component>.py):",
        sep,
        test_file_content,
        "",
    ]
    if hf_reference_excerpt:
        prompt_lines.extend(
            [
                sep,
                "HF REFERENCE FORWARD SIGNATURE (excerpt):",
                sep,
                hf_reference_excerpt,
                "",
            ]
        )
    if captured_inputs_excerpt:
        prompt_lines.extend(
            [
                sep,
                "CAPTURED INPUT MANIFEST (real shapes/dtypes observed):",
                sep,
                captured_inputs_excerpt,
                "",
            ]
        )
    prompt_lines.extend(
        [
            sep,
            "YOUR JOB:",
            sep,
            "Diagnose why pytest SKIPPED this test. The four common causes are:",
            "",
            "  (1) _CANDIDATE_SUBMODULE_PATHS lands on a ModuleList or",
            "      Sequential (a container without forward). Fix: rewrite",
            "      the candidate paths to index into the container, e.g.",
            "      'resblocks' → 'resblocks[0]'.",
            "",
            "  (2) _make_arg_for() builds the wrong inputs for this module's",
            "      forward(...) signature. Fix: edit the sample-kwargs",
            "      construction in the test to match the real signature.",
            "",
            "  (3) The component is genuinely uncallable as one unit (e.g.",
            "      a pure container that has no semantically meaningful",
            "      forward). Verdict: DECOMPOSE (break into sub-components).",
            "",
            "  (4) The component needs hand-authored test logic that no",
            "      reasonable auto-generator can produce. Verdict: MANUAL.",
            "",
            "OUTPUT FORMAT (REQUIRED — your last action must be ONE of):",
            "",
            "  A. If you can fix the test in-place: edit the test file with",
            "     your Edit/Write tool and respond:",
            "        VERDICT: fixed",
            "        SUMMARY: <one-line description of the fix>",
            "",
            "  B. If the component should be decomposed:",
            "        VERDICT: decompose",
            "        SUMMARY: <why one unit isn't testable>",
            "",
            "  C. If the test needs a human:",
            "        VERDICT: manual",
            "        SUMMARY: <what a human needs to write>",
            "",
            "DO NOT edit the stub (_stubs/<component>.py) — that's the",
            "regular iter loop's job. You ONLY edit the test file.",
        ]
    )
    return "\n".join(prompt_lines)


def parse_diagnoser_verdict(agent_output: str) -> Dict[str, str]:
    """Pull the VERDICT and SUMMARY lines from the agent's response.

    Returns a dict with ``verdict`` and ``summary``. ``verdict`` is
    one of the VERDICT_* constants; falls back to VERDICT_UNKNOWN on
    unparseable output.
    """
    verdict = VERDICT_UNKNOWN
    summary = ""
    if not agent_output:
        return {"verdict": verdict, "summary": summary}

    # Find the LAST occurrence — agent may reason aloud earlier.
    verdict_matches = list(re.finditer(r"^\s*VERDICT:\s*(\w+)\s*$", agent_output, re.MULTILINE))
    if verdict_matches:
        raw = verdict_matches[-1].group(1).strip().lower()
        if raw in {VERDICT_FIXED, VERDICT_DECOMPOSE, VERDICT_MANUAL}:
            verdict = raw

    summary_matches = list(re.finditer(r"^\s*SUMMARY:\s*(.+?)\s*$", agent_output, re.MULTILINE))
    if summary_matches:
        summary = summary_matches[-1].group(1).strip()

    return {"verdict": verdict, "summary": summary}


def diagnose_skip(
    *,
    component_name: str,
    skip_reason: str,
    test_file: Path,
    demo_dir: Path,
    agent_bin: str,
    agent_model: str = "sonnet",
    timeout_s: int = 300,
    hf_reference_excerpt: str = "",
) -> Dict[str, Any]:
    """Invoke the LLM diagnoser for one SKIPped component.

    Returns a dict::

        {
            "component": <name>,
            "verdict": "fixed" | "decompose" | "manual" | "unknown",
            "summary": <agent's one-line summary>,
            "agent_stdout": <raw agent output, for debugging>,
            "rc": <agent subprocess return code>,
        }

    The caller is responsible for acting on the verdict (re-running
    pytest after "fixed", spawning the decomposer after "decompose",
    flagging in OUTCOME after "manual").

    Best-effort: any exception during the subprocess call is caught
    and returns verdict=UNKNOWN with the exception text in summary —
    a failed diagnosis is still better than crashing the iter loop.
    """
    try:
        test_content = test_file.read_text(encoding="utf-8") if test_file.is_file() else "(test file not found)"
    except Exception as exc:
        return {
            "component": component_name,
            "verdict": VERDICT_UNKNOWN,
            "summary": f"could not read test file: {exc}",
            "agent_stdout": "",
            "rc": 1,
        }

    # Captured-input manifest excerpt (real shapes + kwargs names).
    captured_excerpt = ""
    try:
        from ..bringup_loop import _safe_id as _safe

        manifest_path = demo_dir / "_captured" / _safe(component_name) / "manifest.json"
        if manifest_path.is_file():
            data = json.loads(manifest_path.read_text())
            captured_excerpt = json.dumps(data, indent=2)[:2000]
    except Exception:
        pass

    prompt = build_diagnoser_prompt(
        component_name=component_name,
        skip_reason=skip_reason,
        test_file_content=test_content,
        hf_reference_excerpt=hf_reference_excerpt,
        captured_inputs_excerpt=captured_excerpt,
    )

    cmd = [
        agent_bin,
        "-p",
        "--dangerously-skip-permissions",
        "--add-dir",
        str(demo_dir.parent.parent.parent),  # add demo dir's repo root
        "--model",
        agent_model,
        "--output-format",
        "text",
        "--tools",
        "Read",
        "Write",
        "Edit",
        "Grep",
    ]
    try:
        proc = _subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        rc = proc.returncode
    except _subprocess.TimeoutExpired:
        return {
            "component": component_name,
            "verdict": VERDICT_UNKNOWN,
            "summary": f"agent timed out after {timeout_s}s",
            "agent_stdout": "",
            "rc": 124,
        }
    except Exception as exc:
        return {
            "component": component_name,
            "verdict": VERDICT_UNKNOWN,
            "summary": f"agent subprocess failed: {type(exc).__name__}: {exc}",
            "agent_stdout": "",
            "rc": 1,
        }

    parsed = parse_diagnoser_verdict(out)
    return {
        "component": component_name,
        "verdict": parsed["verdict"],
        "summary": parsed["summary"],
        "agent_stdout": out,
        "rc": rc,
    }


def diagnose_skips_in_demo(
    *,
    demo_dir: Path,
    skipped_components: List[str],
    skip_reasons: Dict[str, str],
    agent_bin: str,
    agent_model: str = "sonnet",
    timeout_s_per_component: int = 300,
) -> List[Dict[str, Any]]:
    """Convenience wrapper: diagnose every SKIPped component in a demo.

    Returns a list of diagnosis dicts (one per component). Caller
    iterates and acts on verdicts.

    Components whose SKIP reason isn't recognized as a harness pattern
    (per :func:`is_harness_skip`) are silently passed through with
    verdict=UNKNOWN — these aren't candidates for harness-diagnosis
    iteration and the iter loop should treat them as regular stub
    iterations instead.
    """
    results: List[Dict[str, Any]] = []
    for comp in sorted(set(skipped_components)):
        reason = skip_reasons.get(comp, "")
        if not is_harness_skip(reason):
            results.append(
                {
                    "component": comp,
                    "verdict": VERDICT_UNKNOWN,
                    "summary": "skip reason doesn't match harness patterns; not a diagnoser candidate",
                    "agent_stdout": "",
                    "rc": 0,
                }
            )
            continue
        from ..bringup_loop import _safe_id as _safe

        test_file = demo_dir / "tests" / "pcc" / f"test_{_safe(comp)}.py"
        result = diagnose_skip(
            component_name=comp,
            skip_reason=reason,
            test_file=test_file,
            demo_dir=demo_dir,
            agent_bin=agent_bin,
            agent_model=agent_model,
            timeout_s=timeout_s_per_component,
        )
        results.append(result)
    return results
