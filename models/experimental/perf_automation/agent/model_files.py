"""read_model_files (PLAN section 7.3) — discovery sub-agent: GATHERER ONLY.

Division of labor (user decision, 2026-06-10):
  - The SUB-AGENT gathers: it explores the model directory read-only and
    returns structured findings WITH evidence notes. It decides nothing.
  - CODE validates FORM: JSON shape, files exist, case non-empty. Code never
    judges meaning (repo naming conventions are the agent's problem).
  - The LEAD AGENT approves: it reads the evidence notes and makes the
    continue/stop decision (probes.lead_review_gate), recorded in the manifest.
  - One non-negotiable FLOOR stays in code: no end-to-end correctness check
    candidate at all -> fatal. The lead resolves ambiguity, never overrides
    the need for a correctness gate.

Schema (architecture-neutral; fixed keys = harness roles, open `components`):
  perf_test  {path, case, note} | null     -> what tracy profiles
  pcc        {end_to_end: {path, threshold, note}, ...} -> correctness gates
  components {name: {path, note}}          -> isolated tests (open map)
  model_files [...]                        -> where levers get applied
  summary    str                           -> one-paragraph narrative for the lead
  flags      [{level, code, detail}]       -> fatal/warning findings
Path values may be pytest node ids (path::test_fn) — validated on the file part.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


class ModelFilesError(Exception):
    """Raised when the sub-agent's pathmap is malformed or points off-tree."""


PROMPT_TEMPLATE = (
    "You are GATHERING information about a model directory for performance-"
    "optimization tooling. You decide nothing — a reviewer makes the decisions. "
    "Explore the tree rooted at {root} (READ-ONLY) and return ONLY a JSON object:\n"
    '  "perf_test": {{"path": <repo-relative path or pytest node id of the '
    'perf/end-to-end test to profile>, "case": <a pytest -k expression selecting '
    "EXACTLY ONE existing parametrized case — read the @pytest.mark.parametrize "
    "ids; never invent an id; default to the FIRST id in the list unless told "
    'otherwise>, "note": <one '
    "sentence: what it measures, signposts, metrics>}} — or null if no "
    "perf/time-measuring test exists; if the end-to-end PCC test is runnable, "
    "prefer returning IT as perf_test with its FIRST parametrize id as case, "
    "plus a warning flag;\n"
    '  "pcc": object mapping role name to {{"path": <path or node id>, '
    '"threshold": <the numeric PCC threshold the test ENFORCES — read the '
    "assert (assert pcc > X / comp_pcc(..., X) / assert_with_pcc(..., X)) — "
    'null if you cannot find one>, "note": <one sentence: what it compares, '
    "against what reference, citing the threshold line/snippet>}}; "
    'use role "end_to_end" for the full-model correctness check if one exists;\n'
    '  "components": object mapping component name (whatever this model actually '
    "has: attention, backbone, unet_block, ...) to {{path, note}}; empty if none;\n"
    '  "model_files": array of repo-relative paths to the model source files '
    "(the files an optimizer would edit);\n"
    '  "summary": one short paragraph describing what you found and anything '
    "unusual the reviewer should know;\n"
    '  "flags": array of {{"level": "fatal"|"warning", "code": <short id>, '
    '"detail": <one sentence>}}. If there is NO end-to-end correctness check, '
    'emit {{"level": "fatal", "code": "no_end_to_end_pcc", ...}}. If there is no '
    'dedicated perf test, emit {{"level": "warning", "code": "no_perf_test", ...}}. '
    "If the end-to-end check exists but you cannot find the numeric threshold it "
    'enforces, emit {{"level": "fatal", "code": "no_pcc_threshold", ...}}.\n'
    "All paths relative to {root}. No commentary, no code fences."
)


def build_prompt(model_root: str | Path) -> str:
    return PROMPT_TEMPLATE.format(root=str(model_root))


def _require_file(root: Path, rel: Any, what: str) -> None:
    """Accept plain paths AND pytest node ids (path::test_fn) — agents often
    return the more precise form; validate the file part only."""
    file_part = rel.split("::", 1)[0] if isinstance(rel, str) else rel
    if not isinstance(file_part, str) or not (root / file_part).is_file():
        raise ModelFilesError(f"{what} -> {rel!r} is not a file under {root}")


def _norm_entry(value: Any, root: Path, what: str) -> dict[str, str]:
    """Normalize 'path-or-{path,note}' (Postel) into {path, note}."""
    if isinstance(value, str):
        entry = {"path": value, "note": ""}
    elif isinstance(value, dict) and "path" in value:
        entry = {"path": value["path"], "note": str(value.get("note", ""))}
        if "threshold" in value:
            entry["threshold"] = value["threshold"]
    else:
        raise ModelFilesError(f"{what} must be a path string or {{path, note}}")
    _require_file(root, entry["path"], what)
    return entry


def _validate(pathmap: dict[str, Any], model_root: Path) -> dict[str, Any]:
    if not isinstance(pathmap, dict):
        raise ModelFilesError("pathmap must be a JSON object")

    flags = pathmap.get("flags", [])
    if not isinstance(flags, list):
        raise ModelFilesError("pathmap.flags must be an array (may be empty)")
    fatal = [f for f in flags if isinstance(f, dict) and f.get("level") == "fatal"]
    warnings = [f for f in flags if isinstance(f, dict) and f.get("level") == "warning"]

    pcc_raw = pathmap.get("pcc")
    if not isinstance(pcc_raw, dict):
        raise ModelFilesError("pathmap.pcc must be an object")
    # FLOOR (code, not judgment): no e2e correctness candidate -> cannot continue.
    if fatal or "end_to_end" not in pcc_raw:
        details = (
            "; ".join(f"{f.get('code')}: {f.get('detail')}" for f in fatal) or "no 'end_to_end' PCC entry discovered"
        )
        raise ModelFilesError(f"CANNOT CONTINUE — fatal discovery flag(s): {details}")
    pcc = {name: _norm_entry(v, model_root, f"pcc entry {name!r}") for name, v in pcc_raw.items()}
    for name, entry in pcc.items():
        thr = entry.setdefault("threshold", None)
        if thr is not None:
            if isinstance(thr, bool) or not isinstance(thr, (int, float)) or not (0.0 < float(thr) < 1.0):
                raise ModelFilesError(f"pcc entry {name!r} threshold must be a number in (0, 1), got {thr!r}")
            entry["threshold"] = float(thr)
    # FLOOR (code): an e2e check with no enforced threshold gives the loop nothing to gate on.
    if pcc["end_to_end"]["threshold"] is None:
        raise ModelFilesError(
            "CANNOT CONTINUE — fatal discovery flag(s): no_pcc_threshold: "
            "end_to_end PCC test found but no numeric threshold extracted from it"
        )
    missing_thr = sorted(n for n, e in pcc.items() if n != "end_to_end" and e["threshold"] is None)
    if missing_thr and not any(f.get("code") == "no_component_pcc_threshold" for f in warnings):
        warnings.append(
            {
                "level": "warning",
                "code": "no_component_pcc_threshold",
                "detail": "no threshold for component pcc "
                + ", ".join(missing_thr)
                + "; loop gates on end_to_end only",
            }
        )

    perf_raw = pathmap.get("perf_test")
    if perf_raw is None:
        # WARNING path: no dedicated perf test -> profile the e2e PCC test.
        if not any(f.get("code") == "no_perf_test" for f in warnings):
            warnings.append(
                {"level": "warning", "code": "no_perf_test", "detail": "perf_test null; falling back to pcc.end_to_end"}
            )
        perf = {"path": pcc["end_to_end"]["path"], "case": None, "note": "fallback: profiling the end-to-end PCC test"}
    elif isinstance(perf_raw, dict):
        perf = _norm_entry(perf_raw, model_root, "perf_test")
        case = perf_raw.get("case")
        if not isinstance(case, str) or not case.strip():
            raise ModelFilesError("perf_test.case must be a non-empty pytest -k expression")
        perf["case"] = case
    else:
        raise ModelFilesError("pathmap.perf_test must be an object {path, case, note} or null")

    components_raw = pathmap.get("components", {})
    if not isinstance(components_raw, dict):
        raise ModelFilesError("pathmap.components must be an object (may be empty)")
    components = {name: _norm_entry(v, model_root, f"components entry {name!r}") for name, v in components_raw.items()}

    model_files = pathmap.get("model_files")
    if not isinstance(model_files, list) or not model_files:
        raise ModelFilesError("pathmap.model_files must be a non-empty array")
    for rel in model_files:
        _require_file(model_root, rel, "model_files entry")

    return {
        "perf_test": perf,
        "pcc": pcc,
        "components": components,
        "model_files": list(model_files),
        "summary": str(pathmap.get("summary", "")),
        "warnings": warnings,
    }


def read_model_files(
    model_root: str | Path,
    runner: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Run the discovery sub-agent and return the validated pathmap."""
    if runner is None:
        raise ValueError("runner (sub-agent query) required")
    model_root = Path(model_root)
    raw = runner(build_prompt(model_root))
    try:
        pathmap = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ModelFilesError(f"sub-agent did not return valid JSON: {exc}") from exc
    return _validate(pathmap, model_root)
