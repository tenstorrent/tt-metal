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
  perf_test  {path, case, note} | null     -> what tracy profiles (the PRIMARY/first mode)
  perf_tests [{path, case, note}, ...]      -> ALL perf workloads of a MULTI-MODAL model
                                               (e.g. t2t/t2s/s2tt heads); optimize covers EACH so
                                               "optimize the model" = the whole pipeline, not one
                                               mode. Absent -> [perf_test] (single-workload model).
  pcc        {end_to_end: {path, threshold, note}, ...} -> correctness gates
  components {name: {path, note}}          -> isolated tests (open map)
  model_files [...]                        -> where levers get applied
  summary    str                           -> one-paragraph narrative for the lead
  flags      [{level, code, detail}]       -> fatal/warning findings
Path values may be pytest node ids (path::test_fn) — validated on the file part.
"""

from __future__ import annotations

import json
import os
import re as _re
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
    '  "perf_tests": array of {{path, case, note}} — if this model is MULTI-MODAL or '
    "exposes SEVERAL distinct perf workloads (e.g. separate per-mode/per-head tests like "
    "t2t/t2s/s2tt, prefill vs decode, or multiple resolutions), return ALL of them here so the "
    "optimizer covers the WHOLE pipeline, not one mode. List one entry per distinct workload "
    "(each its own path+case). If the model has only one perf workload, omit this (or return "
    "just the single perf_test);\n"
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


def _normalize_relpath(root: Path, rel: Any) -> Any:
    """Resolve a discovery-emitted path to a MODEL-ROOT-RELATIVE one, tolerating the two
    shapes the (non-deterministic) discovery agent emits interchangeably:
      * a pytest node id (path::test_fn) — keep the node suffix, normalize the file part;
      * a REPO-relative path that re-includes the model-dir prefix
        ('models/demos/<model>/tests/x.py') instead of model-relative ('tests/x.py') —
        which would otherwise double the prefix under model_root and fail (the nemotron
        flake). Strip everything up to and including the model-dir name.
    Returns the normalized string if a file resolves, else the original (so _require_file
    raises with the agent's literal path)."""
    if not isinstance(rel, str):
        return rel
    file_part, sep, node = rel.partition("::")
    parts = Path(file_part).parts
    candidates = [file_part]
    if root.name in parts:  # repo-relative path that re-includes the model dir
        idx = len(parts) - 1 - list(reversed(parts)).index(root.name)
        candidates.append(str(Path(*parts[idx + 1 :])) if idx + 1 < len(parts) else "")
    for cand in candidates:
        if cand and (root / cand).is_file():
            return cand + (sep + node if sep else "")
    return rel


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
    entry["path"] = _normalize_relpath(root, entry["path"])  # tolerate repo-relative / node-id forms
    _require_file(root, entry["path"], what)
    return entry


def _enumerate_pcc_components(root: Path) -> dict[str, Any]:
    """DETERMINISTIC component enumeration from the canonical per-component PCC tests
    (<model_root>/tests/pcc/test_*.py) — one component per test, named by the test file. This is the
    source of truth for 'all modules': COMPLETE and IDENTICAL across runs, unlike the discovery
    sub-agent's freehand `components` map (which was non-deterministic — 21 vs 23 across runs — and
    dropped real modules like decoder_layer). Path resolves to the component's impl
    (_stubs/<name>.py or tt/<name>.py) when present, else the PCC test itself. Returns {} for models
    without a tests/pcc/ dir, so the caller falls back to the sub-agent map (stays model-agnostic)."""
    pcc_dir = root / "tests" / "pcc"
    if not pcc_dir.is_dir():
        return {}
    out: dict[str, Any] = {}
    for f in sorted(pcc_dir.glob("test_*.py")):
        name = f.stem[len("test_") :]
        if not name or name == "conftest":
            continue
        impl_rel = None
        for cand in (root / "_stubs" / f"{name}.py", root / "tt" / f"{name}.py"):
            if cand.is_file():
                impl_rel = str(cand.relative_to(root))
                break
        out[name] = {
            "path": impl_rel or str(f.relative_to(root)),
            "note": f"per-component PCC test: tests/pcc/{f.name}",
        }
    return out


def _find_pcc_for_task(root: Path, task: str) -> str | None:
    """Find the end-to-end PCC test that gates a given pipeline/task. Searches the model's tests/ for
    a `def test_...` whose name contains the task and ('pcc' or 'e2e'), e.g. task 't2t' ->
    test_e2e_pcc_t2t. Returns a 'relpath::func' node id (model-root-relative), or None."""
    pat = _re.compile(r"def\s+(test_[A-Za-z0-9_]*)\s*\(")
    cands = []
    tests = root / "tests"
    if tests.is_dir():
        for f in sorted(tests.rglob("*.py")):
            if "pcc" not in f.name and "e2e" not in f.name:
                continue
            try:
                txt = f.read_text(errors="ignore")
            except OSError:
                continue
            for fn in pat.findall(txt):
                low = fn.lower()
                if task.lower() in low and ("pcc" in low or "e2e" in low):
                    cands.append(f"{f.relative_to(root)}::{fn}")
    # prefer the most specific (shortest name that still matches) and stable order
    return sorted(cands, key=len)[0] if cands else None


_NON_PIPELINE_DEMOS = {"common", "audio_loader", "output_validation", "utils", "__init__"}


def _demo_tasks(root: Path) -> list[str]:
    """The pipeline tasks emit-e2e emitted, from demo/demo_<task>.py (the canonical per-Call list).
    Strips a trailing _vN version suffix and dedups, so demo_t2t.py + demo_t2t_v2.py => one 't2t'."""
    demo = root / "demo"
    if not demo.is_dir():
        return []
    seen, tasks = set(), []
    for f in sorted(demo.glob("demo_*.py")):
        task = _re.sub(r"_v\d+$", "", f.stem[len("demo_") :])
        if not task or task in _NON_PIPELINE_DEMOS or task in seen:
            continue
        seen.add(task)
        tasks.append(task)
    return tasks


def _first_test_fn(path: Path) -> str | None:
    """The first `def test_...` in a file (the perf test's pytest function name)."""
    try:
        m = _re.search(r"def\s+(test_[A-Za-z0-9_]*)\s*\(", path.read_text(errors="ignore"))
    except OSError:
        return None
    return m.group(1) if m else None


def _find_perf_test_for_task(root: Path, task: str) -> str | None:
    """Find an EXISTING perf test for a task, by tt-metal/emitted conventions in priority order.
    Returns a 'relpath::testfn' node id, or None (caller will GENERATE one from the demo). Patterns:
    tests/e2e/test_<task>_perf.py (emitted) -> tests/test_*<task>*perf*.py / *_device_perf*.py
    (tt-metal flat) -> tests/perf/test_*<task>*.py. Task match is by substring so single-model demos
    (task may be 'main' / the model name) still resolve their one perf test."""
    t = task.lower()
    tests = root / "tests"
    if not tests.is_dir():
        return None
    pats = [
        f"e2e/test_{task}_perf.py",
        "test_*_perf.py",
        "test_*_device_perf.py",
        "perf/test_*.py",
        "*_perf.py",
        "**/test_*perf*.py",
    ]
    for pat in pats:
        matches = sorted(tests.glob(pat))
        # named task must match its own token (never cross-assign another pipeline's test); 'main' takes first
        if t != "main":
            chosen = next((m for m in matches if t in m.name.lower()), None)
        else:
            chosen = matches[0] if matches else None
        if chosen:
            fn = _first_test_fn(chosen) or chosen.stem
            return f"{chosen.relative_to(root)}::{fn}"
    return None


def _enumerate_pipelines(root: Path) -> list[dict[str, Any]]:
    """DETERMINISTIC pipeline discovery. The perf test is ALWAYS generated from scratch off the demo,
    never an existing test reused. PRIMARY signal = demo/demo_<task>.py: each is ONE pipeline with its
    perf test auto-genned from that demo, plus its e2e PCC. FALLBACK (a single demo.py driving all
    tasks internally): ONE 'main' pipeline, perf test auto-genned from demo/demo.py. Returns [] only
    when there is no demo to generate from (caller then uses the sub-agent's single perf_test)."""
    from .perf_test_gen import generate_perf_test

    out: list[dict[str, Any]] = []
    tasks = _demo_tasks(root)
    if tasks:
        # Always (re)generate each pipeline's perf test from its demo (force=True); never reuse an
        # existing/partial one. No demo file to lift from -> None (caller gates; no silent reuse).
        for task in tasks:
            demo_rel = f"demo/demo_{task}.py"
            perf = generate_perf_test(root, task, demo_rel, force=True) if (root / demo_rel).is_file() else None
            out.append(
                {
                    "task": task,
                    "demo": demo_rel,
                    "perf_test": perf,
                    "pcc_test": _find_pcc_for_task(root, task),
                }
            )
        return out
    # Single demo.py driving all tasks internally: ONE 'main' pipeline, perf test auto-genned from it
    # (same from-scratch rule as above; never reuse an existing test). No demo.py -> [] (caller gates).
    demo_main = "demo/demo.py"
    if (root / demo_main).is_file():
        perf = generate_perf_test(root, "main", demo_main, force=True)
        return [{"task": "main", "demo": demo_main, "perf_test": perf, "pcc_test": _find_pcc_for_task(root, "main")}]
    return []


def _validate(pathmap: dict[str, Any], model_root: Path) -> dict[str, Any]:
    if not isinstance(pathmap, dict):
        raise ModelFilesError("pathmap must be a JSON object")

    flags = pathmap.get("flags", [])
    if not isinstance(flags, list):
        raise ModelFilesError("pathmap.flags must be an array (may be empty)")
    fatal = [f for f in flags if isinstance(f, dict) and f.get("level") == "fatal"]
    warnings = [f for f in flags if isinstance(f, dict) and f.get("level") == "warning"]

    # Module-level optimize (TT_PERF_MODULE_LEVEL) gates each module on its OWN
    # per-component PCC test and never enters the whole-model pipeline, so the
    # absence of a whole-model end_to_end correctness check is NOT fatal here —
    # the per-component test IS the correctness gate. Demote no_end_to_end_pcc to
    # a warning and drop the end_to_end-required floor for module-level runs. The
    # other module-level consumers (perf_test_gen, before_loop, probes) already
    # honor this flag; the discovery floor was the one place that never did.
    _module_level = os.environ.get("TT_PERF_MODULE_LEVEL", "") not in ("", "0", "false", "False")
    if _module_level:
        _demoted = [f for f in fatal if f.get("code") == "no_end_to_end_pcc"]
        fatal = [f for f in fatal if f.get("code") != "no_end_to_end_pcc"]
        warnings.extend(_demoted)

    pcc_raw = pathmap.get("pcc")
    if not isinstance(pcc_raw, dict):
        raise ModelFilesError("pathmap.pcc must be an object")
    # FLOOR (code, not judgment): no e2e correctness candidate -> cannot continue
    # (except module-level, which supplies its own per-component gate — see above).
    if fatal or ("end_to_end" not in pcc_raw and not _module_level):
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
    # FLOOR (code): an e2e check with no enforced threshold gives the loop nothing to gate on
    # (only when an end_to_end entry exists — module-level may legitimately have none).
    if "end_to_end" in pcc and pcc["end_to_end"]["threshold"] is None:
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
        if "end_to_end" in pcc:
            perf = {
                "path": pcc["end_to_end"]["path"],
                "case": None,
                "note": "fallback: profiling the end-to-end PCC test",
            }
        else:
            perf = {
                "path": "",
                "case": None,
                "note": "module-level: perf test generated from --pcc-test in before_loop",
            }
    elif isinstance(perf_raw, dict):
        perf = _norm_entry(perf_raw, model_root, "perf_test")
        case = perf_raw.get("case")
        if not isinstance(case, str) or not case.strip():
            raise ModelFilesError("perf_test.case must be a non-empty pytest -k expression")
        perf["case"] = case
    else:
        raise ModelFilesError("pathmap.perf_test must be an object {path, case, note} or null")

    # MULTI-MODAL: a model may expose several distinct perf workloads (e.g. t2t/t2s/s2tt heads).
    # "Optimize the model" must cover ALL of them, not one. perf_tests is the full list; perf_test
    # stays the PRIMARY (first) for backward compat. Absent -> [perf] (single-workload model).
    perf_tests_raw = pathmap.get("perf_tests")
    if perf_tests_raw is None:
        perf_tests = [perf]
    elif isinstance(perf_tests_raw, list) and perf_tests_raw:
        perf_tests = []
        for i, e in enumerate(perf_tests_raw):
            if not isinstance(e, dict):
                raise ModelFilesError(f"pathmap.perf_tests[{i}] must be an object {{path, case, note}}")
            pe = _norm_entry(e, model_root, f"perf_tests[{i}]")
            c = e.get("case")
            pe["case"] = c if (isinstance(c, str) and c.strip()) else None
            perf_tests.append(pe)
        if perf_raw is None:  # no singular given -> primary is the first listed mode
            perf = perf_tests[0]
    else:
        raise ModelFilesError("pathmap.perf_tests must be a non-empty array or null")

    components_raw = pathmap.get("components", {})
    if not isinstance(components_raw, dict):
        raise ModelFilesError("pathmap.components must be an object (may be empty)")
    # DETERMINISTIC override: when canonical per-component PCC tests exist, enumerate the component
    # set from them (complete + stable every run) instead of trusting the sub-agent's freehand list.
    pcc_components = _enumerate_pcc_components(Path(model_root))
    if pcc_components:
        components_raw = pcc_components
    components = {name: _norm_entry(v, model_root, f"components entry {name!r}") for name, v in components_raw.items()}

    model_files = pathmap.get("model_files")
    if not isinstance(model_files, list) or not model_files:
        raise ModelFilesError("pathmap.model_files must be a non-empty array")
    model_files = [_normalize_relpath(model_root, rel) for rel in model_files]
    for rel in model_files:
        _require_file(model_root, rel, "model_files entry")

    # DETERMINISTIC pipeline detection: each e2e perf test = one stitched (emit-e2e) pipeline, paired
    # with its own end-to-end PCC. One => single-modal; many => multi-modal. This is what discovery is
    # FOR — find the pipeline(s) + per-pipeline PCC; optimize then runs each. Falls back to the single
    # discovered perf_test/end_to_end-PCC when no e2e perf tests exist (model-agnostic).
    pipelines = _enumerate_pipelines(Path(model_root))
    if not pipelines:
        _pcc_node = pcc["end_to_end"]["path"] if "end_to_end" in pcc else perf["path"]
        pipelines = [{"task": "main", "perf_test": perf["path"], "pcc_test": _pcc_node}]
    is_multimodal = len(pipelines) > 1

    # Under mandatory regen, the baseline profiles the regenerated pipeline perf test(s), not the
    # discovery sub-agent's pick (which can be non-deterministic / a partial prefill-only test).
    if os.environ.get("PERF_REGEN_PERF_TEST") == "1":
        pipe_nodes = [p["perf_test"] for p in pipelines if p.get("perf_test")]
        if pipe_nodes:
            reconciled = []
            for node in pipe_nodes:
                path_part, _, fn = str(node).partition("::")
                entry = _norm_entry(path_part, model_root, "pipeline perf_test")
                entry["case"] = fn or None
                reconciled.append(entry)
            perf_tests = reconciled
            perf = reconciled[0]

    return {
        "perf_test": perf,
        "perf_tests": perf_tests,
        "pcc": pcc,
        "components": components,
        "pipelines": pipelines,
        "is_multimodal": is_multimodal,
        "model_files": list(model_files),
        "summary": str(pathmap.get("summary", "")),
        "warnings": warnings,
    }


_PCC_THR_RE = _re.compile(r"(?:pcc|comp_pcc|assert_with_pcc|allclose)[^0-9]{0,48}(0\.9\d+)", _re.IGNORECASE)


def _extract_pcc_threshold(pcc_file: Path, default: float = 0.99) -> float:
    """Numeric PCC threshold lifted from the test text (e.g. assert_with_pcc(..., 0.99)); default 0.99."""
    try:
        txt = pcc_file.read_text(errors="ignore")
    except OSError:
        return default
    m = _PCC_THR_RE.search(txt)
    if m:
        try:
            v = float(m.group(1))
            if 0.0 < v < 1.0:
                return v
        except ValueError:
            pass
    return default


def resolve_pcc_node(
    model_root: str | Path, pcc_node: str, tt_root: str | Path, threshold: float | None = None
) -> tuple[str, float, Path]:
    """Resolve a --pcc-test node ('path::fn', tt-root-relative or absolute) to a (model-root-relative
    node, threshold, absolute file) triple. The PCC test may live outside model_root, so the relative
    path may contain '..' (pytest and _require_file resolve it on the filesystem)."""
    model_root = Path(model_root).resolve()
    tt_root = Path(tt_root).resolve()
    file_part, _, fn = str(pcc_node).partition("::")
    pcc_abs = (Path(file_part) if os.path.isabs(file_part) else tt_root / file_part).resolve()
    if not pcc_abs.is_file():
        raise ModelFilesError(f"--pcc-test file {file_part!r} not found (looked under {tt_root})")
    node_rel = os.path.relpath(pcc_abs, model_root) + (f"::{fn}" if fn else "")
    _require_file(model_root, node_rel, "pcc end_to_end (--pcc-test)")
    thr = float(threshold) if threshold else _extract_pcc_threshold(pcc_abs)
    if not (0.0 < thr < 1.0):
        raise ModelFilesError(f"--pcc-test threshold must be in (0, 1), got {thr!r}")
    return node_rel, thr, pcc_abs


def read_model_files(
    model_root: str | Path,
    runner: Callable[[str], str] | None = None,
    pcc_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the discovery sub-agent and return the validated pathmap. pcc_override (a {path, threshold}
    e2e entry from --pcc-test) is injected as pcc.end_to_end and the other scattered tests are dropped:
    discovery still maps the model, but the correctness gate is the one the user pinned."""
    if runner is None:
        raise ValueError("runner (sub-agent query) required")
    model_root = Path(model_root)
    raw = runner(build_prompt(model_root))
    try:
        pathmap = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ModelFilesError(f"sub-agent did not return valid JSON: {exc}") from exc
    if pcc_override:
        pathmap["pcc"] = {"end_to_end": {"path": pcc_override["path"], "threshold": pcc_override["threshold"]}}
        pathmap["perf_test"] = None
        pathmap["perf_tests"] = None
        pathmap["components"] = {}
    return _validate(pathmap, model_root)
