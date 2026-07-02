# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""bringup-mcp — external stdio MCP server exposing the DETERMINISTIC bring-up gates to a
Claude-Code agent, so `auto-up`/`promote --engine cc` can drive per-component bring-up through the
shared cc harness.

Model-agnostic: no per-model logic. It REUSES the existing deterministic core so cc and fsm behave
identically — the per-component PCC runner (`cli._run_focused_pytest` + `_parse_pytest_report` +
`_scope_report_to_demo`), the component list (`cli._list_component_pcc_tests`, scoped to
`bringup_status.json`), the attempt-cap arithmetic (`_cli_helpers.bringup_ladder`), the loop failure
taxonomy that drives the cap (`cli._classify_failure`), the verdict classifier that gates decompose
(`failure_classifier.classify_failure` + `component_decomposer.failure_class_warrants_decomposition`),
the torch-wrapper detector (`cli._stub_uses_torch_wrapper`), the best-native snapshot rule
(`auto_iterate._should_snapshot_best_native`), the stable CPU-fallback synthesizer
(`cli._rewrite_components_to_stable_fallback`), and the decompose plan / consumer
(`decompose --write-plan` + `decomposition_consumer.consume_decomposition_plan`).

Snapshot contract — identical to the fsm loop (`auto_iterate._skip_component_to_fallback`):
  .py.bak            = original scaffold torch-wrapper (CPU-delegating baseline); scaffold-time only.
  .py.preiter_native = an ALREADY-native stub, snapshotted once BEFORE the agent first edits it.
  .py.best_native    = highest-PCC native body seen this session (gated by _should_snapshot_best_native).
  .py.last_good_native = written on a PCC graduation; the shared marker promote/emit-e2e read.

Fallback = re-testing CASCADE (last_good_native > best_native > preiter_native > bak), then a
synthesized stable CPU wrapper — the SAME mixed-execution terminal the fsm loop produces.

Harness-skip (Tier-2 diagnoser parity with the fsm loop): when a PCC test SKIPs for a harness reason
(uncallable submodule / wrong synthetic inputs / ModuleList — detected via `skip_diagnoser.is_harness_skip`)
the gate routes to `fix_harness` (fix the TEST, not the stub) so the component can graduate NATIVELY
instead of being mis-repaired as a stub bug or silently counted as graduated (the seamless-m4t
false-rc=0 bug). At cap an unfixable harness-skip is retired via `mark_harness_skipped` (writes
harness_skipped.json + skip_diagnosis.json, the SAME artifacts the fsm OUTCOME banner surfaces).

Gate ladder per ungraduated component (mirrors the fsm loop, model-agnostic):
  harness-skip -> fix_harness. Otherwise emit (attempt 0) -> repair (pre-cap). AT cap: if the failure
  verdict warrants it (and not already decomposed) -> decompose (split + retire parent); else if still
  harness-skipping -> mark_manual; else -> fall_back_to_cpu.
can_stop is true ONLY when every material component is graduated OR fallen back to CPU OR
harness-skipped (manual).

Config via env:
  BRINGUP_MCP_DEMO_DIR / BRINGUP_MCP_MODEL_ID / BRINGUP_MCP_STATE (required)
  BRINGUP_MCP_MAX_ATTEMPTS (base cap, default 2)
  BRINGUP_MCP_PCC (default 0.99) · BRINGUP_MCP_TIMEOUT (default 1800)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from scripts.tt_hw_planner import cli as _cli  # noqa: E402
from scripts.tt_hw_planner._cli_helpers import auto_iterate as _auto  # noqa: E402
from scripts.tt_hw_planner._cli_helpers import bringup_ladder  # noqa: E402

from mcp.server.fastmcp import FastMCP  # noqa: E402

mcp = FastMCP("bringup-mcp")

_DEMO_DIR = Path(os.environ.get("BRINGUP_MCP_DEMO_DIR", "") or ".")
_MODEL_ID = os.environ.get("BRINGUP_MCP_MODEL_ID", "")
_STATE_PATH = Path(os.environ.get("BRINGUP_MCP_STATE", "") or (_DEMO_DIR / ".bringup_cc_state.json"))
_MAX_ATTEMPTS = int(os.environ.get("BRINGUP_MCP_MAX_ATTEMPTS", "2"))
_HARD_CAP = max(3, _MAX_ATTEMPTS * 2)
_PCC = float(os.environ.get("BRINGUP_MCP_PCC", "0.99"))
_TIMEOUT = int(os.environ.get("BRINGUP_MCP_TIMEOUT", "1800"))


def _load_state() -> dict:
    try:
        return json.loads(_STATE_PATH.read_text())
    except Exception:
        return {}


def _save_state(st: dict) -> None:
    try:
        _STATE_PATH.write_text(json.dumps(st, indent=2))
    except Exception:
        pass


def _component_of(test_file: str) -> str:
    stem = Path(test_file).stem
    return stem[len("test_") :] if stem.startswith("test_") else stem


def _stub_path(component: str) -> Path:
    return _DEMO_DIR / "_stubs" / f"{component}.py"


def _snap(component: str, suffix: str) -> Path:
    return _stub_path(component).with_suffix(suffix)


def _is_torch_wrapper(stub: Path) -> bool:
    """Delegate to the SAME detector the fsm loop uses so 'native vs CPU-fallback' means the
    same thing in both engines."""
    try:
        return bool(_cli._stub_uses_torch_wrapper(stub))
    except Exception:
        return False


def _is_graduated(component: str) -> bool:
    """Graduation per the SHARED contract emit-e2e/promote read: a `.py.last_good_native` snapshot
    exists next to the stub (written on a PCC pass)."""
    return _snap(component, ".py.last_good_native").is_file()


def _components() -> list[str]:
    try:
        return [_component_of(t) for t in _cli._list_component_pcc_tests(_DEMO_DIR)]
    except Exception:
        return []


def _test_file_for(component: str) -> str | None:
    for t in _cli._list_component_pcc_tests(_DEMO_DIR, only=[component]):
        return t
    return None


def _run_pcc(component: str) -> dict:
    """Run ONE component's PCC test on device via the SAME runner the fsm loop uses and scope the
    report to this demo. Returns {ran, passed, failed, skipped, summary, details, skip_reason}."""
    tf = _test_file_for(component)
    if not tf:
        return {
            "ran": False, "passed": False, "failed": False, "skipped": False,
            "summary": f"no pcc test for '{component}'", "details": "", "skip_reason": "",
        }
    _cli._run_focused_pytest(model_id=_MODEL_ID, test_files=[tf], timeout_s=_TIMEOUT)
    report = _cli._scope_report_to_demo(_cli._parse_pytest_report(), _DEMO_DIR)
    skip_reason = ""
    for entry in (report.get("per_skipped") or {}).values():
        if isinstance(entry, dict) and entry.get("component") == component:
            skip_reason = str(entry.get("message") or entry.get("reason") or "")
            break
    return {
        "ran": True,
        "passed": component in (report.get("passed_components") or []),
        "failed": component in (report.get("failed_components") or []),
        "skipped": component in (report.get("skipped_components") or []),
        "summary": str(report.get("summary", "")),
        "details": str(report.get("details", "")),
        "skip_reason": skip_reason,
    }


def _is_harness_skip(reason_text: str) -> bool:
    """A pytest SKIP whose reason matches the harness-bug markers (uncallable submodule, wrong
    synthetic inputs, ModuleList) — the fix is in the TEST, not the stub. Same predicate the fsm
    loop's skip_diagnoser uses."""
    if not reason_text:
        return False
    try:
        from scripts.tt_hw_planner._cli_helpers.skip_diagnoser import is_harness_skip

        return bool(is_harness_skip(reason_text))
    except Exception:
        return False


def _classify_loop(summary: str, details: str) -> str:
    """The loop failure taxonomy (PCC_ONLY / SHAPE / CRASH / ...) that drives the consecutive-same-
    class counter and the PCC_ONLY cap bonus. Same fn the fsm loop uses."""
    try:
        return str(_cli._classify_failure(summary, details))
    except Exception:
        return ""


def _cap_verdict_warrants_decompose(component: str, st: dict) -> bool:
    """At cap: does the failure verdict warrant splitting this component into children? Uses the SAME
    two-step the fsm loop does — `failure_classifier.classify_failure` on the recent failure text,
    then `component_decomposer.failure_class_warrants_decomposition` on the verdict class. Absent the
    classifiers, do NOT decompose (prefer the plain CPU fallback)."""
    text = (st.get("last_failure_text", {}) or {}).get(component, "")
    try:
        from scripts.tt_hw_planner.failure_classifier import classify_failure
        from scripts.tt_hw_planner.component_decomposer import failure_class_warrants_decomposition

        verdict = classify_failure(reason="exhausted per-component attempt cap", failure_text=text)
        return bool(failure_class_warrants_decomposition(verdict.class_name))
    except Exception:
        return False


@mcp.tool()
def list_components() -> dict:
    """The per-component PCC tests this bring-up must resolve (NEW/ADAPT from bringup_status.json), and
    each component's status: graduated / fallen-back / attempts."""
    st = _load_state()
    comps = _components()
    return {
        "components": comps,
        "graduated": [c for c in comps if _is_graduated(c)],
        "fallen_back": [c for c in comps if c in (st.get("fallback") or [])],
        "attempts": {c: (st.get("attempts", {}) or {}).get(c, 0) for c in comps},
    }


@mcp.tool()
def run_component(component: str) -> dict:
    """Run ONE component's PCC test on device via the SAME runner the fsm loop uses; report {ok,
    graduated, summary, failed, skipped, failure_class}.

    Snapshots on FIRST touch mirror the fsm loop exactly:
      * if the stub is the original CPU-delegating torch-wrapper and no `.py.bak` exists, save it as
        `.py.bak` (the fallback baseline);
      * if the stub is ALREADY native and no `.py.preiter_native` exists, save it as
        `.py.preiter_native` so a later cap-out rolls back to the code we came in with, not the stale
        scaffold.

    On failure it DETERMINISTICALLY classifies the trace (`cli._classify_failure`) and persists the
    class + text — this (not any label you pass) drives the cap and the decompose decision.

    If ok is false and the test could not even run (import/collection/torch-reference error in the
    summary), the blocker is in the TEST HARNESS (tests/pcc/conftest.py) — fix that, not the stub.
    PCC>=threshold is enforced inside the test."""
    stub = _stub_path(component)
    if stub.is_file():
        if _is_torch_wrapper(stub):
            bak = _snap(component, ".py.bak")
            if not bak.is_file():
                try:
                    shutil.copy2(stub, bak)
                except OSError:
                    pass
        else:
            preiter = _snap(component, ".py.preiter_native")
            if not preiter.is_file():
                try:
                    shutil.copy2(stub, preiter)
                except OSError:
                    pass
    res = _run_pcc(component)
    st = _load_state()
    cls = ""
    harness_skip = False
    if res["skipped"] and _is_harness_skip(res["skip_reason"]):
        harness_skip = True
        cls = "HARNESS_SKIP"
        st.setdefault("harness_skip_reason", {})[component] = res["skip_reason"][:2000]
        st.setdefault("last_failure_class", {})[component] = cls
        st.setdefault("last_failure_text", {})[component] = res["skip_reason"][:4000]
    else:
        if component in (st.get("harness_skip_reason", {}) or {}):
            st["harness_skip_reason"].pop(component, None)
        if res["failed"] or (res["ran"] and not res["passed"] and not res["skipped"]):
            cls = _classify_loop(res["summary"], res["details"])
            st.setdefault("last_failure_class", {})[component] = cls
            st.setdefault("last_failure_text", {})[component] = (res["summary"] + "\n" + res["details"])[:4000]
    _save_state(st)
    return {
        "ok": bool(res["passed"]),
        "graduated": bool(res["passed"]),
        "summary": (res["skip_reason"] or res["summary"])[:1000],
        "failed": bool(res["failed"]),
        "skipped": bool(res["skipped"]),
        "harness_skip": harness_skip,
        "failure_class": cls,
    }


@mcp.tool()
def record_result(component: str, ok: bool, pcc: float = 0.0, failure_class: str = "") -> dict:
    """Persist the outcome of working `component`: bump attempts, advance the consecutive-same-class
    counter using the DETERMINISTIC class from run_component (your `failure_class` arg is only a
    fallback if run_component didn't classify), track last PCC, and snapshot the best-PCC NATIVE stub
    as `.py.best_native` using the SAME rule the fsm loop applies (`_should_snapshot_best_native`:
    write on no-prior / prior-None / strict improvement; skip if PCC unmeasured; skip torch-wrappers).
    On ok, writes the `.py.last_good_native` graduation snapshot (the shared contract promote/emit-e2e
    read)."""
    st = _load_state()
    st.setdefault("attempts", {})[component] = (st.get("attempts", {}) or {}).get(component, 0) + 1
    stub = _stub_path(component)

    best = _snap(component, ".py.best_native")
    prior = (st.get("best_pcc", {}) or {}).get(component)
    new_pcc = pcc if pcc else None
    if stub.is_file() and not _is_torch_wrapper(stub):
        if _auto._should_snapshot_best_native(snap_exists=best.is_file(), prior_pcc=prior, new_pcc=new_pcc):
            try:
                shutil.copy2(stub, best)
                if new_pcc is not None and (prior is None or new_pcc > prior):
                    st.setdefault("best_pcc", {})[component] = new_pcc
            except OSError:
                pass

    if ok:
        if stub.is_file():
            try:
                shutil.copy2(stub, _snap(component, ".py.last_good_native"))
            except OSError:
                pass
        st.setdefault("consecutive_same_class", {})[component] = 0
    else:
        cls = (st.get("last_failure_class", {}) or {}).get(component) or failure_class or ""
        prev = (st.get("prev_recorded_class", {}) or {}).get(component, "")
        cur = (st.get("consecutive_same_class", {}) or {}).get(component, 0)
        st.setdefault("consecutive_same_class", {})[component] = (cur + 1) if (cls and cls == prev) else 1
        st.setdefault("prev_recorded_class", {})[component] = cls
        st.setdefault("last_failure_class", {})[component] = cls
        st.setdefault("last_pcc", {})[component] = pcc
    _save_state(st)
    return {"recorded": True, "component": component, "graduated": ok}


@mcp.tool()
def restore_best(component: str) -> dict:
    """Revert the stub to its best-PCC snapshot (`.py.best_native`). Call this when an edit REGRESSED a
    component (new PCC lower than a prior attempt) so you don't lose ground; then try a different fix."""
    best = _snap(component, ".py.best_native")
    stub = _stub_path(component)
    if best.is_file():
        try:
            shutil.copy2(best, stub)
            return {"restored": True, "component": component, "from": "best_native"}
        except OSError as exc:  # noqa: BLE001
            return {"restored": False, "error": str(exc)}
    return {"restored": False, "reason": "no .best_native snapshot yet"}


def _mark_fallback(component: str) -> None:
    st = _load_state()
    fb = set(st.get("fallback") or [])
    fb.add(component)
    st["fallback"] = sorted(fb)
    _save_state(st)


def _unmark_fallback(component: str) -> None:
    st = _load_state()
    fb = set(st.get("fallback") or [])
    fb.discard(component)
    st["fallback"] = sorted(fb)
    _save_state(st)


def _do_fallback(component: str) -> dict:
    """The re-testing CASCADE the fsm loop runs at cap (`_skip_component_to_fallback`):
    restore priority `.py.last_good_native > .py.best_native > .py.preiter_native > .py.bak`, re-test
    each NATIVE restore (re-graduate if it passes — never discard a snapshot that works), accept a
    torch-wrapper restore as CPU fallback without re-test, and if nothing survives write the
    synthesized stable CPU wrapper (`cli._rewrite_components_to_stable_fallback`)."""
    stub = _stub_path(component)
    candidates = [
        (_snap(component, ".py.last_good_native"), "last_good_native"),
        (_snap(component, ".py.best_native"), "best_native"),
        (_snap(component, ".py.preiter_native"), "preiter_native"),
        (_snap(component, ".py.bak"), "bak"),
    ]
    cascade: list[str] = []
    for snap_path, label in candidates:
        if not snap_path.is_file():
            continue
        try:
            stub.write_text(snap_path.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            continue
        if _is_torch_wrapper(stub):
            _mark_fallback(component)
            return {"fallback": True, "component": component, "restored_from": label, "cascade": cascade}
        res = _run_pcc(component)
        if res["passed"]:
            try:
                shutil.copy2(stub, _snap(component, ".py.last_good_native"))
            except OSError:
                pass
            _unmark_fallback(component)
            return {"fallback": False, "regraduated": True, "component": component, "restored_from": label, "cascade": cascade}
        cascade.append(f"{label} re-test did not pass")
    try:
        _cli._rewrite_components_to_stable_fallback(_DEMO_DIR, [component])
        _mark_fallback(component)
        return {"fallback": True, "component": component, "restored_from": "stable_cpu_wrapper", "cascade": cascade}
    except Exception as exc:  # noqa: BLE001
        return {"fallback": False, "component": component, "error": str(exc), "cascade": cascade}


@mcp.tool()
def fall_back_to_cpu(component: str) -> dict:
    """Retire a component that exhausted its attempt cap to CPU (mixed execution) — the SAME
    re-testing CASCADE the fsm loop runs. Only the gate should direct you here (rung 'fallback')."""
    return _do_fallback(component)


@mcp.tool()
def mark_harness_skipped(component: str, verdict: str = "manual", reason: str = "") -> dict:
    """Terminal for a component whose PCC test SKIPs for a HARNESS reason you could not fix (needs a
    human-authored test, or the module is genuinely un-unit-testable). Records it to
    `harness_skipped.json` + `skip_diagnosis.json` (the SAME artifacts the fsm loop's skip_diagnoser
    writes, which the OUTCOME banner surfaces) so the run can complete honestly WITHOUT silently
    counting the skip as graduated. Only the gate should direct you here (rung 'mark_manual'). Prefer
    fixing the harness (fix_harness) or decomposing first."""
    st = _load_state()
    hs = set(st.get("harness_skipped") or [])
    hs.add(component)
    st["harness_skipped"] = sorted(hs)
    _save_state(st)
    try:
        hs_path = _DEMO_DIR / "harness_skipped.json"
        existing = {}
        if hs_path.is_file():
            existing = json.loads(hs_path.read_text())
        comps = set(existing.get("harness_skipped_components") or [])
        comps.add(component)
        hs_path.write_text(json.dumps({"harness_skipped_components": sorted(comps)}, indent=2))
    except Exception:
        pass
    try:
        diag_path = _DEMO_DIR / "skip_diagnosis.json"
        doc = {"diagnoses": []}
        if diag_path.is_file():
            doc = json.loads(diag_path.read_text())
        doc.setdefault("diagnoses", []).append(
            {"component": component, "verdict": (verdict or "manual"),
             "summary": (reason or (st.get("harness_skip_reason", {}) or {}).get(component, ""))[:1000]}
        )
        diag_path.write_text(json.dumps(doc, indent=2))
    except Exception:
        pass
    return {"harness_skipped": True, "component": component, "verdict": verdict or "manual"}


@mcp.tool()
def decompose_component(component: str) -> dict:
    """Split a cap-stuck composite whose failure verdict warrants it into children, then retire the
    parent — the SAME cap-time action the fsm loop auto-spawns (`_skip_component_to_fallback` decompose
    branch). Runs `decompose <model> <component> --write-plan`, applies the plan to bringup_status.json
    via the shared consumer (children then appear as new components in list_components/
    termination_check), and cascade-retires the PARENT to CPU (the fsm loop retires the parent at cap
    regardless). Only the gate should direct you here (rung 'decompose'). rc: children_added>0 =
    decomposed; 0 = primitive/leaf (parent still retired)."""
    st = _load_state()
    decomposed = set(st.get("decomposed") or [])
    decomposed.add(component)
    st["decomposed"] = sorted(decomposed)
    _save_state(st)

    children_added = 0
    notes: list = []
    reason = ""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "scripts.tt_hw_planner", "decompose", _MODEL_ID, component, "--write-plan"],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(_REPO),
        )
        if proc.returncode == 1:
            reason = "primitive/leaf module — no non-trivial children to spawn"
        elif proc.returncode != 0:
            reason = (proc.stderr or "")[:300]
        else:
            from scripts.tt_hw_planner.decomposition_consumer import consume_decomposition_plan

            children_added, notes = consume_decomposition_plan(model_id=_MODEL_ID, demo_dir=_DEMO_DIR)
    except subprocess.TimeoutExpired:
        reason = "decompose subprocess timed out (HF load too slow)"
    except Exception as exc:  # noqa: BLE001
        reason = f"{type(exc).__name__}: {exc}"

    parent = _do_fallback(component)
    return {
        "decomposed": children_added > 0,
        "children_added": children_added,
        "notes": notes[:8],
        "reason": reason,
        "parent_retired": bool(parent.get("fallback")),
    }


@mcp.tool()
def termination_check() -> dict:
    """THE deterministic stop gate for bring-up. can_stop is true ONLY when every material component is
    graduated OR fallen back to CPU. Otherwise returns next_target = the next component + rung. Rung
    ladder (model-agnostic, mirrors the fsm loop): emit (never tried) -> repair (pre-cap); AT cap, if
    the failure verdict warrants it and it hasn't been decomposed -> decompose, else -> fallback. Uses
    the extracted bringup_ladder cap rule. Agent may NOT declare done."""
    st = _load_state()
    comps = _components()
    if not comps:
        return {"can_stop": True, "halt": False, "next_target": None, "reason": "no components to bring up"}
    decomposed = set(st.get("decomposed") or [])
    last_class_map = st.get("last_failure_class", {}) or {}
    harness_skip_reasons = st.get("harness_skip_reason", {}) or {}
    terminal = set(st.get("fallback") or []) | set(st.get("harness_skipped") or [])
    work, needs_cap = [], []
    for c in comps:
        if _is_graduated(c) or c in terminal:
            continue
        eff = bringup_ladder.effective_attempt_cap(
            c,
            max_attempts_per_component=_MAX_ATTEMPTS,
            hard_total_attempt_cap=_HARD_CAP,
            complexity_bonus=0,
            last_failure_class=last_class_map,
            last_pcc=st.get("last_pcc", {}) or {},
        )
        at_cap = bringup_ladder.is_at_cap(
            c,
            attempts_per_component=st.get("attempts", {}) or {},
            consecutive_same_class_attempts=st.get("consecutive_same_class", {}) or {},
            effective_cap=eff,
            hard_total_attempt_cap=_HARD_CAP,
        )
        (needs_cap if at_cap else work).append(c)
    can_stop = not work and not needs_cap
    nxt = None
    if work:
        c = work[0]
        attempts = (st.get("attempts", {}) or {}).get(c, 0)
        last_class = last_class_map.get(c, "")
        if c in harness_skip_reasons:
            nxt = {
                "unit": c,
                "rung": "fix_harness",
                "reason": f"component '{c}' PCC test SKIPPED (harness bug, not a stub bug): "
                f"{harness_skip_reasons[c][:300]}. Fix the TEST HARNESS, NOT the stub — in "
                f"tests/pcc/test_{c}.py adjust the submodule path (_CANDIDATE_SUBMODULE_PATHS, e.g. add "
                f"[0] for a ModuleList) and/or the synthetic-input builder (_make_arg_for / sample "
                f"kwargs) to match this module's real forward signature and shapes (see _captured/{c}/ "
                f"for real shapes); or fix tests/pcc/conftest.py. Re-run run_component. If the module is "
                f"genuinely uncallable as one unit, decompose_component('{c}'); if it needs a "
                f"human-authored test, mark_harness_skipped('{c}', 'manual').",
            }
        else:
            rung = "emit" if attempts == 0 else "repair"
            nxt = {
                "unit": c,
                "rung": rung,
                "reason": f"component '{c}' not graduated (attempts={attempts}, last_class={last_class or 'none'}). "
                f"run_component to see the failure; if it cannot even run, fix tests/pcc/conftest.py; else "
                f"edit _stubs/{c}.py to native ttnn; re-run; record_result. PCC>={_PCC} graduates it.",
            }
    elif needs_cap:
        c = needs_cap[0]
        if c not in decomposed and _cap_verdict_warrants_decompose(c, st):
            nxt = {
                "unit": c,
                "rung": "decompose",
                "reason": f"component '{c}' exhausted its attempt cap and its failure verdict is "
                f"decomposable — call decompose_component('{c}') to split it into children (they become "
                f"new components) and retire the parent to CPU.",
            }
        elif c in harness_skip_reasons:
            nxt = {
                "unit": c,
                "rung": "mark_manual",
                "reason": f"component '{c}' exhausted its attempt cap still HARNESS-SKIPPING and could "
                f"not be split — call mark_harness_skipped('{c}', 'manual') so the run completes "
                f"honestly (surfaced in harness_skipped.json, NOT counted as graduated).",
            }
        else:
            nxt = {
                "unit": c,
                "rung": "fallback",
                "reason": f"component '{c}' exhausted its attempt cap — call fall_back_to_cpu('{c}') to "
                f"retire it to CPU (mixed execution) so the pipeline still works.",
            }
    return {
        "can_stop": can_stop,
        "halt": False,
        "halt_reason": None,
        "graduated": sorted([c for c in comps if _is_graduated(c)]),
        "fallen_back": sorted(st.get("fallback") or []),
        "harness_skipped": sorted(st.get("harness_skipped") or []),
        "next_target": nxt,
    }


if __name__ == "__main__":
    mcp.run()
