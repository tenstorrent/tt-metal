from __future__ import annotations

import json
import os
import re
from pathlib import Path

_REPORT_NAME = "RUN_REPORT.md"


def enumerate_graduated(demo_dir) -> list:
    """Return bring-up components that graduated to standalone native ttnn.

    A component qualifies when its recorded best PCC is >= 0.99 and it is not on
    the torch-fallback list. Decomposed / fallback components are excluded because
    their stubs are torch references, so there is no ttnn implementation to
    optimize. Reads the generic bring-up state, so it is model-agnostic."""
    state = Path(demo_dir) / ".bringup_cc_state.json"
    try:
        d = json.loads(state.read_text())
    except Exception:
        return []
    best = d.get("best_pcc", {}) or {}
    fallback = set(d.get("fallback", []) or [])
    grad = [m for m, p in best.items() if isinstance(p, (int, float)) and p >= 0.99 and m not in fallback]
    return sorted(grad)


def pcc_test_node(demo_dir, module: str, repo_root=None):
    """Resolve the per-component PCC test node id for `module`. The engine resolves
    ``--pcc-test`` relative to the repo root, so return a repo-root-relative path
    (`models/demos/<demo>/tests/pcc/test_<module>.py::<first test_ fn>`) when
    `repo_root` is given, else the demo-relative path. None if the file or a test
    function is missing."""
    rel = Path("tests") / "pcc" / ("test_%s.py" % module)
    path = Path(demo_dir) / rel
    try:
        src = path.read_text()
    except Exception:
        return None
    m = re.search(r"^def (test_[A-Za-z0-9_]+)\s*\(", src, re.MULTILINE)
    if not m:
        return None
    node_path = rel
    if repo_root is not None:
        try:
            node_path = Path(path).resolve().relative_to(Path(repo_root).resolve())
        except Exception:
            node_path = rel
    return "%s::%s" % (node_path.as_posix(), m.group(1))


def _read_section(demo_dir, key: str):
    p = Path(demo_dir) / _REPORT_NAME
    try:
        txt = p.read_text()
    except Exception:
        return None
    begin, end = "<!-- BEGIN %s -->" % key, "<!-- END %s -->" % key
    i, j = txt.find(begin), txt.find(end)
    if i < 0 or j < 0 or j < i:
        return None
    return txt[i + len(begin) : j].strip()


def _module_status(result) -> str:
    if not result:
        return "failed"
    res = result.get("results") or []
    if res and all(r.get("can_stop") for r in res):
        return "converged"
    return "ran"


def run_module_level_optimize(args, demo_dir, repo_root, run_cc) -> int:
    """Optimize graduated modules one at a time by reusing the cc engine per module.

    Each module is optimized against its own per-component PCC test (the engine
    auto-generates a module-scoped perf test from it), so the heavy full-pipeline
    baseline is never entered. After each module the engine's ``optimize`` report
    block is re-keyed to a per-module section, and a roll-up table is written. With
    ``--then-e2e`` a single full-pipeline confirmation pass runs at the end."""
    from ..run_report import upsert_report_section

    demo_dir = Path(demo_dir)
    mods = enumerate_graduated(demo_dir)
    if not mods:
        print("  [optimize/module] no graduated native modules found (need .bringup_cc_state.json).")
        return 1

    want = getattr(args, "modules", None)
    if want:
        wset = {w.strip() for w in str(want).split(",") if w.strip()}
        missing = sorted(wset - set(mods))
        if missing:
            print("  [optimize/module] skipping %s (not graduated / not found)" % missing)
        mods = [m for m in mods if m in wset]
    if not mods:
        print("  [optimize/module] nothing to optimize after filtering.")
        return 1

    os.environ["TT_PERF_MODULE_LEVEL"] = "1"
    print("  [optimize/module] module-level optimize over %d graduated module(s): %s" % (len(mods), mods))
    rows = []
    for i, m in enumerate(mods, 1):
        node = pcc_test_node(demo_dir, m, repo_root)
        if node is None:
            print("  [optimize/module] %d/%d %s: no PCC test node — skipped" % (i, len(mods), m))
            rows.append((m, "no-pcc-test", None))
            continue
        print("\n  [optimize/module] === %d/%d module: %s (pcc %s) ===" % (i, len(mods), m, node))
        result = run_cc(
            demo_dir,
            repo_root,
            devices=args.devices,
            metric=args.metric,
            pcc_test=node,
            max_rounds=getattr(args, "max_rounds", 3),
            model_id_hint=getattr(args, "target", None),
            hitl=getattr(args, "hitl", False),
        )
        status = _module_status(result)
        rows.append((m, status, result))
        _rekey_module_section(demo_dir, m, node, status, upsert_report_section)

    _write_rollup(demo_dir, rows, upsert_report_section)

    if getattr(args, "then_e2e", False):
        print("\n  [optimize/module] --then-e2e: confirming module wins survive the full pipeline")
        run_cc(
            demo_dir,
            repo_root,
            devices=args.devices,
            metric=args.metric,
            e2e_only=True,
            max_rounds=getattr(args, "max_rounds", 3),
            model_id_hint=getattr(args, "target", None),
        )
    return 0


def _rekey_module_section(demo_dir, module, node, status, upsert):
    detail = _read_section(demo_dir, "optimize") or "(no optimize detail recorded)"
    block = "## Module: `%s`\n\n- pcc gate: `%s`\n- outcome: **%s**\n\n%s" % (module, node, status, detail)
    upsert(demo_dir, "module:%s" % module, block)
    upsert(demo_dir, "optimize", "")


def _write_rollup(demo_dir, rows, upsert):
    lines = ["## Module-level optimize — summary", "", "| module | outcome | rounds | can_stop |", "|---|---|---|---|"]
    for m, status, result in rows:
        rounds = can = "-"
        if result:
            res = result.get("results") or []
            if res:
                rounds = ",".join(str(r.get("rounds", "?")) for r in res)
                can = ",".join(str(bool(r.get("can_stop"))) for r in res)
        lines.append("| %s | %s | %s | %s |" % (m, status, rounds, can))
    upsert(demo_dir, "module-summary", "\n".join(lines))
