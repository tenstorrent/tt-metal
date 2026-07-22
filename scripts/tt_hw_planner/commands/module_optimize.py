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


_OPT_STATE_NAME = ".module_optimize_state.json"


def _opt_state_path(demo_dir):
    return Path(demo_dir) / _OPT_STATE_NAME


def _load_optimized(demo_dir) -> dict:
    try:
        data = json.loads(_opt_state_path(demo_dir).read_text())
        return data.get("optimized", {}) or {}
    except Exception:
        return {}


def _mark_optimized(demo_dir, module, status, result) -> None:
    p = _opt_state_path(demo_dir)
    try:
        data = json.loads(p.read_text()) if p.is_file() else {}
    except Exception:
        data = {}
    opt = data.get("optimized") or {}
    rounds = None
    if result:
        res = result.get("results") or []
        if res:
            rounds = ",".join(str(r.get("rounds", "?")) for r in res)
    opt[module] = {"status": status, "rounds": rounds}
    data["optimized"] = opt
    try:
        p.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


def _module_status(result) -> str:
    if not result:
        return "failed"
    res = result.get("results") or []
    if res and all(r.get("can_stop") for r in res):
        return "converged"
    return "ran"


def _clear_profile_cache() -> None:
    """Wipe the on-disk profiling cache before a module-level run.

    The cache is keyed per module + perf-test, but entries persist in /tmp across
    runs; clearing at the start of a fresh module-level optimize guarantees no module
    inherits a stale profile captured for a different module in an earlier run."""
    import shutil
    import tempfile

    cache = Path(tempfile.gettempdir()) / "perf_mcp_profile_cache"
    if cache.is_dir():
        shutil.rmtree(cache, ignore_errors=True)


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
    all_grad = list(mods)
    total = len(all_grad)

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

    reverify = bool(getattr(args, "reverify", False))
    if reverify:
        import tempfile

        for _m in mods:
            try:
                (
                    Path(tempfile.gettempdir())
                    / ("perf_mcp_orig_baseline_%s_%s.json" % (Path(demo_dir).name, _m))
                ).unlink()
            except OSError:
                pass
    if not reverify:
        done = _load_optimized(demo_dir)
        skip = [m for m in mods if m in done]
        if skip:
            print(
                "  [optimize/module] skipping %d already-optimized module(s) (pass --reverify to redo): %s"
                % (len(skip), skip)
            )
        mods = [m for m in mods if m not in done]
    if not mods:
        print("  [optimize/module] all target modules already optimized (pass --reverify to redo).")
        _write_rollup(demo_dir, [], upsert_report_section)
        return 0

    os.environ["TT_PERF_MODULE_LEVEL"] = "1"
    os.environ.setdefault("PERF_MCP_VALIDATE_STALL_SEC", "120")
    _clear_profile_cache()
    print("  [optimize/module] module-level optimize over %d graduated module(s): %s" % (len(mods), mods))
    rows = []
    for m in mods:
        pos = (all_grad.index(m) + 1) if m in all_grad else 0
        idx = "%d/%d" % (pos, total)
        node = pcc_test_node(demo_dir, m, repo_root)
        if node is None:
            print("  [optimize/module] %s %s: no PCC test node — skipped" % (idx, m))
            rows.append((m, "no-pcc-test", None))
            continue
        print("\n  [optimize/module] === %s module: %s (pcc %s) ===" % (idx, m, node))
        os.environ["PERF_MCP_REPORT_KEY"] = "module:%s" % m
        os.environ["PERF_MCP_REPORT_MODULE"] = m
        os.environ["PERF_MCP_REPORT_PCC"] = node
        os.environ["PERF_MCP_REPORT_INDEX"] = idx
        os.environ["PERF_MCP_TASK"] = m
        try:
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
        finally:
            for _k in (
                "PERF_MCP_REPORT_KEY",
                "PERF_MCP_REPORT_MODULE",
                "PERF_MCP_REPORT_PCC",
                "PERF_MCP_REPORT_INDEX",
                "PERF_MCP_TASK",
            ):
                os.environ.pop(_k, None)
        status = _module_status(result)
        rows.append((m, status, result))
        if status in ("converged", "ran"):
            _mark_optimized(demo_dir, m, status, result)
        _rekey_module_section(demo_dir, m, node, status, upsert_report_section, index=idx)
        _reorder_module_sections(demo_dir)
        _pin_bringup_top(demo_dir)

    _write_rollup(demo_dir, rows, upsert_report_section)
    _reorder_module_sections(demo_dir)
    _pin_bringup_top(demo_dir)

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


def _rekey_module_section(demo_dir, module, node, status, upsert, index=""):
    """Finalize the module's section. The live optimize detail is written in-place
    into ``module:<module>`` during the run (so it renders in the module's own
    section, labelled, correctly positioned). Here we just flip the transient
    ``optimizing…`` outcome to the final status. When the run recorded no detail
    (0 attempts), re-seed the placeholder so the section still names the module."""
    section = _read_section(demo_dir, "module:%s" % module)
    if section and ("Optimization summary" in section or "# Optimize (perf)" in section):
        section = re.sub(r"- outcome: \*\*[^*]*\*\*", "- outcome: **%s**" % status, section, count=1)
        upsert(demo_dir, "module:%s" % module, section)
        return
    idx = (" — %s" % index) if index else ""
    block = "## Module: `%s`%s\n\n- pcc gate: `%s`\n- outcome: **%s**\n\n%s" % (
        module,
        idx,
        node,
        status,
        "(no optimize detail recorded)",
    )
    upsert(demo_dir, "module:%s" % module, block)


def _reorder_module_sections(demo_dir) -> None:
    """Sort the ``module:<name>`` blocks in RUN_REPORT.md by their ``— N/total`` index,
    so the report always reads 1,2,3,… regardless of write-order. Non-module sections
    (bring-up, rollup) keep their positions; blocks are only rewritten into their existing
    slots in sorted order, so no content is added or dropped. No-op if already sorted or
    fewer than two module blocks. Best-effort: never raises."""
    p = Path(demo_dir) / _REPORT_NAME
    try:
        txt = p.read_text()
    except Exception:
        return
    pat = re.compile(r"<!-- BEGIN (module:[A-Za-z0-9_]+) -->.*?<!-- END \1 -->", re.S)
    matches = list(pat.finditer(txt))
    if len(matches) < 2:
        return

    def _idx(b):
        mm = re.search(r"## Module: `[^`]+` — (\d+)/", b)
        return int(mm.group(1)) if mm else 9999

    blocks = [m.group(0) for m in matches]
    order = [_idx(b) for b in blocks]
    if order == sorted(order):
        return
    sorted_blocks = [b for _, b in sorted(enumerate(blocks), key=lambda kv: (_idx(kv[1]), kv[0]))]
    out, last = [], 0
    for m, sb in zip(matches, sorted_blocks):
        out.append(txt[last : m.start()])
        out.append(sb)
        last = m.end()
    out.append(txt[last:])
    try:
        p.write_text("".join(out))
    except Exception:
        pass


def _pin_bringup_top(demo_dir) -> None:
    """Hoist the ``bringup`` section above every ``module:*`` optimize block so the report always
    reads bring-up first, then per-module optimize — regardless of write order. A fresh optimize on a
    deleted report writes module 1 before the bring-up section is re-added, which otherwise leaves
    bring-up wedged in the middle. Best-effort; no-op if bring-up is absent or already at the top.
    Never raises."""
    p = Path(demo_dir) / _REPORT_NAME
    try:
        txt = p.read_text()
    except Exception:
        return
    bm = re.search(r"<!-- BEGIN bringup -->.*?<!-- END bringup -->", txt, re.S)
    if bm is None or txt[: bm.start()].strip() == "":
        return
    block = bm.group(0).strip()
    rest = (txt[: bm.start()] + txt[bm.end() :]).strip()
    try:
        p.write_text(block + "\n\n" + rest + "\n")
    except Exception:
        pass


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
