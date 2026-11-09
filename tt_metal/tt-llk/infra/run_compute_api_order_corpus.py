#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run the compute-kernel API ordering checker over a corpus and render a report.

Two input modes:

  * ``--compile-db`` / ``--log``: replay real JIT compile commands (the recommended
    corpus run; coverage is exactly the kernels the workload compiled).
  * ``--discover``: scan ``ttnn/ models/ tests/ tt-train/`` for compute kernels and
    synthesize JIT-like flags for each.  This is a wide but approximate sweep: a
    kernel that needs host-generated defines the scan cannot know about is
    reported as a coverage hole, never silently treated as clean.

The report is Markdown with per-rule counts and a triage column.  ``--triage``
takes a YAML mapping from finding fingerprint to ``{true_positive, note}`` and
computes the false-positive rate over reviewed findings.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import importlib.util
import json
import multiprocessing as mp
import os
import re
import signal
import sys
import time
from multiprocessing.connection import wait
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CHECKER = HERE / "check_compute_api_order.py"

DISCOVER_ROOTS = ("ttnn", "models", "tests", "tt-train", "tt_metal")
KERNEL_PATH_RE = re.compile(
    r"/(?:kernels|kernels_ng|kernels_dfb|unified_kernels|[^/]*_impl_kernels|kernels_qsr|test_kernels)/"
)
COMPUTE_RE = re.compile(r"api/compute/|compute_kernel_hw_startup|compute_kernel_api")

_CHECKER_MODULE = None
_TABLE_CACHE = {}
_SFPI_CACHE = None


def _checker():
    global _CHECKER_MODULE
    if _CHECKER_MODULE is None:
        spec = importlib.util.spec_from_file_location("compute_api_order", CHECKER)
        module = importlib.util.module_from_spec(spec)
        sys.modules["compute_api_order"] = module
        spec.loader.exec_module(module)
        module._configure_libclang(None)
        _CHECKER_MODULE = module
    return _CHECKER_MODULE


def discover(roots) -> list[Path]:
    files = []
    for root in roots:
        base = REPO / root
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.suffix not in (".cpp", ".cc", ".cxx"):
                continue
            p = str(path).replace(os.sep, "/")
            if not KERNEL_PATH_RE.search(p + "/"):
                continue
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            if not COMPUTE_RE.search(text):
                continue
            if "kernel_main" not in text and "TT_KERNEL" not in text and "KERNEL(" not in text:
                continue
            files.append(path)
    return sorted(set(files))


def _fingerprint(finding) -> str:
    return f"{finding.arch}|{finding.thread}|{finding.rule}|{finding.file}:{finding.line}|{finding.message}"


def _triage_entry(finding, triage):
    """Exact fingerprint, then rule|file:line, then a rule-wide default."""
    keys = [
        _fingerprint(finding),
        f"{finding.rule}|{finding.file}:{finding.line}",
    ]
    try:
        relative = Path(finding.file).resolve().relative_to(REPO)
        keys.append(f"{finding.rule}|{relative}:{finding.line}")
    except ValueError:
        pass
    keys.append(finding.rule)
    for key in keys:
        entry = triage.get(key)
        if entry is not None:
            return entry
    return {}


def _table(arch: str):
    if arch not in _TABLE_CACHE:
        _TABLE_CACHE[arch] = _checker().RuleTable.load(arch)
    return _TABLE_CACHE[arch]


def _sfpi_root():
    global _SFPI_CACHE
    if _SFPI_CACHE is None:
        _SFPI_CACHE = _checker()._find_sfpi(os.environ.get("TT_METAL_SFPI_ROOT"))
    return _SFPI_CACHE


def _run_one(task, sfpi_root=None, max_paths=24):
    arch, thread, src = task
    cc = _checker()
    sfpi = Path(sfpi_root) if sfpi_root else _sfpi_root()
    if sfpi is None:
        return src, arch, thread, [], {}, "SFPI headers not found"
    args = cc.synth_args(src, arch, thread, sfpi)
    try:
        analyzer = cc.Analyzer(args, thread, arch, _table(arch), src)
        analyzer.analyze_paths(max_paths=max_paths)
    except Exception as exc:  # noqa: BLE001
        return src, arch, thread, [], {}, f"{type(exc).__name__}: {exc}"
    return src, arch, thread, [f.as_json() for f in analyzer.findings], {
        "entry": analyzer._entry_name,
        "parse_errors": len(analyzer.parse_errors),
    }, None


def _finding_from_json(d: dict):
    cc = _checker()
    return cc.Finding(**d)


def _alarm_handler(signum, frame):
    raise TimeoutError("unit exceeded the per-unit timeout")


def _worker_main(conn, sfpi_root, unit_timeout, max_tasks, max_paths):
    """Reuse one libclang environment for many units; recycle before memory grows."""
    signal.signal(signal.SIGALRM, _alarm_handler)
    done = 0
    try:
        while True:
            try:
                task = conn.recv()
            except EOFError:
                break
            if task is None:
                break
            if unit_timeout:
                signal.setitimer(signal.ITIMER_REAL, unit_timeout)
            try:
                result = _run_one(task, sfpi_root, max_paths)
            except Exception as exc:  # noqa: BLE001
                result = (task[2], task[0], task[1], [], {}, f"{type(exc).__name__}: {exc}")
            finally:
                if unit_timeout:
                    signal.setitimer(signal.ITIMER_REAL, 0)
            done += 1
            recycle = done >= max_tasks
            try:
                conn.send((*result, recycle))
            except (BrokenPipeError, EOFError, OSError):
                break
            if recycle:
                break
    finally:
        try:
            conn.close()
        except OSError:
            pass


def _run_pool(tasks, jobs, sfpi_root, unit_timeout, max_tasks, max_paths, start_method="fork"):
    """Supervised process pool: a crashed worker is a coverage hole, not a lost run."""
    ctx = mp.get_context(start_method)
    pending = collections.deque(tasks)
    results = []
    holes = []
    workers = {}
    next_wid = 0

    def spawn():
        nonlocal next_wid
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(
            target=_worker_main,
            args=(child_conn, sfpi_root, unit_timeout, max_tasks, max_paths),
            daemon=True,
        )
        proc.start()
        child_conn.close()
        wid = next_wid
        next_wid += 1
        workers[wid] = {"conn": parent_conn, "proc": proc, "task": None}
        return wid

    def retire(wid):
        worker = workers.pop(wid, None)
        if worker is None:
            return
        try:
            worker["conn"].close()
        except OSError:
            pass
        if worker["proc"].is_alive():
            worker["proc"].join(timeout=2)
        if worker["proc"].is_alive():
            worker["proc"].terminate()
            worker["proc"].join(timeout=2)

    def assign(wid):
        worker = workers.get(wid)
        if worker is None:
            return
        if pending:
            task = pending.popleft()
            worker["task"] = task
            try:
                worker["conn"].send(task)
            except (BrokenPipeError, EOFError, OSError):
                holes.append(
                    {
                        "file": task[2],
                        "arch": task[0],
                        "thread": task[1],
                        "error": "worker pipe closed before the task started",
                    }
                )
                worker["task"] = None
                retire(wid)
                if pending:
                    assign(spawn())
        else:
            worker["task"] = None
            try:
                worker["conn"].send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass

    def crash(wid, reason):
        worker = workers.get(wid)
        if worker is not None and worker["task"] is not None:
            task = worker["task"]
            holes.append({"file": task[2], "arch": task[0], "thread": task[1], "error": reason})
        retire(wid)
        if pending:
            assign(spawn())

    for _ in range(min(jobs, len(pending))):
        assign(spawn())

    try:
        while workers:
            conns = {worker["conn"]: wid for wid, worker in workers.items()}
            ready = wait(list(conns), timeout=2.0)
            if not ready:
                for wid in list(workers):
                    if not workers[wid]["proc"].is_alive():
                        crash(wid, f"worker exited with code {workers[wid]['proc'].exitcode}")
                continue
            for conn in ready:
                wid = conns[conn]
                if wid not in workers:
                    continue
                try:
                    item = conn.recv()
                except (EOFError, OSError):
                    crash(wid, f"worker exited with code {workers[wid]['proc'].exitcode}")
                    continue
                path, arch, thread, fjson, meta, error, recycle = item
                results.append((path, arch, thread, fjson, meta, error))
                if len(results) % 100 == 0:
                    print(f"  {len(results)}/{len(tasks)} units", file=sys.stderr)
                workers[wid]["task"] = None
                if recycle:
                    retire(wid)
                    if pending:
                        assign(spawn())
                else:
                    assign(wid)
            while pending and len(workers) < jobs:
                assign(spawn())
    finally:
        for wid in list(workers):
            retire(wid)
    return results, holes


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument("--from-json", help="render an existing findings JSON")
    src.add_argument("--compile-db")
    src.add_argument("--log")
    src.add_argument("--discover", action="store_true")
    parser.add_argument("--roots", nargs="+", default=list(DISCOVER_ROOTS))
    parser.add_argument("--arch", action="append", choices=("wormhole_b0", "blackhole"))
    parser.add_argument("--threads", nargs="+", default=["UNPACK", "MATH", "PACK"])
    parser.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 4) // 2)))
    parser.add_argument("--worker-max-tasks", type=int, default=50, help="recycle each worker after N units")
    parser.add_argument("--unit-timeout", type=float, default=600.0, help="per-unit timeout in seconds (0 disables)")
    parser.add_argument("--max-paths", type=int, default=8, help="correlated condition paths explored per unit")
    parser.add_argument("--start-method", choices=("fork", "spawn"), default="fork" if sys.platform.startswith("linux") else "spawn")
    parser.add_argument("--limit", type=int, help="analyse at most N kernel files (per arch)")
    parser.add_argument("--output", default="tt_metal/tt-llk/infra/compute_api_order_corpus.md")
    parser.add_argument("--json-output", help="write the raw findings JSON here")
    parser.add_argument("--triage", help="YAML fingerprint -> {true_positive, note}")
    parser.add_argument("--runtime-seconds", type=float, help="analysis wall time for the report")
    parser.add_argument("--sfpi", help="SFPI root")
    args = parser.parse_args(argv)

    if args.sfpi:
        os.environ["TT_METAL_SFPI_ROOT"] = args.sfpi

    arches = args.arch or ["wormhole_b0", "blackhole"]
    started = dt.datetime.now(dt.timezone.utc)

    findings = []
    holes = []
    units = []

    if args.from_json:
        payload = json.loads(Path(args.from_json).read_text())
        findings = [_finding_from_json(d) for d in payload.get("findings", [])]
        holes = payload.get("holes", [])
        units = payload.get("units", [])
        args.kernel_files = payload.get("kernel_files") or len({u.get("file") for u in units})
    elif args.discover:
        files = discover(args.roots)
        if args.limit:
            files = files[: args.limit]
        args.kernel_files = len(files)
        tasks = [(arch, thread, str(f)) for arch in arches for f in files for thread in args.threads]
    else:
        cc = _checker()
        entries = (
            cc.entries_from_compile_db(Path(args.compile_db))
            if args.compile_db
            else cc.entries_from_log(Path(args.log))
        )
        tasks = []
        args.kernel_files = len({cc._entry_source(e) for e in entries})
        for entry in entries:
            arch, thread = cc.detect_from_args(entry.get("arguments", []), entry.get("file", ""))
            if arch is None:
                for a in arches:
                    tasks.append((a, thread or "MATH", cc._entry_source(entry)))
            else:
                tasks.append((arch, thread or "MATH", cc._entry_source(entry)))

    if args.from_json:
        pass
    elif args.jobs > 1:
        results, pool_holes = _run_pool(
            tasks,
            args.jobs,
            args.sfpi,
            args.unit_timeout,
            args.worker_max_tasks,
            args.max_paths,
            args.start_method,
        )
        holes.extend(pool_holes)
        finding_dicts = []
        for path, arch, thread, fjson, meta, error in results:
            if error:
                holes.append({"file": path, "arch": arch, "thread": thread, "error": error})
            else:
                units.append(
                    {
                        "file": path,
                        "arch": arch,
                        "thread": thread,
                        "entry": meta.get("entry"),
                        "parse_errors": meta.get("parse_errors", 0),
                        "findings": len(fjson),
                    }
                )
                finding_dicts.extend(fjson)
        findings = [_finding_from_json(d) for d in finding_dicts]
    else:
        finding_dicts = []
        for i, task in enumerate(tasks, 1):
            path, arch, thread, fjson, meta, error = _run_one(task, args.sfpi, args.max_paths)
            if error:
                holes.append({"file": path, "arch": arch, "thread": thread, "error": error})
            else:
                units.append(
                    {
                        "file": path,
                        "arch": arch,
                        "thread": thread,
                        "entry": meta.get("entry"),
                        "parse_errors": meta.get("parse_errors", 0),
                        "findings": len(fjson),
                    }
                )
                finding_dicts.extend(fjson)
            if i % 100 == 0:
                print(f"  {i}/{len(tasks)} units", file=sys.stderr)
        findings = [_finding_from_json(d) for d in finding_dicts]

    if not (args.compile_db or args.log or args.discover or args.from_json):
        parser.error("one of --from-json, --compile-db, --log or --discover is required")

    findings.sort(key=lambda f: (f.file, f.line, f.thread, f.rule))
    runtime_seconds = args.runtime_seconds
    if runtime_seconds is None:
        runtime_seconds = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
    triage = {}
    if args.triage:
        triage = yaml.safe_load(Path(args.triage).read_text()) or {}

    report = render_report(arches, args, started, findings, holes, units, triage, runtime_seconds)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    if args.json_output:
        Path(args.json_output).write_text(
            json.dumps(
                {
                    "kernel_files": getattr(args, "kernel_files", None),
                    "findings": [f.as_json() for f in findings],
                    "holes": holes,
                    "units": units,
                },
                indent=2,
            )
        )
    print(f"wrote {out} ({len(findings)} findings, {len(holes)} holes, {len(units)} units)")
    return 0


def _count_triage(findings, triage):
    counts = collections.Counter()
    reviewed = 0
    false_positives = 0
    for f in findings:
        entry = _triage_entry(f, triage)
        verdict = entry.get("true_positive")
        if verdict is True:
            counts["true_positive"] += 1
            reviewed += 1
        elif verdict is False:
            counts["false_positive"] += 1
            reviewed += 1
            false_positives += 1
        else:
            counts["unreviewed"] += 1
    return counts, reviewed, false_positives


def _rule_totals(findings, triage, rule) -> tuple[int, int, int, int]:
    """Return (findings, true positives, false positives, unreviewed) for a rule."""
    entries = [f for f in findings if f.rule == rule]
    tp = sum(1 for f in entries if _triage_entry(f, triage).get("true_positive") is True)
    fp = sum(1 for f in entries if _triage_entry(f, triage).get("true_positive") is False)
    return len(entries), tp, fp, len(entries) - tp - fp


def render_report(arches, args, started, findings, holes, units, triage, runtime_seconds=None) -> str:
    analyzed_files = {u["file"] for u in units} - {h["file"] for h in holes}
    error_findings = [f for f in findings if f.severity == "error"]
    warning_findings = [f for f in findings if f.severity != "error"]
    _ec, error_reviewed, error_fps = _count_triage(error_findings, triage)
    error_by_rule = collections.Counter(f.rule for f in error_findings)
    warning_by_rule = collections.Counter(f.rule for f in warning_findings)
    error_tp = sum(1 for f in error_findings if _triage_entry(f, triage).get("true_positive") is True)
    error_fp = sum(1 for f in error_findings if _triage_entry(f, triage).get("true_positive") is False)
    error_unreviewed = len(error_findings) - error_tp - error_fp
    lines = [
        "# Compute-kernel API ordering corpus report",
        "",
        f"- Generated: {started.isoformat(timespec='seconds')}",
        f"- Architectures: {', '.join(arches)}",
        f"- Threads: {', '.join(args.threads)}",
        (
            f"- Input: `{args.from_json or args.compile_db or args.log}`"
            if not args.discover
            else f"- Input: discovered kernels under {', '.join(args.roots)}"
        ),
        (
            f"- Kernel files discovered: {getattr(args, 'kernel_files', '?')}"
            if getattr(args, "kernel_files", None) is not None
            else "- Kernel files discovered: not recorded"
        ),
        (
            f"- Kernel files analysed: {len(analyzed_files)}/"
            f"{getattr(args, 'kernel_files', '?')} "
            f"({100.0 * len(analyzed_files) / args.kernel_files:.1f}%)"
            if getattr(args, "kernel_files", None)
            else f"- Kernel files analysed: {len(analyzed_files)}"
        ),
        f"- Units analysed: {len(units)}",
        f"- Units that could not be analysed: {len(holes)}",
        f"- Error findings: {len(error_findings)}",
        f"- Advisory warnings: {len(warning_findings)}",
        f"- Findings: {len(findings)}",
        (f"- Analysis wall time: {runtime_seconds:.1f}s" if runtime_seconds else "- Analysis wall time: not recorded for this replay"),
        "",
        "The checker is advisory. A finding is not a verdict: review the source and",
        "record the verdict in the triage file. Unreviewed findings are not counted as",
        "false positives. Advisory warnings are listed separately and are excluded from",
        "the error totals.",
        "",
        "## Error findings by rule",
        "",
        "| Rule | Findings | True positives | False positives | Unreviewed |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for rule, _n in sorted(error_by_rule.items()):
        n, tp, fp, unreviewed = _rule_totals(error_findings, triage, rule)
        lines.append(f"| `{rule}` | {n} | {tp} | {fp} | {unreviewed} |")
    lines.append(
        f"| **Total** | **{len(error_findings)}** | **{error_tp}** | **{error_fp}** | "
        f"**{error_unreviewed}** |"
    )
    lines += [
        "",
        (
            f"Error findings: reviewed {error_reviewed}; false positives {error_fps}; "
            f"false-positive rate {error_fps / error_reviewed:.1%}."
            if error_reviewed
            else "No error findings triaged yet."
        ),
    ]
    if warning_findings:
        lines += [
            "",
            "## Advisory warnings (excluded from error totals)",
            "",
            "| Rule | Findings | True positives | False positives | Unreviewed |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for rule, _n in sorted(warning_by_rule.items()):
            n, tp, fp, unreviewed = _rule_totals(warning_findings, triage, rule)
            lines.append(f"| `{rule}` | {n} | {tp} | {fp} | {unreviewed} |")
        warning_fp = sum(
            1 for f in warning_findings if _triage_entry(f, triage).get("true_positive") is False
        )
        lines += [
            "",
            f"Advisory warnings: {len(warning_findings)} findings; triaged false positives "
            f"{warning_fp}; not included in the error false-positive rate.",
        ]
    lines += [
        "",
        "## Findings",
        "",
        "| # | Severity | Rule | Thread | Location | Message | Earlier call | Triage |",
        "| ---: | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, f in enumerate(findings, 1):
        entry = _triage_entry(f, triage)
        verdict = entry.get("true_positive")
        mark = "true positive" if verdict is True else ("false positive" if verdict is False else "")
        note = entry.get("note", "")
        if note:
            mark += f": {note}"
        loc = f"`{f.file.replace(str(REPO) + '/', '')}:{f.line}`"
        msg = f.message.replace("|", "\\|")
        earlier = (f.earlier or "").replace("|", "\\|").replace(str(REPO) + "/", "")
        lines.append(
            f"| {i} | {f.severity} | `{f.rule}` | {f.thread} | {loc} | {msg} | {earlier} | {mark} |"
        )

    if holes:
        lines += [
            "",
            "## Coverage holes",
            "",
            "| File | Arch | Thread | Reason |",
            "| --- | --- | --- | --- |",
        ]
        for h in holes[:500]:
            lines.append(
                f"| `{h['file'].replace(str(REPO) + '/', '')}` | {h['arch']} | {h['thread']} | "
                f"{str(h['error']).replace('|', '\\|')} |"
            )
        if len(holes) > 500:
            lines.append(f"| ... | | | {len(holes) - 500} more |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
