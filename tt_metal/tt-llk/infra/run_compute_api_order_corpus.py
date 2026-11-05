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
import concurrent.futures
import datetime as dt
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CHECKER = HERE / "check_compute_api_order.py"

DISCOVER_ROOTS = ("ttnn", "models", "tests", "tt-train", "tt_metal")
KERNEL_PATH_RE = re.compile(r"/(?:kernels|kernels_ng|kernels_dfb|unified_kernels)/")
COMPUTE_RE = re.compile(r"api/compute/|compute_kernel_hw_startup|compute_kernel_api")

_CHECKER_MODULE = None


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


def _run_one(task):
    arch, thread, src = task
    cc = _checker()
    sfpi = cc._find_sfpi(os.environ.get("TT_METAL_SFPI_ROOT"))
    if sfpi is None:
        return src, arch, thread, [], {}, "SFPI headers not found"
    args = cc.synth_args(src, arch, thread, sfpi)
    try:
        analyzer = cc.Analyzer(args, thread, arch, cc.RuleTable.load(arch), src)
        analyzer.analyze_paths()
    except Exception as exc:  # noqa: BLE001
        return src, arch, thread, [], {}, f"{type(exc).__name__}: {exc}"
    return src, arch, thread, analyzer.findings, {
        "entry": analyzer._entry_name,
        "parse_errors": len(analyzer.parse_errors),
    }, None


def _finding_from_json(d: dict):
    cc = _checker()
    return cc.Finding(**d)


def _run_subprocess(task):
    """Run one unit in a child process so a libclang crash is a hole, not a lost run."""
    arch, thread, src = task
    cc = _checker()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out = tmp.name
    cmd = [
        sys.executable,
        str(CHECKER),
        "--kernel",
        src,
        "--arch",
        arch,
        "--thread",
        thread,
        "--format",
        "json",
        "--output",
        out,
        "--quiet",
    ]
    sfpi = os.environ.get("TT_METAL_SFPI_ROOT")
    if sfpi:
        cmd += ["--sfpi", sfpi]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            return src, arch, thread, [], {}, (proc.stderr.strip() or f"exit {proc.returncode}")[-500:]
        payload = json.loads(Path(out).read_text())
        findings = [_finding_from_json(d) for d in payload.get("findings", [])]
        meta = {
            "entry": (payload.get("units") or [{}])[0].get("entry"),
            "parse_errors": (payload.get("units") or [{}])[0].get("parse_errors", 0),
        }
        return src, arch, thread, findings, meta, None
    except Exception as exc:  # noqa: BLE001
        return src, arch, thread, [], {}, f"{type(exc).__name__}: {exc}"
    finally:
        try:
            os.unlink(out)
        except OSError:
            pass


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
    parser.add_argument("--jobs", type=int, default=max(1, min(6, (os.cpu_count() or 4) // 2)))
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
    elif args.discover:
        files = discover(args.roots)
        if args.limit:
            files = files[: args.limit]
        tasks = [(arch, thread, str(f)) for arch in arches for f in files for thread in args.threads]
    else:
        cc = _checker()
        entries = (
            cc.entries_from_compile_db(Path(args.compile_db))
            if args.compile_db
            else cc.entries_from_log(Path(args.log))
        )
        tasks = []
        for entry in entries:
            arch, thread = cc.detect_from_args(entry.get("arguments", []), entry.get("file", ""))
            if arch is None:
                for a in arches:
                    tasks.append((a, thread or "MATH", cc._entry_source(entry)))
            else:
                tasks.append((arch, thread or "MATH", cc._entry_source(entry)))

    if args.from_json:
        executor = None
        results = []
        runner = None
    elif args.jobs > 1:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs)
        runner = _run_subprocess
    else:
        executor = None
        runner = _run_one

    if not args.from_json:
        if executor is None:
            results = map(runner, tasks)
        else:
            results = executor.map(runner, tasks)

    try:
        for i, (path, arch, thread, unit_findings, meta, error) in enumerate(results, 1):
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
                        "findings": len(unit_findings),
                    }
                )
                findings.extend(unit_findings)
            if i % 50 == 0:
                print(f"  {i}/{len(tasks)} units", file=sys.stderr)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    if not (args.compile_db or args.log or args.discover or args.from_json):
        parser.error("one of --from-json, --compile-db, --log or --discover is required")

    findings.sort(key=lambda f: (f.file, f.line, f.thread, f.rule))
    triage = {}
    if args.triage:
        triage = yaml.safe_load(Path(args.triage).read_text()) or {}

    report = render_report(arches, args, started, findings, holes, units, triage, args.runtime_seconds)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    if args.json_output:
        Path(args.json_output).write_text(
            json.dumps(
                {
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
        entry = triage.get(_fingerprint(f)) or {}
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


def render_report(arches, args, started, findings, holes, units, triage, runtime_seconds=None) -> str:
    counts, reviewed, fps = _count_triage(findings, triage)
    error_findings = [f for f in findings if f.severity == "error"]
    _ec, error_reviewed, error_fps = _count_triage(error_findings, triage)
    by_rule = collections.Counter(f.rule for f in findings)
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
        f"- Units analysed: {len(units)}",
        f"- Units that could not be analysed: {len(holes)}",
        f"- Findings: {len(findings)}",
        (f"- Analysis wall time: {runtime_seconds:.1f}s" if runtime_seconds else "- Analysis wall time: not recorded for this replay"),
        "",
        "The checker is advisory. A finding is not a verdict: review the source and",
        "record the verdict in the triage file. Unreviewed findings are not counted as",
        "false positives.",
        "",
        "## Findings by rule",
        "",
        "| Rule | Findings | True positives | False positives | Unreviewed |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for rule, n in sorted(by_rule.items()):
        tp = sum(
            1
            for f in findings
            if f.rule == rule and (triage.get(_fingerprint(f)) or {}).get("true_positive") is True
        )
        fp = sum(
            1
            for f in findings
            if f.rule == rule and (triage.get(_fingerprint(f)) or {}).get("true_positive") is False
        )
        lines.append(f"| `{rule}` | {n} | {tp} | {fp} | {n - tp - fp} |")
    lines += [
        "",
        (
            f"Reviewed: {reviewed}; false positives: {fps}; "
            f"false-positive rate: {fps / reviewed:.1%}"
            if reviewed
            else "No findings triaged yet."
        ),
        (
            f"Errors only (excludes advisory warnings): reviewed {error_reviewed}; "
            f"false positives {error_fps}; false-positive rate {error_fps / error_reviewed:.1%}."
            if error_reviewed
            else "No error findings triaged yet."
        ),
        "",
        "## Findings",
        "",
        "| # | Rule | Thread | Location | Message | Earlier call | Triage |",
        "| ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for i, f in enumerate(findings, 1):
        entry = triage.get(_fingerprint(f)) or {}
        verdict = entry.get("true_positive")
        mark = "true positive" if verdict is True else ("false positive" if verdict is False else "")
        note = entry.get("note", "")
        if note:
            mark += f": {note}"
        loc = f"`{f.file.replace(str(REPO) + '/', '')}:{f.line}`"
        msg = f.message.replace("|", "\\|")
        earlier = (f.earlier or "").replace("|", "\\|").replace(str(REPO) + "/", "")
        lines.append(f"| {i} | `{f.rule}` | {f.thread} | {loc} | {msg} | {earlier} | {mark} |")

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
