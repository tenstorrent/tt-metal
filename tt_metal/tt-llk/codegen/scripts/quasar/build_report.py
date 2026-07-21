#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Build the human-readable codegen run report from a run's artifacts.

Everything the orchestrator used to hand-assemble for §5f is derived here:
the fixed sections come straight from run.json; Assumptions / Reasoning are
pulled from the per-agent self-logs (agent_*.md); the tool histogram and top
commands come from the extracted transcripts. Writes the report to --out and
prints it to stdout.

Usage: build_report.py --log-dir <LOG_DIR> --out <path/to/report.md>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys


def _d(v) -> dict:
    """Return v if it's a dict, else {} — run.json fields can be the wrong type."""
    return v if isinstance(v, dict) else {}


def _l(v) -> list:
    return v if isinstance(v, list) else []


def _i(v, default=0) -> int:
    """Coerce to int; None / missing / non-numeric -> default (run.json fields
    are frequently null before a step populates them)."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _f(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _dur(seconds) -> str:
    try:
        s = int(seconds)
    except (TypeError, ValueError):
        return "?"
    return f"{s // 3600}h{(s % 3600) // 60}m{s % 60}s"


def _section(text: str, header_re: str) -> str:
    """Return the body between a `## <header>` line and the next `## ` line."""
    m = re.search(rf"^#{{2,3}}\s*{header_re}.*$", text, re.M | re.I)
    if not m:
        return ""
    start = m.end()
    nxt = re.search(r"^#{2,3}\s", text[start:], re.M)
    body = text[start : start + nxt.start()] if nxt else text[start:]
    return body.strip()


def _agent_label(stem: str) -> str:
    """agent_writer_cycle1 -> 'writer cycle1'; agent_analysis_refiner_v1 -> 'refiner v1'."""
    name = stem[len("agent_") :]
    name = name.replace("analysis_refiner", "refiner")
    return name.replace("_", " ")


def _first_paragraph(body: str) -> str:
    para = body.split("\n\n", 1)[0].strip() if body else ""
    return " ".join(para.split())


def _collect_agent_logs(log_dir: str):
    logs = []
    for path in sorted(glob.glob(os.path.join(log_dir, "agent_*.md"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        logs.append((_agent_label(stem), text))
    return logs


def _assumptions(logs) -> list[str]:
    out = []
    bullet = re.compile(r"^\s*(?:[-*]\s+|\d+\.\s+)")
    for label, text in logs:
        body = _section(text, r"Assumptions")
        if not body or body.strip().lower() in {"none", "none.", "n/a"}:
            continue
        # Group wrapped continuation lines back into one bullet each.
        entries: list[str] = []
        cur = ""
        for line in body.splitlines():
            s = line.strip()
            if not s:
                continue
            if bullet.match(line):
                if cur:
                    entries.append(cur)
                cur = bullet.sub("", line).strip()
            else:
                cur = f"{cur} {s}".strip()
        if cur:
            entries.append(cur)
        for e in entries:
            out.append(f"  [{label}] {' '.join(e.split())}")
    return out


def _reasoning(logs) -> list[str]:
    out = []
    for label, text in logs:
        para = _first_paragraph(_section(text, r"Reasoning summary"))
        if para:
            out.append(f"  [{label}] {para}")
    return out


def _transcripts(log_dir: str) -> list[str]:
    tdir = os.path.join(log_dir, "transcripts")
    if not os.path.isdir(tdir):
        return ["  (transcript extraction skipped)"]
    out = []
    for tools in sorted(glob.glob(os.path.join(tdir, "*_tools.md"))):
        base = os.path.basename(tools)[: -len("_tools.md")]
        slug = re.sub(r"^\d+_", "", base).replace("_", " ")
        hist = re.findall(
            r"^\|\s*`([^`]+)`\s*\|\s*(\d+)\s*\|",
            open(tools, errors="replace").read(),
            re.M,
        )
        hist_str = ", ".join(f"{t}×{n}" for t, n in hist[:10]) or "(none)"
        out.append(f"  {slug}:")
        out.append(f"    Tool histogram: {hist_str}")
        cmds_path = os.path.join(tdir, base + "_commands.md")
        if os.path.isfile(cmds_path):
            blocks = re.findall(
                r"###\s*\d+\.\s*(.+?)\n+```bash\n(.*?)\n```",
                open(cmds_path, errors="replace").read(),
                re.S,
            )
            top = sorted(blocks, key=lambda b: len(b[1]), reverse=True)[:5]
            if top:
                out.append("    Key bash:")
                for title, cmd in top:
                    out.append(f"      - {cmd.strip().splitlines()[0][:200]}")
                    out.append(f"        ({title.strip()})")
    return out or ["  (no transcripts found)"]


def build(d: dict, log_dir: str) -> str:
    logs = _collect_agent_logs(log_dir)
    tok = _d(d.get("tokens"))
    L = []
    A = L.append

    A("========================================")
    A("  LLK CodeGen — Generation Complete")
    A("========================================")
    A(f"Prompt:           {d.get('prompt', '')}")
    A(f"Kernel:           {d.get('kernel', '')}")
    A(f"Kernel Type:      {d.get('kernel_type', '')}")
    A(f"Target Arch:      {d.get('arch', '')}")
    A(f"Reference:        {d.get('reference_file', '')}")
    A(f"Generated File:   {d.get('generated_file', '')}")
    A(f"Lines Generated:  {_i(d.get('lines_generated'))}")
    A("----------------------------------------")
    A("Timing:")
    A(f"  Start:          {d.get('start_time', '')}")
    A(f"  End:            {d.get('end_time', '')}")
    A(f"  Duration:       {_dur(d.get('duration_seconds'))}")
    A("----------------------------------------")
    A("Tokens:")
    A(f"  Input:          {_i(tok.get('input'))}")
    A(f"  Output:         {_i(tok.get('output'))}")
    A(f"  Cache Read:     {_i(tok.get('cache_read'))}")
    A(f"  Cache Creation: {_i(tok.get('cache_creation'))}")
    A(f"  Total:          {_i(tok.get('total'))}")
    A(
        f"  Cost:           ${_f(d.get('cost_usd')):.2f} USD  (estimate; see Claude Console for billing)"
    )
    A("----------------------------------------")
    A("Flow:")
    A(
        f"  Cycles Used:       {_i(d.get('cycles_attempted'), 1)}/{_i(d.get('cycles_cap'), 3)}"
    )
    A(f"  Refinements:       {_i(d.get('refinement_count'))}")
    A(f"  Status:            {d.get('status', '')}")
    A("----------------------------------------")
    tt, tp = _i(d.get("tests_total")), _i(d.get("tests_passed"))
    tests = "NOT_AVAILABLE" if not tt else ("PASSED" if tp == tt else "FAILED")
    A("Quality:")
    A(
        f"  Compile Attempts:  {_i(d.get('compilation_attempts'))} (across writer + tester internal loop)"
    )
    A(
        f"  Compilation:       {'PASSED' if d.get('status') in ('success', 'compiled') else 'FAILED'}"
    )
    A(f"  Functional Tests:  {tests} ({tp}/{tt})")
    A(
        f"  Tests Source:      {'GENERATED' if d.get('tests_generated') else 'PRE-EXISTING'}"
    )
    A(f"  Formatted:         {'YES' if d.get('formatted') else 'NO'}")
    A(
        f"  Optimized:         {'YES' if d.get('optimized') else 'NO'} ({d.get('optimization_type', 'none')})"
    )
    A("----------------------------------------")
    A("Per Cycle:")
    for ph in _l(d.get("per_phase")):
        if not isinstance(ph, dict):
            continue
        A(
            f"  Cycle {ph.get('phase')} ({ph.get('name', '')}): "
            f"compiles={_i(ph.get('compilation_attempts'))}, "
            f"debug_cycles={_i(ph.get('debug_cycles'))}, "
            f"result={ph.get('test_result', '')}"
        )
    A("----------------------------------------")
    failures = [f for f in _l(d.get("failures")) if isinstance(f, dict)]
    if failures:
        resolved = sum(1 for f in failures if f.get("resolved"))
        A(
            f"Failures: {len(failures)} ({resolved} resolved, {len(failures) - resolved} unresolved)"
        )
        for f in failures:
            tag = "RESOLVED" if f.get("resolved") else "UNRESOLVED"
            A(
                f"  [{f.get('type')}] {f.get('step')} ({f.get('agent')}): {f.get('message')} — {tag}"
            )
        A("----------------------------------------")
    assumptions = _assumptions(logs)
    if assumptions:
        A("Assumptions made during the run:")
        L.extend(assumptions)
        A("----------------------------------------")
    reasoning = _reasoning(logs)
    if reasoning:
        A("Reasoning highlights:")
        L.extend(reasoning)
        A("----------------------------------------")
    A("Commands & tools summary:")
    L.extend(_transcripts(log_dir))
    A("----------------------------------------")
    ft = _l(d.get("formats_tested"))
    fx = _d(d.get("formats_excluded"))
    if ft or fx:
        A("Formats Tested:")
        A(f"  {', '.join(str(x) for x in ft) if ft else '(none recorded)'}")
        if fx:
            A(f"  Excluded: {', '.join(f'{k} ({v})' for k, v in fx.items())}")
        A("----------------------------------------")
    A("Artifacts:")
    A(f"  Generated File: {d.get('generated_file', '')}")
    A(f"  Metrics:        {log_dir}/  (run.json, generated.patch)")
    A(f"  Branch:         {d.get('git_branch', '')}")
    if d.get("obstacle"):
        A("----------------------------------------")
        A(f"Obstacle: {d['obstacle']}")
    A("========================================")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run_json = os.path.join(args.log_dir, "run.json")
    try:
        with open(run_json, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"build_report: cannot read {run_json}: {exc}", file=sys.stderr)
        return 1

    report = build(data, args.log_dir)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
