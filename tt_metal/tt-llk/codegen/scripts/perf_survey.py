#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Headroom survey: one wave diagnosis per implemented Quasar kernel, ranked.

For every perf test in tests/python_tests/quasar (SFPU tests first), collect its
variants, pick one per op (SFPU tests: per MathOperation, preferring Float16_b;
other tests: one per module), and run perf_diagnose.py on it, one at a time. After
every op it rewrites <out-dir>/survey.csv and survey.md, ranked by stall headroom.
An op whose diag.json already exists is skipped (``--retry-failed`` re-runs failed
and skipped ones), so the survey can be stopped and resumed.

    python codegen/scripts/perf_survey.py --out-dir /proj_sw/.../survey [--module perf_x_quasar.py ...]

Each op keeps ~50 KB; the emulator dump and wave file are deleted by perf_diagnose.
A row is a ``candidate`` when the diagnosis is ok, the kernel is issue-bound (its stall
headroom is judged; unit- and sync-bound kernels are reported but not judged) and the
stall headroom is at least --candidate-pct (default 5, the Step 6c threshold).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_HERE = Path(__file__).resolve().parent
_LLK = _HERE.parents[1]
SKIP_MODULES = {"perf_fused_quasar.py"}  # fused pipelines, not one kernel
_OP_RE = re.compile(r"MathOperation\.([A-Za-z0-9_]+)")
# key:value ids (e.g. reduce): the fields that name the operation.
_KEYED_RE = re.compile(r"\b(reduce_pool|mathop|op):([A-Za-z0-9_]+)")
_FORMAT_RE = re.compile(r"^(Float|Int|UInt|Bfp|Tf32|MxFp|Bfloat|Lf8)")


def _op_key(test_id: str) -> str | None:
    """MathOperation.<Op>, else the parametrize id's leading tokens before the first format."""
    m = _OP_RE.search(test_id)
    if m:
        return m.group(1)
    keyed = _KEYED_RE.findall(test_id)
    if keyed:
        return "_".join(value for _key, value in keyed)
    bracket = (
        test_id[test_id.find("[") + 1 : test_id.rfind("]")] if "[" in test_id else ""
    )
    tokens = []
    for token in bracket.split("-"):
        if _FORMAT_RE.match(token):
            break
        tokens.append(token)
    key = "-".join(tokens)
    # A real op name is a short dash-separated token list; anything with structure is not one.
    return key if key and re.fullmatch(r"[A-Za-z0-9_-]{1,40}", key) else None


COLUMNS = [
    "module",
    "op",
    "status",
    "run_type",
    "measured",
    "floor",
    "stall_headroom_pct",
    "bound",
    "top_cause",
    "candidate",
    "reason",
    "test_id",
    "dir",
]


@dataclass(frozen=True)
class Entry:
    module: str
    op: str
    test_id: str
    run_type: str
    sfpu: bool


def _is_sfpu(module: str) -> bool:
    return "sfpu" in module and "matmul" not in module


def parse_collected(log: str, module: str) -> list[str]:
    pat = re.compile(rf"^(quasar/)?{re.escape(module)}::")
    return [line.strip() for line in log.splitlines() if pat.match(line.strip())]


def plan(collected: dict[str, list[str]]) -> list[Entry]:
    """One entry per op (SFPU modules) or per module (others); SFPU modules first."""
    entries: list[Entry] = []
    modules = sorted(collected, key=lambda m: (not _is_sfpu(m), m))
    for module in modules:
        ids = collected[module]
        if not ids:
            continue
        sfpu = _is_sfpu(module)
        groups: dict[str, list[str]] = {}
        for tid in ids:
            key = _op_key(tid) if sfpu else None
            groups.setdefault(key or Path(module).stem, []).append(tid)
        for op in sorted(groups):
            choices = groups[op]
            pick = next((t for t in choices if "Float16_b" in t), choices[0])
            entries.append(
                Entry(module, op, pick, "MATH_ISOLATE" if sfpu else "L1_TO_L1", sfpu)
            )
    return entries


def _row(entry: Entry, out_dir: Path, candidate_pct: float) -> dict:
    diag_path = out_dir / "diag.json"
    diag = (
        json.loads(diag_path.read_text())
        if diag_path.is_file()
        else {"status": "missing"}
    )
    top = (diag.get("top_causes") or [[None, None]])[0]
    hr = diag.get("stall_headroom_pct")
    bound = diag.get("bound")
    # Only an issue-bound kernel has a meaningful stall headroom. Results written before the
    # bound existed fall back to the old rule (SFPU tests only).
    judged = bound["kind"] == "issue" if bound else entry.sfpu
    candidate = (
        diag.get("status") == "ok" and judged and hr is not None and hr >= candidate_pct
    )
    return {
        "module": entry.module,
        "op": entry.op,
        "status": diag.get("status"),
        "run_type": entry.run_type,
        "measured": diag.get("measured"),
        "floor": diag.get("floor"),
        "stall_headroom_pct": hr,
        "bound": f"{bound['kind']}({bound['thread']})" if bound else "",
        "top_cause": f"{top[0]}: {top[1]}" if top[0] else "",
        "candidate": "yes" if candidate else "no",
        "reason": diag.get("reason", ""),
        "test_id": entry.test_id,
        "dir": str(out_dir),
    }


def _rank(rows: list[dict]) -> list[dict]:
    def key(r):
        judged = (
            r["status"] in ("ok", "not_reconciled")
            and r["stall_headroom_pct"] is not None
        )
        return (
            not judged,
            r["status"] != "ok",
            -(r["stall_headroom_pct"] or 0),
            r["module"],
            r["op"],
        )

    return sorted(rows, key=key)


def _write_tables(root: Path, rows: list[dict]) -> None:
    with open(root / "survey.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    lines = [
        "| op | module | status | bound | measured cyc/tile | floor | stall headroom % | top cause | candidate |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['op']} | {r['module']} | {r['status']} | {r['bound']} | {r['measured'] or ''} | {r['floor'] or ''} | "
            f"{r['stall_headroom_pct'] if r['stall_headroom_pct'] is not None else ''} | "
            f"{r['top_cause'] or r['reason']} | {r['candidate']} |"
        )
    (root / "survey.md").write_text("\n".join(lines) + "\n")


def _entry_dir(root: Path, entry: Entry) -> Path:
    return root / Path(entry.module).stem / entry.op


def run_survey(
    entries: list[Entry],
    root: Path,
    diagnose: Callable[[Entry, Path], None],
    *,
    retry_failed: bool = False,
    candidate_pct: float = 5.0,
    log: Callable[[str], None] = lambda _m: None,
) -> list[dict]:
    root.mkdir(parents=True, exist_ok=True)
    # Every result already on disk is in the table from the start; the loop fills in the rest.
    rows: dict[int, dict] = {
        i: _row(e, _entry_dir(root, e), candidate_pct)
        for i, e in enumerate(entries)
        if (_entry_dir(root, e) / "diag.json").is_file()
    }
    _write_tables(root, _rank(list(rows.values())))
    for i, entry in enumerate(entries):
        out = _entry_dir(root, entry)
        existing = out / "diag.json"
        status = (
            json.loads(existing.read_text()).get("status")
            if existing.is_file()
            else None
        )
        if status is None or (retry_failed and status in ("failed", "skipped")):
            log(f"[survey {i + 1}/{len(entries)}] {entry.module} {entry.op}")
            diagnose(entry, out)
            rows[i] = _row(entry, out, candidate_pct)
            _write_tables(root, _rank(list(rows.values())))
    return _rank(list(rows.values()))


def _collect(llk: Path, module: str) -> list[str]:
    done = subprocess.run(
        [
            "bash",
            str(llk / ".claude/scripts/run_test.sh"),
            "count",
            "--worktree",
            str(llk),
            "--arch",
            "quasar",
            "--test",
            module,
        ],
        capture_output=True,
        text=True,
    )
    return parse_collected(done.stderr + "\n" + done.stdout, module)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--llk",
        type=Path,
        default=_LLK,
        help="tt-llk checkout to survey (default: this one)",
    )
    p.add_argument(
        "--module", action="append", help="perf test file(s) to survey (default: all)"
    )
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--candidate-pct", type=float, default=5.0)
    p.add_argument("--plan-only", action="store_true", help="print the plan and exit")
    args = p.parse_args(argv)

    modules = args.module or sorted(
        f.name
        for f in (args.llk / "tests/python_tests/quasar").glob("perf_*_quasar.py")
        if f.name not in SKIP_MODULES
    )
    collected = {m: _collect(args.llk, m) for m in modules}
    for m, ids in collected.items():
        if not ids:
            print(f"[survey] {m}: no variants collected", file=sys.stderr)
    entries = plan(collected)
    if args.plan_only:
        for e in entries:
            print(f"{e.module}\t{e.op}\t{e.run_type}\t{e.test_id}")
        return 0

    def diagnose(entry: Entry, out: Path) -> None:
        subprocess.run(
            [
                sys.executable,
                str(_HERE / "perf_diagnose.py"),
                "--llk",
                str(args.llk),
                "--module",
                entry.module,
                "--test-id",
                entry.test_id,
                "--run-type",
                entry.run_type,
                "--out-dir",
                str(out),
            ],
            env=dict(os.environ),
        )

    rows = run_survey(
        entries,
        args.out_dir,
        diagnose,
        retry_failed=args.retry_failed,
        candidate_pct=args.candidate_pct,
        log=lambda m: print(m, file=sys.stderr, flush=True),
    )
    print(
        f"[survey] {len(rows)} ops; candidates: "
        f"{', '.join(r['op'] for r in rows if r['candidate'] == 'yes') or 'none'}; "
        f"table: {args.out_dir / 'survey.md'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
