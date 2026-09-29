#!/usr/bin/env python3
"""Lint tests/pipeline_reorg/*.yaml cmd blocks: a job must stop at the first failed test command.

Every cmd block runs under ``bash -eo pipefail`` (see .github/actions/run-with-log), so a
plain sequence of pytest lines already stops at the first failure. The job-start hook
resets the devices once per job. Nothing resets them between the commands of one block.

A block that records a failure and continues, for example::

    pytest tests/a.py || fail=1
    pytest tests/b.py || fail=1
    exit $fail

starts the second pytest on devices that a hang, an abort or a triage dump left in an
unknown state. The second command then fails on device init, or hangs until the job
timeout (tt-metal#57706). This lint rejects the patterns that keep a block running:

* ``|| <name>=...`` after any command (records the failure, continues)
* ``|| true`` or ``|| :`` after a test runner (pytest, tracy, ctest, ./build/test/...)
* ``set +e``

Use ``|| exit $?`` instead, as in tt-metal#57710, or split the block into separate legs.

A baseline file (``--baseline``) lists entries that predate this lint, keyed by yaml
basename. A baselined entry that no longer triggers a finding is reported as stale, so
the baseline can only shrink.

Usage::

    lint_pipeline_cmds.py [--baseline FILE] [--tests-dir DIR | FILE ...]
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TESTS_DIR = REPO_ROOT / "tests" / "pipeline_reorg"
DEFAULT_BASELINE = REPO_ROOT / ".github" / "pipeline_cmd_lint_baseline.yaml"

# ``|| fail=1``, ``|| exit_code=$?``, ``|| rc=$?`` -- the failure is stored, the block goes on.
_OR_ASSIGNMENT = re.compile(r"\|\|\s*[A-Za-z_][A-Za-z0-9_]*=[^\s;}]*")
# ``|| true`` / ``|| :`` -- the failure is discarded. Only a problem after a test runner.
_OR_SWALLOW = re.compile(r"\|\|\s*(true|:)(?=\s|;|}|$)")
_TEST_RUNNER = re.compile(r"(^|[\s;&|(])(pytest|tracy|ctest|gtest|python3?\s+-m\s+(pytest|tracy))\b|/build/test/")
_SET_PLUS_E = re.compile(r"(^|[\s;&|{])set\s+\+e\b")


@dataclass(frozen=True)
class Finding:
    line: int
    message: str


def _logical_lines(cmd: str) -> list[tuple[int, str]]:
    """Join ``\\``-continued lines; return (first physical line number, joined text)."""
    out: list[tuple[int, str]] = []
    buf: list[str] = []
    start = 0
    for idx, raw in enumerate(cmd.splitlines(), start=1):
        if not buf:
            start = idx
        stripped = raw.rstrip()
        if stripped.endswith("\\"):
            buf.append(stripped[:-1])
            continue
        buf.append(stripped)
        out.append((start, " ".join(part.strip() for part in buf)))
        buf = []
    if buf:
        out.append((start, " ".join(part.strip() for part in buf)))
    return out


def lint_entry(entry: dict) -> list[Finding]:
    cmd = entry.get("cmd") or ""
    findings: list[Finding] = []
    for line_no, text in _logical_lines(cmd):
        code = text.strip()
        if not code or code.startswith("#"):
            continue
        if _SET_PLUS_E.search(code):
            findings.append(
                Finding(line_no, "`set +e` disables errexit; the block keeps running after a failed command")
            )
            continue
        match = _OR_ASSIGNMENT.search(code)
        if match:
            snippet = code[match.start() : match.end()].strip()
            findings.append(
                Finding(
                    line_no,
                    f"`{snippet}` records the failure and continues; use `|| exit $?` "
                    "so the job stops at the first failed command (tt-metal#57706)",
                )
            )
            continue
        swallow = _OR_SWALLOW.search(code)
        if swallow and _TEST_RUNNER.search(code):
            snippet = code[swallow.start() : swallow.end()].strip()
            findings.append(
                Finding(
                    line_no,
                    f"`{snippet}` discards a test runner's exit code; a failed or hung test must fail the job",
                )
            )
    return findings


def _load_entries(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text()) or []
    if not isinstance(data, list):
        return []
    return [e for e in data if isinstance(e, dict) and "cmd" in e]


def lint_files(paths: list[Path], baseline: dict[str, list[str]]) -> list[str]:
    """Return human-readable problems: findings outside the baseline plus stale baseline rows."""
    problems: list[str] = []
    for path in sorted(paths):
        allowed = set(baseline.get(path.name, []) or [])
        seen: set[str] = set()
        for entry in _load_entries(path):
            name = str(entry.get("name", "<unnamed>"))
            findings = lint_entry(entry)
            if not findings:
                continue
            seen.add(name)
            if name in allowed:
                continue
            for finding in findings:
                problems.append(f"{path.name}: {name}: cmd line {finding.line}: {finding.message}")
        for name in sorted(allowed - seen):
            problems.append(
                f"{path.name}: {name}: listed in the lint baseline but has no finding; " "remove it from the baseline"
            )
    return problems


def _load_baseline(path: Path | None) -> dict[str, list[str]]:
    if path is None or not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: baseline must be a mapping of yaml basename -> [entry names]")
    return {str(k): list(v or []) for k, v in data.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", type=Path, help="yaml files to lint (default: every yaml in --tests-dir)")
    parser.add_argument("--tests-dir", type=Path, default=DEFAULT_TESTS_DIR)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    args = parser.parse_args(argv)

    files = args.files or sorted(args.tests_dir.glob("*.yaml"))
    problems = lint_files(files, _load_baseline(args.baseline))
    for problem in problems:
        print(f"::error::{problem}" if _in_github_actions() else problem)
    if problems:
        print(f"\n{len(problems)} problem(s). Test commands must stop the block on failure: use `|| exit $?`.")
        return 1
    print(f"lint_pipeline_cmds: {len(files)} file(s) OK")
    return 0


def _in_github_actions() -> bool:
    import os

    return os.environ.get("GITHUB_ACTIONS") == "true"


if __name__ == "__main__":
    sys.exit(main())
