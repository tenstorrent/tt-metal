#!/usr/bin/env python3
"""Lint tests/pipeline_reorg/*.yaml cmd blocks: a job must stop at the first failed test command.

Every cmd block runs under ``bash -eo pipefail`` (see .github/actions/run-with-log), so a
plain sequence of pytest lines already stops at the first failure. The job-start hook
resets the devices once per job. Nothing resets them between the commands of one block.

A block that records a failure and then runs another test command, for example::

    pytest tests/a.py || fail=1
    pytest tests/b.py || fail=1
    exit $fail

starts the second pytest on devices that a hang, an abort or a triage dump left in an
unknown state. The second command then fails on device init, or hangs until the job
timeout (tt-metal#57706). This lint rejects the patterns that keep a block running past a
failed test:

* ``|| <name>=...`` or ``|| { ...; }`` (without an ``exit``) after a command, when a later
  line of the block runs a test (directly, or through a shell function defined in the
  block that runs one). Recording the exit code of the *last* test command and then doing
  post-processing before ``exit $rc`` is fine: nothing runs on the device after it.
* ``|| true`` or ``|| :`` applied to a test runner (pytest, tracy, ctest, ./build/test/...)
* ``set +e``

Use ``|| exit $?`` instead, as in tt-metal#57710, or split the block into separate legs.

A baseline file (``--baseline``) lists entries that predate this lint, keyed by yaml
basename. A baselined entry that no longer triggers a finding is reported as stale, and
with ``--base-baseline`` (the baseline as it is on the target branch) any entry that is
new relative to that file is rejected, so the baseline can only shrink.

Usage::

    lint_pipeline_cmds.py [--baseline FILE] [--base-baseline FILE] [--tests-dir DIR | FILE ...]
"""

from __future__ import annotations

import argparse
import os
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
# ``|| { rc=$?; ...; }`` -- same thing behind a brace group, unless the group exits.
_OR_GROUP_OPEN = re.compile(r"\|\|\s*\{")
# ``|| true`` / ``|| :`` -- the failure is discarded. Only a problem when applied to a test runner.
_OR_SWALLOW = re.compile(r"\|\|\s*(true|:)(?=\s|;|}|$)")
_TEST_RUNNER = re.compile(r"(^|[\s;&|(\"'])(pytest|tracy|ctest|gtest|python3?\s+-m\s+(pytest|tracy))\b|/build/test/")
_SET_PLUS_E = re.compile(r"(^|[\s;&|{])set\s+\+e\b")
# WORKAROUND: Blaze prefill has too few Galaxies to give each section its own leg, so a leg
# that exports this runs every section even after a failure, on devices nothing has reset.
# Remove it once CI has enough Galaxies or the models are stable enough to split the legs.
_CONTINUE_MARKER = re.compile(r"^\s*export\s+CI_CONTINUE_ON_TEST_FAILURE=1\b", re.MULTILINE)
_CONTINUE_ALLOWED_YAMLS = {"blaze_models_prefill_tests.yaml"}
_EXIT = re.compile(r"(^|[\s;{])exit\b")
_FUNC_DEF = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(\)\s*\{")
_FIRST_WORD = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)")
# Command boundaries for "what does this `||` apply to": `;` and pipes start a new command,
# `&&` does not (`pytest a && rm x || true` still swallows the pytest failure).
_SEGMENT_SPLIT = re.compile(r";|\|(?!\|)|\(|\{")


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


def _code_lines(cmd: str) -> list[tuple[int, str]]:
    return [(no, text.strip()) for no, text in _logical_lines(cmd) if text.strip() and not text.strip().startswith("#")]


def _test_helpers(lines: list[tuple[int, str]]) -> set[str]:
    """Names of shell functions defined in the block whose body runs a test runner."""
    helpers: set[str] = set()
    current: str | None = None
    body: list[str] = []
    for _, text in lines:
        if current is None:
            match = _FUNC_DEF.match(text)
            if not match:
                continue
            current, body = match.group(1), [text]
            if "}" in text[match.end() :]:  # one-line body
                if _TEST_RUNNER.search(text):
                    helpers.add(current)
                current, body = None, []
            continue
        body.append(text)
        if text.startswith("}"):
            if any(_TEST_RUNNER.search(part) for part in body):
                helpers.add(current)
            current, body = None, []
    return helpers


def _runs_tests(text: str, helpers: set[str]) -> bool:
    if _TEST_RUNNER.search(text):
        return True
    first = _FIRST_WORD.match(text)
    return bool(first and first.group(1) in helpers)


def _brace_group(code: str, start: int) -> tuple[int, int, str] | None:
    """(start, end, body) of the ``|| { ... }`` group whose ``||`` begins at ``start``.

    Tracks nesting so ``${VAR}`` and inner groups inside the body do not end it early.
    """
    match = _OR_GROUP_OPEN.match(code, start)
    if not match:
        return None
    depth = 1
    for index in range(match.end(), len(code)):
        char = code[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return start, index + 1, code[match.end() : index]
    return start, len(code), code[match.end() :]


def _recording_group(code: str) -> re.Match | tuple[int, int] | None:
    """The first ``|| { ... }`` group that does not exit, as a (start, end) span."""
    for match in _OR_GROUP_OPEN.finditer(code):
        group = _brace_group(code, match.start())
        if group is None:
            continue
        start, end, body = group
        if not _EXIT.search(body):
            return start, end
    return None


def _command_before(code: str, index: int) -> str:
    """The shell command the operator at ``index`` applies to: the last segment before it."""
    return _SEGMENT_SPLIT.split(code[:index])[-1]


def lint_entry(entry: dict) -> list[Finding]:
    cmd = entry.get("cmd") or ""
    lines = _code_lines(cmd)
    helpers = _test_helpers(lines)
    findings: list[Finding] = []
    for position, (line_no, code) in enumerate(lines):
        if _SET_PLUS_E.search(code):
            findings.append(
                Finding(line_no, "`set +e` disables errexit; the block keeps running after a failed command")
            )
            continue
        recorder: tuple[int, int] | None = None
        assignment = _OR_ASSIGNMENT.search(code)
        if assignment:
            recorder = (assignment.start(), assignment.end())
        else:
            recorder = _recording_group(code)
        if recorder:
            later_tests = any(_runs_tests(text, helpers) for _, text in lines[position + 1 :])
            if later_tests or _FUNC_DEF.match(code):
                snippet = code[recorder[0] : recorder[1]].strip()
                if len(snippet) > 40:
                    snippet = snippet[:37] + "..."
                findings.append(
                    Finding(
                        line_no,
                        f"`{snippet}` records the failure and the block goes on to run another test; "
                        "use `|| exit $?` so the job stops at the first failed command (tt-metal#57706)",
                    )
                )
            continue
        swallow = _OR_SWALLOW.search(code)
        if swallow and _runs_tests(_command_before(code, swallow.start()), helpers):
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
            if _CONTINUE_MARKER.search(entry.get("cmd") or ""):
                if path.name not in _CONTINUE_ALLOWED_YAMLS:
                    problems.append(f"{path.name}: {name}: CI_CONTINUE_ON_TEST_FAILURE is not allowed in this yaml")
                continue
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
                f"{path.name}: {name}: listed in the lint baseline but has no finding; remove it from the baseline"
            )
    return problems


def baseline_growth(baseline: dict[str, list[str]], base_baseline: dict[str, list[str]]) -> list[str]:
    """Entries present in ``baseline`` but not in ``base_baseline``: the list can only shrink."""
    problems: list[str] = []
    for yaml_name, names in sorted(baseline.items()):
        allowed = set(base_baseline.get(yaml_name, []) or [])
        for name in names:
            if name not in allowed:
                problems.append(
                    f"{yaml_name}: {name}: new lint baseline entry; the baseline can only shrink, "
                    "fix the cmd block instead (use `|| exit $?` or split the legs)"
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
    parser.add_argument(
        "--base-baseline",
        type=Path,
        default=None,
        help="the baseline file as on the target branch; entries not in it are rejected "
        "(a missing or empty file skips the check, e.g. before the lint has landed)",
    )
    args = parser.parse_args(argv)

    files = args.files or sorted(args.tests_dir.glob("*.yaml"))
    baseline = _load_baseline(args.baseline)
    problems = lint_files(files, baseline)
    if args.base_baseline is not None:
        if args.base_baseline.exists() and args.base_baseline.stat().st_size > 0:
            problems.extend(baseline_growth(baseline, _load_baseline(args.base_baseline)))
        else:
            print(f"lint_pipeline_cmds: no baseline at {args.base_baseline}; skipping the shrink-only check")
    for problem in problems:
        print(f"::error::{problem}" if _in_github_actions() else problem)
    if problems:
        print(f"\n{len(problems)} problem(s). Test commands must stop the block on failure: use `|| exit $?`.")
        return 1
    print(f"lint_pipeline_cmds: {len(files)} file(s) OK")
    return 0


def _in_github_actions() -> bool:
    return os.environ.get("GITHUB_ACTIONS") == "true"


if __name__ == "__main__":
    sys.exit(main())
