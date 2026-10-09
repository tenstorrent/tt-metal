# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Runs a suite's benchmark binary and returns the JSON files it wrote.

Google Benchmark suites run once with --benchmark_repetitions. Contract suites, such as a gtest, run one process
per repetition and write to $TT_PERF_OUTPUT. Also builds the exact-name filter used to re-run only the cases
that came out of tolerance.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from tests.perf.registry import REPO_ROOT, Suite

# The framework sets these itself. Suite args may still narrow the cases with --benchmark_filter.
RESERVED_ARGS = ("--benchmark_out", "--benchmark_repetitions")
_REGEX_SPECIAL = set(".^$|()[]{}*+?\\")
# Benchmark binaries must come from the build tree; suites.yaml cannot point anywhere else.
BINARY_ROOT = REPO_ROOT / "build"


class RunError(RuntimeError):
    pass


def _checked_binary(suite: Suite) -> Path:
    binary = suite.binary.resolve()
    if not binary.is_relative_to(BINARY_ROOT.resolve()):
        raise RunError(f"suite {suite.name} binary {binary} is outside {BINARY_ROOT}")
    if not binary.is_file():
        raise RunError(f"benchmark binary does not exist: {binary}; build it first")
    return binary


def _checked_filter(case_filter: str) -> str:
    """Accepts only a printable regex, since the filter comes from the command line."""
    if not case_filter.isprintable():
        raise RunError(f"case filter {case_filter!r} contains control characters")
    try:
        re.compile(case_filter)
    except re.error as error:
        raise RunError(f"case filter {case_filter!r} is not a valid regex: {error}") from None
    return case_filter


def exact_filter(cases: list[str]) -> str:
    """A --benchmark_filter regex that selects exactly these case names."""
    escaped = ["".join("\\" + c if c in _REGEX_SPECIAL else c for c in case) for case in sorted(cases)]
    return "^(" + "|".join(escaped) + ")$"


def run(suite: Suite, environment: str, out_dir: Path, *, tag: str, case_filter: str | None = None) -> list[Path]:
    """Runs the suite once (or once per process repetition) and returns its output files."""
    binary = _checked_binary(suite)
    reserved = [arg for arg in suite.args if arg.startswith(RESERVED_ARGS)]
    if reserved:
        raise RunError(f"suite {suite.name} args must not set {reserved}; the framework owns them")
    out_dir.mkdir(parents=True, exist_ok=True)
    if suite.kind == "contract":
        return [
            _run_contract_process(suite, binary, out_dir / f"{tag}_rep{rep}", rep)
            for rep in range(suite.process_repetitions)
        ]
    output = out_dir / f"{tag}.json"
    command = [str(binary), *suite.args, "--benchmark_out_format=json", f"--benchmark_out={output}"]
    repetitions = suite.repetitions_for(environment)
    if repetitions is not None:
        command.append(f"--benchmark_repetitions={repetitions}")
    if case_filter is not None:
        # Google Benchmark keeps the last --benchmark_filter, so this narrows any filter in the suite args.
        command.append(f"--benchmark_filter={_checked_filter(case_filter)}")
    _execute(command, os.environ.copy(), out_dir / f"{tag}.log")
    if not output.is_file():
        raise RunError(f"{suite.binary.name} wrote no output to {output}")
    return [output]


def _run_contract_process(suite: Suite, binary: Path, rep_dir: Path, rep: int) -> Path:
    rep_dir.mkdir(parents=True, exist_ok=True)
    output = rep_dir / "result.json"
    env = os.environ.copy()
    for name in suite.env_unset:
        env.pop(name, None)
    # {scratch} is for bulky per-repetition state such as caches, which must not end up in the CI artifacts.
    with tempfile.TemporaryDirectory(prefix=f"perf_{suite.name}_") as scratch:
        for name, value in suite.env.items():
            try:
                env[name] = value.format(rep=rep, rep_dir=rep_dir, scratch=scratch, env=os.environ)
            except KeyError as missing:
                raise RunError(f"suite {suite.name} needs ${missing.args[0]} to set {name}") from None
        env["TT_PERF_OUTPUT"] = str(output)
        _execute([str(binary), *suite.args], env, rep_dir / "run.log")
    if not output.is_file():
        raise RunError(f"{suite.binary.name} wrote no output to {output}; it must honor TT_PERF_OUTPUT")
    return output


def _execute(command: list[str], env: dict, log_path: Path) -> None:
    # The binary is checked to be under BINARY_ROOT, the args come from the checked-in suites.yaml, a user filter is
    # validated, and there is no shell, so no outside text can change what runs.
    print("+ " + " ".join(command), flush=True)
    with open(log_path, "w") as log:
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=False
        )
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        returncode = process.wait()
    if returncode:
        raise RunError(f"{Path(command[0]).name} exited with {returncode}; see {log_path}")
