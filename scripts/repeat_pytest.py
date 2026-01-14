#!/usr/bin/env python3
"""
Repeat a pytest invocation N times, stopping on first failure.

Default test target:
  tests/ttnn/tracy/test_perf_op_report.py

Examples:
  ./scripts/repeat_pytest.py -n 50
  ./scripts/repeat_pytest.py -n 50 -- -k "test_something" -vv
  ./scripts/repeat_pytest.py -n 10 --test tests/ttnn/tracy/test_trace_runs.py::test_with_ops
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def _banner(msg: str, width: int = 100) -> None:
    line = "=" * width
    print("\n" + line)
    print(msg)
    print(line + "\n", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Repeat pytest N times, stop on first failure.")
    parser.add_argument(
        "-n",
        "--num-runs",
        type=int,
        required=True,
        help="Number of iterations to run.",
    )
    parser.add_argument(
        "--test",
        default="tests/ttnn/tracy/test_perf_op_report.py",
        help="Pytest test path/nodeid to run each iteration.",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to pytest after '--'.",
    )
    args = parser.parse_args()

    if args.num_runs <= 0:
        raise SystemExit("--num-runs must be >= 1")

    passthrough = list(args.pytest_args)
    if passthrough[:1] == ["--"]:
        passthrough = passthrough[1:]

    cmd = [sys.executable, "-m", "pytest", args.test, *passthrough]

    for i in range(1, args.num_runs + 1):
        _banner(f"PYTEST REPEAT RUN {i}/{args.num_runs}\n\nCommand:\n  {' '.join(cmd)}")
        try:
            completed = subprocess.run(cmd, check=False)
        except KeyboardInterrupt:
            _banner(f"INTERRUPTED on iteration {i}/{args.num_runs}")
            return 130

        if completed.returncode != 0:
            _banner(f"FAILED on iteration {i}/{args.num_runs} (exit={completed.returncode})")
            return completed.returncode

    _banner(f"ALL PASSED: {args.num_runs}/{args.num_runs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
