# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Do the derived bring-up ops' tests cover every call this model makes? Optionally run those tests.

    python -m models.demos.common.bringup.testing.fork_cases --capture <bringup>/results/fork_calls.json \
        [--spec <spec.yaml> | --model <name>] [--run-tests]

- ``--capture``: the file ``testing/fork_capture.py`` wrote, one entry per distinct call signature (``sig``).
- Test cases: each fork ``ttnn/ttnn/bringup/<fork>/tests/cases.py`` defines ``CASES``, a list of dicts, each with at
  least ``model`` and ``sig``: the model that made the call, and the captured signature the case reproduces.
- Uncovered: a call this model makes that no case of its fork carries (same model, same sig).
- ``--run-tests``: runs ``scripts/run_safe_pytest.sh --run-all`` on the tests of every fork the model uses, every
  model's cases included.
Records forks_used, fork_calls, fork_calls_uncovered and fork_tests_failed (a gate wants 0 of each of the last two)
and prints what is missing.
"""

from __future__ import annotations

import argparse
import json
import re
import runpy
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
FORKS = REPO / "ttnn" / "ttnn" / "bringup"
BIND = re.compile(r'bind_function<\s*"(\w+)"\s*,\s*"ttnn\.bringup\."\s*>')


def op_to_fork() -> dict[str, str]:
    """ttnn.bringup.<op> -> fork folder, from each fork's nanobind sources."""
    out = {}
    for f in sorted(FORKS.glob("*/*_nanobind.cpp")):
        for op in BIND.findall(f.read_text()):
            out[f"ttnn.bringup.{op}"] = f.parent.name
    return out


def cases(fork: str) -> list[dict]:
    p = FORKS / fork / "tests" / "cases.py"
    return list(runpy.run_path(str(p)).get("CASES", [])) if p.exists() else []


def check(capture: Path, model: str) -> tuple[dict, dict[str, list[dict]]]:
    calls = json.loads(capture.read_text())["calls"]
    owner = op_to_fork()
    by_fork: dict[str, list[dict]] = {}
    for c in calls:
        fork = owner.get(c["op"])
        if fork is None:
            raise SystemExit(f"{c['op']}: no fork binds this op (INDEX.md)")
        by_fork.setdefault(fork, []).append(c)
    missing: dict[str, list[dict]] = {}
    for fork, cs in by_fork.items():
        have = {k["sig"] for k in cases(fork) if k.get("model") == model}
        gap = [c for c in cs if c["sig"] not in have]
        if gap:
            missing[fork] = gap
    stats = {
        "forks_used": len(by_fork),
        "fork_calls": len(calls),
        "fork_calls_uncovered": sum(len(v) for v in missing.values()),
    }
    return {"stats": stats, "forks": sorted(by_fork)}, missing


def run_tests(forks: list[str]) -> int:
    dirs = [str((FORKS / f / "tests").relative_to(REPO)) for f in forks if (FORKS / f / "tests").is_dir()]
    if len(dirs) < len(forks):
        print(f"forks without tests/: {sorted(set(forks) - {Path(d).parent.name for d in dirs})}")
        return 1
    if not dirs:
        return 0
    # --no-precompile: the up-front collect pass would run each test's host-side input building a second time.
    return subprocess.run(["scripts/run_safe_pytest.sh", "--run-all", "--no-precompile", *dirs], cwd=REPO).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--spec")
    g.add_argument("--model")
    ap.add_argument("--run-tests", action="store_true")
    a = ap.parse_args(argv)

    spec = None
    if a.model is None:
        from models.demos.common.bringup.reference.golden import load_spec

        spec = load_spec(a.spec)
        model = spec.model_dir.name
    else:
        model = a.model
    res, missing = check(Path(a.capture), model)
    stats = res["stats"]
    print(f"{model}: {stats['fork_calls']} distinct call(s) to {stats['forks_used']} fork(s): {res['forks']}")
    for fork, gap in missing.items():
        print(f"  {fork}: {len(gap)} call(s) with no case for {model}:")
        for c in gap:
            print(f"    sig {c['sig']} (x{c['count']}): {c['op']}")
    failed = 0
    if a.run_tests and res["forks"]:
        failed = int(run_tests(res["forks"]) != 0)
    stats["fork_tests_failed"] = failed
    if spec is not None:
        from models.demos.common.bringup.core import metrics

        for k, v in stats.items():
            metrics.record(k, v)
    print(json.dumps(stats))
    return 1 if stats["fork_calls_uncovered"] or failed else 0


if __name__ == "__main__":
    sys.exit(main())
