# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F199: every forked op's every test, strictly: the closing check of a bring-up run and of a rebase.

    python -m models.demos.common.bringup.testing.fork_suite [--fork <f> ...] [--spec <spec.yaml>]

Per fork (default: every folder under ttnn/ttnn/bringup with a tests/ folder):
- unit: each ``tests/unit/test_*.py`` file, perf files included, one run each (every card visible);
- cases: the model cases ``tests/test_*.py``, grouped by each case's mesh (fork_tests.py);
- source: the original op's tests the fork carries (``tests/source.yaml``, fork_source.py), on the fork.
Every test is counted: passed, failed, errors, skipped (with pytest's reason), not run (collected, never reported) and
collection errors (an import error, nothing collected). A failing test is never accepted because it failed before
(owner, 2026-10-27); the only way to pass with one is an owner waiver in the spec, ``tests.fork_waivers: {<test id>:
<reason>}`` (the node id as printed, or the source key), reported as WAIVED. Skips are listed with their reasons.

Records fork_suite_failed (gate: == 0), fork_suite_passed, fork_suite_skipped, fork_suite_waived and writes the full
list to ``<results>/<task>_fork_suite.json``. Owner (2026-10-27): it runs once, after a completed run (task F.1, after
Z.1) and in the rebase skill, not on every fork change: it takes about an hour on a 2x2 QuietBox.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from models.demos.common.bringup.testing import fork_tests as FT

REPO = Path(__file__).resolve().parents[5]
FORKS = REPO / "ttnn" / "ttnn" / "bringup"


def all_forks() -> list[str]:
    return sorted(p.parent.name for p in FORKS.glob("*/tests") if p.is_dir())


def unit(fork: str, runner=None) -> dict:
    t = FT.empty_tally()
    for f in sorted((FORKS / fork / "tests" / "unit").glob("test_*.py")):
        nodes, errors = FT.collect([str(f.relative_to(REPO))])
        t["collect_errors"] += len(errors)
        t["problems"] += [f"collection error: {e}" for e in errors]
        if nodes:
            env = dict(os.environ)
            env.pop("TT_VISIBLE_DEVICES", None)
            print(f"[{fork}] unit {f.name}: {len(nodes)} test(s)", flush=True)
            FT.tally_group(t, fork, nodes, env, [], runner, f.name)
    return t


def source(fork: str) -> dict:
    """The fork's run of its source selection, tallied: anything not passed or skipped is a failure (fork_source.report
    labels them against the baseline)."""
    from models.demos.common.bringup.testing import fork_source as FS

    t = FT.empty_tally()
    m = FS.manifest(fork)
    if not m.get("tests"):
        return t
    scratch = REPO / "generated" / "bringup_fork_source" / fork
    scratch.mkdir(parents=True, exist_ok=True)
    got = FS.run(m, True, "fork", scratch)
    base_p = FORKS / fork / "tests" / "source_baseline.json"
    FS.report(json.loads(base_p.read_text())["tests"] if base_p.exists() else {}, got, f"{fork} source")
    for k, v in sorted(got.items()):
        if v in ("passed", "skipped"):
            t[v] += 1
            if v == "skipped":
                t["problems"].append(f"skipped: {k}")
        elif k.endswith("::<no results>"):
            t["collect_errors"] += 1
            t["problems"].append(f"collection error: {k}")
        else:
            t["failed"] += 1
            t["problems"].append(f"{v}: {k}")
    return t


def waive(t: dict, waivers: dict[str, str]) -> int:
    """Move failures the owner waived out of the failure count; returns how many."""
    n = 0
    for i, p in enumerate(t["problems"]):
        kind, _, rest = p.partition(": ")
        test = rest.split(": ", 1)[0].split(" (", 1)[0]
        if kind in ("failed", "error", "not run") and test in waivers:
            key = {"failed": "failed", "error": "error", "not run": "not_run"}[kind]
            t[key] -= 1
            t["problems"][i] = f"WAIVED {p} (owner: {waivers[test]})"
            n += 1
    return n


def run_suite(forks: list[str], waivers: dict[str, str], runner=None, parts=None) -> dict:
    """{fork: {part: tally}} with the waivers applied. ``parts`` (selftests): {part: fn(fork) -> tally}."""
    parts = parts or {"unit": lambda f: unit(f, runner), "cases": None, "source": source}
    out = {}
    for fork in forks:
        out[fork] = {}
        for part, fn in parts.items():
            if part == "cases":
                if not list((FORKS / fork / "tests").glob("test_*.py")):
                    continue
                t = FT.run([fork], runner=runner)[fork]
            else:
                t = fn(fork)
            t["waived"] = waive(t, waivers)
            out[fork][part] = t
    return out


def totals(res: dict) -> dict:
    ts = [t for parts in res.values() for t in parts.values()]
    return {
        "fork_suite_failed": sum(FT.bad(t) for t in ts),
        "fork_suite_passed": sum(t["passed"] for t in ts),
        "fork_suite_skipped": sum(t["skipped"] for t in ts),
        "fork_suite_waived": sum(t.get("waived", 0) for t in ts),
    }


def report(res: dict) -> None:
    print("\nfork suite:")
    for fork, parts in res.items():
        for part, t in parts.items():
            print(
                f"  {fork:28} {part:6} passed {t['passed']:5}  failed {t['failed']:3}  errors {t['error']:3}  "
                f"skipped {t['skipped']:4}  not run {t['not_run']:3}  collection errors {t['collect_errors']:2}"
                f"  waived {t.get('waived', 0)}"
            )
    for fork, parts in res.items():
        for part, t in parts.items():
            for p in t["problems"]:
                print(f"    [{fork} {part}] {p}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fork", action="append", help="only these forks (default: every fork with tests)")
    ap.add_argument("--spec", help="the model spec: its tests.fork_waivers, and the metrics go to its task")
    a = ap.parse_args(argv)
    waivers = {}
    a.spec = a.spec or os.environ.get("BRINGUP_SPEC")
    if a.spec:
        from models.demos.common.bringup.core.spec import Spec

        waivers = dict(Spec.load(a.spec).get("tests.fork_waivers") or {})
    res = run_suite(a.fork or all_forks(), waivers)
    report(res)
    st = totals(res)
    print(json.dumps(st))
    if os.environ.get("BRINGUP_RESULTS_DIR"):
        from models.demos.common.bringup.core import metrics

        for k, v in st.items():
            metrics.record(k, v)
        metrics.record_json("fork_suite", res)
    return 1 if st["fork_suite_failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
