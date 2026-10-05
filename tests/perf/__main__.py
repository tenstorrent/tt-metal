# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CLI for perf results outside pytest.

  python -m tests.perf update --from-run <run-id> [--suite NAME] [--force]
  python -m tests.perf report measurements.json [base_measurements.json] [--all]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from tests.perf import compare as cmp
from tests.perf import report, session, update
from tests.perf.golden import Environment, Golden
from tests.perf.registry import load as load_registry

ARTIFACT_PATTERN = "perf_results_*"


def _download_run(run_id: str, dest: Path) -> list[Path]:
    subprocess.run(["gh", "run", "download", run_id, "--pattern", ARTIFACT_PATTERN, "--dir", str(dest)], check=True)
    return sorted(dest.rglob("measurements.json"))


def cmd_update(args) -> int:
    update.refuse_in_ci()
    suites = load_registry()
    with tempfile.TemporaryDirectory() as tmp:
        records = [session.Record.load(p) for p in _download_run(args.from_run, Path(tmp))]
    records = [r for r in records if args.suite is None or r.suite == args.suite]
    if not records:
        print(f"run {args.from_run} has no perf records" + (f" for suite {args.suite}" if args.suite else ""))
        return 1
    status = 0
    for record in records:
        suite = suites[record.suite]
        golden, comparison = session.evaluate(record, suite)
        session.publish(record, suite, golden, comparison, show_all=False)
        try:
            session.update_golden(record, suite, golden, comparison, force=args.force, source=f"CI run {args.from_run}")
        except update.UpdateRefused as refused:
            print(f"{record.suite}/{record.environment}: golden not modified: {refused}. Pass --force to accept.")
            status = 1
    return status


def cmd_report(args) -> int:
    run = session.Record.load(args.run)
    suites = load_registry()
    suite = suites.get(run.suite)
    policy = suite.policy if suite else cmp.Policy(regression_pct=5, improvement_pct=5)
    units = {name: spec.unit for name, spec in run.metrics.items()}
    if args.base is None:
        if suite is None:
            print(f"unknown suite {run.suite}; pass a base measurements.json to compare against")
            return 1
        golden, comparison = session.evaluate(run, suite)
        env = golden.environments.get(run.environment)
        print(
            report.render(
                f"{run.suite} / {run.environment}",
                comparison,
                units,
                context=run.context,
                golden_context=env.context if env else {},
                show_all=args.all,
            )
        )
        return 0
    base = session.Record.load(args.base)
    baseline = Golden(
        suite=run.suite,
        metrics=base.metrics,
        environments={
            run.environment: Environment(repetitions=base.repetitions, context=base.context, cases=base.cases)
        },
    )
    comparison = cmp.compare(
        run.cases,
        baseline,
        run.environment,
        run.metrics,
        policy,
        errors=run.errors,
        case_filter=run.case_filter,
        repetitions=run.repetitions,
    )
    print(
        report.render(
            f"{run.suite}: {args.run} vs {args.base}",
            comparison,
            units,
            context=run.context,
            golden_context=base.context,
            show_all=args.all,
            baseline_label="base",
        )
    )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tests.perf", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    up = sub.add_parser("update", help="update goldens from a CI run's perf artifacts")
    up.add_argument("--from-run", required=True, help="GitHub Actions run id")
    up.add_argument("--suite", help="only update this suite")
    up.add_argument("--force", action="store_true", help="accept regressions, new and missing cases")
    up.set_defaults(func=cmd_update)
    rep = sub.add_parser("report", help="compare a run against its golden, or against another run")
    rep.add_argument("run", type=Path)
    rep.add_argument("base", type=Path, nargs="?")
    rep.add_argument("--all", action="store_true", help="list every case")
    rep.set_defaults(func=cmd_report)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except update.UpdateRefused as refused:
        print(refused)
        return 1


if __name__ == "__main__":
    sys.exit(main())
