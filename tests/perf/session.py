# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Shared flow for pytest and the CLI: run, record, compare, report, update."""

from __future__ import annotations

import datetime
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tests.perf import compare as cmp
from tests.perf import contract, golden as golden_io, report, runner, update
from tests.perf.contract import MetricSpec
from tests.perf.registry import REPO_ROOT, Suite

RECORD_SCHEMA_VERSION = 1
OUTPUT_ROOT = REPO_ROOT / "generated" / "perf"
RETRY_MAX_FRACTION = 0.25


@dataclass
class Record:
    """Everything needed to compare or update later, written as measurements.json."""

    suite: str
    environment: str
    commit: str
    metrics: dict[str, MetricSpec]
    cases: dict[str, dict[str, float]]
    repetitions: int | None = None
    context: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    retry: dict[str, dict[str, float]] = field(default_factory=dict)
    # Number of out-of-band cases that were not re-run because too much of the suite moved.
    retry_skipped: int = 0
    case_filter: str | None = None

    @property
    def filtered(self) -> bool:
        return self.case_filter is not None

    def save(self, path: Path) -> None:
        document = asdict(self)
        document["metrics"] = {k: v.as_dict() for k, v in self.metrics.items()}
        document["schema_version"] = RECORD_SCHEMA_VERSION
        path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")

    @staticmethod
    def load(path: Path) -> "Record":
        document = json.loads(Path(path).read_text())
        if document.pop("schema_version", None) != RECORD_SCHEMA_VERSION:
            raise ValueError(f"{path} is not a version {RECORD_SCHEMA_VERSION} perf record")
        document["metrics"] = {k: MetricSpec.from_dict(v) for k, v in document["metrics"].items()}
        return Record(**document)


def output_dir(suite: str, environment: str) -> Path:
    return OUTPUT_ROOT / suite / environment


def _commit() -> str:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def execute(suite: Suite, environment: str, case_filter: str | None = None) -> Record:
    """Runs the suite, re-runs cases outside tolerance once, and writes measurements.json."""
    out_dir = output_dir(suite.name, environment)
    raw = contract.read(runner.run(suite, environment, out_dir, tag="run", case_filter=case_filter))
    record = Record(
        suite=suite.name,
        environment=environment,
        commit=_commit(),
        metrics=raw.metrics,
        cases=contract.aggregate(raw),
        repetitions=contract.repetitions(raw),
        context=contract.run_context(raw),
        errors=raw.errors,
        case_filter=case_filter,
    )
    _, comparison = evaluate(record, suite)
    # Report-only suites never fail on these statuses, so a re-run would only cost CI time.
    retry_cases = []
    if suite.policy.enforce:
        retry_cases = sorted({r.case for r in comparison.results if r.status in (cmp.REGRESSION, cmp.STALE)})
    # A shift across much of the suite is systematic rather than noise, so a re-run would not change the outcome.
    if len(retry_cases) > 1 and len(retry_cases) > RETRY_MAX_FRACTION * len(record.cases):
        print(f"Not re-running: {len(retry_cases)} of {len(record.cases)} cases are outside tolerance", flush=True)
        record.retry_skipped = len(retry_cases)
    elif retry_cases:
        print(f"Re-running {len(retry_cases)} cases outside tolerance to rule out noise", flush=True)
        retry_filter = runner.exact_filter(retry_cases) if suite.kind == "google_benchmark" else None
        retry_raw = contract.read(runner.run(suite, environment, out_dir, tag="retry", case_filter=retry_filter))
        record.retry = {case: v for case, v in contract.aggregate(retry_raw).items() if case in retry_cases}
    record.save(out_dir / "measurements.json")
    return record


def evaluate(record: Record, suite: Suite) -> tuple[golden_io.Golden, cmp.Comparison]:
    golden = golden_io.load(suite.golden, suite.name)
    comparison = cmp.compare(
        record.cases,
        golden,
        record.environment,
        record.metrics,
        suite.policy,
        errors=record.errors,
        retry=record.retry,
        case_filter=record.case_filter,
        repetitions=record.repetitions,
    )
    return golden, comparison


def failures(record: Record, suite: Suite, golden: golden_io.Golden, comparison: cmp.Comparison) -> list[str]:
    """Reasons the run fails. Report-only suites still fail on benchmark errors and changed case sets."""
    ignored = set()
    if not suite.policy.enforce:
        ignored |= {cmp.REGRESSION, cmp.STALE}
        if record.environment not in golden.environments:
            ignored |= {cmp.NEW}
    reasons = []
    for status in cmp.FAILING:
        count = sum(1 for r in comparison.results if r.status == status)
        if count and status not in ignored:
            reasons.append(f"{count} {status}")
    if suite.policy.enforce:
        reasons += comparison.config_errors
    return reasons


def publish(record: Record, suite: Suite, golden: golden_io.Golden, comparison: cmp.Comparison, show_all: bool):
    env = golden.environments.get(record.environment)
    kwargs = dict(
        context=record.context,
        golden_context=env.context if env else {},
        enforce=suite.policy.enforce,
        show_all=show_all,
        notes=[f"Not re-run: {record.retry_skipped} of {len(record.cases)} cases moved, so the shift is systematic"]
        if record.retry_skipped
        else [],
    )
    units = {name: spec.unit for name, spec in record.metrics.items()}
    title = f"{record.suite} / {record.environment}"
    print("\n" + report.render(title, comparison, units, **kwargs), flush=True)
    markdown = report.render(title, comparison, units, markdown=True, **kwargs)
    out_dir = output_dir(record.suite, record.environment)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(markdown)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
            summary.write(markdown + "\n")


def update_golden(
    record: Record, suite: Suite, golden: golden_io.Golden, comparison: cmp.Comparison, *, force: bool, source: str
) -> update.UpdatePlan:
    plan = update.apply(
        golden,
        record.environment,
        comparison,
        record.cases,
        record.metrics,
        repetitions=record.repetitions,
        context=record.context,
        provenance={"commit": record.commit, "date": datetime.date.today().isoformat(), "source": source},
        filtered=record.filtered,
        force=force,
    )
    if plan.empty():
        print(f"{suite.name}/{record.environment}: golden already matches; nothing to update")
        return plan
    golden_io.save(golden, suite.golden)
    if plan.replaced_environment:
        print(f"{suite.name}/{record.environment}: recorded {len(record.cases)} cases afresh")
    else:
        golden_path = suite.golden.relative_to(REPO_ROOT) if REPO_ROOT in suite.golden.parents else suite.golden
        print(
            f"{suite.name}/{record.environment}: wrote {len(plan.written)} values, "
            f"removed {len(plan.removed)}; review `git diff {golden_path}`"
        )
    return plan
