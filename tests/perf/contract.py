# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Reads benchmark output that follows the perf contract (tests/perf/perf_contract.hpp).

The output is Google Benchmark JSON. Gated metrics are declared in the context block as
``perf.metric.<name> = "unit=..;better=..;aggregate=.."``. Each case reports one value per repetition, either as
iteration rows or, for binaries that only report aggregates, as the aggregate row named by the declared
aggregation.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path

METRIC_PREFIX = "perf.metric."
RUN_CONTEXT_PREFIX = "perf.context."
CASE_CONTEXT_PREFIX = "ctx_"
AGGREGATES = ("min", "median", "max")
BETTER = ("lower", "higher")


class ContractError(ValueError):
    """Benchmark output does not follow the perf contract."""


@dataclass(frozen=True)
class MetricSpec:
    unit: str
    better: str
    aggregate: str

    def as_dict(self) -> dict:
        return {"unit": self.unit, "better": self.better, "aggregate": self.aggregate}

    @staticmethod
    def from_dict(d: dict) -> "MetricSpec":
        return MetricSpec(unit=d["unit"], better=d["better"], aggregate=d["aggregate"])


@dataclass
class RawRun:
    """Samples from one or more benchmark output files of the same binary."""

    metrics: dict[str, MetricSpec] = field(default_factory=dict)
    samples: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    # Pre-aggregated values and their repetition counts, for binaries that only report aggregate rows.
    aggregated: dict[str, dict[str, float]] = field(default_factory=dict)
    repetitions: dict[str, int] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    context: dict[str, str] = field(default_factory=dict)
    case_context: dict[str, set] = field(default_factory=dict)


def parse_metric_decl(name: str, value: str) -> MetricSpec:
    fields = dict(part.split("=", 1) for part in value.split(";") if "=" in part)
    try:
        spec = MetricSpec(unit=fields["unit"], better=fields["better"], aggregate=fields["aggregate"])
    except KeyError as missing:
        raise ContractError(f"metric {name!r} declaration {value!r} is missing {missing}") from None
    if spec.better not in BETTER:
        raise ContractError(f"metric {name!r} has better={spec.better!r}; expected one of {BETTER}")
    if spec.aggregate not in AGGREGATES:
        raise ContractError(f"metric {name!r} has aggregate={spec.aggregate!r}; expected one of {AGGREGATES}")
    return spec


def _valid(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def read(paths: list[Path]) -> RawRun:
    """Merges benchmark output files into one RawRun. Repetitions from separate processes are concatenated."""
    run = RawRun()
    for path in paths:
        try:
            document = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ContractError(f"cannot read benchmark output {path}: {error}") from error
        context = document.get("context", {})
        for key, value in context.items():
            if key.startswith(METRIC_PREFIX):
                name = key[len(METRIC_PREFIX) :]
                spec = parse_metric_decl(name, str(value))
                if run.metrics.setdefault(name, spec) != spec:
                    raise ContractError(f"metric {name!r} is declared differently across output files")
            elif key.startswith(RUN_CONTEXT_PREFIX):
                run.context[key[len(RUN_CONTEXT_PREFIX) :]] = str(value)
        if not run.metrics:
            raise ContractError(f"{path} declares no perf.metric.* entries; see tests/perf/perf_contract.hpp")
        rows = document.get("benchmarks")
        if not isinstance(rows, list):
            raise ContractError(f"{path} has no 'benchmarks' list")
        _read_rows(rows, run, path)
    return run


def _read_rows(rows: list, run: RawRun, path: Path) -> None:
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise ContractError(f"{path} has a benchmark row without a name")
        case = row.get("run_name") or row["name"]
        if row.get("error_occurred"):
            run.errors.setdefault(case, str(row.get("error_message", "benchmark reported an error")))
            continue
        run_type = row.get("run_type", "iteration")
        if run_type == "aggregate":
            _read_aggregate_row(row, case, run)
            continue
        if run_type != "iteration":
            continue
        reported = False
        for metric in run.metrics:
            if metric in row:
                if not _valid(row[metric]):
                    raise ContractError(f"case {case!r} has invalid {metric}: {row[metric]!r}")
                run.samples.setdefault(case, {}).setdefault(metric, []).append(float(row[metric]))
                reported = True
        if not reported:
            raise ContractError(f"case {case!r} reports none of the declared metrics {sorted(run.metrics)}")
        _read_case_context(row, case, run)


def _read_aggregate_row(row: dict, case: str, run: RawRun) -> None:
    for metric, spec in run.metrics.items():
        if row.get("aggregate_name") == spec.aggregate and metric in row:
            if not _valid(row[metric]):
                raise ContractError(f"case {case!r} has invalid {spec.aggregate} {metric}: {row[metric]!r}")
            run.aggregated.setdefault(case, {})[metric] = float(row[metric])
            run.repetitions[case] = int(row.get("repetitions", 0))
            _read_case_context(row, case, run)


def _read_case_context(row: dict, case: str, run: RawRun) -> None:
    for key, value in row.items():
        if key.startswith(CASE_CONTEXT_PREFIX) and isinstance(value, (int, float)):
            run.case_context.setdefault(key[len(CASE_CONTEXT_PREFIX) :], set()).add(value)


def aggregate(run: RawRun) -> dict[str, dict[str, float]]:
    """Returns one value per case and metric using each metric's declared aggregation."""
    values: dict[str, dict[str, float]] = {}
    for case, per_metric in run.samples.items():
        for metric, samples in per_metric.items():
            values.setdefault(case, {})[metric] = _AGGREGATORS[run.metrics[metric].aggregate](samples)
    for case, per_metric in run.aggregated.items():
        for metric, value in per_metric.items():
            values.setdefault(case, {}).setdefault(metric, value)
    return values


_AGGREGATORS = {"min": min, "max": max, "median": statistics.median}


def run_context(run: RawRun) -> dict[str, str]:
    """Run-wide context plus per-case context collapsed to a single value or a range."""
    context = dict(run.context)
    for key, values in run.case_context.items():
        lo, hi = min(values), max(values)
        context[key] = _fmt_number(lo) if lo == hi else f"{_fmt_number(lo)}..{_fmt_number(hi)}"
    return context


def _fmt_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def repetitions(run: RawRun) -> int | None:
    """The single repetition count shared by all cases, or None if the output does not say."""
    counts = set(run.repetitions.values())
    counts |= {len(samples) for per_metric in run.samples.values() for samples in per_metric.values()}
    if not counts:
        return None
    if len(counts) > 1:
        raise ContractError(f"cases ran with different repetition counts: {sorted(counts)}")
    return counts.pop()


SUFFIX_SEGMENTS = ("manual_time", "real_time", "process_time")


def group_of(case: str) -> str:
    """Group key for a case: the name segments before the first key:value argument."""
    head = []
    for segment in case.split("/"):
        if ":" in segment or segment in SUFFIX_SEGMENTS:
            break
        head.append(segment)
    return "/".join(head) or case
