# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Classifies every case against its golden value using the suite's policy."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from tests.perf.contract import MetricSpec, group_of
from tests.perf.golden import Environment, Golden

PASS = "PASS"
REGRESSION = "REGRESSION"
STALE = "STALE"
NEW = "NEW"
MISSING = "MISSING"
ERROR = "ERROR"
FAILING = (REGRESSION, STALE, NEW, MISSING, ERROR)


@dataclass(frozen=True)
class Policy:
    regression_pct: float
    improvement_pct: float
    enforce: bool = True
    # Ordered (regex, {"regression_pct": .., "improvement_pct": ..}) pairs; the first match wins.
    overrides: tuple = ()

    def thresholds(self, case: str) -> tuple[float, float]:
        for pattern, values in self.overrides:
            if re.search(pattern, case):
                return (
                    float(values.get("regression_pct", self.regression_pct)),
                    float(values.get("improvement_pct", self.improvement_pct)),
                )
        return self.regression_pct, self.improvement_pct


@dataclass
class CaseResult:
    case: str
    metric: str
    status: str
    actual: float | None = None
    golden: float | None = None
    retry_actual: float | None = None
    # Status of the first run when the retry disagreed with it, so the report can show the flake.
    first_status: str | None = None
    error: str | None = None

    @property
    def group(self) -> str:
        return group_of(self.case)

    @property
    def change_pct(self) -> float | None:
        if self.actual is None or self.golden is None:
            return None
        return (self.actual / self.golden - 1) * 100

    @property
    def retry_change_pct(self) -> float | None:
        if self.retry_actual is None or self.golden is None:
            return None
        return (self.retry_actual / self.golden - 1) * 100


def classify(actual: float, golden: float, spec: MetricSpec, regression_pct: float, improvement_pct: float) -> str:
    # Rounded so a change of exactly the threshold passes despite floating point error.
    worse_pct = round((actual / golden - 1) * 100, 9)
    if spec.better == "higher":
        worse_pct = -worse_pct
    if worse_pct > regression_pct:
        return REGRESSION
    if worse_pct < -improvement_pct:
        return STALE
    return PASS


@dataclass
class Comparison:
    results: list[CaseResult] = field(default_factory=list)
    config_errors: list[str] = field(default_factory=list)


def config_errors(golden: Golden, environment: str, metrics: dict[str, MetricSpec], repetitions: int | None) -> list:
    errors = []
    for name, spec in metrics.items():
        if name in golden.metrics and golden.metrics[name] != spec:
            errors.append(f"metric {name} is now {spec.as_dict()} but the golden has {golden.metrics[name].as_dict()}")
    env = golden.environments.get(environment)
    if env is not None and env.repetitions is not None and repetitions is not None and env.repetitions != repetitions:
        errors.append(f"run used {repetitions} repetitions but the golden was recorded with {env.repetitions}")
    return errors


def compare(
    measurements: dict[str, dict[str, float]],
    golden: Golden,
    environment: str,
    metrics: dict[str, MetricSpec],
    policy: Policy,
    *,
    errors: dict[str, str] | None = None,
    retry: dict[str, dict[str, float]] | None = None,
    case_filter: str | None = None,
    repetitions: int | None = None,
) -> Comparison:
    errors = errors or {}
    retry = retry or {}
    env = golden.environments.get(environment) or Environment()
    expected = {
        (case, metric)
        for case, values in env.cases.items()
        if case_filter is None or re.search(case_filter, case)
        for metric in values
    }
    produced = {(case, metric) for case, values in measurements.items() for metric in values}
    comparison = Comparison(config_errors=config_errors(golden, environment, metrics, repetitions))
    for case, metric in sorted(expected | produced):
        golden_value = env.cases.get(case, {}).get(metric)
        actual = measurements.get(case, {}).get(metric)
        result = CaseResult(case, metric, PASS, actual=actual, golden=golden_value)
        if case in errors:
            result.status, result.error = ERROR, errors[case]
        elif actual is None:
            result.status = MISSING
        elif golden_value is None:
            result.status = NEW
        else:
            regression_pct, improvement_pct = policy.thresholds(case)
            result.status = classify(actual, golden_value, metrics[metric], regression_pct, improvement_pct)
            retry_value = retry.get(case, {}).get(metric)
            if result.status != PASS and retry_value is not None:
                result.retry_actual = retry_value
                retry_status = classify(retry_value, golden_value, metrics[metric], regression_pct, improvement_pct)
                # A case only fails when both runs agree, so a single noisy run cannot fail or update it.
                if retry_status != result.status:
                    result.first_status, result.status = result.status, PASS
        comparison.results.append(result)
    # Errored cases with no golden and no measurement still need a row.
    reported = {r.case for r in comparison.results}
    for case, message in sorted(errors.items()):
        if case not in reported:
            comparison.results.append(CaseResult(case, "-", ERROR, error=message))
    return comparison
