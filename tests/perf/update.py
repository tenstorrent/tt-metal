# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Golden updates. All-or-nothing: anything that needs consent refuses the whole update unless forced."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from tests.perf.compare import ERROR, MISSING, NEW, REGRESSION, STALE, CaseResult, Comparison
from tests.perf.contract import MetricSpec
from tests.perf.golden import Environment, Golden


class UpdateRefused(Exception):
    pass


@dataclass
class UpdatePlan:
    written: list[CaseResult] = field(default_factory=list)
    removed: list[CaseResult] = field(default_factory=list)
    replaced_environment: bool = False

    def empty(self) -> bool:
        return not (self.written or self.removed or self.replaced_environment)


def refuse_in_ci() -> None:
    if os.environ.get("GITHUB_ACTIONS") or os.environ.get("CI"):
        raise UpdateRefused("golden updates are never run in CI; update from a CI run with `--from-run` locally")


def _closest_to_golden(result: CaseResult) -> float:
    """Of the run and its retry, the value nearer the old golden, so one lucky run never sets the baseline."""
    candidates = [v for v in (result.actual, result.retry_actual) if v is not None]
    if result.golden is None:
        return candidates[0]
    return min(candidates, key=lambda v: abs(v - result.golden))


def apply(
    golden: Golden,
    environment: str,
    comparison: Comparison,
    measurements: dict[str, dict[str, float]],
    metrics: dict[str, MetricSpec],
    *,
    repetitions: int | None,
    context: dict[str, str],
    provenance: dict[str, str],
    filtered: bool,
    force: bool,
) -> UpdatePlan:
    """Applies an update to ``golden`` in place and returns what changed. Raises UpdateRefused on refusal."""
    refuse_in_ci()
    by_status: dict[str, list[CaseResult]] = {}
    for result in comparison.results:
        by_status.setdefault(result.status, []).append(result)
    missing = [] if filtered else by_status.get(MISSING, [])

    reasons = []
    if comparison.config_errors:
        reasons += [f"measurement configuration changed: {e}" for e in comparison.config_errors]
    if by_status.get(REGRESSION):
        reasons.append(f"{len(by_status[REGRESSION])} regressed beyond tolerance")
    if by_status.get(NEW):
        reasons.append(f"{len(by_status[NEW])} new cases are not in the golden")
    if missing:
        reasons.append(f"{len(missing)} golden cases were not produced by this unfiltered run")
    if by_status.get(ERROR):
        reasons.append(f"{len(by_status[ERROR])} cases reported benchmark errors")
    if reasons and not force:
        raise UpdateRefused("; ".join(reasons))
    if comparison.config_errors and filtered:
        raise UpdateRefused("a measurement configuration change must be recorded from an unfiltered run")

    plan = UpdatePlan()
    if comparison.config_errors:
        # A different measurement makes every old value incomparable, so the environment is recorded afresh.
        golden.environments[environment] = Environment(
            cases={case: dict(values) for case, values in measurements.items()}
        )
        golden.metrics.update(metrics)
        plan.replaced_environment = True
    else:
        env = golden.environments.setdefault(environment, Environment())
        for name, spec in metrics.items():
            golden.metrics.setdefault(name, spec)
        for result in by_status.get(STALE, []) + (
            by_status.get(REGRESSION, []) + by_status.get(NEW, []) if force else []
        ):
            env.cases.setdefault(result.case, {})[result.metric] = _closest_to_golden(result)
            plan.written.append(result)
        if force:
            for result in missing:
                env.cases.get(result.case, {}).pop(result.metric, None)
                if not env.cases.get(result.case):
                    env.cases.pop(result.case, None)
                plan.removed.append(result)
    if plan.empty():
        return plan
    env = golden.environments[environment]
    env.repetitions = repetitions
    env.context = dict(context)
    env.provenance = dict(provenance)
    return plan
