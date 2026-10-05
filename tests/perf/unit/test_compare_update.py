# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.perf import compare as cmp
from tests.perf import update
from tests.perf.contract import MetricSpec
from tests.perf.golden import Environment, Golden
from tests.perf.unit.expect import expect_error  # noqa: F401

LOWER = {"t": MetricSpec("s", "lower", "min")}
HIGHER = {"bw": MetricSpec("B/s", "higher", "median")}
POLICY = cmp.Policy(regression_pct=5, improvement_pct=5)
ENV = "wh"


@pytest.fixture(autouse=True)
def not_in_ci(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)


def golden_of(cases, metrics=LOWER, repetitions=2):
    return Golden("s", dict(metrics), {ENV: Environment(repetitions=repetitions, cases=cases)})


def statuses(comparison):
    return {r.case: r.status for r in comparison.results}


@pytest.mark.parametrize("actual, status", [(1.05, "PASS"), (1.0501, "REGRESSION"), (0.95, "PASS"), (0.9499, "STALE")])
def test_lower_is_better_boundaries(actual, status):
    result = cmp.compare({"a": {"t": actual}}, golden_of({"a": {"t": 1.0}}), ENV, LOWER, POLICY)
    assert statuses(result) == {"a": status}


def test_higher_is_better_inverts_direction():
    golden = golden_of({"a": {"bw": 100.0}, "b": {"bw": 100.0}}, HIGHER)
    result = cmp.compare({"a": {"bw": 94.0}, "b": {"bw": 106.0}}, golden, ENV, HIGHER, POLICY)
    assert statuses(result) == {"a": "REGRESSION", "b": "STALE"}


def test_override_thresholds_apply_by_pattern():
    policy = cmp.Policy(5, 5, overrides=(("page_size:32/", {"regression_pct": 15}),))
    golden = golden_of({"Read/page_size:32/x": {"t": 1.0}, "Read/page_size:64/x": {"t": 1.0}})
    result = cmp.compare(
        {"Read/page_size:32/x": {"t": 1.1}, "Read/page_size:64/x": {"t": 1.1}}, golden, ENV, LOWER, policy
    )
    assert statuses(result) == {"Read/page_size:32/x": "PASS", "Read/page_size:64/x": "REGRESSION"}


def test_retry_must_agree_for_a_case_to_fail():
    golden = golden_of({"flaky": {"t": 1.0}, "real": {"t": 1.0}})
    result = cmp.compare(
        {"flaky": {"t": 1.2}, "real": {"t": 1.2}},
        golden,
        ENV,
        LOWER,
        POLICY,
        retry={"flaky": {"t": 1.01}, "real": {"t": 1.19}},
    )
    by_case = {r.case: r for r in result.results}
    assert (by_case["flaky"].status, by_case["flaky"].first_status) == ("PASS", "REGRESSION")
    assert by_case["real"].status == "REGRESSION"


def test_missing_new_error_and_filter():
    golden = golden_of({"kept": {"t": 1.0}, "gone": {"t": 1.0}, "other/x": {"t": 1.0}, "broken": {"t": 1.0}})
    result = cmp.compare(
        {"kept": {"t": 1.0}, "added": {"t": 1.0}},
        golden,
        ENV,
        LOWER,
        POLICY,
        errors={"broken": "TT_FATAL"},
        case_filter="^(kept|gone|added|broken)$",
    )
    assert statuses(result) == {"kept": "PASS", "gone": "MISSING", "added": "NEW", "broken": "ERROR"}


def test_measurement_config_mismatch_is_reported():
    golden = golden_of({"a": {"t": 1.0}}, {"t": MetricSpec("s", "lower", "median")}, repetitions=3)
    result = cmp.compare({"a": {"t": 1.0}}, golden, ENV, LOWER, POLICY, repetitions=2)
    assert len(result.config_errors) == 2


def apply(golden, measurements, *, retry=None, filtered=False, force=False, errors=None, repetitions=2):
    comparison = cmp.compare(
        measurements,
        golden,
        ENV,
        LOWER,
        POLICY,
        retry=retry,
        errors=errors,
        repetitions=repetitions,
        case_filter="^(a|b|c|new)$" if filtered else None,
    )
    return update.apply(
        golden,
        ENV,
        comparison,
        measurements,
        LOWER,
        repetitions=repetitions,
        context={},
        provenance={"commit": "abc"},
        filtered=filtered,
        force=force,
    )


def test_update_writes_only_out_of_band_improvements():
    golden = golden_of({"a": {"t": 1.0}, "b": {"t": 1.0}, "c": {"t": 1.0}})
    plan = apply(golden, {"a": {"t": 0.80}, "b": {"t": 1.04}, "c": {"t": 0.97}}, retry={"a": {"t": 0.85}})
    assert golden.environments[ENV].cases == {"a": {"t": 0.85}, "b": {"t": 1.0}, "c": {"t": 1.0}}
    assert [r.case for r in plan.written] == ["a"]
    assert golden.environments[ENV].provenance == {"commit": "abc"}


def test_update_refuses_everything_when_any_case_regressed(expect_error):
    golden = golden_of({"a": {"t": 1.0}, "b": {"t": 1.0}})
    with expect_error(update.UpdateRefused, "regressed"):
        apply(golden, {"a": {"t": 0.5}, "b": {"t": 2.0}})
    assert golden.environments[ENV].cases == {"a": {"t": 1.0}, "b": {"t": 1.0}}
    apply(golden, {"a": {"t": 0.5}, "b": {"t": 2.0}}, force=True)
    assert golden.environments[ENV].cases == {"a": {"t": 0.5}, "b": {"t": 2.0}}


@pytest.mark.parametrize(
    "measurements, errors, message",
    [
        ({"a": {"t": 1.0}, "b": {"t": 1.0}, "new": {"t": 1.0}}, None, "new cases"),
        ({"a": {"t": 1.0}}, None, "not produced"),
        ({"a": {"t": 1.0}, "b": {"t": 1.0}}, {"c": "boom"}, "benchmark errors"),
    ],
)
def test_new_missing_and_errored_cases_need_force(expect_error, measurements, errors, message):
    golden = golden_of({"a": {"t": 1.0}, "b": {"t": 1.0}})
    with expect_error(update.UpdateRefused, message):
        apply(golden, measurements, errors=errors)


def test_force_adds_new_and_prunes_missing_on_unfiltered_runs():
    golden = golden_of({"a": {"t": 1.0}, "b": {"t": 1.0}})
    apply(golden, {"a": {"t": 1.0}, "new": {"t": 3.0}}, force=True)
    assert golden.environments[ENV].cases == {"a": {"t": 1.0}, "new": {"t": 3.0}}


def test_filtered_run_leaves_unselected_and_unproduced_cases_alone():
    golden = golden_of({"a": {"t": 1.0}, "b": {"t": 1.0}, "z": {"t": 1.0}})
    apply(golden, {"a": {"t": 0.5}}, filtered=True)
    assert golden.environments[ENV].cases == {"a": {"t": 0.5}, "b": {"t": 1.0}, "z": {"t": 1.0}}


def test_config_change_rerecords_environment_only_with_force(expect_error):
    golden = golden_of({"a": {"t": 1.0}, "b": {"t": 1.0}}, repetitions=3)
    with expect_error(update.UpdateRefused, "configuration changed"):
        apply(golden, {"a": {"t": 2.0}})
    apply(golden, {"a": {"t": 2.0}}, force=True)
    assert golden.environments[ENV].cases == {"a": {"t": 2.0}}
    assert golden.environments[ENV].repetitions == 2


def test_new_environment_needs_force(expect_error):
    golden = Golden("s")
    with expect_error(update.UpdateRefused, "new cases"):
        apply(golden, {"a": {"t": 1.0}})
    apply(golden, {"a": {"t": 1.0}}, force=True)
    assert golden.metrics == LOWER and golden.environments[ENV].cases == {"a": {"t": 1.0}}


def test_update_is_refused_in_ci(expect_error, monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    with expect_error(update.UpdateRefused, "never run in CI"):
        apply(golden_of({"a": {"t": 1.0}}), {"a": {"t": 0.5}}, force=True)
