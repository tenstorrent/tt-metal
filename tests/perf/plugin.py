# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""pytest plugin for runtime perf suites. Load it with ``-p tests.perf.plugin`` (CI also passes --noconftest)."""

from __future__ import annotations

import os
import socket

import pytest

from tests.perf import contract, session, update
from tests.perf.golden import GoldenError
from tests.perf.registry import RegistryError, Suite
from tests.perf.runner import RunError


def pytest_addoption(parser):
    group = parser.getgroup("perf")
    group.addoption(
        "--perf-environment",
        default=os.environ.get("PERF_ENVIRONMENT"),
        help="Golden environment to compare against, e.g. wh_n300_civ2. Defaults to $PERF_ENVIRONMENT.",
    )
    group.addoption("--perf-update", action="store_true", help="Update the golden from this run (never in CI).")
    group.addoption(
        "--force", action="store_true", help="With --perf-update, accept regressions, new and missing cases."
    )
    group.addoption(
        "--perf-filter",
        default=None,
        help="Regex of cases to run, replacing any filter in the suite's args (Google Benchmark suites only).",
    )
    group.addoption("--perf-all", action="store_true", help="List every case in the report, not just failures.")


class Perf:
    def __init__(self, config):
        self.config = config

    def check(self, suite: Suite) -> None:
        option = self.config.getoption
        environment = option("--perf-environment")
        if not environment:
            pytest.fail("set --perf-environment or $PERF_ENVIRONMENT to choose the golden environment")
        case_filter = option("--perf-filter")
        if case_filter is not None and suite.kind != "google_benchmark":
            pytest.fail(f"suite {suite.name} cannot be filtered")
        try:
            if option("--perf-update"):
                update.refuse_in_ci()
            record = session.execute(suite, environment, case_filter)
            golden, comparison = session.evaluate(record, suite)
            session.publish(record, suite, golden, comparison, show_all=option("--perf-all"))
            if option("--perf-update"):
                session.update_golden(
                    record,
                    suite,
                    golden,
                    comparison,
                    force=option("--force"),
                    source=f"local run on {socket.gethostname()}",
                )
                golden, comparison = session.evaluate(record, suite)
        except update.UpdateRefused as refused:
            pytest.fail(f"golden not modified: {refused}. Re-run to rule out noise, or pass --force.", pytrace=False)
        except (RunError, contract.ContractError, GoldenError, RegistryError) as error:
            pytest.fail(str(error), pytrace=False)
        reasons = session.failures(record, suite, golden, comparison)
        if reasons:
            pytest.fail(f"{suite.name}: {', '.join(reasons)}; see the summary above", pytrace=False)


@pytest.fixture
def perf(request) -> Perf:
    return Perf(request.config)
