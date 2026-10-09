# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The pytest entry point, with one test per suite in suites.yaml.

CI runs a single suite by node id, e.g. ``tests/perf/test_suites.py::test_perf[pgm_dispatch]``.
"""

import pytest

from tests.perf import registry

# Suites are bounded by the CI job timeout; the repository's 300 s per-test default is far too short.
SUITES = [pytest.param(suite, id=name, marks=pytest.mark.timeout(0)) for name, suite in registry.load().items()]


@pytest.mark.parametrize("suite", SUITES)
def test_perf(suite, perf):
    perf.check(suite)
