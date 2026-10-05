# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Entry point for every registered perf suite. Select one by node id, e.g. ``test_perf[pgm_dispatch]``."""

import pytest

from tests.perf import registry

# Suites are bounded by the CI job timeout; the repository's 300 s per-test default is far too short.
SUITES = [pytest.param(suite, id=name, marks=pytest.mark.timeout(0)) for name, suite in registry.load().items()]


@pytest.mark.parametrize("suite", SUITES)
def test_perf(suite, perf):
    perf.check(suite)
