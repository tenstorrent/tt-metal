# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""In-tree pytest entry. Hooks live in ``helpers.llk_pytest_plugin`` so the
same harness can be driven from outside this repo — though an out-of-tree suite
should load it as ``tt_llk_harness.plugin``, the supported spelling, rather than
naming the implementation module. See ``docs/tests/getting_started.md`` §9.

``from conftest import skip_for_wormhole`` (and the other arch markers) keeps
working for existing tests.
"""

pytest_plugins = ["helpers.llk_pytest_plugin", "fuser.pytest_plugin"]

import math  # noqa: E402
from contextlib import contextmanager  # noqa: E402

import pytest  # noqa: E402
from helpers.llk_pytest_plugin import (  # noqa: E402
    blackhole_only,
    quasar_only,
    skip_for_blackhole,
    skip_for_coverage,
    skip_for_quasar,
    skip_for_wormhole,
    wormhole_only,
)
from helpers.ulp_domains import Domain, Problem, TableError  # noqa: E402

__all__ = [
    "TANH_DOMAINS",
    "blackhole_only",
    "problem_id",
    "refuses",
    "quasar_only",
    "refused",
    "skip_for_blackhole",
    "skip_for_coverage",
    "skip_for_quasar",
    "skip_for_wormhole",
    "wormhole_only",
]

# A list of the shape the sweep writes for tanh on a bf16 output: gated at 3 steps on
# both tails, 10 (past bf16's ceiling of 6) on [-1, 1).
TANH_DOMAINS = (Domain(-math.inf, -1, 3), Domain(-1, 1, 10), Domain(1, math.inf, 3))


@contextmanager
def refused(problem: Problem):
    """The block raises a ``TableError`` whose problem is *problem*."""
    with pytest.raises(TableError) as caught:  # allow-pytest.raises: host-only test
        yield
    assert caught.value.problem is problem, caught.value


def refuses(match: str):
    """The block raises a ``ValueError`` whose message matches *match*; for the metric
    and the gate, which raise plain ``ValueError``."""
    return pytest.raises(ValueError, match=match)  # allow-pytest.raises: host-only test


def problem_id(value) -> str | None:
    """A parametrize id: the ``Problem`` member's name; ``None`` (pytest's own
    numbering) for other values."""
    return value.name if isinstance(value, Problem) else None
