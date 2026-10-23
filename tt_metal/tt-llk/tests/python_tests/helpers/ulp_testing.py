# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What the ULP host tests share: asserting a refusal by its typed reason, naming a
parametrized case by it, and one well-formed ``domains`` list."""

import math
from contextlib import contextmanager
from typing import Optional

import pytest

from .ulp_domains import Domain, Problem, TableError

#: A list of the shape the sweep writes for tanh on a bf16 output: gated at 3 steps on
#: both tails, 10 (past bf16's ceiling of 6) on [-1, 1).
TANH_DOMAINS = (
    Domain(-math.inf, -1, 3),
    Domain(-1, 1, 10),
    Domain(1, math.inf, 3),
)


@contextmanager
def refused(problem: Problem):
    """The block refuses something with a :class:`TableError` for *problem*."""
    with pytest.raises(TableError) as caught:  # allow-pytest.raises: host-only test
        yield
    assert caught.value.problem is problem, caught.value


def problem_id(value) -> Optional[str]:
    """A pytest ``ids=`` callable: a :class:`Problem` is named by its member, any other
    parameter is left to pytest's own numbering (which it applies only to ``None``)."""
    return value.name if isinstance(value, Problem) else None
