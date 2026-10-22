# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What the ULP host tests share: asserting a refusal by its typed reason."""

from contextlib import contextmanager

import pytest

from .ulp_domains import Problem, TableError


@contextmanager
def refused(problem: Problem):
    """The block refuses something with a :class:`TableError` for *problem*."""
    with pytest.raises(TableError) as caught:  # allow-pytest.raises: host-only test
        yield
    assert caught.value.problem is problem, caught.value
