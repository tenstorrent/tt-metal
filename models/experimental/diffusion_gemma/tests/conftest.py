# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import re
from contextlib import contextmanager


@contextmanager
def _expect_error(expected_exception, match=None):
    try:
        yield
    except expected_exception as exc:
        if match is not None and re.search(match, str(exc)) is None:
            raise AssertionError(f"Exception message did not match {match!r}: {exc}") from exc
    else:
        raise AssertionError(f"Expected {expected_exception} to be raised")


@pytest.fixture
def expect_error():
    return _expect_error
