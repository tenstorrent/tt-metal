# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

from contextlib import contextmanager

import pytest


@pytest.fixture
def expect_error():
    @contextmanager
    def check(error, message):
        try:
            yield
        except error as caught:
            assert message in str(caught)
        else:
            pytest.fail(f"Expected {error.__name__}")

    return check
