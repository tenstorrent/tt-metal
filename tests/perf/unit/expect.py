# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The repository's expect_error fixture, for perf unit tests that run with --noconftest."""

import contextlib

import pytest


@pytest.fixture
def expect_error():
    @contextlib.contextmanager
    def expect_error_(error, message=None):
        with pytest.raises(error, match=message) as exc_info:  # allow-pytest.raises: fixture implementation
            yield exc_info

    return expect_error_
