# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--dummy_weights",
        action="store",
        default=False,
        type=bool,
        help="Use dummy/random weights instead of loading checkpoints in tests that support it.",
    )


@pytest.fixture
def dummy_weights(request):
    return request.config.getoption("--dummy_weights") or False
