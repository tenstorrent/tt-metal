# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.dirname(HERE)
sys.path[:0] = [os.path.dirname(TOOL), HERE]

from hal_explorer import pipeline  # noqa: E402  pylint: disable=wrong-import-position
from hal_explorer.toolchain import (  # noqa: E402  pylint: disable=wrong-import-position
    build,
)

needs_toolchain = pytest.mark.skipif(
    not os.path.isfile(build.compile_flags()[0]),
    reason="SFPI toolchain not installed in tests/sfpi",
)


@pytest.fixture(scope="session")
def tables():
    return pipeline.load_tables()
