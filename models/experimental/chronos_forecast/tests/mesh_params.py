# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Mesh shapes for data-parallel Chronos tests.

The conftest ``mesh_device`` fixture skips shapes larger than the machine's
system mesh; ``(1, N)`` opens as a line and 2D shapes may be rotated to fit
(e.g. ``(4, 8)`` on an 8x4 Blackhole Galaxy).
"""

import pytest

DATA_PARALLEL_MESHES = [
    pytest.param(1, id="1chip"),
    pytest.param(4, id="4chip"),
    pytest.param((1, 8), id="8chip"),
    pytest.param((2, 8), id="16chip"),
    pytest.param((4, 8), id="32chip"),
]
MULTI_CHIP_MESHES = DATA_PARALLEL_MESHES[1:]
