# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Open the device once per module for this op's unit tests.

Applies @pytest.mark.use_module_device to every collected test; the root
`device` fixture is function-scoped and the marker switches it to module scope.
Do not define a local `device` fixture here — it shadows the root one and
disables the marker.
"""
import pytest

# The golden suite's ring mesh fixtures, defined once here so every module of this directory shares ONE session
# cache (importing them per module creates one cache per module, and the second re-opens the fabric under the first).
from eval.golden_tests.all_to_all_matmul.conftest import _fabric_mesh_cache, mesh_device  # noqa: F401


def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(pytest.mark.use_module_device)
