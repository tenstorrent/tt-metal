# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Open the device once per module for this op's unit tests.

Applies @pytest.mark.use_module_device to every collected test; the root
`device` fixture is function-scoped and the marker switches it to module scope.
Do not define a local `device` fixture here — it shadows the root one and
disables the marker.
"""
import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(pytest.mark.use_module_device)


# The multi-device fabric fixtures, registered ONCE for the whole directory. Importing them into each test module
# instead registers one session-scoped `_fabric_mesh_cache` per module: the first module's mesh then stays open
# while the next module's cache calls set_fabric_config, which fails with `!devices_still_open` when the files run
# in one session (as the acceptance gate runs them).
from eval.golden_tests.ring_joint_scaled_dot_product_attention.conftest import (  # noqa: E402,F401 — fixtures
    _fabric_mesh_cache,
    mesh_device,
)
