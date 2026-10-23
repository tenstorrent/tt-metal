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


# Fabric mesh fixtures, registered ONCE for the whole directory. Importing the golden session fixture
# `_fabric_mesh_cache` into each test module instead gives every module its own session FixtureDef (its own
# cache), and the second module then re-sets the fabric config while the first module's mesh is still open
# (TT_FATAL) when the directory runs in one pytest session.
from eval.golden_tests.distributed_rms_norm.conftest import _fabric_mesh_cache, mesh_device  # noqa: E402,F401
