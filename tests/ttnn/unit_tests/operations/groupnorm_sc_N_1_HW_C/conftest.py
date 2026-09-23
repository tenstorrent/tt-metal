# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Open the device once per module for this op's unit tests.

Applies @pytest.mark.use_module_device to every test collected from this directory; the root
`device` fixture is function-scoped and the marker switches it to module scope.
Do not define a local `device` fixture here — it shadows the root one and
disables the marker.

pytest_collection_modifyitems sees every item of the session, not only this directory's, so the
marker is applied by path; marking other directories' tests would break the ones that parametrize
device_params.
"""
import pathlib

import pytest

_HERE = pathlib.Path(__file__).resolve().parent


def pytest_collection_modifyitems(items):
    for item in items:
        if _HERE in pathlib.Path(item.path).resolve().parents:
            item.add_marker(pytest.mark.use_module_device)
