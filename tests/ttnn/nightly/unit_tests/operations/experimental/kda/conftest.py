# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.ttnn.nightly.unit_tests.operations.experimental.kda.kda_op_checklist import (
    GROUPS,
    SHAPES,
    binding_of,
    item_for,
)


def pytest_configure(config):
    for name, meaning in {**GROUPS, **SHAPES}.items():
        config.addinivalue_line("markers", f"{name}: KDA op test, {meaning} (README.md)")


@pytest.hookimpl(tryfirst=True)  # before -m deselects, so selection sees these markers
def pytest_collection_modifyitems(config, items):
    """Mark every test of a module under the standard (one that declares NOT_APPLICABLE) with its checklist group,
    and with ``host``, ``unit`` or ``production``. Names alone decide the group, so marks and names cannot disagree."""
    for item in items:
        module = getattr(item, "module", None)
        if module is None or not hasattr(module, "NOT_APPLICABLE"):
            continue
        checklist_item = item_for(item.originalname, binding_of(module.__name__))
        if checklist_item is not None:
            item.add_marker(checklist_item.group)
        if not {"device", "mesh_device"} & set(item.fixturenames):
            item.add_marker("host")
        elif item.get_closest_marker("production") is None:
            item.add_marker("unit")


@pytest.fixture
def isolated_program_cache(device):
    # Setup: give the test an enabled, empty program cache.
    device.disable_and_clear_program_cache()
    device.enable_program_cache()

    # Yield control to pytest to execute the test.
    yield

    # Teardown: remove programs created by the test and restore the enabled, empty state.
    device.disable_and_clear_program_cache()
    device.enable_program_cache()


@pytest.fixture
def zero_actual_start(device):
    from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start

    return make_actual_start(device, 0)
