# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A device test that ends in anything but an assertion (an operation timeout, "device is unrecoverable", a failed
open) stops every later device test of the session: a stuck kernel must not be followed by more programs."""

import pytest

from models.experimental.hy4_preview.fused.model import hostwrites
from models.experimental.hy4_preview.fused.moe import prepare

_device_fault = []


@pytest.fixture(autouse=True)
def _host_write_registry_ends_with_the_test():
    """A model installs the process's host-write registry (model/hostwrites.py); it must not outlive the test's
    device, or it registers and keeps the tensors of the tests that follow in the session."""
    yield
    hostwrites.uninstall()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.failed and call.excinfo is not None and not call.excinfo.errisinstance(AssertionError):
        _device_fault.append(f"{item.nodeid}: {call.excinfo.typename}: {str(call.excinfo.value)[:200]}")


def pytest_runtest_setup(item):
    if "mesh_device" in item.fixturenames and not prepare.stream_paths(prepare.DEFAULT_OUT, 0, "tp32", 0)[0].exists():
        pytest.skip(
            f"the converted expert streams are not under {prepare.DEFAULT_OUT} (model/convert.py writes them; "
            "$HY4_MOE_STREAMS names another place)"
        )
    if _device_fault and "device" in item.fixturenames:
        pytest.skip(f"an earlier device test did not end cleanly, no further device work: {_device_fault[0]}")
