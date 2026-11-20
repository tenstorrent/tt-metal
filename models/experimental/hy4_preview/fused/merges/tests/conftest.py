# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A device test that ends in anything but an assertion (an operation timeout, "device is unrecoverable", a
failed open) stops every later device test of the session: a stuck kernel must not be followed by more programs."""

import pytest

_device_fault = []


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.failed and call.excinfo is not None and not call.excinfo.errisinstance(AssertionError):
        _device_fault.append(f"{item.nodeid}: {call.excinfo.typename}: {str(call.excinfo.value)[:200]}")


def pytest_runtest_setup(item):
    if _device_fault and ("device" in item.fixturenames or "mesh_device" in item.fixturenames):
        pytest.skip(f"an earlier device test did not end cleanly, no further device work: {_device_fault[0]}")
