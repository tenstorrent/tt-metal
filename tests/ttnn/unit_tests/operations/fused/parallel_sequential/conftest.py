# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Stress-test hook for #41622: after each test passes normally, re-run it 2x
with program cache disabled.  The fusion build cache persists (same device
within a single test invocation), so re-runs hit the cache with stale
CBDescriptor.buffer pointers and runtime arg addresses — exercising the
patch_stale_descriptor path in patchable_generic_op.
"""

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    # Let the test run normally first.
    outcome = yield
    if outcome.excinfo is not None:
        return  # failed on the normal run; skip stress re-runs

    device = item.funcargs.get("device")
    if device is None:
        return

    device.disable_and_clear_program_cache()
    try:
        for _ in range(2):
            item.runtest()
    finally:
        device.enable_program_cache()
