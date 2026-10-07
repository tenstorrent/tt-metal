# SPDX-License-Identifier: Apache-2.0
import pytest

from models.demos.blackhole.mimo_v26.tests.harness import assert_integ_lib


@pytest.fixture(scope="session", autouse=True)
def _integ_lib_guard():
    """every mimo_v26 test: refuse to run on the shared-build ttnn library (see harness.assert_integ_lib; MIMO_ALLOW_SHARED_LIB=1 overrides)"""
    assert_integ_lib()
    yield
