# The kernel sweep of tools/mock_mm.py (311 bmm cases) and tools/mock_ops.py (86 other ops) run on a CI device, so the
# ELF identity of the merged head against main covers the same kernels as the mock-cluster sweep.
import os
import runpy

import pytest

D = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.timeout(2400)
def test_mock_mm():
    runpy.run_path(os.path.join(D, "mock_mm.py"), run_name="__main__")


@pytest.mark.timeout(2400)
def test_mock_ops():
    runpy.run_path(os.path.join(D, "mock_ops.py"), run_name="__main__")
