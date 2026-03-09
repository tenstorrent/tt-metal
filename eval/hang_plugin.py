"""Pytest plugin: skip remaining parametrizations after a hang.

Load with: pytest -p eval.hang_plugin ...

When a parametrized test fails due to a device hang (detected by timeout
keywords in the traceback), all remaining parametrizations of that test
function are skipped. Tests in other functions continue normally.
"""

import pytest

_HANG_PATTERNS = [
    "operation timeout",
    "Operation timed out",
    "TT_METAL_OPERATION_TIMEOUT",
    "Timeout waiting for",
    "dispatch timeout",
]

# Track test function names that have experienced a hang
_hung_functions: set = set()


def _is_hang(report) -> bool:
    """Check if a test failure was caused by a hang/timeout."""
    if report.longrepr:
        text = str(report.longrepr).lower()
        return any(p.lower() in text for p in _HANG_PATTERNS)
    return False


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" and report.failed and _is_hang(report):
        _hung_functions.add(item.originalname)


def pytest_runtest_setup(item):
    if item.originalname in _hung_functions:
        pytest.skip(f"Skipped: previous parametrization of {item.originalname} hung")
