# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# conftest_power.py — pytest fixture for in-process power monitoring
#
# Copy (or symlink) this file into your test directory's conftest.py, or
# add its path to ``conftest.py`` via ``pytest_plugins``.
#
# Usage inside a test:
#
#     def test_matmul(power_monitor):
#         # ... run workload ...
#         pass
#
#     # After the test body finishes the fixture tears down automatically.
#     # The JSON report is available at ``power_monitor.report`` and is also
#     # written to a temp file whose path is ``power_monitor.report_path``.
#
# Device detection, backend selection, and throttling are all handled by
# tt_power_sidecar.py (single source of truth).  This file contains only
# the pytest-specific wiring: the Poller wrapper, result container, and
# fixture itself.

import json
import sys
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

# Import shared backend infrastructure from the sidecar script (same directory).
_SIDECAR_DIR = Path(__file__).resolve().parent
if str(_SIDECAR_DIR) not in sys.path:
    sys.path.insert(0, str(_SIDECAR_DIR))

try:
    from tt_power_sidecar import detect_devices, PowerPoller, compute_report  # noqa: E402
except ImportError as _e:
    raise ImportError(
        "conftest_power.py requires tt_power_sidecar.py in the same directory (%s). "
        "Original error: %s" % (_SIDECAR_DIR, _e)
    )


# Result container and report builder (fixture-specific — not in sidecar)


class PowerMonitorResult:
    """Container returned by the ``power_monitor`` fixture."""

    def __init__(self) -> None:
        self.report: dict[str, Any] | None = None
        self.report_path: Path | None = None


# pytest fixture


@pytest.fixture
def power_monitor(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> Generator[PowerMonitorResult, None, None]:
    """Poll Tenstorrent device power for the duration of the test.

    Yields a ``PowerMonitorResult`` whose ``.report`` is populated after the
    test body completes.  Uses ``backend="auto"`` so N300 reports both the
    local chip (sysfs) and the remote chip (pyluwen, throttled to 1/s).
    """
    interval_s = 0.1  # 100 ms default

    devices = detect_devices(backend="auto")
    result = PowerMonitorResult()

    if not devices:
        # No hardware — yield a no-op result so tests still run in CI.
        result.report = {
            "test_name": request.node.nodeid,
            "duration_s": 0.0,
            "poll_interval_ms": int(interval_s * 1000),
            "devices": {},
        }
        yield result
        return

    poller = PowerPoller(devices, interval_s)
    poller.start()
    wall_start = time.monotonic()

    yield result  # test body runs here

    wall_end = time.monotonic()
    poller.stop()

    report = compute_report(
        command=["<pytest-fixture>", request.node.nodeid],
        exit_code=0,
        wall_start=wall_start,
        wall_end=wall_end,
        poll_interval_ms=int(interval_s * 1000),
        devices=devices,
        poller=poller,
    )
    report["test_name"] = request.node.nodeid
    result.report = report

    # Write JSON report to pytest's tmp directory.
    report_file: Path = tmp_path / "power_report.json"
    report_file.write_text(json.dumps(report, indent=2) + "\n")
    result.report_path = report_file
