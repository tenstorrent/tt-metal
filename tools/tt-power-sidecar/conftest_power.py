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
    from tt_power_sidecar import detect_devices, PowerPoller  # noqa: E402
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


def _build_report(
    poller: PowerPoller,
    duration_s: float,
    devices: list[Any],
    test_name: str,
    poll_interval_ms: int,
) -> dict[str, Any]:
    """Build the JSON-serialisable report dict from a completed poller run."""
    device_reports: dict[str, Any] = {}
    for dev in devices:
        # Defensive None filter — poller already drops them, but guard aggregates.
        s = [(ts, w) for ts, w in poller.samples.get(dev.index, []) if w is not None]
        n = len(s)
        if n == 0:
            device_reports[str(dev.index)] = {
                "energy_J": 0.0,
                "energy_Wh": 0.0,
                "avg_power_W": 0.0,
                "peak_power_W": 0.0,
                "min_power_W": 0.0,
                "sample_count": 0,
                "backend": dev.backend_name,
            }
            continue
        powers = [x[1] for x in s]
        energy = 0.0
        for i in range(1, n):
            dt = s[i][0] - s[i - 1][0]
            energy += (s[i][1] + s[i - 1][1]) / 2.0 * dt
        sampling_span = s[-1][0] - s[0][0] if n > 1 else 0.0
        avg = energy / sampling_span if sampling_span > 0 else powers[0]
        device_reports[str(dev.index)] = {
            "energy_J": round(energy, 3),
            "energy_Wh": round(energy / 3600.0, 6),
            "avg_power_W": round(avg, 3),
            "peak_power_W": round(max(powers), 3),
            "min_power_W": round(min(powers), 3),
            "sample_count": n,
            "backend": dev.backend_name,
        }
    return {
        "test_name": test_name,
        "duration_s": round(duration_s, 3),
        "poll_interval_ms": poll_interval_ms,
        "devices": device_reports,
    }


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
        yield result
        return

    poller = PowerPoller(devices, interval_s)
    poller.start()
    wall_start = time.monotonic()

    yield result  # test body runs here

    wall_end = time.monotonic()
    poller.stop()

    report = _build_report(
        poller,
        wall_end - wall_start,
        devices,
        request.node.nodeid,
        int(interval_s * 1000),
    )
    result.report = report

    # Write JSON report to pytest's tmp directory.
    report_file: Path = tmp_path / "power_report.json"
    report_file.write_text(json.dumps(report, indent=2) + "\n")
    result.report_path = report_file
