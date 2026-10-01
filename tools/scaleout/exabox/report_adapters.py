#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Map per-test analyzer / script exit codes to cluster health ``status``.

Analyzer codes are **not** interchangeable across ``test_type``. Each mapping
follows that test's analyze script (or recover shell RC). This module does not
read logs or files.
"""

from __future__ import annotations

from cluster_health_schema import TEST_TYPES

# ---------------------------------------------------------------------------
# physical — analyze_validation_results.py
#  0 healthy (≥80% success)
#  1 unhealthy links (repeated on same link)
#  2 unhealthy links (scattered)
#  3 DRAM training failures
#  4 missing connections (FSD vs discovered)
#  5 extra connections
#  6 missing global connection
#  7 FSD error
#  8 MGD error
#  9 workload timeout
# 10 ARC timeout
# 11 AICLK timeout
# 12 network errors (MPI/SSH)
# 13 device init error
# 50 inconclusive
# 66 input error
# Any non-zero (including 50 and 66) → failed.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# fabric — analyze_fabric_results.py (different table than physical)
#  0 all tests passed
#  1 MGD error (topology mismatch)
#  2 firmware initialization failed
#  3 fabric router sync timeout
#  4 test hanging (incomplete log)
#  5 NOC address conflict
#  6 Ethernet core timeout
# 50 inconclusive (manual review) → degraded
# 66 input error (log file not found) → skipped
# 1–6 and any other non-zero → failed
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# dispatch — analyze_dispatch_results.py
#  0 passed (no failed tests and at least one passed)
#  1 failed
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# recover — recover.sh success/fail (not an analyze script)
#  0 or None → passed; any other int → failed
# The health record omits analyzer_code for recover.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# host — analyze_host_health_results.py (diag_report.json overall_status)
#  0 PASS → passed
#  1 FAIL → failed
#  2 WARN → degraded
# Any other non-zero → failed.
# Do not use the diag process RC (0 for both PASS and WARN).
# ---------------------------------------------------------------------------

_FABRIC_INCONCLUSIVE = 50
_FABRIC_INPUT_ERROR = 66
_HOST_WARN = 2

_PHYSICAL_REASONS = {
    1: "Unhealthy links (repeated - same links failing)",
    2: "Unhealthy links (scattered - failures across different links)",
    3: "DRAM training failures",
    4: "Missing connections",
    5: "Extra connections",
    6: "Missing global connection",
    7: "FSD configuration error",
    8: "MGD topology mismatch",
    9: "Workload timeout",
    10: "ARC timeout",
    11: "AICLK timeout",
    12: "Network error",
    13: "Device init error (missing devices)",
    50: "Inconclusive - unrecognized errors",
    66: "No log files found to analyze",
}

_FABRIC_REASONS = {
    1: "MGD error (topology mismatch)",
    2: "Firmware initialization failed",
    3: "Fabric router sync timeout",
    4: "Test hanging (incomplete log)",
    5: "NOC address conflict",
    6: "Ethernet core timeout",
    _FABRIC_INCONCLUSIVE: "Inconclusive (manual review required)",
    _FABRIC_INPUT_ERROR: "Input error (log file not found)",
}


def status_for(test_type: str, analyzer_code: int | None) -> str:
    """Return a schema ``status`` for ``test_type`` and that test's exit code.

    Raises ValueError if ``test_type`` is unknown.
    """
    if test_type not in TEST_TYPES:
        raise ValueError(f"test_type: must be one of {sorted(TEST_TYPES)}")

    if test_type == "physical":
        if analyzer_code is None:
            raise ValueError("analyzer_code: required for physical")
        return "passed" if analyzer_code == 0 else "failed"

    if test_type == "fabric":
        if analyzer_code is None:
            raise ValueError("analyzer_code: required for fabric")
        if analyzer_code == 0:
            return "passed"
        if analyzer_code == _FABRIC_INCONCLUSIVE:
            return "degraded"
        if analyzer_code == _FABRIC_INPUT_ERROR:
            return "skipped"
        return "failed"

    if test_type == "dispatch":
        if analyzer_code is None:
            raise ValueError("analyzer_code: required for dispatch")
        return "passed" if analyzer_code == 0 else "failed"

    if test_type == "host":
        if analyzer_code is None:
            raise ValueError("analyzer_code: required for host")
        if analyzer_code == 0:
            return "passed"
        if analyzer_code == _HOST_WARN:
            return "degraded"
        return "failed"

    # recover
    if analyzer_code is None or analyzer_code == 0:
        return "passed"
    return "failed"


def reason_for(test_type: str, analyzer_code: int | None) -> str:
    """Return a concise non-passing reason for this test result.

    The analyzer code namespace is local to ``test_type``. Passed results have
    no reason. Unknown non-zero codes retain the code for operator triage.
    """
    if test_type not in TEST_TYPES:
        raise ValueError(f"test_type: must be one of {sorted(TEST_TYPES)}")
    if analyzer_code is None or analyzer_code == 0:
        return ""

    if test_type == "physical":
        return _PHYSICAL_REASONS.get(analyzer_code, f"Unknown physical analysis error (rc={analyzer_code})")
    if test_type == "fabric":
        return _FABRIC_REASONS.get(analyzer_code, f"Unknown fabric analysis error (rc={analyzer_code})")
    if test_type == "dispatch":
        if analyzer_code == 1:
            return "One or more dispatch tests failed"
        if analyzer_code == 66:
            return "No dispatch test log file found"
        return f"Dispatch failed (rc={analyzer_code})"
    if test_type == "host":
        if analyzer_code == _HOST_WARN:
            return "Diagnostic warning"
        if analyzer_code == 1:
            return "Diagnostic failed"
        return f"Diagnostic failed (code={analyzer_code})"

    return f"Recover failed (rc={analyzer_code})"
