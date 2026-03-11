# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Tests for hang report JUnit XML generation.

Requires:
  - The hang app binary to be built (build/tools/tests/triage/hang_apps/add_2_integers_hang/triage_hang_app_add_2_integers_hang)
  - tt-exalens to be installed (scripts/install_debugger.sh)
  - A Tenstorrent device available

Run:
  pytest tools/tests/triage/test_hang_report.py -x --timeout 120
"""

import os
import subprocess
from pathlib import Path

import pytest
from defusedxml.ElementTree import parse as XMLParse


METAL_HOME = Path(__file__).resolve().parent.parent.parent.parent
HANG_APP = METAL_HOME / "build" / "tools" / "tests" / "triage" / "hang_apps" / "add_2_integers_hang" / "triage_hang_app_add_2_integers_hang"
TRIAGE_SCRIPT = METAL_HOME / "tools" / "tt-triage.py"
REPORT_DIR = METAL_HOME / "generated" / "test_reports"


@pytest.mark.skipif(not HANG_APP.exists(), reason="Hang app binary not built")
@pytest.mark.skipif(not TRIAGE_SCRIPT.exists(), reason="tt-triage.py not found")
def test_hang_generates_junit_xml():
    """Run the hang app with auto-triage and verify a hang report XML is generated."""
    # Record existing hang reports so we only check newly created ones
    existing = set(REPORT_DIR.glob("hang_report_*.xml")) if REPORT_DIR.exists() else set()

    env = {
        **os.environ,
        "TT_METAL_OPERATION_TIMEOUT_SECONDS": "0.5",
        "TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE": (
            f"python3 {TRIAGE_SCRIPT} --disable-progress --skip-version-check 1>&2"
        ),
        "PYTEST_CURRENT_TEST": "fake/test_hang.py::test_hung_operation[device0] (call)",
    }

    result = subprocess.run(
        [str(HANG_APP)],
        cwd=str(METAL_HOME),
        env=env,
        capture_output=True,
        timeout=90,
    )

    assert result.returncode == 0, f"Hang app failed: stderr={result.stderr.decode()[-500:]}"

    # Find newly created hang report XML
    current = set(REPORT_DIR.glob("hang_report_*.xml"))
    new_files = sorted(current - existing)
    assert len(new_files) >= 1, f"Expected a new hang report XML in {REPORT_DIR}, found none"
    xml_path = new_files[0]

    try:
        tree = XMLParse(str(xml_path))
        root = tree.getroot()
        assert root.tag == "testsuites"

        testsuite = root.find("testsuite")
        assert testsuite is not None
        assert testsuite.get("failures") == "1"

        testcase = testsuite.find("testcase")
        assert testcase is not None
        assert testcase.get("classname") == "fake.test_hang"
        assert testcase.get("name") == "test_hung_operation[device0]"

        props = {p.get("name"): p.get("value") for p in testcase.findall("properties/property")}
        assert props.get("failure_type") == "hang", f"Expected failure_type=hang, got properties: {props}"
        assert "start_timestamp" in props
        assert "end_timestamp" in props

        failure = testcase.find("failure")
        assert failure is not None
        assert "[HANG DETECTED]" in failure.get("message", "")
        assert "Triage summary" in (failure.text or "")
    finally:
        xml_path.unlink(missing_ok=True)
