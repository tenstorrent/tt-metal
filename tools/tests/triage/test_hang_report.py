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
import tempfile
from pathlib import Path

import pytest
from defusedxml.ElementTree import parse as XMLParse


METAL_HOME = Path(__file__).resolve().parent.parent.parent.parent
HANG_APP = METAL_HOME / "build" / "tools" / "tests" / "triage" / "hang_apps" / "add_2_integers_hang" / "triage_hang_app_add_2_integers_hang"
TRIAGE_SCRIPT = METAL_HOME / "tools" / "tt-triage.py"


@pytest.fixture
def report_dir(tmp_path):
    """Provide a temporary directory for generated test reports."""
    d = tmp_path / "generated" / "test_reports"
    d.mkdir(parents=True)
    return d


@pytest.mark.skipif(not HANG_APP.exists(), reason="Hang app binary not built")
@pytest.mark.skipif(not TRIAGE_SCRIPT.exists(), reason="tt-triage.py not found")
def test_hang_generates_junit_xml(report_dir):
    """Run the hang app with auto-triage and verify a hang report XML is generated."""
    env = {
        **os.environ,
        "TT_METAL_OPERATION_TIMEOUT_SECONDS": "0.5",
        "TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE": (
            f"python3 {TRIAGE_SCRIPT} --disable-progress --skip-version-check 1>&2"
        ),
        "PYTEST_CURRENT_TEST": "fake/test_hang.py::test_hung_operation[device0] (call)",
    }

    # Patch triage.py to write reports to our temp dir by setting cwd
    # The hang app catches the timeout and exits cleanly
    result = subprocess.run(
        [str(HANG_APP)],
        cwd=str(report_dir.parent.parent),
        env=env,
        capture_output=True,
        timeout=90,
    )

    # The hang app exits 0 after catching the timeout
    assert result.returncode == 0, f"Hang app failed: stderr={result.stderr.decode()[-500:]}"

    # Verify a hang report XML was generated
    xml_files = list(report_dir.glob("hang_report_*.xml"))
    assert len(xml_files) == 1, f"Expected 1 hang report XML, found {len(xml_files)} in {report_dir}: {list(report_dir.iterdir())}"

    # Parse and validate the XML
    tree = XMLParse(str(xml_files[0]))
    root = tree.getroot()
    assert root.tag == "testsuites"

    testsuite = root.find("testsuite")
    assert testsuite is not None
    assert testsuite.get("failures") == "1"

    testcase = testsuite.find("testcase")
    assert testcase is not None
    assert testcase.get("classname") == "fake.test_hang"
    assert testcase.get("name") == "test_hung_operation[device0]"

    # Verify failure_type=hang property
    props = {p.get("name"): p.get("value") for p in testcase.findall("properties/property")}
    assert props.get("failure_type") == "hang", f"Expected failure_type=hang, got properties: {props}"
    assert "start_timestamp" in props
    assert "end_timestamp" in props

    # Verify failure message
    failure = testcase.find("failure")
    assert failure is not None
    assert "[HANG DETECTED]" in failure.get("message", "")
    assert "Triage summary" in (failure.text or "")
