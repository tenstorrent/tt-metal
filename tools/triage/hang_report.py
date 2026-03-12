# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""Generate a JUnit XML test report for a hung test so it appears in CI artifacts.

When tt-triage is invoked during a dispatch timeout, this module writes a JUnit XML
file that marks the currently-running pytest test as failed with failure_type=hang.
This guarantees the hung test appears in test reports even if the process is later
killed and pytest never finalizes its own XML.
"""

import os
from datetime import datetime, timezone
from html import escape
from textwrap import dedent

from triage import utils

REPORT_DIR = "generated/test_reports"


def write_hang_junit_xml(triage_summary: str) -> None:
    """Write a JUnit XML report for the hung test identified by PYTEST_CURRENT_TEST.

    No-op if PYTEST_CURRENT_TEST is not set (i.e. not running under pytest).
    """
    current_test = os.environ.get("PYTEST_CURRENT_TEST")
    if not current_test:
        return

    test_id = current_test.rsplit(" (", 1)[0]
    filepath, sep, test_name = test_id.partition("::")
    if not sep:
        test_name = "unknown"

    classname = filepath.replace("/", ".").removesuffix(".py")
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    hostname = os.environ.get("HOSTNAME", "unknown")

    failure_message = "[HANG DETECTED] Card hang detected during test execution. tt-triage was invoked."

    xml_content = dedent("""\
        <?xml version="1.0" encoding="utf-8"?>
        <testsuites name="pytest tests">
          <testsuite name="pytest" errors="0" failures="1" skipped="0" tests="1" time="0" timestamp="{timestamp}" hostname="{hostname}">
            <testcase classname="{classname}" name="{test_name}" time="0">
              <properties>
                <property name="failure_type" value="hang"/>
                <property name="start_timestamp" value="{timestamp}"/>
                <property name="end_timestamp" value="{timestamp}"/>
              </properties>
              <failure message="{failure_message}">{failure_message}

        Triage summary:
        {triage_summary}
              </failure>
            </testcase>
          </testsuite>
        </testsuites>""").format(
        timestamp=timestamp,
        hostname=escape(hostname),
        classname=escape(classname),
        test_name=escape(test_name),
        failure_message=escape(failure_message),
        triage_summary=escape(triage_summary),
    )

    try:
        os.makedirs(REPORT_DIR, exist_ok=True)
        report_path = os.path.join(REPORT_DIR, f"hang_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xml")
        with open(report_path, "w") as f:
            f.write(xml_content)
        utils.INFO(f"Hang report written to {report_path}")
    except Exception as e:
        utils.WARN(f"Failed to write hang JUnit XML report: {e}")
