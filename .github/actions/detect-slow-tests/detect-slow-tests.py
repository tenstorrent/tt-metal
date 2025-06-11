import os
import xml.etree.ElementTree as ET
import sys


class SlowTestDetectionError(Exception):
    pass


class NoTestReportsFoundError(SlowTestDetectionError):
    pass


class TestReportParsingError(SlowTestDetectionError):
    pass


class SlowTestsExceededError(SlowTestDetectionError):
    pass


def detect_slow_tests(report_dir, timeout):
    # Find all XML files in the report directory
    report_files = [
        os.path.join(root, file) for root, dirs, files in os.walk(report_dir) for file in files if file.endswith(".xml")
    ]
    if not report_files:
        raise NoTestReportsFoundError("No test reports found.")

    slow_tests = []

    for report_file in report_files:
        try:
            tree = ET.parse(report_file)
            root = tree.getroot()
            for tc in root.findall(".//testcase"):
                time = float(tc.get("time", 0))
                if time > timeout:
                    slow_tests.append(
                        f"{report_file}: {tc.get('classname', 'Unknown')}.{tc.get('name', 'Unknown')} ({time:.3f}s)"
                    )
        except Exception as e:
            raise TestReportParsingError(f"Error parsing {report_file}: {e}")

    if slow_tests:
        raise SlowTestsExceededError(f"Some tests exceeded {timeout}s:\n" + "\n".join(slow_tests))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python detect-slow-tests.py <report_dir> <timeout>")
        sys.exit(1)

    report_dir = sys.argv[1]
    timeout = float(sys.argv[2])

    try:
        detect_slow_tests(report_dir, timeout)
    except SlowTestDetectionError as e:
        print(e)
        sys.exit(1)
    print("No slow tests slower than {timeout}s detected.")
