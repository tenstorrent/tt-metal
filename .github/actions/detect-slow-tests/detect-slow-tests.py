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
    # Hardcoded list of tests to exclude from slow test detection (Don't use this unless you HAVE TO)
    exceptions = [
        "DispatchFixture.TensixFailOnDuplicateKernelCreationDataflow",
    ]

    # Find all XML files in the report directory
    report_files = [
        os.path.join(root, file) for root, dirs, files in os.walk(report_dir) for file in files if file.endswith(".xml")
    ]
    if not report_files:
        raise NoTestReportsFoundError("No test reports found.")

    excluded_tests = []
    slow_tests = []

    for report_file in report_files:
        try:
            tree = ET.parse(report_file)
            root = tree.getroot()
            for tc in root.findall(".//testcase"):
                time = float(tc.get("time", 0))
                if time > timeout:
                    test_name = f"{tc.get('classname', 'Unknown')}.{tc.get('name', 'Unknown')}"

                    # Check if this test is in the exceptions list
                    if test_name in exceptions:
                        excluded_tests.append(f"{report_file}: {test_name} ({time:.3f}s) - EXCLUDED")
                    else:
                        slow_tests.append(f"{report_file}: {test_name} ({time:.3f}s)")
        except Exception as e:
            raise TestReportParsingError(f"Error parsing {report_file}: {e}")

    # Print excluded tests for visibility
    if excluded_tests:
        print("Excluded slow tests:")
        for test in excluded_tests:
            print(f"  {test}")
        print()

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
