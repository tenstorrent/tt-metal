import argparse
import xmltodict
import glob
import os
from typing import Union


def _guaranteed_list(x):
    if not x:
        return []
    elif isinstance(x, list):
        return x
    else:
        return [x]


def _build_workflow_command(
    command_name: str,
    file: str,
    line: int,
    end_line: Union[int, None] = None,
    column: Union[int, None] = None,
    end_column: Union[int, None] = None,
    title: Union[str, None] = None,
    message: Union[str, None] = None,
):
    result = f"::{command_name} "

    entries = [
        ("file", file),
        ("line", line),
        ("endLine", end_line),
        ("col", column),
        ("endColumn", end_column),
        ("title", title),
    ]

    result = result + ",".join(f"{k}={v}" for k, v in entries if v is not None)

    if message is not None:
        result = result + "::" + _escape(message)

    return result


def _escape(s: str) -> str:
    return s.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


if __name__ == "__main__":
    # Get xml dir path from cmdline
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str, help="Path to the GoogleTest XML directory")
    args = parser.parse_args()

    # Path to the directory containing XML files
    xml_dir = args.directory

    # Use glob to find all XML files in the directory
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))

    # Iterate through each XML file
    for xml_file in xml_files:
        with open(xml_file, "r") as f:
            results = xmltodict.parse(f.read())

        # Check for failed tests
        failed_tests = []
        for testsuite in _guaranteed_list(results["testsuites"]["testsuite"]):
            for testcase in _guaranteed_list(testsuite["testcase"]):
                if "failure" in testcase:
                    failed_tests.append(testcase)

        # Create error annotations for each failed test
        for failed_test in failed_tests:
            failure_messages = _guaranteed_list(failed_test["failure"])
            if failure_messages:
                # first message is often enough
                failure_message = failure_messages[0]["@message"]
            else:
                failure_message = "unknown_failure_message"

            msg = _build_workflow_command(
                command_name="error",
                file=failed_test["@file"].lstrip("/work/"),
                line=int(failed_test["@line"]),
                message=failure_message,
            )
            print(msg)
