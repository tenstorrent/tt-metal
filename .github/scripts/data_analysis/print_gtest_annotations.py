import argparse
import xml.etree.ElementTree as ET
import glob
import os
from typing import Union


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
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for testsuite in root.findall("testsuite"):
            # Skip printing out pytest information
            # It's already handled by pytest-github-actions-annotate-failures plugin
            if testsuite.attrib.get("name") == "pytest":
                continue
            for testcase in testsuite.findall("testcase"):
                failure = testcase.find("failure")
                # If failure exists, print the failure message
                if failure is not None:
                    failure_message = failure.attrib.get("message")
                    msg = _build_workflow_command(
                        command_name="error",
                        file=testcase.attrib.get("file", "").lstrip("/work/"),
                        line=int(testcase.attrib["line"]),
                        message=failure_message,
                    )
                    print(msg)
