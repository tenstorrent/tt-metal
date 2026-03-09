"""Classify pytest failures from JUnit XML into categories.

Categories (checked in priority order):
  hang        - operation timeout / dispatch timeout
  OOM         - L1 or DRAM allocation failure
  compilation - kernel build or link failure
  numerical   - allclose / PCC / tolerance mismatch
  other       - anything else
"""

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

# Classification patterns, checked in priority order (first match wins)
PATTERNS = [
    (
        "hang",
        [
            r"[Oo]peration timeout",
            r"Operation timed out",
            r"TT_METAL_OPERATION_TIMEOUT",
            r"Timeout waiting for",
            r"[Dd]ispatch timeout",
        ],
    ),
    (
        "OOM",
        [
            r"Out of Memory",
            r"out of memory",
            r"Statically allocated circular buffers",
            r"L1 allocation",
            r"not enough space",
            r"DRAM allocation",
            r"\bOOM\b",
            r"Cannot allocate",
        ],
    ),
    (
        "compilation",
        [
            r"CompilationError",
            r"compilation error",
            r"kernel build failed",
            r"CQ Compile",
            r"linking failed",
            r"compile_program_with_kernel",
        ],
    ),
    (
        "numerical",
        [
            r"allclose",
            r"\bPCC\b",
            r"[Nn]umerical [Mm]ismatch",
            r"max_diff=",
            r"mean_diff=",
            r"atol=",
            r"rtol=",
        ],
    ),
]


def classify(traceback_text: str) -> str:
    """Classify a failure traceback into a category."""
    for category, patterns in PATTERNS:
        for pattern in patterns:
            if re.search(pattern, traceback_text):
                return category
    return "other"


def extract_shape(test_name: str) -> Optional[str]:
    """Extract shape info from a parametrized test name.

    Examples:
        "test_foo[minimal_1x1x32x32]" -> "minimal_1x1x32x32"
        "test_foo[w512]" -> "w512"
        "test_foo[b2c3_32x32]" -> "b2c3_32x32"
    """
    match = re.search(r"\[(.+)\]$", test_name)
    return match.group(1) if match else None


def parse_junit_xml(xml_path: Path) -> list:
    """Parse a JUnit XML file and classify each test result.

    Returns a list of dicts with keys:
        test_name, test_file, shape, status, failure_category, failure_message
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    results = []

    for tc in root.iter("testcase"):
        name = tc.get("name", "")
        classname = tc.get("classname", "")

        # Extract test file from classname (e.g. "test_golden_shapes" from
        # "eval.golden_tests.layer_norm_rm.test_golden_shapes")
        test_file = classname.rsplit(".", 1)[0] if "." in classname else classname

        shape = extract_shape(name)

        failure = tc.find("failure")
        error = tc.find("error")
        skipped = tc.find("skipped")

        if failure is not None:
            message = failure.get("message", "")
            traceback = failure.text or ""
            full_text = f"{message}\n{traceback}"
            category = classify(full_text)
            results.append(
                {
                    "test_name": name,
                    "test_file": test_file,
                    "shape": shape,
                    "status": "failed",
                    "failure_category": category,
                    "failure_message": full_text[:500],
                }
            )
        elif error is not None:
            message = error.get("message", "")
            traceback = error.text or ""
            full_text = f"{message}\n{traceback}"
            category = classify(full_text)
            results.append(
                {
                    "test_name": name,
                    "test_file": test_file,
                    "shape": shape,
                    "status": "error",
                    "failure_category": category,
                    "failure_message": full_text[:500],
                }
            )
        elif skipped is not None:
            message = skipped.get("message", "")
            category = "hang" if "hung" in message.lower() else None
            results.append(
                {
                    "test_name": name,
                    "test_file": test_file,
                    "shape": shape,
                    "status": "skipped",
                    "failure_category": category,
                    "failure_message": message[:500] if message else None,
                }
            )
        else:
            results.append(
                {
                    "test_name": name,
                    "test_file": test_file,
                    "shape": shape,
                    "status": "passed",
                    "failure_category": None,
                    "failure_message": None,
                }
            )

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Classify pytest failures from JUnit XML")
    parser.add_argument("xml_path", help="Path to JUnit XML file")
    parser.add_argument("--output", "-o", help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    results = parse_junit_xml(Path(args.xml_path))

    output = json.dumps(results, indent=2)
    if args.output:
        Path(args.output).write_text(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
