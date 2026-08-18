#!/usr/bin/env python3
"""
Assert a workflow's `test-type` picker help text lists exactly the stages that exist.

The validator (validate_test_type_selection.py) derives its accepted tokens from the
test matrix, so a dispatch can never run a stage that does not exist. What it cannot
do is keep the workflow_dispatch input *description* honest: that list is static text,
and it drifted -- advertising three stages that had been renamed away while omitting
three that existed. A wrong list costs whoever uses the picker a failed dispatch, and
hides runnable stages, so check it in CI instead.

Only the token SET is compared. Ordering and the surrounding prose are free-form.

Usage:
    python check_test_type_help_text.py <tests_yaml_path> <workflow_yaml_path>
"""

import re
import sys

import yaml

MARKER = "Valid stages:"


def error(message):
    """Emit a GitHub Actions error annotation and exit non-zero."""
    safe = str(message).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::error::{safe}", file=sys.stderr)
    sys.exit(1)


def matrix_test_types(path):
    with open(path) as f:
        tests = yaml.safe_load(f)
    if not isinstance(tests, list):
        error(f"Test matrix file must contain a list of test entries: {path}")
    return {t["test_type"] for t in tests if t.get("test_type")}


def advertised_test_types(path):
    """Tokens listed after "Valid stages:" in the test-type input description."""
    with open(path) as f:
        workflow = yaml.safe_load(f)
    # "on" is the YAML 1.1 boolean true, so PyYAML may key it either way.
    triggers = workflow.get("on", workflow.get(True)) or {}
    try:
        description = triggers["workflow_dispatch"]["inputs"]["test-type"]["description"]
    except (KeyError, TypeError):
        error(f"{path} has no workflow_dispatch input named 'test-type' with a description")
    if MARKER not in description:
        error(f"The test-type description in {path} must list its stages after a {MARKER!r} marker")
    listed = description.split(MARKER, 1)[1]
    # The list ends at the first period that closes it; the block scalar folds to one line.
    listed = listed.split(".")[0]
    return {tok.strip() for tok in listed.split(",") if re.fullmatch(r"[a-z0-9_]+", tok.strip())}


def main():
    if len(sys.argv) != 3:
        error("Usage: check_test_type_help_text.py <tests_yaml_path> <workflow_yaml_path>")
    tests_yaml, workflow_yaml = sys.argv[1], sys.argv[2]

    real = matrix_test_types(tests_yaml)
    listed = advertised_test_types(workflow_yaml)

    phantom = sorted(listed - real)
    missing = sorted(real - listed)
    if phantom or missing:
        parts = [f"The test-type help text in {workflow_yaml} disagrees with {tests_yaml}."]
        if phantom:
            parts.append(f"Advertised but no such stage: {', '.join(phantom)}")
        if missing:
            parts.append(f"Stage exists but is not advertised: {', '.join(missing)}")
        parts.append("Update the 'Valid stages:' list in the test-type input description.")
        error("\n".join(parts))

    print(f"test-type help text matches the matrix ({len(real)} stages)")


if __name__ == "__main__":
    main()
