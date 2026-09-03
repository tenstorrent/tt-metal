#!/usr/bin/env python3
"""
Validate and normalize a comma-separated test-type selection against a test matrix.

The set of supported tokens is derived directly from the test matrix YAML (the same
file consumed by prepare_test_matrix.py), so this validator never drifts from the
tests that actually exist. Supported tokens are:

  * "all"                -> run every stage
  * each entry's test_type   (e.g. mla, prefill_block, kimi_moe, ...)
  * each entry's test_group  (e.g. module -> run every stage in that group)

Selection rules:
  * Empty / whitespace-only input          -> normalized to "all".
  * A selection that contains "all"        -> normalized to "all".
  * Otherwise                              -> de-duplicated, order-preserving list
                                              of the validated tokens.

On success the normalized selection is printed to stdout and, when running under
GitHub Actions, written to GITHUB_OUTPUT as `selection=<value>`.
On failure a GitHub `::error::` annotation is emitted and the script exits 1.

Usage:
    python validate_test_type_selection.py <tests_yaml_path> <selection>

Examples:
    python validate_test_type_selection.py tests/pipeline_reorg/blaze_models_prefill_tests.yaml "mla,prefill_block"
    python validate_test_type_selection.py tests/pipeline_reorg/blaze_models_prefill_tests.yaml ""        # -> all
    python validate_test_type_selection.py tests/pipeline_reorg/blaze_models_prefill_tests.yaml "all"
"""

import os
import sys

import yaml

ALL = "all"


def error(message):
    """Emit a GitHub Actions error annotation and exit non-zero."""
    safe = str(message).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::error::{safe}", file=sys.stderr)
    sys.exit(1)


def load_supported_tokens(tests_yaml_path):
    """
    Return (test_types, test_groups, supported) derived from the test matrix YAML.

    `supported` is the full set of accepted tokens (test_types + test_groups + "all").
    """
    if not os.path.exists(tests_yaml_path):
        error(f"Test matrix file not found: {tests_yaml_path}")

    with open(tests_yaml_path, "r") as f:
        tests = yaml.safe_load(f)

    if not isinstance(tests, list):
        error(f"Test matrix file must contain a list of test entries: {tests_yaml_path}")

    test_types = sorted({t["test_type"] for t in tests if t.get("test_type")})
    test_groups = sorted({t["test_group"] for t in tests if t.get("test_group")})
    supported = {ALL, *test_types, *test_groups}
    return test_types, test_groups, supported


def parse_selection(selection):
    """Split a comma-separated selection into cleaned, non-empty tokens (order preserved)."""
    if not selection:
        return []
    return [tok.strip() for tok in selection.split(",") if tok.strip()]


def dedupe(tokens):
    """De-duplicate while preserving first-seen order."""
    seen = set()
    result = []
    for tok in tokens:
        if tok not in seen:
            seen.add(tok)
            result.append(tok)
    return result


def write_output(selection):
    """Print the normalized selection and expose it as a GitHub Actions step output."""
    print(selection)
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"selection={selection}\n")


def main():
    if len(sys.argv) != 3:
        error("Usage: validate_test_type_selection.py <tests_yaml_path> <selection>")

    tests_yaml_path = sys.argv[1]
    raw_selection = sys.argv[2]

    test_types, test_groups, supported = load_supported_tokens(tests_yaml_path)
    tokens = parse_selection(raw_selection)

    # Empty selection or an explicit "all" anywhere in the list means "run everything".
    if not tokens or ALL in tokens:
        write_output(ALL)
        return

    invalid = dedupe([tok for tok in tokens if tok not in supported])
    if invalid:
        valid_list = "\n".join(
            [
                f"  all                          -> run every stage",
                *[f"  {t}" for t in test_types],
                *[f"  {g:<28} -> test group (runs every stage in the group)" for g in test_groups],
            ]
        )
        error(
            f"Invalid test-type token(s): {', '.join(invalid)}\n"
            f"Selection was: '{raw_selection}'\n"
            f"Supported tokens (from {tests_yaml_path}):\n{valid_list}\n"
            "Provide a comma-separated subset of the above, or leave empty / use 'all' to run everything."
        )

    write_output(",".join(dedupe(tokens)))


if __name__ == "__main__":
    main()
