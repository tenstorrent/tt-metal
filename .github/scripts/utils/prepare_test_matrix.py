#!/usr/bin/env python3
"""
Build a filtered test matrix based on enabled SKUs.

This script:
1. Loads test definitions from a YAML file
2. Filters tests based on enabled SKUs (comma-separated string)
3. Adds runs_on labels from the SKU configuration
4. Outputs the filtered matrix as JSON

Usage:
    python prepare_test_matrix.py <tests_yaml_path> <enabled_skus> <sku_config_yaml_path>

Example:
    python prepare_test_matrix.py tests/pipeline_reorg/galaxy_e2e_tests.yaml "wh_galaxy,bh_galaxy" .github/sku_config.yaml
"""

import yaml
import json
import sys
import os


def parse_enabled_skus(enabled_skus_str):
    """
    Parse comma-separated SKU string into a list.

    Args:
        enabled_skus_str: Comma-separated string of SKUs (e.g., "wh_galaxy,bh_galaxy")

    Returns:
        List of SKU strings, or empty list if input is empty
    """
    if not enabled_skus_str or not enabled_skus_str.strip():
        return []

    # Split by comma, strip whitespace, filter out empty strings
    skus = [sku.strip() for sku in enabled_skus_str.split(",") if sku.strip()]
    return skus


def load_sku_config(sku_config_path):
    """
    Load SKU configuration from YAML file.

    Args:
        sku_config_path: Path to SKU configuration YAML file

    Returns:
        Dictionary mapping SKU names to their configuration
    """
    if not os.path.exists(sku_config_path):
        print(f"::error::SKU config file not found at: {sku_config_path}")
        sys.exit(1)

    with open(sku_config_path, "r") as f:
        config = yaml.safe_load(f)

    if "skus" not in config:
        print(f"::error::SKU config file must contain a 'skus' key: {sku_config_path}")
        sys.exit(1)

    return config["skus"]


def load_tests(tests_yaml_path):
    """
    Load test definitions from YAML file.

    Args:
        tests_yaml_path: Path to tests YAML file

    Returns:
        List of test dictionaries
    """
    if not os.path.exists(tests_yaml_path):
        print(f"::error::Test matrix file not found at: {tests_yaml_path}")
        sys.exit(1)

    with open(tests_yaml_path, "r") as f:
        tests = yaml.safe_load(f)

    if not isinstance(tests, list):
        print(f"::error::Test matrix file must contain a list of tests: {tests_yaml_path}")
        sys.exit(1)

    return tests


def build_test_matrix(tests, enabled_skus, sku_config):
    """
    Filter tests based on enabled SKUs and expand multi-SKU entries into flat matrix entries.

    Each test entry may define multiple SKUs in its 'skus' dict. This function
    expands each test into one matrix entry per enabled SKU, with the appropriate
    timeout and runs_on labels.

    Args:
        tests: List of test dictionaries (with 'skus' dict)
        enabled_skus: List of enabled SKU strings
        sku_config: Dictionary mapping SKU names to their configuration

    Returns:
        Filtered list of flat test dictionaries. Each entry has all keys from the
        test (e.g. name, cmd, model, owner_id, team) with skus removed and sku,
        timeout, and runs_on set for the selected SKU.
    """
    if not enabled_skus:
        print("::error::No SKUs enabled. At least one SKU must be specified.")
        sys.exit(1)

    # Validate that all enabled SKUs exist in config
    for sku in enabled_skus:
        if sku not in sku_config:
            print(f"::error::SKU '{sku}' not found in SKU configuration. Available SKUs: {list(sku_config.keys())}")
            sys.exit(1)

    filtered_tests = []

    for test in tests:
        test_name = test.get("name", "Unnamed Test")
        test_skus = test.get("skus")

        # Skip tests without skus
        if not test_skus or not isinstance(test_skus, dict):
            print(f"::warning::Test '{test_name}' has no valid 'skus' mapping, skipping")
            continue

        # Determine which of this test's SKUs are enabled
        matching_skus = [s for s in test_skus if s in enabled_skus]

        # Append SKU to name when the same test runs on more than one SKU
        append_sku_to_name = len(matching_skus) > 1

        for sku_name in matching_skus:
            sku_test_config = test_skus[sku_name]

            if sku_name not in sku_config:
                print(f"::warning::SKU '{sku_name}' for test '{test_name}' not found in SKU config, skipping")
                continue

            # Start from test copy so all keys (model, arch, etc.) are preserved
            entry = test.copy()
            entry.pop("skus", None)
            entry["sku"] = sku_name
            entry["timeout"] = sku_test_config.get("timeout", 0)
            entry["runs_on"] = sku_config[sku_name].get("runs_on", [])
            if append_sku_to_name:
                entry["name"] = f"{test_name} [{sku_name}]"
            for key, value in sku_test_config.items():
                if key != "timeout" and value is not None:
                    entry[key] = value
            filtered_tests.append(entry)

    if not filtered_tests:
        print(f"::error::No tests selected for enabled SKUs '{','.join(enabled_skus)}'. Failing pipeline.")
        sys.exit(1)

    return filtered_tests


def main():
    if len(sys.argv) != 4:
        print("Usage: python prepare_test_matrix.py <tests_yaml_path> <enabled_skus> <sku_config_yaml_path>")
        print(
            'Example: python prepare_test_matrix.py tests/pipeline_reorg/galaxy_e2e_tests.yaml "wh_galaxy,bh_galaxy" .github/sku_config.yaml'
        )
        sys.exit(1)

    tests_yaml_path = sys.argv[1]
    enabled_skus_str = sys.argv[2]
    sku_config_path = sys.argv[3]

    print(f"Loading tests from: {tests_yaml_path}")
    print(f"Loading SKU config from: {sku_config_path}")
    print(f"Enabled SKUs: '{enabled_skus_str}'")

    # Parse enabled SKUs
    enabled_skus = parse_enabled_skus(enabled_skus_str)
    print(f"Parsed enabled SKUs: {enabled_skus}")

    # Load configurations
    sku_config = load_sku_config(sku_config_path)
    tests = load_tests(tests_yaml_path)

    # Build filtered matrix
    filtered_matrix = build_test_matrix(tests, enabled_skus, sku_config)

    # Output as JSON
    print(f"\nFiltered test matrix ({len(filtered_matrix)} tests):")
    json_output_pretty = json.dumps(filtered_matrix, indent=2)
    print(json_output_pretty)

    # Output for GitHub Actions (using multiline output format)
    # This writes to GITHUB_OUTPUT file for job outputs
    # Use compact JSON for GITHUB_OUTPUT to match other workflows
    json_output_compact = json.dumps(filtered_matrix)
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"matrix<<EOF\n{json_output_compact}\nEOF\n")
    else:
        # Fallback: output to stdout (for testing)
        print(f"\nmatrix={json_output_compact}")


if __name__ == "__main__":
    main()
