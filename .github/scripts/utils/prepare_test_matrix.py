#!/usr/bin/env python3
"""
Build a filtered test matrix based on enabled SKUs.

This script:
1. Loads test definitions from a YAML file
2. Filters tests based on enabled SKUs (comma-separated string)
3. Adds runs_on labels from the SKU configuration
4. Optionally annotates each entry with weights-cache-mode + system-name
   from a systems-config YAML (used by the Blackhole demo pipeline so the
   per-job container volume mount can pick the right cache source)
5. Outputs the filtered matrix as JSON

Usage:
    python prepare_test_matrix.py <tests_yaml_path> <enabled_skus> <sku_config_yaml_path> [<systems_config_yaml_path>]

enabled_skus is a comma-separated list, or the literal ALL_SKUS_IN_TESTS to enable every SKU
key that appears under any test entry's skus mapping in the tests YAML.

systems_config_yaml_path is optional. When provided, it must contain a top-level "systems"
list whose entries are mappings with keys: sku, name, weights-cache-mode (and optionally
num_devices). Each output matrix entry will have weights-cache-mode and system-name
fields populated from this table for its SKU.

Examples:
    python prepare_test_matrix.py tests/pipeline_reorg/galaxy_e2e_tests.yaml "wh_galaxy,bh_galaxy" .github/sku_config.yaml
    python prepare_test_matrix.py tests/pipeline_reorg/galaxy_demo_tests.yaml ALL_SKUS_IN_TESTS .github/sku_config.yaml
    python prepare_test_matrix.py tests/pipeline_reorg/blackhole_demo_tests.yaml ALL_SKUS_IN_TESTS .github/sku_config.yaml .github/blackhole_demo_systems.yaml
"""

import yaml
import json
import sys
import os

ALL_SKUS_IN_TESTS = "ALL_SKUS_IN_TESTS"


def collect_skus_from_tests(tests):
    """Return sorted unique SKU names referenced in tests' skus mappings."""
    names = set()
    for test in tests:
        test_skus = test.get("skus")
        if isinstance(test_skus, dict):
            names.update(test_skus.keys())
    return sorted(names)


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


def load_systems_config(systems_config_path):
    """
    Load optional systems configuration mapping SKU → cache-mode + display name.

    Returns a dict keyed by SKU name with values {name, weights-cache-mode, num_devices}.
    """
    if not os.path.exists(systems_config_path):
        print(f"::error::Systems config file not found at: {systems_config_path}")
        sys.exit(1)

    with open(systems_config_path, "r") as f:
        config = yaml.safe_load(f)

    if "systems" not in config or not isinstance(config["systems"], list):
        print(f"::error::Systems config file must contain a 'systems' list: {systems_config_path}")
        sys.exit(1)

    by_sku = {}
    for entry in config["systems"]:
        sku = entry.get("sku")
        if not sku:
            print(f"::warning::systems entry missing 'sku' key, skipping: {entry}")
            continue
        by_sku[sku] = entry
    return by_sku


def substitute_cmd_placeholders(entry):
    """
    Replace placeholders in entry["cmd"] with values from the same entry.

    Placeholders use the form {key_name}; e.g. {tt_cache_path} is replaced with
    entry["tt_cache_path"]. This allows per-SKU values (e.g. different TT_CACHE_PATH
    paths) to be injected into the same base command.
    """
    cmd = entry.get("cmd")
    if not cmd or not isinstance(cmd, str):
        raise ValueError(f"cmd is not a string: {cmd}")
    for key, value in entry.items():
        placeholder = "{" + key + "}"
        if placeholder in cmd:
            entry["cmd"] = entry["cmd"].replace(placeholder, str(value))


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


def build_test_matrix(tests, enabled_skus, sku_config, systems_by_sku=None):
    """
    Filter tests based on enabled SKUs and expand multi-SKU entries into flat matrix entries.

    Each test entry may define multiple SKUs in its 'skus' dict. This function
    expands each test into one matrix entry per enabled SKU, with the appropriate
    timeout and runs_on labels.

    Args:
        tests: List of test dictionaries (with 'skus' dict)
        enabled_skus: List of enabled SKU strings
        sku_config: Dictionary mapping SKU names to their configuration
        systems_by_sku: Optional dict mapping SKU → {name, weights-cache-mode, ...}.
            When provided, each output entry gets weights-cache-mode and system-name
            fields populated from this table. SKUs not present in the table fall
            through with no annotation (matrix builder doesn't fail).

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

        # Always append SKU to name for clarity in CI
        append_sku_to_name = True

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
            if systems_by_sku is not None and sku_name in systems_by_sku:
                sys_entry = systems_by_sku[sku_name]
                entry["weights-cache-mode"] = sys_entry.get("weights-cache-mode")
                entry["system-name"] = sys_entry.get("name")
            substitute_cmd_placeholders(entry)
            filtered_tests.append(entry)

    if not filtered_tests:
        print(f"::error::No tests selected for enabled SKUs '{','.join(enabled_skus)}'. Failing pipeline.")
        sys.exit(1)

    return filtered_tests


def main():
    if len(sys.argv) not in (4, 5):
        print(
            "Usage: python prepare_test_matrix.py <tests_yaml_path> <enabled_skus> <sku_config_yaml_path> [<systems_config_yaml_path>]"
        )
        print("  enabled_skus: comma-separated list, or ALL_SKUS_IN_TESTS")
        print("  systems_config_yaml_path: optional. When provided, each output entry gets")
        print("    weights-cache-mode + system-name from the SKU's entry in that file.")
        print(
            'Example: python prepare_test_matrix.py tests/pipeline_reorg/galaxy_e2e_tests.yaml "wh_galaxy,bh_galaxy" .github/sku_config.yaml'
        )
        sys.exit(1)

    tests_yaml_path = sys.argv[1]
    enabled_skus_str = sys.argv[2]
    sku_config_path = sys.argv[3]
    systems_config_path = sys.argv[4] if len(sys.argv) == 5 else None

    print(f"Loading tests from: {tests_yaml_path}")
    print(f"Loading SKU config from: {sku_config_path}")
    if systems_config_path:
        print(f"Loading systems config from: {systems_config_path}")
    print(f"Enabled SKUs: '{enabled_skus_str}'")

    sku_config = load_sku_config(sku_config_path)
    systems_by_sku = load_systems_config(systems_config_path) if systems_config_path else None
    tests = load_tests(tests_yaml_path)

    if enabled_skus_str.strip().upper() == ALL_SKUS_IN_TESTS:
        enabled_skus = collect_skus_from_tests(tests)
        print(f"Resolved {ALL_SKUS_IN_TESTS} to: {enabled_skus}")
        if not enabled_skus:
            print("::error::No SKU keys found under skus in the tests YAML.")
            sys.exit(1)
    else:
        enabled_skus = parse_enabled_skus(enabled_skus_str)
        print(f"Parsed enabled SKUs: {enabled_skus}")

    filtered_matrix = build_test_matrix(tests, enabled_skus, sku_config, systems_by_sku=systems_by_sku)

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
