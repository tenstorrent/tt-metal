#!/usr/bin/env python3
"""
Build a filtered test matrix based on enabled SKUs.

This script:
1. Loads test definitions from a YAML file
2. Filters tests based on enabled SKUs (comma-separated string)
3. Optionally intersects with --sku-allowlist (change gating)
4. Optionally rewrites SKUs to merge_queue_sku when --event is merge_group
5. Adds runs_on labels from the SKU configuration
6. Annotates each entry with weights-cache-mode from sku_config.yaml
   (used by the Blackhole demo pipeline so the per-job container volume mount can
   pick the right cache source)
7. Outputs the filtered matrix as JSON

Usage:
    python prepare_test_matrix.py <tests_yaml_path> <enabled_skus> <sku_config_yaml_path>
        [--event EVENT] [--sku-allowlist LIST]

enabled_skus is a comma-separated list, or the literal ALL_SKUS_IN_TESTS to enable every SKU
key that appears under any test entry's skus mapping in the tests YAML. An empty / placeholder
tests YAML resolves ALL_SKUS_IN_TESTS to an empty matrix (matrix=[], exit 0) rather than failing.

--event: when set to merge_group, logical SKUs with merge_queue_sku in sku_config are
rewritten to that concrete prio SKU before runs_on lookup.

--sku-allowlist: omit for no extra filter; empty string skips all tests (matrix=[]
exit 0); otherwise comma-separated logical SKUs intersected with coverage.

`weights-cache-mode` is an optional per-SKU field in sku_config.yaml; when present,
it is copied into each output matrix entry.

Examples:
    python prepare_test_matrix.py tests/pipeline_reorg/galaxy_e2e_tests.yaml "wh_galaxy,bh_galaxy" .github/sku_config.yaml
    python prepare_test_matrix.py tests/pipeline_reorg/galaxy_demo_tests.yaml ALL_SKUS_IN_TESTS .github/sku_config.yaml
    python prepare_test_matrix.py tests/pipeline_reorg/blackhole_demo_tests.yaml ALL_SKUS_IN_TESTS .github/sku_config.yaml --event merge_group
"""

import argparse
import json
import os
import sys

import yaml

ALL_SKUS_IN_TESTS = "ALL_SKUS_IN_TESTS"
MERGE_GROUP_EVENT = "merge_group"


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


def apply_sku_allowlist(enabled_skus, sku_allowlist):
    """
    Intersect coverage SKUs with an optional allowlist.

    Args:
        enabled_skus: Coverage SKU list (logical names from tests / enabled_skus)
        sku_allowlist: None = no filter; "" or whitespace = skip all; else CSV

    Returns:
        Filtered SKU list (possibly empty)
    """
    if sku_allowlist is None:
        return enabled_skus

    allow = parse_enabled_skus(sku_allowlist)
    if not allow:
        print("SKU allowlist is empty; skipping all tests (matrix=[]).")
        return []

    filtered = [s for s in enabled_skus if s in allow]
    print(f"SKU allowlist {allow} → {filtered}")
    return filtered


def resolve_sku_for_event(sku_name, sku_config, event):
    """
    Map a logical SKU to its concrete runner SKU for the given event.

    On merge_group, if sku_config[sku].merge_queue_sku is set, return that prio SKU.
    """
    if event != MERGE_GROUP_EVENT:
        return sku_name

    entry = sku_config.get(sku_name) or {}
    alias = entry.get("merge_queue_sku")
    if not alias:
        return sku_name

    if alias not in sku_config:
        print(f"::error::SKU '{sku_name}' has merge_queue_sku '{alias}' " f"which is not defined in SKU configuration.")
        sys.exit(1)

    print(f"Event '{event}': rewriting SKU '{sku_name}' → '{alias}'")
    return alias


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

    # An empty or comment-only YAML parses to None; treat it as a placeholder
    # (no tests) rather than a malformed file.
    if tests is None:
        return []

    if not isinstance(tests, list):
        print(f"::error::Test matrix file must contain a list of tests: {tests_yaml_path}")
        sys.exit(1)

    return tests


def build_test_matrix(tests, enabled_skus, sku_config, event=None):
    """
    Filter tests based on enabled SKUs and expand multi-SKU entries into flat matrix entries.

    Each test entry may define multiple SKUs in its 'skus' dict. This function
    expands each test into one matrix entry per enabled SKU, with the appropriate
    timeout and runs_on labels. If the SKU entry in sku_config carries a weights-cache-mode
    field, it is copied into the output entry.

    When event is merge_group, logical SKUs are rewritten via merge_queue_sku before
    runs_on lookup. Timeout and other per-SKU fields still come from the logical key
    in the tests YAML.

    Args:
        tests: List of test dictionaries (with 'skus' dict)
        enabled_skus: List of enabled logical SKU strings
        sku_config: Dictionary mapping SKU names to their configuration
        event: Optional GitHub event name (e.g. merge_group)

    Returns:
        Filtered list of flat test dictionaries. Each entry has all keys from the
        test (e.g. name, cmd, model, owner_id, team) with skus removed and sku,
        timeout, and runs_on set for the selected SKU.
    """
    if not enabled_skus:
        print("::error::No SKUs enabled. At least one SKU must be specified.")
        sys.exit(1)

    # Validate that all enabled (logical) SKUs exist in config
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

        # Determine which of this test's SKUs are enabled (logical names)
        matching_skus = [s for s in test_skus if s in enabled_skus]

        # Always append SKU to name for clarity in CI
        append_sku_to_name = True

        for logical_sku in matching_skus:
            sku_test_config = test_skus[logical_sku]
            # logical_sku is guaranteed to be in sku_config (enabled_skus is validated
            # above); resolve_sku_for_event returns either logical_sku or an alias it
            # has already validated (else it exits), so concrete_sku is always valid.
            concrete_sku = resolve_sku_for_event(logical_sku, sku_config, event)

            # Start from test copy so all keys (model, arch, etc.) are preserved
            entry = test.copy()
            entry.pop("skus", None)
            entry["sku"] = concrete_sku
            if concrete_sku != logical_sku:
                entry["logical_sku"] = logical_sku
            entry["timeout"] = sku_test_config.get("timeout", 0)
            entry["runs_on"] = sku_config[concrete_sku].get("runs_on", [])
            if append_sku_to_name:
                entry["name"] = f"{test_name} [{concrete_sku}]"
            for key, value in sku_test_config.items():
                if key != "timeout" and value is not None:
                    entry[key] = value
            sku_entry = sku_config[concrete_sku]
            if "weights-cache-mode" in sku_entry:
                entry["weights-cache-mode"] = sku_entry["weights-cache-mode"]
            substitute_cmd_placeholders(entry)
            filtered_tests.append(entry)

    if not filtered_tests:
        print(f"::error::No tests selected for enabled SKUs '{','.join(enabled_skus)}'. Failing pipeline.")
        sys.exit(1)

    return filtered_tests


def write_matrix_output(filtered_matrix):
    """Print and optionally write matrix to GITHUB_OUTPUT."""
    print(f"\nFiltered test matrix ({len(filtered_matrix)} tests):")
    json_output_pretty = json.dumps(filtered_matrix, indent=2)
    print(json_output_pretty)

    json_output_compact = json.dumps(filtered_matrix)
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"matrix<<EOF\n{json_output_compact}\nEOF\n")
    else:
        print(f"\nmatrix={json_output_compact}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a filtered test matrix based on enabled SKUs.",
    )
    parser.add_argument("tests_yaml_path", help="Path to pipeline_reorg tests yaml")
    parser.add_argument(
        "enabled_skus",
        help="Comma-separated SKUs, or ALL_SKUS_IN_TESTS",
    )
    parser.add_argument("sku_config_yaml_path", help="Path to sku_config.yaml")
    parser.add_argument(
        "--event",
        default=os.environ.get("MATRIX_EVENT_NAME") or None,
        help="GitHub event name; merge_group triggers merge_queue_sku rewrite",
    )
    parser.add_argument(
        "--sku-allowlist",
        default=None,
        help="Omit for no filter; empty string skips all; else CSV of logical SKUs",
    )
    args = parser.parse_args(argv)

    print(f"Loading tests from: {args.tests_yaml_path}")
    print(f"Loading SKU config from: {args.sku_config_yaml_path}")
    print(f"Enabled SKUs: '{args.enabled_skus}'")
    if args.event:
        print(f"Event: '{args.event}'")
    if args.sku_allowlist is not None:
        print(f"SKU allowlist: '{args.sku_allowlist}'")

    sku_config = load_sku_config(args.sku_config_yaml_path)
    tests = load_tests(args.tests_yaml_path)

    if args.enabled_skus.strip().upper() == ALL_SKUS_IN_TESTS:
        enabled_skus = collect_skus_from_tests(tests)
        print(f"Resolved {ALL_SKUS_IN_TESTS} to: {enabled_skus}")
        if not enabled_skus:
            # Empty / placeholder test list (e.g. a team's not-yet-populated gate
            # yaml). Emit an empty matrix and skip rather than failing the job.
            print("No SKU keys found under skus in the tests YAML; skipping all tests (matrix=[]).")
            write_matrix_output([])
            return 0
    else:
        enabled_skus = parse_enabled_skus(args.enabled_skus)
        print(f"Parsed enabled SKUs: {enabled_skus}")

    enabled_skus = apply_sku_allowlist(enabled_skus, args.sku_allowlist)
    if not enabled_skus:
        # Empty after allowlist (including explicit empty allowlist) → skip, do not fail.
        write_matrix_output([])
        return 0

    filtered_matrix = build_test_matrix(tests, enabled_skus, sku_config, event=args.event)
    write_matrix_output(filtered_matrix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
