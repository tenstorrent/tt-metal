#!/usr/bin/env python3
"""Extract allocation configuration from SKU config file."""

import yaml
import sys
import os


def main():
    if len(sys.argv) != 2:
        print(
            "Usage: extract_allocation_from_sku.py <sku_name>",
            file=sys.stderr,
        )
        sys.exit(1)

    sku_name = sys.argv[1]

    # Hardcoded path to SKU config file
    # Script is in .github/scripts/utils/
    script_dir = os.path.dirname(os.path.realpath(__file__))
    repo_root = os.path.realpath(os.path.join(script_dir, "..", "..", ".."))
    sku_config_path = os.path.join(repo_root, ".github", "sku_config.yaml")

    # Verify the file exists
    if not os.path.exists(sku_config_path):
        print(
            f"::error::SKU config file not found at {sku_config_path}", file=sys.stderr
        )
        sys.exit(1)

    # Open and parse the YAML file
    try:
        with open(sku_config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except (OSError, IOError) as e:
        print(f"::error::Failed to read SKU config file: {e}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"::error::Invalid YAML in SKU config file: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(config, dict):
        print(
            f"::error::SKU config file must contain a YAML dictionary", file=sys.stderr
        )
        sys.exit(1)

    skus = config.get("skus", {})
    if sku_name not in skus:
        print(f"::error::SKU '{sku_name}' not found in config file", file=sys.stderr)
        sys.exit(1)

    if "allocation" not in skus[sku_name]:
        print(f"::error::SKU '{sku_name}' has no 'allocation' field", file=sys.stderr)
        sys.exit(1)

    allocation = skus[sku_name]["allocation"]

    # Build the allocation spec
    lines = []
    lines.append(f"allocationType: {allocation['type']}")

    if "count" in allocation:
        lines.append(f"count: {allocation['count']}")

    if "match_labels" in allocation:
        lines.append("matchLabels:")
        for key, value in allocation["match_labels"].items():
            # Quote keys that contain special characters like dots
            if "." in key or "/" in key:
                lines.append(f'  "{key}": {value}')
            else:
                lines.append(f"  {key}: {value}")

    if "topology_keys" in allocation:
        lines.append("topologyKeys:")
        for key in allocation["topology_keys"]:
            lines.append(f"  - {key}")

    # Add MGD section if present
    if "mgd" in skus[sku_name]:
        lines.append("mgd: |")
        mgd_content = skus[sku_name]["mgd"]
        # Add 2-space indentation to each line of MGD content
        for line in mgd_content.splitlines():
            lines.append(f"  {line}")

    # Print the spec
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
