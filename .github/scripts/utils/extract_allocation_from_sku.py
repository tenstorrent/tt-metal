#!/usr/bin/env python3
"""Extract allocation configuration from SKU config file."""

import yaml
import sys
import os


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: extract_allocation_from_sku.py <sku_name> <sku_config_path>",
            file=sys.stderr,
        )
        sys.exit(1)

    sku_name = sys.argv[1]
    sku_config_path = sys.argv[2]

    if not os.path.exists(sku_config_path):
        print(
            f"::error::SKU config file not found at {sku_config_path}", file=sys.stderr
        )
        sys.exit(1)

    with open(sku_config_path) as f:
        config = yaml.safe_load(f)

    skus = config.get("skus", {})
    if sku_name not in skus:
        print(
            f"::error::SKU '{sku_name}' not found in {sku_config_path}", file=sys.stderr
        )
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
            lines.append(f"  {key}: {value}")

    if "topology_keys" in allocation:
        lines.append("topologyKeys:")
        for key in allocation["topology_keys"]:
            lines.append(f"  - {key}")

    # Print the spec
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
