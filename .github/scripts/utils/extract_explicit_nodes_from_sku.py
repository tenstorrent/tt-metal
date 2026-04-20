#!/usr/bin/env python3
"""Extract explicit_nodes configuration from SKU config file."""

import yaml
import sys
import os


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: extract_explicit_nodes_from_sku.py <sku_name> <sku_config_path>",
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

    # Print explicit_nodes if present, empty string otherwise (not an error)
    explicit_nodes = skus[sku_name].get("explicit_nodes", "")
    print(explicit_nodes)


if __name__ == "__main__":
    main()
