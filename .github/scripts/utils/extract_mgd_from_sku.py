#!/usr/bin/env python3
"""Extract MGD configuration from SKU config file."""

import yaml
import sys
import os


def main():
    if len(sys.argv) != 3:
        print("Usage: extract_mgd_from_sku.py <sku_name> <sku_config_path>", file=sys.stderr)
        sys.exit(1)

    sku_name = sys.argv[1]
    sku_config_path = sys.argv[2]

    if not os.path.exists(sku_config_path):
        print(f"::error::SKU config file not found at {sku_config_path}", file=sys.stderr)
        sys.exit(1)

    with open(sku_config_path) as f:
        config = yaml.safe_load(f)

    skus = config.get("skus", {})
    if sku_name in skus and "mgd" in skus[sku_name]:
        print(skus[sku_name]["mgd"])
    else:
        print(f"::error::MGD not found for SKU '{sku_name}'", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
