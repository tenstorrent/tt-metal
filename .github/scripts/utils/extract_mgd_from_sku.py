#!/usr/bin/env python3
"""Extract MGD configuration from SKU config file."""

import yaml
import sys


def main():
    if len(sys.argv) != 2:
        print("Usage: extract_mgd_from_sku.py <sku_name>", file=sys.stderr)
        sys.exit(1)

    sku_name = sys.argv[1]

    with open(".github/sku_config.yaml") as f:
        config = yaml.safe_load(f)

    skus = config.get("skus", {})
    if sku_name in skus and "mgd" in skus[sku_name]:
        print(skus[sku_name]["mgd"])
    else:
        print(f"Error: MGD not found for SKU {sku_name}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
