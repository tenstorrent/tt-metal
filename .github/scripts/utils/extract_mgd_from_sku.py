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
    if sku_name not in skus:
        print(f"::error::SKU '{sku_name}' not found in {sku_config_path}", file=sys.stderr)
        sys.exit(1)

    if "mgd" not in skus[sku_name]:
        print(f"::error::SKU '{sku_name}' has no 'mgd' field", file=sys.stderr)
        sys.exit(1)

    # Print MGD content preserving exact whitespace structure
    # The MGD value from YAML should already have proper relative indentation
    mgd_content = skus[sku_name]["mgd"]
    # Ensure we output it exactly as stored (with trailing newline if present)
    if mgd_content.endswith("\n"):
        print(mgd_content, end="")
    else:
        print(mgd_content)


if __name__ == "__main__":
    main()
