#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Fix shard_spec format in master JSON files after DB reconstruction.

The DB stores shard_spec as the string "None" when there is no shard spec,
but the sweep tracer produces JSON null.  This script converts string "None"
to actual null so the validator sees matching values.

Only uses stdlib — safe to run on ubuntu-latest.
"""

import json
import sys


def _fix_shard_spec(obj):
    """Recursively convert "shard_spec": "None" (string) to null."""
    if isinstance(obj, dict):
        if "shard_spec" in obj and obj["shard_spec"] == "None":
            obj["shard_spec"] = None
        for v in obj.values():
            _fix_shard_spec(v)
    elif isinstance(obj, list):
        for item in obj:
            _fix_shard_spec(item)


def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_shard_spec_format.py <json_file>", file=sys.stderr)
        return 1

    json_file = sys.argv[1]
    with open(json_file) as f:
        data = json.load(f)

    _fix_shard_spec(data)

    with open(json_file, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)

    print(f"Fixed shard_spec format in {json_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
