#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Load issue data from a frozen snapshot or GitHub."""

import argparse
import json
import os
import sys
from pathlib import Path


def load_issue(number: int) -> str:
    snapshot = os.environ.get("CODEGEN_ISSUE_SNAPSHOT")
    if not snapshot:
        raise ValueError("CODEGEN_ISSUE_SNAPSHOT is not set")
    payload = Path(snapshot).read_text()
    issue = json.loads(payload)
    if not isinstance(issue, dict) or issue.get("number") != number:
        raise ValueError(f"snapshot issue number does not match {number}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("number", type=int)
    args = parser.parse_args()
    try:
        payload = load_issue(args.number)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"failed to load issue {args.number}: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(payload)
    if not payload.endswith("\n"):
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
