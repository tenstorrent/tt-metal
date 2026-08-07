#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

# Allow running this file directly as a script: ensure its own directory is
# importable so the sibling validation module resolves regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Re-exported so existing callers/tests can keep importing from this module.
from trace_manifest_validation import (  # noqa: E402
    Report,
    format_report,
    print_resolved,
    resolve_artifact_path,
    validate_manifest,
)

__all__ = [
    "Report",
    "format_report",
    "print_resolved",
    "resolve_artifact_path",
    "validate_manifest",
    "main",
]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to the Phase-1 tracer manifest.json to validate.",
    )
    parser.add_argument(
        "--print-resolved",
        type=int,
        metavar="N",
        default=0,
        help="Print resolved artifact paths for the first N records.",
    )
    parser.add_argument(
        "--no-shape-check",
        action="store_true",
        help="Skip loading tensors to compare on-disk shapes against the manifest.",
    )
    args = parser.parse_args(argv)

    report = validate_manifest(
        args.manifest,
        check_shapes=not args.no_shape_check,
    )

    print("")
    print("=" * 60)
    print(format_report(args.manifest, report), end="")
    print("=" * 60)

    if args.print_resolved > 0:
        print_resolved(args.manifest, args.print_resolved)

    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
