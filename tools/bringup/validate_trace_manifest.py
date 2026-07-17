#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fail-fast validator for Phase-1 tracer manifests (CLI entry point).

The Phase-1 tracer (``phase1_record_ops.py``) records each traced op/module into
a ``manifest.json`` plus a directory of ``.pt`` tensor artifacts. Phase-2 parity
tests (``tracer_test_harness.py`` / generated pytest files) replay those ops and
compare against the recorded golden tensors. When the manifest is malformed, an
artifact is missing, or a recorded shape disagrees with the tensor on disk, the
failure otherwise only surfaces deep inside ``pytest`` as noisy, hard-to-triage
errors.

This tool validates the manifest up-front so problems can be fixed locally
before launching a test run. It runs independently of ``pytest`` and never
imports ``ttnn`` (``torch`` is imported lazily and only for shape checks).

The validation core lives in ``trace_manifest_validation`` and the shared op
knowledge (supported kinds, per-kind required params) lives in
``tracer_op_specs`` so the validator and the runtime harness stay in agreement.

Manifest schema (see ``phase1_record_ops.py`` and ``tools/bringup/README.md``):
    {
      "input_shape": [B, C, H, W],
      "num_records": N,
      "records": [
        {
          "idx": int, "name": str, "kind": str, "params": {...},
          "in_shape": [n,c,h,w], "out_shape": [n,c,h,w],
          "in_path": str, "out_path": str,
          "w_path": str|null, "b_path": str|null
        }, ...
      ]
    }

Artifact path contract (Option 1): all artifact paths (``in_path`` /
``out_path`` / ``w_path`` / ``b_path``) are resolved relative to the directory
containing the manifest file. Absolute paths are used as-is.

Usage:
    python tools/bringup/validate_trace_manifest.py --manifest <manifest.json>
    python tools/bringup/validate_trace_manifest.py --manifest <manifest.json> --print-resolved 5
    python tools/bringup/validate_trace_manifest.py --manifest <manifest.json> --no-shape-check

Exit codes:
    0  Manifest is valid (no errors)
    1  Validation failed, or the manifest could not be found / parsed
"""

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
