# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point for offline fabric capture decode."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tt_metal.fabric.debug.visualizer.decode.inputs import DecodeError, discover_inputs
from tt_metal.fabric.debug.visualizer.decode.output import build_decoded, write_decoded


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Decode fabric debug captures without device access.")
    parser.add_argument("inputs", nargs="+", type=Path, help="directories or manifest/snapshot JSON files")
    parser.add_argument("-o", "--output", required=True, type=Path, help="decoded JSON output")
    parser.add_argument(
        "--allow-manifest-mismatch",
        action="store_true",
        help="pair by run identity when the exact manifest hash is unavailable",
    )
    parser.add_argument(
        "--skip-bin-hash",
        action="store_true",
        help="check raw sidecar size but skip its SHA-256 pass",
    )
    parser.add_argument(
        "--expert-raw",
        action="store_true",
        help="embed hex for captured regions up to 64 bytes",
    )
    parser.add_argument(
        "--slots",
        choices=("none", "headers"),
        default="headers",
        help="decode packet headers in every ring slot (default: headers)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        inputs = discover_inputs(
            args.inputs,
            allow_manifest_mismatch=args.allow_manifest_mismatch,
            skip_bin_hash=args.skip_bin_hash,
        )
        decoded = build_decoded(inputs, expert_raw=args.expert_raw, slots=args.slots)
        write_decoded(args.output, decoded)
    except (DecodeError, OSError) as error:
        print(f"fabric decode failed: {error}", file=sys.stderr)
        return 1
    coverage = decoded["coverage"]
    print(
        f"Wrote {args.output}: {coverage['captured']}/{coverage['routers_total']} routers captured, "
        f"{coverage['ok']} ok"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
