# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Takes a snapshot of each fabric manifest given, reading its local chips' routers through ttexalens.

Usage, from the repository root:
    python3 -m tt_metal.fabric.debug.visualizer.capture MANIFEST|DIR [MANIFEST|DIR ...] [-o DIR]

A DIR stands for the fabric_manifest_rank_*.json files in it, such as the run's generated/fabric. Each snapshot goes
next to its manifest, or into -o's directory. Prints one line per snapshot and one per router error. Exits non-zero when
ttexalens doesn't start or a manifest can't be read or its snapshot written; a router error alone doesn't.
"""

import argparse
import json
import pathlib
import socket
import sys
import time

from tt_metal.fabric.debug.visualizer.capture.exalens import open_exalens
from tt_metal.fabric.debug.visualizer.capture.reads import ManifestError
from tt_metal.fabric.debug.visualizer.capture.snapshot import take_snapshot


def find_manifests(paths: list) -> list:
    """Each file in `paths`, and each directory's fabric_manifest_rank_*.json files, in name order."""
    manifests = []
    for path in paths:
        manifests.extend(sorted(path.glob("fabric_manifest_rank_*.json")) if path.is_dir() else [path])
    return manifests


def main(argv: list) -> int:
    # Arg parsing
    parser = argparse.ArgumentParser(
        prog="python3 -m tt_metal.fabric.debug.visualizer.capture",
        description="Takes a snapshot of each fabric manifest's local routers through ttexalens.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=pathlib.Path,
        metavar="MANIFEST|DIR",
        help="a fabric manifest, or a directory of fabric_manifest_rank_*.json files",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=pathlib.Path,
        help="the directory to write snapshots to (default: next to each manifest)",
    )
    args = parser.parse_args(argv)

    # Check the paths exist
    missing = [str(path) for path in args.paths if not path.exists()]
    if missing:
        print(f"Fabric capture: {' '.join(missing)} does not exist")
        return 1
    manifests = find_manifests(args.paths)
    if not manifests:
        print(f"Fabric capture: no fabric_manifest_rank_*.json in {' '.join(map(str, args.paths))}")
        return 1

    # Open the device using exalens
    started = time.monotonic()
    try:
        device = open_exalens()
    except Exception as error:
        print(f"Fabric capture: ttexalens didn't start: {type(error).__name__}: {error}")
        return 1
    print(f"ttexalens {device.version} started in {time.monotonic() - started:.1f} s")

    # Take the snapshots, print a high level summary of what was captured, and create the snapshot file
    failed = 0
    for manifest_path in manifests:
        started = time.monotonic()
        try:
            snapshot = take_snapshot(
                manifest_path, args.out_dir or manifest_path.parent, device, host=socket.gethostname(), argv=sys.argv
            )
        except (ManifestError, json.JSONDecodeError, OSError) as error:
            failed += 1
            print(f"Fabric capture: no snapshot of {manifest_path}: {error}")
            continue
        print(
            f"{snapshot.json_path}: {snapshot.routers} routers, {len(snapshot.errors)} with errors; "
            f".bin {snapshot.bin_path.stat().st_size:,} bytes; taken in {time.monotonic() - started:.1f} s"
        )
        for path, error in snapshot.errors.items():
            where = f" {error['path']}" if error["path"] is not None else ""
            print(f"  {path}: {error['stage']}{where}: {error['message']}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
