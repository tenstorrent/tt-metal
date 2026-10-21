# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Decodes each fabric snapshot given, with the manifest it was taken with.

Usage, from the repository root:
    python3 -m tt_metal.fabric.debug.visualizer.decode SNAPSHOT|DIR [SNAPSHOT|DIR ...] [-o DIR]
        [--manifest MANIFEST|DIR ...]

A DIR stands for the fabric_snapshot_*.json files in it. Each decoded file goes next to its snapshot, or into -o's
directory, named after the snapshot: fabric_decoded_rank_<r>_of_<n>_<UTC>.json. A snapshot's manifest is the one it
names, one of that name beside it, or, with --manifest, whichever of those manifests has its sha256. Prints one line per
decoded file, then its warnings and router errors. Exits non-zero when a snapshot is refused or its decoded file can't be
written.
"""

import argparse
import datetime
import pathlib
import sys
import time

from tt_metal.fabric.debug.visualizer.decode.decoded import decode_snapshot, get_decoded_name, write_decoded
from tt_metal.fabric.debug.visualizer.decode.inputs import get_checkout, load_input


def find_files(paths: list, pattern: str) -> list:
    """Each file in `paths`, and each directory's files matching `pattern`, in name order."""
    files = []
    for path in paths:
        files.extend(sorted(path.glob(pattern)) if path.is_dir() else [path])
    return files


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m tt_metal.fabric.debug.visualizer.decode",
        description="Decodes each fabric snapshot given, with the manifest it was taken with.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=pathlib.Path,
        metavar="SNAPSHOT|DIR",
        help="a snapshot's .json, or a directory of fabric_snapshot_*.json files",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=pathlib.Path,
        help="the directory to write decoded files to (default: next to each snapshot)",
    )
    parser.add_argument(
        "--manifest",
        nargs="+",
        action="extend",
        type=pathlib.Path,
        default=[],
        metavar="MANIFEST|DIR",
        help="manifests to choose from by sha256, or directories of fabric_manifest_rank_*.json files, for snapshots "
        "whose manifest has moved",
    )
    args = parser.parse_args(argv)

    missing = [str(path) for path in [*args.paths, *args.manifest] if not path.exists()]
    if missing:
        print(f"Fabric decode: {' '.join(missing)} does not exist")
        return 1
    snapshots = find_files(args.paths, "fabric_snapshot_*.json")
    if not snapshots:
        print(f"Fabric decode: no fabric_snapshot_*.json in {' '.join(map(str, args.paths))}")
        return 1
    manifests = find_files(args.manifest, "fabric_manifest_rank_*.json")
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    checkout = get_checkout(pathlib.Path(__file__).resolve().parent)

    failed = 0
    for snapshot_path in snapshots:
        started = time.monotonic()
        out_path = (args.out_dir or snapshot_path.parent) / get_decoded_name(snapshot_path)
        try:
            loaded = load_input(snapshot_path, checkout, manifests)
            decoded = decode_snapshot(loaded, checkout, datetime.datetime.now(datetime.timezone.utc))
            write_decoded(decoded, out_path)
        except (ValueError, OSError) as error:
            failed += 1
            print(f"Fabric decode: {snapshot_path} not decoded: {error}")
            continue
        errors = {path: router["error"] for path, router in decoded["routers"].items() if router["error"] is not None}
        print(
            f"{out_path}: {len(decoded['routers'])} routers, {len(errors)} with capture errors; "
            f"decoded in {time.monotonic() - started:.1f} s"
        )
        for warning in loaded.warnings:
            print(f"  warning: {warning}")
        for path, error in errors.items():
            where = f" {error['path']}" if error.get("path") is not None else ""
            print(f"  {path}: {error['stage']}{where}: {error['message']}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
