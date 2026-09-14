# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point for one manifest-driven fabric capture."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]
_DEBUG_DIR = Path(__file__).resolve().parents[2]
for directory in (_REPO_ROOT, _DEBUG_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from tt_metal.fabric.debug.visualizer.capture.manifest import ManifestError, load_manifest
from tt_metal.fabric.debug.visualizer.capture.peek import CaptureError, peek_manifest
from tt_metal.fabric.debug.visualizer.capture.snapshot import build_snapshot


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Capture fabric-router state selected by a fabric debug manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="manifest emitted by ControlPlane",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="snapshot JSON file to write",
    )
    return parser.parse_args(argv)


def make_read_u32(context, read_from_device):
    def read_u32(device, location, address):
        try:
            raw = read_from_device(location, address, device.id, 4, context)
        except Exception as error:
            print(
                f"Warning: read failed on device {device.id} at 0x{address:08x}: {error}",
                file=sys.stderr,
            )
            return None

        if isinstance(raw, int):
            return raw
        if isinstance(raw, bytes) and len(raw) >= 4:
            return int.from_bytes(raw[:4], byteorder="little")
        if isinstance(raw, (list, tuple)) and len(raw) >= 4:
            return int.from_bytes(bytes(raw[:4]), byteorder="little")

        print(
            f"Warning: read on device {device.id} at 0x{address:08x} returned "
            f"{type(raw).__name__}, expected four bytes or an integer",
            file=sys.stderr,
        )
        return None

    return read_u32


def write_snapshot(path, snapshot):
    """Atomically replace *path* with a formatted snapshot JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary_path.open("w", encoding="utf-8") as output_file:
            json.dump(snapshot, output_file, indent=2)
            output_file.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def capture(manifest_path, output_path, init_ttexalens, read_from_device):
    manifest = load_manifest(manifest_path)
    context = init_ttexalens()
    sample = peek_manifest(
        manifest,
        context.devices,
        make_read_u32(context, read_from_device),
    )
    snapshot = build_snapshot(manifest, [sample])
    write_snapshot(output_path, snapshot)
    return snapshot


def main(argv=None):
    args = parse_args(argv)
    try:
        # Keep hardware imports out of module scope so --help and offline tests do
        # not require a working ttexalens installation.
        from ttexalens.tt_exalens_init import init_ttexalens  # pyright: ignore[reportMissingImports]
        from ttexalens.tt_exalens_lib import read_from_device  # pyright: ignore[reportMissingImports]

        snapshot = capture(args.manifest, args.output, init_ttexalens, read_from_device)
    except (ManifestError, CaptureError, OSError) as error:
        print(f"fabric capture failed: {error}", file=sys.stderr)
        return 1

    routers = snapshot["samples"][0]["routers"]
    successful = sum(router["ok"] for router in routers)
    print(f"Wrote {args.output}: attempted {len(routers)} routers, {successful} fully readable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
