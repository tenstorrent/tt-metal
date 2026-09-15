# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point for one manifest-driven fabric capture."""

from __future__ import annotations

import argparse
import collections
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
from tt_metal.fabric.debug.visualizer.capture.rawfile import RawBlobWriter
from tt_metal.fabric.debug.visualizer.capture.snapshot import build_snapshot, capture_provenance


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Capture fabric-router state selected by a fabric debug manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="fabric instance manifest describing the fabric topology and router targets",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="snapshot JSON file to write",
    )
    parser.add_argument(
        "--liveness-samples",
        type=int,
        default=3,
        help="heartbeat/wall-clock rounds before the main capture (default: 3)",
    )
    parser.add_argument(
        "--liveness-interval",
        type=float,
        default=1.0,
        help="seconds between liveness rounds (default: 1.0)",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="raw blob sidecar path (default: output with .bin suffix)",
    )
    parser.add_argument(
        "--read-chunk",
        type=int,
        default=65536,
        help="bulk L1 read size in bytes (default: 65536)",
    )
    parser.add_argument(
        "--no-l1-image",
        action="store_true",
        help="skip the UNRESERVED L1 image; still capture HAL siblings",
    )
    parser.add_argument(
        "--streams",
        choices=("layout", "all"),
        default="layout",
        help="overlay streams to peek: layout-allocated (default) or 0-31",
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


def make_read_block(context, read_from_device):
    def read_bytes(device, location, address, size):
        try:
            raw = read_from_device(location, address, device.id, size, context)
        except Exception as error:
            print(
                f"Warning: read failed on device {device.id} at 0x{address:08x}: {error}",
                file=sys.stderr,
            )
            return None

        if isinstance(raw, bytes):
            return raw
        if isinstance(raw, bytearray):
            return bytes(raw)
        if isinstance(raw, int):
            return raw.to_bytes(4, "little")[:size].ljust(size, b"\x00")
        if isinstance(raw, (list, tuple)):
            return bytes(raw[:size])

        print(
            f"Warning: bulk read on device {device.id} at 0x{address:08x} returned "
            f"{type(raw).__name__}, expected bytes",
            file=sys.stderr,
        )
        return None

    return read_bytes


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


def capture(
    manifest_path,
    output_path,
    init_ttexalens,
    read_from_device,
    provenance=None,
    liveness_samples=3,
    liveness_interval=1.0,
    sleep=None,
    raw_path=None,
    read_chunk=65536,
    include_unreserved=True,
    streams="layout",
):
    manifest = load_manifest(manifest_path)
    context = init_ttexalens()
    peek_arguments = {
        "liveness_samples": liveness_samples,
        "liveness_interval": liveness_interval,
        "read_bytes": make_read_block(context, read_from_device),
        "read_chunk": read_chunk,
        "include_unreserved": include_unreserved,
        "streams": streams,
    }
    if sleep is not None:
        peek_arguments["sleep"] = sleep
    sidecar = Path(output_path).with_suffix(".bin") if raw_path is None else Path(raw_path)
    writer = RawBlobWriter(sidecar)
    try:
        sample = peek_manifest(
            manifest,
            context,
            make_read_u32(context, read_from_device),
            blob_writer=writer,
            **peek_arguments,
        )
        raw_size, raw_sha256 = writer.close()
    except Exception:
        writer.abort()
        raise
    snapshot = build_snapshot(
        manifest,
        [sample],
        capture_provenance() if provenance is None else provenance,
        raw={"file": sidecar.name, "size": raw_size, "sha256": raw_sha256},
    )
    write_snapshot(output_path, snapshot)
    return snapshot


def main(argv=None):
    args = parse_args(argv)
    try:
        # Keep hardware imports out of module scope so --help and offline tests do
        # not require a working ttexalens installation.
        from ttexalens.tt_exalens_init import init_ttexalens  # pyright: ignore[reportMissingImports]
        from ttexalens.tt_exalens_lib import read_from_device  # pyright: ignore[reportMissingImports]

        snapshot = capture(
            args.manifest,
            args.output,
            init_ttexalens,
            read_from_device,
            liveness_samples=args.liveness_samples,
            liveness_interval=args.liveness_interval,
            raw_path=args.raw,
            read_chunk=args.read_chunk,
            include_unreserved=not args.no_l1_image,
            streams=args.streams,
        )
    except (ManifestError, CaptureError, OSError) as error:
        print(f"fabric capture failed: {error}", file=sys.stderr)
        return 1

    routers = snapshot["samples"][0]["routers"]
    status_counts = collections.Counter(router["status"] for router in routers)
    summary = ", ".join(
        f"{count} {status}" for status, count in sorted(status_counts.items())
    )
    print(f"Wrote {args.output}: {len(routers)} routers: {summary}")
    print(f"Wrote {snapshot['raw']['file']}: {snapshot['raw']['size']} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
