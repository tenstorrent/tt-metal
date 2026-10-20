# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Takes a fabric snapshot of information, where L1 addresses and registers are named in the manifest, on every router core
through a device, and writes the L1 bytes to a .bin file and everything else to a .json file beside it. README.md in
this directory describes the format.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import pathlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from tt_metal.fabric.debug.visualizer.capture.reads import (
    REGISTER_BYTES,
    L1Read,
    ReadPlan,
    RegisterRead,
    RouterPlan,
    get_read_plan,
)

SNAPSHOT_VERSION = 1


class CaptureError(Exception):
    """A device couldn't open a router's core or read from it."""


@dataclass(frozen=True)
class OpenedRouter:
    """A router core a device has found."""

    handle: object  # what the device needs to read the core
    ttexalens_id: int  # ttexalens's id for the core's chip


class Device(Protocol):
    version: str  # the version of the library that reads the device

    def open_router(self, router: RouterPlan) -> OpenedRouter:
        """Finds the router's chip and core, or raises CaptureError."""

    def read(self, opened: OpenedRouter, address: int, size: int) -> bytes:
        """`size` bytes at `address` on the router's core, or raises CaptureError."""


@dataclass(frozen=True)
class Snapshot:
    json_path: pathlib.Path
    bin_path: pathlib.Path
    errors: dict[str, dict]  # each failed router's error, by the router's path


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _utc_text(moment: datetime.datetime) -> str:
    return moment.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def snapshot_name(manifest_path: pathlib.Path, started_at: datetime.datetime) -> str:
    """The snapshot's file name without its suffix, such as fabric_snapshot_rank_1_of_1_20261019T153000Z."""
    stem = manifest_path.stem
    rank = stem[len("fabric_manifest") :] if stem.startswith("fabric_manifest") else f"_{stem}"
    return f"fabric_snapshot{rank}_{started_at.astimezone(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


class _RawFile:
    """The .bin file, written to a temporary file as the reads arrive and renamed into place when complete."""

    def __init__(self, path: pathlib.Path):
        self.path = path
        self.temporary = path.with_name(path.name + ".tmp")
        self.file = open(self.temporary, "wb")
        self.sha256 = hashlib.sha256()
        self.size = 0

    def append(self, data: bytes) -> int:
        """Appends `data` and returns its offset in the file."""
        offset = self.size
        self.file.write(data)
        self.sha256.update(data)
        self.size += len(data)
        return offset

    def complete(self):
        self.file.close()
        os.replace(self.temporary, self.path)

    def discard(self):
        self.file.close()
        self.temporary.unlink(missing_ok=True)


def _write_file(path: pathlib.Path, data: bytes):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _read_bytes(device: Device, opened: OpenedRouter, read: L1Read) -> bytes:
    data = device.read(opened, read.address, read.size)
    if len(data) != read.size:
        raise CaptureError(f"read {len(data)} bytes at {read.address:#x}, not {read.size}")
    return data


def _read_word(device: Device, opened: OpenedRouter, register: RegisterRead) -> int:
    data = device.read(opened, register.address, REGISTER_BYTES)
    if len(data) != REGISTER_BYTES:
        raise CaptureError(f"read {len(data)} bytes at {register.address:#x}, not {REGISTER_BYTES}")
    return int.from_bytes(data, "little")


def _router_entry(router: RouterPlan) -> dict:
    return {
        "device": None,
        "core": {"translated": list(router.virtual_core), "logical": list(router.logical_core)},
        "started_ns": None,
        "finished_ns": None,
        "core_registers": {},
        "areas": {},
        "regions": {},
        "streams": {},
        "error": None,
    }


def _read_router(
    device: Device,
    plan: ReadPlan,
    router: RouterPlan,
    opened: OpenedRouter,
    entry: dict,
    raw: _RawFile,
    monotonic_ns: Callable[[], int],
):
    """The main pass on one router: core registers, stream registers, areas, regions, then stream registers again."""
    entry["started_ns"] = monotonic_ns()
    stage, key = "", ""
    try:
        stage = "core_registers"
        for key, register in plan.core_registers.items():
            entry["core_registers"][key] = _read_word(device, opened, register)

        stage = "streams_before"
        for (stream_id, register_name), register in router.streams.items():
            key = f"{stream_id}/{register_name}"
            value = _read_word(device, opened, register)
            entry["streams"].setdefault(str(stream_id), {})[register_name] = {"before": value}

        stage = "areas"
        for key, area in plan.areas.items():
            offset = raw.append(_read_bytes(device, opened, area))
            entry["areas"][key] = {"address": area.address, "size": area.size, "offset": offset}

        stage = "regions"
        for key, region in router.regions.items():
            offset = raw.append(_read_bytes(device, opened, region))
            entry["regions"][key] = {"address": region.address, "size": region.size, "offset": offset}

        stage = "streams_after"
        for (stream_id, register_name), register in router.streams.items():
            key = f"{stream_id}/{register_name}"
            entry["streams"][str(stream_id)][register_name]["after"] = _read_word(device, opened, register)
    except CaptureError as error:
        entry["error"] = {"stage": stage, "path": key, "message": str(error)}
    entry["finished_ns"] = monotonic_ns()


def take_snapshot(
    manifest_path: pathlib.Path,
    out_dir: pathlib.Path,
    device: Device,
    *,
    host: str,
    argv: list[str],
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    utc_now: Callable[[], datetime.datetime] = _utc_now,
) -> Snapshot:
    """
    Reads every local router of the manifest at `manifest_path` through `device`, and writes the snapshot to `out_dir`.
    A router whose core can't be opened or read records the error and the rest carry on. Raises ManifestError, or
    json.JSONDecodeError, when the manifest can't be used, before anything is read or written.
    """
    # Load the manifest
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)

    # Construct the read plan, which tells us what to read on each router core, by parsing the manifest
    plan = get_read_plan(manifest)

    # Set up the snapshot file names and directories
    started_at = utc_now()
    name = snapshot_name(manifest_path, started_at)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{name}.json"
    bin_path = out_dir / f"{name}.bin"
    for path in (json_path, bin_path):
        if path.exists():
            raise FileExistsError(f"{path} already exists")

    # Set up the router entries, and find every router's chip and core
    routers = {router.path: _router_entry(router) for router in plan.routers}
    opened = {}
    for router in plan.routers:
        entry = routers[router.path]
        try:
            opened[router.path] = device.open_router(router)
        except CaptureError as error:
            entry["error"] = {"stage": "open", "path": None, "message": str(error)}
            continue
        entry["device"] = {"unique_id": f"0x{router.asic_id:016x}", "ttexalens_id": opened[router.path].ttexalens_id}

    # Read the router information
    raw = _RawFile(bin_path)
    try:
        for router in plan.routers:
            if router.path in opened:
                _read_router(device, plan, router, opened[router.path], routers[router.path], raw, monotonic_ns)
    except BaseException:
        raw.discard()
        raise
    raw.complete()

    # Write the snapshot
    snapshot = {
        "kind": "fabric_snapshot",
        "snapshot_version": SNAPSHOT_VERSION,
        "manifest": {
            "file": str(manifest_path.resolve()),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "run": manifest.get("run"),
        },
        "capture": {
            "started_at": _utc_text(started_at),
            "finished_at": _utc_text(utc_now()),
            "host": host,
            "argv": list(argv),
            "ttexalens": device.version,
        },
        "raw": {"file": bin_path.name, "size": raw.size, "sha256": raw.sha256.hexdigest()},
        "routers": routers,
    }
    _write_file(json_path, (json.dumps(snapshot, indent=2) + "\n").encode())
    errors = {path: entry["error"] for path, entry in routers.items() if entry["error"] is not None}

    return Snapshot(json_path=json_path, bin_path=bin_path, errors=errors)
