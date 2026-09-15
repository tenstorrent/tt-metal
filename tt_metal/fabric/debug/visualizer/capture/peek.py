# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Peek health and overlay stream registers on manifest-selected fabric routers using ttexalens."""

from __future__ import annotations

import sys
import time
from pathlib import Path

_DEBUG_DIR = Path(__file__).resolve().parents[2]
if str(_DEBUG_DIR) not in sys.path:
    sys.path.insert(0, str(_DEBUG_DIR))

from fabric_erisc_constants import ERISC_REGISTERS, STREAM_REGISTER_MASK
from fabric_erisc_utils import detect_device_architecture, get_stream_reg_address, normalize_architecture

from .snapshot import utc_timestamp

ALL_STREAM_IDS = tuple(range(32))
BUF_SPACE_AVAILABLE = "BUF_SPACE_AVAILABLE"


class CaptureError(RuntimeError):
    """Raised when a capture cannot produce trustworthy data at all."""


def _logical_xy(loc):
    info = loc.to("logical")
    if isinstance(info, tuple) and len(info) >= 1:
        first = info[0]
        if isinstance(first, tuple) and len(first) == 2:
            return (int(first[0]), int(first[1]))
        if len(info) == 2 and not isinstance(first, str):
            return (int(info[0]), int(info[1]))
    raise ValueError(f"unexpected logical coordinate {info!r}")


def _lookup_device(devices, physical_chip_id):
    try:
        return devices[physical_chip_id]
    except (KeyError, IndexError, TypeError):
        return None


def _device_by_unique_id(context, asic_id):
    try:
        return context.device_by_unique_id.get(asic_id)
    except (AttributeError, TypeError):
        return None


def resolve_device(target, context):
    """Resolve a target by ASIC identity and cross-check its old cluster chip id."""

    physical_device = _lookup_device(context.devices, target.physical_chip_id)
    if target.asic_id is None:
        print(
            f"Warning: router {target.endpoint()} has no asic_id; "
            f"falling back to physical_chip_id {target.physical_chip_id}",
            file=sys.stderr,
        )
        return physical_device, None

    device = _device_by_unique_id(context, target.asic_id)
    if device is None:
        return None, False
    return device, physical_device is not None and physical_device is device


def resolve_overlay_arch(manifest_arch, device):
    try:
        arch = normalize_architecture(manifest_arch)
    except KeyError as error:
        raise CaptureError(f"manifest run.arch {manifest_arch!r} is not a supported capture architecture") from error

    if device is not None:
        live = detect_device_architecture(device)
        if live != arch:
            print(
                f"Warning: manifest arch {arch!r} disagrees with detected arch {live!r} "
                f"on device {getattr(device, 'id', '<unknown>')}; using {arch!r} from the manifest"
            )
    return arch


def _target_asic_id(target):
    return None if target.asic_id is None else f"0x{target.asic_id:016x}"


def _location_for_target(target, device):
    if device is None:
        return None, (
            f"ASIC 0x{target.asic_id:016x} is not visible to ttexalens"
            if target.asic_id is not None
            else f"device {target.physical_chip_id} is not visible to ttexalens"
        )

    locations = device.get_block_locations("eth")
    if target.eth_chan < 0 or target.eth_chan >= len(locations):
        return None, (
            f"eth_chan {target.eth_chan} is out of range for device {target.physical_chip_id} "
            f"({len(locations)} eth blocks)"
        )

    location = locations[target.eth_chan]
    try:
        logical = _logical_xy(location)
    except (TypeError, ValueError) as error:
        return None, str(error)

    expected = (int(target.logical_core[0]), int(target.logical_core[1]))
    if logical != expected:
        return None, (
            f"logical core mismatch: manifest {list(expected)} vs ttexalens {list(logical)}"
        )
    return location, None


def _read_wall_clock(device, location, read_u32):
    low = read_u32(device, location, ERISC_REGISTERS["ETH_RISC_WALL_CLOCK_0"])
    high = read_u32(device, location, ERISC_REGISTERS["ETH_RISC_WALL_CLOCK_1"])
    if low is None or high is None:
        return None
    return (int(high) << 32) | int(low)


def _reset_bits(reset, overlay_arch):
    if reset is None:
        return {"erisc0": None, "erisc1": None}
    return {
        "erisc0": bool(reset & (1 << 11)),
        "erisc1": bool(reset & (1 << 12)) if overlay_arch == "blackhole" else None,
    }


def _failed_sample(target, error, asic_id_matches_physical=None, liveness=None):
    return {
        "id": target.endpoint(),
        "physical_chip_id": target.physical_chip_id,
        "asic_id": _target_asic_id(target),
        "asic_id_matches_physical": asic_id_matches_physical,
        "status": "unknown",
        "error": error,
        "health": {
            "reset": None,
            "reset_bits": {"erisc0": None, "erisc1": None},
            "wall_clock": None,
        },
        "lifecycle": {
            "edm_status": None,
            "termination_signal": None,
            "go_signal": None,
        },
        "liveness": [] if liveness is None else liveness,
        "streams": {"pre": {}, "post": {}, "torn": False},
        "blobs": {},
        "identity": {
            "my_mesh_id": None,
            "my_device_id": None,
            "matches_manifest": None,
        },
    }


def read_block(device, location, address, size, chunk, read_bytes):
    """Read *size* bytes in *chunk*-sized pieces. Failed pieces are zero-filled."""

    if size < 0:
        raise ValueError("block size must be non-negative")
    if chunk < 1:
        raise ValueError("read chunk must be at least 1")
    if size == 0:
        return b"", "unsupported", "region size is 0"

    payload = bytearray(size)
    errors = []
    offset = 0
    while offset < size:
        piece_size = min(chunk, size - offset)
        piece = read_bytes(device, location, address + offset, piece_size)
        if piece is None:
            errors.append(f"chunk at +{offset} unreadable")
        else:
            if not isinstance(piece, (bytes, bytearray)):
                errors.append(f"chunk at +{offset} returned {type(piece).__name__}")
            elif len(piece) != piece_size:
                errors.append(f"chunk at +{offset} returned {len(piece)} of {piece_size} bytes")
                payload[offset : offset + min(len(piece), piece_size)] = piece[:piece_size]
            else:
                payload[offset : offset + piece_size] = piece
        offset += piece_size

    if errors:
        return bytes(payload), "unreadable", "; ".join(errors)
    return bytes(payload), "ok", None


def _identity_from_routing_table(payload, mesh_id, chip_id):
    if len(payload) < 36:
        return {
            "my_mesh_id": None,
            "my_device_id": None,
            "matches_manifest": None,
        }
    my_mesh_id = int.from_bytes(payload[32:34], "little")
    my_device_id = int.from_bytes(payload[34:36], "little")
    return {
        "my_mesh_id": my_mesh_id,
        "my_device_id": my_device_id,
        "matches_manifest": my_mesh_id == mesh_id and my_device_id == chip_id,
    }


def _default_read_bytes(device, location, address, size):
    return bytes(size)


def stream_ids_for_target(manifest, target, streams="layout"):
    if streams == "all":
        return ALL_STREAM_IDS
    if streams != "layout":
        raise ValueError("streams must be 'layout' or 'all'")
    return manifest.stream_regs_for_router(target.mesh_id, target.chip_id, target.eth_chan)


def _read_streams(device, loc, overlay_arch, stream_ids, read_u32):
    values = {}
    errors = []
    for stream_id in stream_ids:
        address = get_stream_reg_address(stream_id, BUF_SPACE_AVAILABLE, overlay_arch)
        value = read_u32(device, loc, address)
        if value is None:
            errors.append(f"stream {stream_id} unreadable")
            continue
        values[str(stream_id)] = {"buf_space_available": int(value) & STREAM_REGISTER_MASK}
    return values, errors


def peek_router(
    target,
    device,
    asic_id_matches_physical,
    read_u32,
    overlay_arch,
    manifest,
    liveness,
    blob_writer=None,
    read_bytes=None,
    read_chunk=65536,
    include_unreserved=True,
    stream_ids=(),
):
    """Read one fabric-router ERISC. Failures are recorded on the sample, not raised.

    *device* is the already-resolved ttexalens device for this target, or None if
    this host cannot see it.
    """

    loc, location_error = _location_for_target(target, device)
    if location_error is not None:
        return _failed_sample(
            target,
            location_error,
            asic_id_matches_physical,
            liveness,
        )

    errors = []
    reset = read_u32(device, loc, ERISC_REGISTERS["ETH_RISC_RESET"])
    if reset is None:
        errors.append("ETH_RISC_RESET unreadable")

    wall_clock = _read_wall_clock(device, loc, read_u32)
    if wall_clock is None:
        errors.append("wall clock unreadable")

    edm_status = read_u32(device, loc, manifest.router_template["edm_status_address"])
    if edm_status is None:
        errors.append("EDMStatus unreadable")
    termination_signal = read_u32(
        device,
        loc,
        manifest.router_template["termination_signal_address"],
    )
    if termination_signal is None:
        errors.append("termination signal unreadable")
    go_word = read_u32(device, loc, manifest.hal["go_msg"]["base"])
    if go_word is None:
        errors.append("go message unreadable")
    go_signal = None if go_word is None else (int(go_word) >> 24) & 0xFF

    pre_streams, pre_errors = _read_streams(device, loc, overlay_arch, stream_ids, read_u32)
    errors.extend(pre_errors)

    blobs = {}
    identity = {
        "my_mesh_id": None,
        "my_device_id": None,
        "matches_manifest": None,
    }
    bulk_reader = _default_read_bytes if read_bytes is None else read_bytes
    for name, address, size in manifest.hal_regions():
        if name == "unreserved" and not include_unreserved:
            blobs[name] = {
                "address": address,
                "size": size,
                "offset": None,
                "sha256": None,
                "status": "unsupported",
                "error": "skipped (--no-l1-image)",
            }
            continue
        payload, blob_status, blob_error = read_block(
            device,
            loc,
            address,
            size,
            read_chunk,
            bulk_reader,
        )
        offset = None
        digest = None
        if blob_writer is not None and size != 0:
            offset, _, digest = blob_writer.add(payload)
        blobs[name] = {
            "address": address,
            "size": size,
            "offset": offset,
            "sha256": digest,
            "status": blob_status,
            "error": blob_error,
        }
        if blob_status == "unreadable":
            errors.append(f"{name} unreadable")
        if name == "routing_table" and blob_status in ("ok", "unreadable"):
            identity = _identity_from_routing_table(payload, target.mesh_id, target.chip_id)

    post_streams, post_errors = _read_streams(device, loc, overlay_arch, stream_ids, read_u32)
    errors.extend(post_errors)
    torn = pre_streams != post_streams

    for index, liveness_sample in enumerate(liveness):
        if liveness_sample["heartbeat"] is None or liveness_sample["wall_clock"] is None:
            errors.append(f"liveness sample {index} unreadable")

    reset_bits = _reset_bits(reset, overlay_arch)
    unknown_status = (
        edm_status is not None
        and edm_status not in manifest.enums["EDMStatus"].values()
    )
    if errors:
        status = "unreadable"
        error = "; ".join(errors)
    elif reset_bits["erisc0"] or reset_bits["erisc1"]:
        status = "reset"
        error = "one or more ethernet RISC reset bits are asserted"
    elif torn:
        status = "torn"
        error = "stream values moved between pre and post image reads"
    elif unknown_status:
        status = "unknown"
        error = f"EDMStatus value {edm_status} is not in manifest.enums.EDMStatus"
    else:
        status = "ok"
        error = None

    return {
        "id": target.endpoint(),
        "physical_chip_id": target.physical_chip_id,
        "asic_id": _target_asic_id(target),
        "asic_id_matches_physical": asic_id_matches_physical,
        "status": status,
        "error": error,
        "health": {
            "reset": reset,
            "reset_bits": reset_bits,
            "wall_clock": wall_clock,
        },
        "lifecycle": {
            "edm_status": edm_status,
            "termination_signal": termination_signal,
            "go_signal": go_signal,
        },
        "liveness": liveness,
        "streams": {
            "pre": pre_streams,
            "post": post_streams,
            "torn": torn,
        },
        "blobs": blobs,
        "identity": identity,
    }


def peek_manifest(
    manifest,
    context,
    read_u32,
    liveness_samples=1,
    liveness_interval=0.0,
    sleep=time.sleep,
    timestamp=utc_timestamp,
    blob_writer=None,
    read_bytes=None,
    read_chunk=65536,
    include_unreserved=True,
    streams="layout",
):
    """Peek every local router named by *manifest* and return one snapshot sample.

    The sample is stamped with the time the reads began, not the time the JSON
    is later assembled. Reads against a wedged device can block for a while, and
    when a sample was taken is evidence in a hang investigation.
    """

    if liveness_samples < 1:
        raise ValueError("liveness_samples must be at least 1")
    if read_chunk < 1:
        raise ValueError("read_chunk must be at least 1")
    if streams not in ("layout", "all"):
        raise ValueError("streams must be 'layout' or 'all'")

    capture_time = timestamp()
    resolved = {}
    for target in manifest.router_targets:
        chip = (target.mesh_id, target.chip_id)
        if chip not in resolved:
            device, matches_physical = resolve_device(target, context)
            resolved[chip] = (
                device,
                matches_physical,
                resolve_overlay_arch(manifest.run["arch"], device),
            )

    liveness_by_router = {target.sort_key(): [] for target in manifest.router_targets}
    for round_index in range(liveness_samples):
        round_time = timestamp()
        for target in manifest.router_targets:
            device, _, _ = resolved[(target.mesh_id, target.chip_id)]
            location, _ = _location_for_target(target, device)
            if location is None:
                heartbeat = None
                wall_clock = None
            else:
                heartbeat = read_u32(device, location, manifest.heartbeat["address"])
                wall_clock = _read_wall_clock(device, location, read_u32)
            liveness_by_router[target.sort_key()].append(
                {
                    "t": round_time,
                    "heartbeat": heartbeat,
                    "wall_clock": wall_clock,
                }
            )
        if round_index + 1 < liveness_samples:
            sleep(liveness_interval)

    routers = []
    for target in manifest.router_targets:
        chip = (target.mesh_id, target.chip_id)
        device, matches_physical, overlay_arch = resolved[chip]
        routers.append(
            peek_router(
                target,
                device,
                matches_physical,
                read_u32,
                overlay_arch,
                manifest,
                liveness_by_router[target.sort_key()],
                blob_writer=blob_writer,
                read_bytes=read_bytes,
                read_chunk=read_chunk,
                include_unreserved=include_unreserved,
                stream_ids=stream_ids_for_target(manifest, target, streams),
            )
        )
    return {"capture_time": capture_time, "routers": routers}
