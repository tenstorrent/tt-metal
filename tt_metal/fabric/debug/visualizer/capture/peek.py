# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Peek health and overlay stream registers on manifest-selected fabric routers using ttexalens."""

from __future__ import annotations

import sys
from pathlib import Path

_DEBUG_DIR = Path(__file__).resolve().parents[2]
if str(_DEBUG_DIR) not in sys.path:
    sys.path.insert(0, str(_DEBUG_DIR))

from fabric_erisc_constants import ERISC_REGISTERS, STREAM_REGISTER_MASK
from fabric_erisc_utils import detect_device_architecture, get_stream_reg_address, normalize_architecture

from .snapshot import utc_timestamp

FABRIC_STREAM_IDS = range(30)
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


def _failed_sample(target, error):
    return {
        "id": target.endpoint(),
        "physical_chip_id": target.physical_chip_id,
        "ok": False,
        "error": error,
        "health": {"reset": None, "wall_clock": None},
        "streams": {},
    }


def peek_router(target, device, read_u32, overlay_arch):
    """Read one fabric-router ERISC. Failures are recorded on the sample, not raised.

    *device* is the already-resolved ttexalens device for this target, or None if
    this host cannot see it.
    """

    if device is None:
        return _failed_sample(
            target,
            f"device {target.physical_chip_id} is not visible to ttexalens",
        )

    locations = device.get_block_locations("eth")
    if target.eth_chan < 0 or target.eth_chan >= len(locations):
        return _failed_sample(
            target,
            f"eth_chan {target.eth_chan} is out of range for device {target.physical_chip_id} "
            f"({len(locations)} eth blocks)",
        )

    loc = locations[target.eth_chan]
    try:
        logical = _logical_xy(loc)
    except (TypeError, ValueError) as error:
        return _failed_sample(target, str(error))

    expected = (int(target.logical_core[0]), int(target.logical_core[1]))
    if logical != expected:
        return _failed_sample(
            target,
            f"logical core mismatch: manifest {list(expected)} vs ttexalens {list(logical)}",
        )

    errors = []
    reset = read_u32(device, loc, ERISC_REGISTERS["ETH_RISC_RESET"])
    if reset is None:
        errors.append("ETH_RISC_RESET unreadable")

    wall_low = read_u32(device, loc, ERISC_REGISTERS["ETH_RISC_WALL_CLOCK_0"])
    wall_high = read_u32(device, loc, ERISC_REGISTERS["ETH_RISC_WALL_CLOCK_1"])
    wall_clock = None
    if wall_low is None or wall_high is None:
        errors.append("wall clock unreadable")
    else:
        wall_clock = (int(wall_high) << 32) | int(wall_low)

    streams = {}
    for stream_id in FABRIC_STREAM_IDS:
        address = get_stream_reg_address(stream_id, BUF_SPACE_AVAILABLE, overlay_arch)
        value = read_u32(device, loc, address)
        if value is None:
            errors.append(f"stream {stream_id} unreadable")
            continue
        streams[str(stream_id)] = {"buf_space_available": int(value) & STREAM_REGISTER_MASK}

    return {
        "id": target.endpoint(),
        "physical_chip_id": target.physical_chip_id,
        "ok": len(errors) == 0,
        "error": None if not errors else "; ".join(errors),
        "health": {"reset": reset, "wall_clock": wall_clock},
        "streams": streams,
    }


def peek_manifest(manifest, devices, read_u32):
    """Peek every local router named by *manifest* and return one snapshot sample.

    The sample is stamped with the time the reads began, not the time the JSON
    is later assembled. Reads against a wedged device can block for a while, and
    when a sample was taken is evidence in a hang investigation.
    """

    capture_time = utc_timestamp()
    resolved = {}
    routers = []
    for target in manifest.router_targets:
        chip = target.physical_chip_id
        if chip not in resolved:
            device = _lookup_device(devices, chip)
            resolved[chip] = (device, resolve_overlay_arch(manifest.run["arch"], device))
        device, overlay_arch = resolved[chip]
        routers.append(peek_router(target, device, read_u32, overlay_arch))
    return {"capture_time": capture_time, "routers": routers}
