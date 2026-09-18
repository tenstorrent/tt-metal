# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Decode packet headers found at the front of fabric ring slots."""

from __future__ import annotations

import struct
from typing import Any

NOC_SEND_TYPES = {
    0: "NOC_UNICAST_WRITE",
    1: "NOC_UNICAST_INLINE_WRITE",
    2: "NOC_UNICAST_ATOMIC_INC",
    3: "NOC_FUSED_UNICAST_ATOMIC_INC",
    4: "NOC_UNICAST_SCATTER_WRITE",
    5: "NOC_MULTICAST_WRITE",
    6: "NOC_MULTICAST_ATOMIC_INC",
    7: "NOC_UNICAST_READ",
    8: "NOC_SPARSE_MCAST_WRITE",
}

# 1D LowLatencyRoutingFields: 2 bits per hop, LSB-first. Matches RoutingFieldsConstants::LowLatency.
_HOP_ACTIONS = {
    0b00: "noop",
    0b01: "write",
    0b10: "forward",
    0b11: "write_and_forward",
}
_HOPS_PER_WORD = 16

# 2D HybridMeshPacketHeader route_buffer action bits. Matches Routing2DCodec.
_ACTION_BITS = (
    (0, "E"),
    (1, "W"),
    (2, "N"),
    (3, "S"),
    (4, "Z"),
    (5, "local"),
)


def noc_address(raw: int) -> dict[str, int]:
    return {
        "raw": raw,
        "x": (raw >> 36) & 0x3F,
        "y": (raw >> 42) & 0x3F,
        "local_addr": raw & ((1 << 36) - 1),
    }


def packet_header_shape(run: dict[str, Any], context: dict[str, Any]) -> tuple[str, int]:
    """Return the manifest-selected packet header name and computed byte size."""

    if context["is_2d_routing"]:
        route_bytes = int(context["routing_2d_route_buffer_size"])
        base_size = ((60 + route_bytes + 15) // 16) * 16
        if run.get("udm_mode") == "ENABLED":
            return f"UDMHybridMeshPacketHeaderT<{route_bytes}>", base_size + 16
        return f"HybridMeshPacketHeaderT<{route_bytes}>", base_size
    extension_words = int(context["routing_1d_extension_words"])
    return f"LowLatencyPacketHeaderT<{extension_words}>", 48 if extension_words == 0 else 64


def _command(payload: bytes, send_type: int) -> dict[str, Any] | None:
    if send_type in (0, 7):
        return {"noc_address": noc_address(struct.unpack_from("<Q", payload, 0)[0])}
    if send_type == 1:
        return {
            "noc_address": noc_address(struct.unpack_from("<Q", payload, 0)[0]),
            "value": struct.unpack_from("<I", payload, 8)[0],
        }
    if send_type == 2:
        return {
            "noc_address": noc_address(struct.unpack_from("<Q", payload, 0)[0]),
            "value": struct.unpack_from("<I", payload, 8)[0],
            "flush": bool(payload[12]),
        }
    if send_type == 3:
        return {
            "noc_address": noc_address(struct.unpack_from("<Q", payload, 0)[0]),
            "semaphore_noc_address": noc_address(struct.unpack_from("<Q", payload, 8)[0]),
            "value": struct.unpack_from("<I", payload, 16)[0],
            "flush": bool(payload[20]),
        }
    if send_type == 4:
        return {
            "noc_addresses": [noc_address(value) for value in struct.unpack_from("<4Q", payload, 0)],
            "chunk_sizes": list(struct.unpack_from("<3H", payload, 32)),
            "chunk_count": payload[38],
            "chunk_encoding": payload[39],
        }
    if send_type == 5:
        address, x_start, y_start, size_x, size_y = struct.unpack_from("<IBBBB", payload, 0)
        return {
            "local_addr": address,
            "noc_x_start": x_start,
            "noc_y_start": y_start,
            "size_x": size_x,
            "size_y": size_y,
        }
    if send_type == 6:
        address, value, x_start, y_start, size_x, size_y = struct.unpack_from("<IIBBBB", payload, 0)
        return {
            "local_addr": address,
            "value": value,
            "noc_x_start": x_start,
            "noc_y_start": y_start,
            "size_x": size_x,
            "size_y": size_y,
        }
    if send_type == 8:
        return {
            "noc_addresses": [noc_address(value) for value in struct.unpack_from("<4Q", payload, 0)],
            "counts": list(payload[32:36]),
            "num_dests": payload[36],
            "num_chips": payload[37],
            "write_index": payload[38],
            "chip_index": payload[39],
        }
    return None


def decode_1d_hops(value: int, extra_words: list[int] | None = None) -> dict[str, Any]:
    """Unpack the 1D hop tape. Trailing NOOPs are dropped; remaining hops are from this router."""

    hops = []
    for word in [value, *(extra_words or [])]:
        for index in range(_HOPS_PER_WORD):
            hops.append(_HOP_ACTIONS[(word >> (index * 2)) & 0b11])
    last = max((index for index, hop in enumerate(hops) if hop != "noop"), default=-1)
    remaining = hops[: last + 1]
    if not remaining:
        kind = "empty"
    elif remaining[-1] == "write" and all(hop == "forward" for hop in remaining[:-1]):
        kind = "unicast"
    elif "write_and_forward" in remaining or remaining.count("write") > 1:
        kind = "multicast"
    else:
        kind = "other"
    return {
        "value": value,
        "hops": remaining,
        "kind": kind,
        "hops_remaining": len(remaining),
    }


_STEP = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}


def decode_2d_path(
    buffer: bytes,
    *,
    mesh_shape: dict[str, Any] | None,
    coord: dict[str, Any] | None,
    torus: bool = False,
) -> dict[str, Any] | None:
    """Walk the [Y | X] action map from this router. Returns None when the walk
    cannot start (no coord/shape or a short buffer). Otherwise a hop list where a
    multi-direction byte is one 'N+E' split hop and the walk then stops, as do Z
    chords, loops, and leaving a non-torus mesh."""

    if not mesh_shape or coord is None:
        return None
    mesh_y = int(mesh_shape.get("y") or 0)
    mesh_x = int(mesh_shape.get("x") or 0)
    try:
        y = int(coord.get("y"))
        x = int(coord.get("x"))
    except (TypeError, ValueError):
        return None
    if mesh_y <= 0 or mesh_x <= 0 or mesh_y + mesh_x > len(buffer):
        return None
    if not (0 <= y < mesh_y and 0 <= x < mesh_x):
        return None
    hops: list[str] = []
    visited = set()
    while len(hops) <= mesh_y + mesh_x:
        if (y, x) in visited:
            return {"hops": hops, "complete": False}
        visited.add((y, x))
        # The Y byte wins when nonzero, else the X byte carries it.
        raw = buffer[y] or buffer[mesh_y + x]
        dirs = [name for bit, name in _ACTION_BITS if raw & (1 << bit)]
        if not dirs:
            return {"hops": hops, "complete": False}
        if len(dirs) > 1:
            hops.append("+".join(dirs))
            return {"hops": hops, "complete": False}
        (step,) = dirs
        hops.append(step)
        if step == "local":
            return {"hops": hops, "complete": True}
        if step == "Z":
            return {"hops": hops, "complete": False}
        dy, dx = _STEP[step]
        y, x = y + dy, x + dx
        if torus:
            y %= mesh_y
            x %= mesh_x
        elif not (0 <= y < mesh_y and 0 <= x < mesh_x):
            return {"hops": hops, "complete": False}
    return {"hops": hops, "complete": False}


def decode_packet_header(
    payload: bytes,
    *,
    run: dict[str, Any],
    context: dict[str, Any],
    mesh_ids: set[int],
    mesh_shape: dict[str, Any] | None = None,
    mesh_coord: dict[str, Any] | None = None,
    torus: bool = False,
) -> dict[str, Any]:
    """Decode one header; payload bytes after the header are never inspected."""

    name, computed_size = packet_header_shape(run, context)
    expected_size = int(context["packet_header_size_bytes"])
    if computed_size != expected_size:
        raise ValueError(
            f"{name} computes to {computed_size} bytes, manifest says {expected_size}"
        )
    if len(payload) < expected_size:
        raise ValueError(f"{name} needs {expected_size} bytes, got {len(payload)}")

    payload_size, send_type, src_channel = struct.unpack_from("<HBB", payload, 40)
    result = {
        "type": name,
        "payload_size_bytes": payload_size,
        "noc_send_type": {"raw": send_type, "name": NOC_SEND_TYPES.get(send_type)},
        "src_channel": src_channel,
        "command": _command(payload[:40], send_type),
    }
    if context["is_2d_routing"]:
        route_bytes = int(context["routing_2d_route_buffer_size"])
        route_end = 48 + route_bytes
        dst_chip, dst_mesh = struct.unpack_from("<HH", payload, route_end)
        east, west, north, south = struct.unpack_from("<4H", payload, route_end + 4)
        result["routing"] = {
            "destination": {"mesh_id": dst_mesh, "chip_id": dst_chip},
            "mcast": {"E": east, "W": west, "N": north, "S": south},
            "path": decode_2d_path(
                payload[48:route_end], mesh_shape=mesh_shape, coord=mesh_coord, torus=torus
            ),
        }
        if run.get("udm_mode") == "ENABLED":
            result["udm_control_raw"] = payload[computed_size - 16 : computed_size].hex()
        destination_valid = dst_mesh in mesh_ids
    else:
        extension_words = int(context["routing_1d_extension_words"])
        extra = (
            list(struct.unpack_from(f"<{extension_words}I", payload, 48)) if extension_words else []
        )
        result["routing"] = decode_1d_hops(struct.unpack_from("<I", payload, 44)[0], extra)
        destination_valid = True
    result["plausible"] = (
        send_type in NOC_SEND_TYPES
        and payload_size <= int(context["max_payload_size_bytes"])
        and destination_valid
    )
    return result
