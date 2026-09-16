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


def decode_packet_header(
    payload: bytes,
    *,
    run: dict[str, Any],
    context: dict[str, Any],
    mesh_ids: set[int],
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
        routing_value = struct.unpack_from("<I", payload, 44)[0]
        route_end = 48 + route_bytes
        dst_chip, dst_mesh = struct.unpack_from("<HH", payload, route_end)
        result["routing"] = {
            "value": routing_value,
            "route_buffer": payload[48:route_end].hex(),
            "destination": {"mesh_id": dst_mesh, "chip_id": dst_chip},
            "mcast_params": list(struct.unpack_from("<4H", payload, route_end + 4)),
        }
        if run.get("udm_mode") == "ENABLED":
            result["udm_control_raw"] = payload[computed_size - 16 : computed_size].hex()
        destination_valid = dst_mesh in mesh_ids
    else:
        extension_words = int(context["routing_1d_extension_words"])
        result["routing"] = {
            "value": struct.unpack_from("<I", payload, 44)[0],
            "route_buffer": list(struct.unpack_from(f"<{extension_words}I", payload, 48))
            if extension_words
            else [],
        }
        destination_valid = True
    result["plausible"] = (
        send_type in NOC_SEND_TYPES
        and payload_size <= int(context["max_payload_size_bytes"])
        and destination_valid
    )
    return result
