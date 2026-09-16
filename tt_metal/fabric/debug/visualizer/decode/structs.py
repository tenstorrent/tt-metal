# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Little-endian decoders for schemas named by router debug layouts."""

from __future__ import annotations

import struct
from typing import Any

ROUTER_STATE = {
    0: "INITIALIZING",
    1: "RUNNING",
    2: "PAUSED",
    3: "DRAINING",
    4: "RETRAINING",
}
CONNECTION_STATE = {0: "unused", 1: "open", 2: "close_request"}
DIRECTION = {0: "E", 1: "W", 2: "N", 3: "S", 4: "Z", 5: "INVALID_DIRECTION", 6: "INVALID_ENTRY"}

RAW_SCHEMAS = {
    "raw",
    "perf_telemetry",
    "code_profiling",
    "channel_trimming",
    "packet_ring",
}
SUPPORTED_SCHEMAS = RAW_SCHEMAS | {
    "u32",
    "EDMStatus",
    "TerminationSignal",
    "handshake_info_t",
    "EDMChannelWorkerLocationInfo",
    "SenderChannelProducerCursor",
    "u32_counter_array",
    "fabric_telemetry",
    "routing_l1_info_t",
}


def _need(payload: bytes, size: int, schema: str) -> None:
    if len(payload) < size:
        raise ValueError(f"{schema} needs {size} bytes, got {len(payload)}")


def enum_value(raw: int, table: dict[str, int]) -> dict[str, Any]:
    return {"raw": raw, "name": next((name for name, value in table.items() if value == raw), None)}


def unpack_direction_table(payload: bytes, count: int) -> list[str]:
    """Port direction_table_t::get_direction/decompress_value."""

    result = []
    for index in range(count):
        bit_index = index * 3
        byte_index, bit_offset = divmod(bit_index, 8)
        word = payload[byte_index]
        if byte_index + 1 < len(payload):
            word |= payload[byte_index + 1] << 8
        compressed = (word >> bit_offset) & 0x7
        result.append(DIRECTION.get(compressed, "INVALID_ENTRY"))
    return result


def _u64(payload: bytes, offset: int) -> int:
    return struct.unpack_from("<Q", payload, offset)[0]


def _bandwidth(payload: bytes, offset: int) -> dict[str, int]:
    return {
        "elapsed_active_cycles": _u64(payload, offset),
        "elapsed_cycles": _u64(payload, offset + 8),
        "num_words_sent": _u64(payload, offset + 16),
        "num_packets_sent": _u64(payload, offset + 24),
    }


def decode_fabric_telemetry(payload: bytes) -> dict[str, Any]:
    _need(payload, 160, "fabric_telemetry")
    version, mesh_id, neighbor_mesh_id, device_id, neighbor_device_id, direction, stats, fabric_config = (
        struct.unpack_from("<IHHBBBBI", payload, 0)
    )
    eriscs = []
    for offset in (80, 104):
        router_state = struct.unpack_from("<I", payload, offset)[0]
        eriscs.append(
            {
                "router_state": {"raw": router_state, "name": ROUTER_STATE.get(router_state)},
                "tx_heartbeat": _u64(payload, offset + 8),
                "rx_heartbeat": _u64(payload, offset + 16),
            }
        )
    return {
        "static_info": {
            "version": version,
            "mesh_id": mesh_id,
            "neighbor_mesh_id": neighbor_mesh_id,
            "device_id": device_id,
            "neighbor_device_id": neighbor_device_id,
            "direction": direction,
            "supported_stats": stats,
            "fabric_config": fabric_config,
        },
        "dynamic_info": {
            "tx_bandwidth": _bandwidth(payload, 16),
            "rx_bandwidth": _bandwidth(payload, 48),
            "erisc": eriscs,
        },
        "postcode": struct.unpack_from("<I", payload, 128)[0],
        "scratch": list(struct.unpack_from("<7I", payload, 132)),
    }


def decode_routing_l1_info(
    payload: bytes,
    enums: dict[str, dict[str, int]],
    *,
    mesh_count: int,
) -> dict[str, Any]:
    _need(payload, 2704, "routing_l1_info_t")
    state = struct.unpack_from("<I", payload, 0)[0]
    command = struct.unpack_from("<I", payload, 16)[0]
    mesh_id, device_id = struct.unpack_from("<HH", payload, 32)
    mesh_y, mesh_x = payload[2702], payload[2703]
    active_chips = min(mesh_y * mesh_x, 256)
    return {
        "state": {"raw": state, "name": ROUTER_STATE.get(state)},
        "command": enum_value(command, enums.get("RouterCommand", {})),
        "my_mesh_id": mesh_id,
        "my_device_id": device_id,
        "intra_mesh_directions": unpack_direction_table(payload[36:132], active_chips),
        "inter_mesh_directions": unpack_direction_table(payload[132:516], min(mesh_count, 1024)),
        "route_table_raw_offset": 516,
        "exit_node_table_raw_offset": 1676,
        "my_mesh_coord": {"y": payload[2700], "x": payload[2701]},
        "mesh_shape": {"y": mesh_y, "x": mesh_x},
    }


def decode_payload(
    schema: str,
    payload: bytes,
    *,
    enums: dict[str, dict[str, int]],
    region: dict[str, Any],
    mesh_count: int,
) -> Any:
    """Decode one captured region. Unknown schemas are rejected by the caller."""

    if schema in RAW_SCHEMAS:
        return None
    if schema == "u32":
        words = list(struct.unpack(f"<{len(payload) // 4}I", payload[: len(payload) // 4 * 4]))
        return {"word": words[0] if words else None, "words": words}
    if schema in ("EDMStatus", "TerminationSignal"):
        _need(payload, 4, schema)
        return enum_value(struct.unpack_from("<I", payload)[0], enums.get(schema, {}))
    if schema == "handshake_info_t":
        _need(payload, 8, schema)
        local, neighbor_mesh, neighbor_device = struct.unpack_from("<IHB", payload)
        return {
            "local_value": local,
            "neighbor_mesh_id": neighbor_mesh,
            "neighbor_device_id": neighbor_device,
        }
    if schema == "EDMChannelWorkerLocationInfo":
        _need(payload, 52, schema)
        x, y = struct.unpack_from("<HH", payload, 32)
        return {
            "worker_semaphore_address": struct.unpack_from("<I", payload, 0)[0],
            "worker_teardown_semaphore_address": struct.unpack_from("<I", payload, 16)[0],
            "worker_xy": {"x": x, "y": y},
            "edm_read_counter": struct.unpack_from("<I", payload, 48)[0],
        }
    if schema == "SenderChannelProducerCursor":
        _need(payload, 8, schema)
        write_counter, write_index = struct.unpack_from("<II", payload)
        return {"write_counter": write_counter, "write_index": write_index}
    if schema == "u32_counter_array":
        count = min(int(region.get("count", len(payload) // 4)), len(payload) // 4)
        return {"counters": list(struct.unpack_from(f"<{count}I", payload))}
    if schema == "fabric_telemetry":
        return decode_fabric_telemetry(payload)
    if schema == "routing_l1_info_t":
        return decode_routing_l1_info(payload, enums, mesh_count=mesh_count)
    raise KeyError(schema)
