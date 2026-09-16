# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Decode packet headers from manifest-described channel rings."""

from __future__ import annotations

from typing import Any

from .headers import decode_packet_header, packet_header_shape
from .regions import RegionDecoder


def decode_rings(
    router: dict[str, Any],
    region_decoder: RegionDecoder,
    *,
    run: dict[str, Any],
    context: dict[str, Any],
    mesh_ids: set[int],
    slots: str,
) -> list[dict[str, Any]]:
    result = []
    input_index = router["capture"]["snapshot_index"]
    if input_index is None:
        return result

    _, computed_size = packet_header_shape(run, context)
    expected_size = int(context["packet_header_size_bytes"])
    for region in router["regions"]:
        if region.get("schema") != "packet_ring":
            continue
        ring = {
            "id": region["id"],
            "status": region["status"],
            "error": region["error"],
            "depth": int(region.get("count", 0)),
            "stride": int(region.get("stride", 0)),
            "occupied_count": None,
            "occupancy_source": None,
            "occupancy_status": "unknown",
            "raw_ref": region.get("raw_ref"),
        }
        if slots == "none":
            result.append(ring)
            continue
        ring["slots"] = []
        if region["status"] != "ok" or region.get("raw_ref") is None:
            result.append(ring)
            continue
        if computed_size != expected_size:
            ring.update(
                status="unsupported",
                error=f"computed packet header size {computed_size} != manifest size {expected_size}",
            )
            result.append(ring)
            continue
        payload = region_decoder.raw_slice(input_index, region["raw_ref"])
        if payload is None:
            ring.update(status="not_captured", error="raw sidecar is unavailable")
            result.append(ring)
            continue
        stride = ring["stride"]
        for index in range(ring["depth"]):
            offset = index * stride
            header_bytes = payload[offset : offset + expected_size]
            slot = {
                "index": index,
                "slot_state": "unknown",
                "raw_ref": {
                    "file": region["raw_ref"]["file"],
                    "offset": region["raw_ref"]["offset"] + offset,
                    "size": expected_size,
                },
                "header": None,
                "error": None,
            }
            try:
                slot["header"] = decode_packet_header(
                    header_bytes,
                    run=run,
                    context=context,
                    mesh_ids=mesh_ids,
                )
            except ValueError as error:
                slot["error"] = str(error)
            ring["slots"].append(slot)
        result.append(ring)
    return result
