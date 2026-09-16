# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Assemble and atomically write the decoded-state skeleton."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .credits import annotate_links, decode_channels, stall_score
from .inputs import DecodeInput
from .headers import packet_header_shape
from .liveness import classify_liveness, decode_lifecycle
from .merge import merge_inputs
from .regions import RegionDecoder
from .rings import decode_rings
from .structs import CONNECTION_STATE, ROUTER_STATE

DECODED_VERSION = 1


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _packet_header_type(run: dict[str, Any], context: dict[str, Any]) -> str:
    return packet_header_shape(run, context)[0]


def _input_description(item: DecodeInput) -> dict[str, Any]:
    manifest = item.manifest.manifest
    if item.snapshot is None:
        return {
            "manifest": {
                "path": str(item.manifest.path),
                "sha256": manifest.sha256,
                "mpi_rank": manifest.run["mpi_rank"],
                "host_rank": manifest.run["host_rank"],
                "sha_verified": None,
            },
            "snapshot": None,
            "raw": None,
        }
    raw_reference = item.snapshot["raw"]
    return {
        "manifest": {
            "path": str(item.manifest.path),
            "sha256": manifest.sha256,
            "mpi_rank": manifest.run["mpi_rank"],
            "host_rank": manifest.run["host_rank"],
            "sha_verified": item.manifest_sha_verified,
        },
        "snapshot": {
            "path": str(item.snapshot_path),
            "sha256": item.snapshot_sha256,
            "captured_at": item.snapshot["captured_at"],
            "provenance": dict(item.snapshot.get("provenance", {})),
        },
        "raw": {
            "file": raw_reference["file"],
            "size": raw_reference["size"],
            "sha256": raw_reference["sha256"],
            "present": item.raw is not None,
            "verified": item.raw.verified if item.raw is not None else False,
        },
    }


def build_decoded(
    inputs: tuple[DecodeInput, ...],
    *,
    generated_at: str | None = None,
    expert_raw: bool = False,
    slots: str = "headers",
) -> dict[str, Any]:
    """Build merged and typed offline fabric state."""

    merged = merge_inputs(inputs)
    first_manifest = inputs[0].manifest.manifest
    run = first_manifest.run
    context = dict(first_manifest.fabric_context)
    context["packet_header_type"] = _packet_header_type(run, context)
    region_decoder = RegionDecoder(inputs, expert_raw=expert_raw)
    mesh_ids = {
        int(mesh["mesh_id"])
        for item in inputs
        for mesh in item.manifest.manifest.data["meshes"]
    }
    for router in merged["routers"]:
        router["regions"] = region_decoder.decode_router(router)
        router["warnings"] = []
        input_index = router["capture"]["snapshot_index"]
        if input_index is None:
            router["lifecycle"] = None
            router["liveness"] = None
            router["rings"] = []
            router["channels"] = decode_channels(router)
            router["stall_score"] = None
            continue
        sample = region_decoder.sample_for(input_index, router["id"])
        manifest = inputs[input_index].manifest.manifest
        router["lifecycle"] = decode_lifecycle(sample, manifest.enums)
        router["liveness"] = classify_liveness(sample, manifest.heartbeat)
        routing_region = next(
            (
                region
                for region in router["regions"]
                if region["id"] == "hal.routing_table" and isinstance(region.get("value"), dict)
            ),
            None,
        )
        if routing_region is not None:
            value = routing_region["value"]
            router["identity"] = {
                "my_mesh_id": value["my_mesh_id"],
                "my_device_id": value["my_device_id"],
                "matches": (
                    value["my_mesh_id"] == router["id"]["mesh_id"]
                    and value["my_device_id"] == router["id"]["chip_id"]
                ),
            }
        router["rings"] = decode_rings(
            router,
            region_decoder,
            run=manifest.run,
            context=manifest.fabric_context,
            mesh_ids=mesh_ids,
            slots=slots,
        )
        router["channels"] = decode_channels(router)
        router["stall_score"] = stall_score(router)
    merged["coverage"]["identity_mismatch"] = sum(
        router["identity"]["matches"] is False for router in merged["routers"]
    )
    enums = dict(first_manifest.enums)
    enums.setdefault("RouterState", {name: raw for raw, name in ROUTER_STATE.items()})
    enums.setdefault("ConnectionState", {name: raw for raw, name in CONNECTION_STATE.items()})
    decoded = {
        "decoded_version": DECODED_VERSION,
        "kind": "fabric_debug_decoded",
        "generated_at": generated_at or utc_timestamp(),
        "inputs": [_input_description(item) for item in inputs],
        "run": {
            "arch": run["arch"],
            "fabric_config": run["fabric_config"],
            "topology": context["topology"],
            "is_2d_routing": context["is_2d_routing"],
            "udm_mode": run.get("udm_mode", "DISABLED"),
            "world_size": run["world_size"],
        },
        "fabric_context": context,
        "enums": enums,
        **merged,
    }
    annotate_links(decoded)
    return decoded


def write_decoded(path: str | Path, decoded: dict[str, Any]) -> None:
    """Atomically write decoded JSON."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(decoded, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
