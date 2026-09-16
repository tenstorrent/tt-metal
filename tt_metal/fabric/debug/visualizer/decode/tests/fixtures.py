# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Small valid artifacts shared by decode tests."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

from tt_metal.fabric.debug.visualizer.capture.tests.test_manifest import chip, manifest, router


def router_sample(chip_id: int, eth_chan: int, *, status: str = "ok") -> dict:
    return {
        "id": {"mesh_id": 0, "chip_id": chip_id, "eth_chan": eth_chan},
        "physical_chip_id": chip_id,
        "asic_id": f"0x{chip_id + 1:016x}",
        "asic_id_matches_physical": True,
        "status": status,
        "error": None if status == "ok" else status,
        "health": {
            "reset": 0,
            "reset_bits": {"erisc0": status == "reset", "erisc1": None},
            "wall_clock": 123,
        },
        "lifecycle": {
            "edm_status": 0xA3B3C3D3,
            "termination_signal": 0,
            "go_signal": 0x80,
        },
        "liveness": [{"t": "2026-09-15T00:00:00Z", "heartbeat": 0xDCBA0040, "wall_clock": 123}],
        "streams": {
            "pre": {
                "4": {"buf_space_available": 0},
                "13": {"buf_space_available": 4},
                "22": {"buf_space_available": 2},
                "23": {"buf_space_available": 11987},
            },
            "post": {
                "4": {"buf_space_available": 0},
                "13": {"buf_space_available": 4},
                "22": {"buf_space_available": 2},
                "23": {"buf_space_available": 11987},
            },
            "torn": status == "torn",
        },
        "blobs": {
            "unreserved": {
                "address": 0,
                "size": 16,
                "offset": 0,
                "sha256": hashlib.sha256(bytes(16)).hexdigest(),
                "status": "ok",
                "error": None,
            }
        },
        "identity": {
            "my_mesh_id": 0,
            "my_device_id": chip_id,
            "matches_manifest": True,
        },
    }


def write_input(
    directory: Path,
    *,
    rank: int = 0,
    local_chip: int = 0,
    all_chip_ids: tuple[int, ...] = (0,),
    status: str = "ok",
) -> tuple[Path, Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    chips = []
    for chip_id in all_chip_ids:
        is_local = chip_id == local_chip
        chips.append(
            chip(
                chip_id,
                is_local=is_local,
                physical_chip_id=chip_id if is_local else None,
                asic_id=f"0x{chip_id + 1:016x}" if is_local else None,
                routers=[router(chip_id + 1)] if is_local else [],
            )
        )
    manifest_data = manifest(chips)
    manifest_data["run"].update({"mpi_rank": rank, "host_rank": rank, "world_size": len(all_chip_ids)})
    manifest_data["fabric_context"].update(
        {
            "topology": "Mesh",
            "packet_header_size_bytes": 96,
            "routing_2d_route_buffer_size": 36,
        }
    )
    manifest_data["hal"]["fabric_telemetry"] = {"base": 2000, "size": 160}
    manifest_data["hal"]["routing_table"] = {"base": 3000, "size": 2704}
    manifest_data["links"] = [
        {
            "src": {"mesh_id": 0, "chip_id": local_chip, "eth_chan": local_chip + 1},
            "dst": {"mesh_id": 0, "chip_id": local_chip, "eth_chan": local_chip + 1},
        }
    ]
    for chip_row in manifest_data["meshes"][0]["chips"]:
        for router_row in chip_row.get("routers", []):
            router_row["instance"]["worker_sender_channel"] = 0
            router_row["instance"]["credit_plan"] = {
                "vc0_uses_counters": False,
                "vc1_uses_counters": False,
                "vc2_uses_counters": False,
            }
    regions = manifest_data["layouts"]["L0123456789abcdef"]["regions"]
    regions.extend(
        [
            {
                "id": "lifecycle.edm_status",
                "parent": "lifecycle",
                "backing": "unreserved_l1",
                "address": 32,
                "size": 16,
                "allocated": True,
                "enabled": True,
                "writer": "any_erisc",
                "schema": "EDMStatus",
            },
            {
                "id": "sender.0.control.conn_info",
                "parent": "",
                "backing": "unreserved_l1",
                "address": 64,
                "size": 64,
                "allocated": True,
                "enabled": True,
                "writer": "any_erisc",
                "schema": "EDMChannelWorkerLocationInfo",
            },
            {
                "id": "sender.0.control.cursor",
                "parent": "",
                "backing": "unreserved_l1",
                "address": 128,
                "size": 16,
                "allocated": True,
                "enabled": True,
                "writer": "worker",
                "schema": "SenderChannelProducerCursor",
            },
            {
                "id": "credits.counters",
                "parent": "",
                "backing": "unreserved_l1",
                "address": 144,
                "size": 16,
                "count": 4,
                "stride": 4,
                "allocated": True,
                "enabled": True,
                "writer": "peer",
                "schema": "u32_counter_array",
            },
            {
                "id": "future.schema",
                "parent": "",
                "backing": "unreserved_l1",
                "address": 160,
                "size": 4,
                "allocated": True,
                "enabled": True,
                "writer": "any_erisc",
                "schema": "future_type",
            },
            {
                "id": "sender.0.control.connection_sem",
                "parent": "",
                "backing": "unreserved_l1",
                "address": 176,
                "size": 16,
                "allocated": True,
                "enabled": True,
                "writer": "worker",
                "schema": "u32",
            },
            {
                "id": "sender.0.ring",
                "parent": "",
                "backing": "unreserved_l1",
                "address": 256,
                "size": 192,
                "count": 2,
                "stride": 96,
                "allocated": True,
                "enabled": True,
                "writer": "any_erisc",
                "schema": "packet_ring",
            },
            {
                "id": "receiver.0.ring",
                "parent": "",
                "backing": "group",
                "count": 8,
                "allocated": False,
                "enabled": True,
                "writer": "none",
            },
            {
                "id": "receiver.0.pkts_sent",
                "parent": "",
                "backing": "stream_reg",
                "stream_id": 4,
                "allocated": True,
                "enabled": True,
                "writer": "any_erisc",
            },
            {
                "id": "credits.downstream.vc0.edge1.free_slots",
                "parent": "",
                "backing": "stream_reg",
                "stream_id": 13,
                "allocated": True,
                "enabled": True,
                "writer": "peer",
            },
            {
                "id": "hal",
                "parent": "",
                "backing": "group",
                "allocated": False,
                "enabled": True,
                "writer": "none",
            },
            {
                "id": "hal.telemetry",
                "parent": "hal",
                "backing": "fixed_l1",
                "address": 2000,
                "size": 160,
                "allocated": True,
                "enabled": True,
                "writer": "host",
                "schema": "fabric_telemetry",
            },
            {
                "id": "hal.routing_table",
                "parent": "hal",
                "backing": "fixed_l1",
                "address": 3000,
                "size": 2704,
                "allocated": True,
                "enabled": True,
                "writer": "host",
                "schema": "routing_l1_info_t",
            },
        ]
    )
    manifest_path = directory / f"manifest_{rank}.json"
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    raw_path = directory / f"snapshot_{rank}.bin"
    unreserved = bytearray(1024)
    struct.pack_into("<IHB", unreserved, 16, 0xAA, 7, 8)
    struct.pack_into("<I", unreserved, 32, 0xA3B3C3D3)
    struct.pack_into("<I", unreserved, 64, 0x1234)
    struct.pack_into("<I", unreserved, 80, 0x5678)
    struct.pack_into("<HH", unreserved, 96, 3, 4)
    struct.pack_into("<I", unreserved, 112, 99)
    struct.pack_into("<II", unreserved, 128, 17, 1)
    struct.pack_into("<4I", unreserved, 144, 10, 20, 30, 40)
    struct.pack_into("<I", unreserved, 176, 0)
    noc_address = 0x1234 | (5 << 36) | (6 << 42)
    struct.pack_into("<Q", unreserved, 256, noc_address)
    struct.pack_into("<HBB", unreserved, 256 + 40, 64, 0, 2)
    struct.pack_into("<I", unreserved, 256 + 44, 0)
    unreserved[256 + 48 : 256 + 84] = bytes(range(36))
    struct.pack_into("<HH4H", unreserved, 256 + 84, local_chip, 0, 1, 2, 3, 4)
    struct.pack_into("<HBB", unreserved, 352 + 40, 5000, 255, 0)
    struct.pack_into("<HH4H", unreserved, 352 + 84, 99, 99, 0, 0, 0, 0)
    telemetry = bytearray(160)
    struct.pack_into("<IHHBBBBI", telemetry, 0, 1, 0, 1, local_chip, 2, 3, 0xF, 4)
    struct.pack_into("<4Q", telemetry, 16, 1, 2, 3, 4)
    struct.pack_into("<4Q", telemetry, 48, 5, 6, 7, 8)
    struct.pack_into("<I", telemetry, 80, 1)
    struct.pack_into("<QQ", telemetry, 88, 9, 10)
    struct.pack_into("<I", telemetry, 104, 2)
    struct.pack_into("<QQ", telemetry, 112, 11, 12)
    struct.pack_into("<I7I", telemetry, 128, 0xCAFE, *range(7))
    routing = bytearray(2704)
    struct.pack_into("<I", routing, 0, 1)
    struct.pack_into("<I", routing, 16, 0)
    struct.pack_into("<HH", routing, 32, 0, local_chip)
    routing[36] = 0b10001000
    routing[2700:2704] = bytes((0, local_chip, 1, len(all_chip_ids)))
    raw_payload = bytes(unreserved + telemetry + routing)
    raw_path.write_bytes(raw_payload)
    raw_sha = hashlib.sha256(raw_payload).hexdigest()
    sample = router_sample(local_chip, local_chip + 1, status=status)
    sample["blobs"] = {
        "unreserved": {
            "address": 0,
            "size": len(unreserved),
            "offset": 0,
            "sha256": hashlib.sha256(unreserved).hexdigest(),
            "status": "ok",
            "error": None,
        },
        "fabric_telemetry": {
            "address": 2000,
            "size": len(telemetry),
            "offset": len(unreserved),
            "sha256": hashlib.sha256(telemetry).hexdigest(),
            "status": "ok",
            "error": None,
        },
        "routing_table": {
            "address": 3000,
            "size": len(routing),
            "offset": len(unreserved) + len(telemetry),
            "sha256": hashlib.sha256(routing).hexdigest(),
            "status": "ok",
            "error": None,
        },
    }
    snapshot = {
        "snapshot_version": 1,
        "kind": "fabric_debug_snapshot",
        "captured_at": "2026-09-15T00:00:00Z",
        "manifest": {
            "path": "ignored.json",
            "manifest_version": 1,
            "sha256": manifest_sha,
            "run": {
                "arch": manifest_data["run"]["arch"],
                "fabric_config": manifest_data["run"]["fabric_config"],
                "host_rank": rank,
                "mpi_rank": rank,
                "world_size": len(all_chip_ids),
            },
        },
        "provenance": {
            "ttexalens_version": "test",
            "tt_umd_version": "test",
            "hostname": f"host-{rank}",
            "owner_alive": False,
            "argv": ["capture"],
        },
        "raw": {"file": raw_path.name, "size": len(raw_payload), "sha256": raw_sha},
        "samples": [{"capture_time": "2026-09-15T00:00:00Z", "routers": [sample]}],
    }
    snapshot_path = directory / f"snapshot_{rank}.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    return manifest_path, snapshot_path, raw_path
