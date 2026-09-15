# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tt_metal.fabric.debug.visualizer.capture.manifest import ManifestError, load_manifest

MINIMAL_LAYOUT_ID = "L0123456789abcdef"


def default_instance() -> dict:
    return {
        "handshake": 16,
        "sender_channels_per_vc": [1, 0, 0, 0],
        "receiver_channels_per_vc": [1, 0],
        "downstream_edm_mask_per_vc": [1, 0, 0, 0],
        "tensix_extension": False,
        "udm_mode": False,
    }


def minimal_layout(*, router_count: int = 1) -> dict:
    return {
        "regions": [
            {
                "id": "lifecycle",
                "parent": "",
                "backing": "group",
                "allocated": False,
                "enabled": True,
                "writer": "none",
            },
            {
                "id": "lifecycle.handshake",
                "parent": "lifecycle",
                "backing": "unreserved_l1",
                "address": 16,
                "size": 16,
                "allocated": True,
                "enabled": True,
                "writer": "any_erisc",
                "schema": "handshake_info_t",
            },
            {
                "id": "credits.sender.0.free_slots",
                "parent": "",
                "backing": "stream_reg",
                "stream_id": 22,
                "allocated": True,
                "enabled": True,
                "writer": "worker",
            },
            {
                "id": "credits.sender.1.free_slots",
                "parent": "",
                "backing": "stream_reg",
                "stream_id": 23,
                "allocated": True,
                "enabled": False,
                "writer": "worker",
            },
        ],
        "router_count": router_count,
    }


def required_blocks() -> dict:
    region = {"base": 0, "size": 0}
    return {
        "hal": {
            "unreserved": {"base": 0, "size": 1024},
            "go_msg": region,
            "launch": region,
            "fabric_telemetry": region,
            "routing_table": region,
            "router_state": region,
            "router_command": region,
            "eth_fw_mailbox": region,
        },
        "heartbeat": {
            "address": 0x1F80,
            "magic": 0xDCBA0000,
            "magic_mask": 0xFFFF0000,
            "period_iters": 64,
        },
        "fabric_context": {
            "topology": "Linear",
            "is_2d_routing": True,
            "packet_header_size_bytes": 32,
            "max_payload_size_bytes": 4320,
            "channel_buffer_size_bytes": 4352,
            "tensix_enabled": False,
            "bubble_flow_control": False,
        },
        "router_template": {
            "edm_status_address": 1,
            "termination_signal_address": 2,
            "edm_local_sync_address": 3,
            "handshake_address": 4,
            "unused_config_handshake_address": 4,
            "edm_channel_ack_addr": 5,
            "diagnostics": {
                "perf_telemetry": region,
                "code_profiling": region,
                "trimming": region,
            },
            "addresses_to_clear": [],
            "router_buffer_clear_size_words": 1,
        },
        "stream_assignment": {"0": {"VC0_ACK_STREAM": 0}},
        "enums": {
            "EDMStatus": {"READY_FOR_TRAFFIC": 0xA3B3C3D3},
            "TerminationSignal": {"IMMEDIATELY_TERMINATE": 2},
            "RouterCommand": {"RUN": 0},
            "RunMsg": {"RUN_MSG_GO": 0x80, "RUN_MSG_DONE": 0},
        },
        "layouts": {MINIMAL_LAYOUT_ID: minimal_layout()},
    }


def router(eth_chan: int, direction: str = "E") -> dict:
    return {
        "eth_chan": eth_chan,
        "direction": direction,
        "routing_plane": 0,
        "link_class": "intramesh",
        "logical_core": [0, eth_chan],
        "virtual_core": [18 + eth_chan, 16],
        "layout_id": MINIMAL_LAYOUT_ID,
        "instance": default_instance(),
    }


def chip(
    chip_id: int,
    *,
    is_local: bool,
    physical_chip_id: int | None,
    routers: list[dict],
    master_router_chan: int | None = None,
) -> dict:
    if master_router_chan is None:
        master_router_chan = (routers[0]["eth_chan"] if routers else 0) if is_local else None
    return {
        "fabric_chip_id": chip_id,
        "mesh_coord": [0, chip_id],
        "physical_chip_id": physical_chip_id,
        "asic_id": None,
        "is_local": is_local,
        "master_router_chan": master_router_chan,
        "routers": routers,
    }


def manifest(chips: list[dict]) -> dict:
    data = {
        "manifest_version": 1,
        "kind": "fabric_debug_manifest",
        "run": {
            "arch": "WORMHOLE_B0",
            "fabric_config": "FABRIC_2D",
            "fabric_type": "MESH",
            "host_rank": 0,
            "mpi_rank": 0,
            "world_size": 2,
            "written_at": "2026-09-15T00:00:00Z",
        },
        "meshes": [
            {
                "mesh_id": 0,
                "shape": [1, len(chips)],
                "chips": chips,
            }
        ],
        "links": [],
    }
    data.update(required_blocks())
    return data


class ManifestTest(unittest.TestCase):
    def load(self, data: dict):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            return load_manifest(path)

    def test_enumerates_only_this_hosts_routers(self):
        loaded = self.load(
            manifest(
                [
                    chip(0, is_local=True, physical_chip_id=4, routers=[router(1)]),
                    chip(1, is_local=False, physical_chip_id=None, routers=[]),
                    chip(2, is_local=True, physical_chip_id=7, routers=[router(8, "W")]),
                ]
            )
        )

        # physical chip 7 models a UMD-remote chip on the same T3K host. It is
        # still local to this process and therefore remains a legal peek target.
        self.assertEqual(
            [(target.chip_id, target.physical_chip_id, target.eth_chan) for target in loaded.router_targets],
            [(0, 4, 1), (2, 7, 8)],
        )
        self.assertEqual(
            loaded.router_targets[1].endpoint(),
            {"mesh_id": 0, "chip_id": 2, "eth_chan": 8},
        )
        self.assertEqual(loaded.heartbeat["address"], 0x1F80)
        self.assertEqual(loaded.enums["EDMStatus"]["READY_FOR_TRAFFIC"], 0xA3B3C3D3)
        self.assertEqual(loaded.router_targets[0].layout_id, MINIMAL_LAYOUT_ID)

    def test_targets_are_sorted_by_global_endpoint(self):
        loaded = self.load(
            manifest(
                [
                    chip(
                        1,
                        is_local=True,
                        physical_chip_id=0,
                        routers=[router(9, "W"), router(0)],
                    ),
                    chip(0, is_local=True, physical_chip_id=4, routers=[router(6)]),
                ]
            )
        )

        self.assertEqual(
            [(target.chip_id, target.eth_chan) for target in loaded.router_targets],
            [(0, 6), (1, 0), (1, 9)],
        )

    def test_rejects_duplicate_router_endpoint(self):
        data = manifest(
            [chip(0, is_local=True, physical_chip_id=4, routers=[router(1), router(1)])]
        )

        with self.assertRaisesRegex(ManifestError, "duplicate local router endpoint"):
            self.load(data)

    def test_rejects_local_chip_without_physical_id(self):
        data = manifest([chip(0, is_local=True, physical_chip_id=None, routers=[router(1)])])

        with self.assertRaisesRegex(ManifestError, "physical_chip_id must be an integer"):
            self.load(data)

    def test_rejects_unsupported_manifest_version(self):
        data = manifest([])
        data["manifest_version"] = 123456

        with self.assertRaisesRegex(ManifestError, "unsupported manifest_version"):
            self.load(data)

    def test_rejects_missing_required_block(self):
        data = manifest([])
        del data["hal"]

        with self.assertRaisesRegex(ManifestError, "manifest.hal is required"):
            self.load(data)

    def test_sha256_is_hash_of_file_bytes(self):
        data = manifest([chip(0, is_local=True, physical_chip_id=4, routers=[router(1)])])
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.json"
            encoded = json.dumps(data).encode("utf-8")
            path.write_bytes(encoded)
            loaded = load_manifest(path)

        self.assertEqual(loaded.sha256, hashlib.sha256(encoded).hexdigest())
        self.assertRegex(loaded.sha256, r"^[0-9a-f]{64}$")

    def test_layout_id_must_exist(self):
        data = manifest([chip(0, is_local=True, physical_chip_id=4, routers=[router(1)])])
        data["meshes"][0]["chips"][0]["routers"][0]["layout_id"] = "Ldeadbeefdeadbeef"

        with self.assertRaisesRegex(ManifestError, "is not in manifest.layouts"):
            self.load(data)

    def test_unreserved_region_out_of_bounds_rejected(self):
        data = manifest([chip(0, is_local=True, physical_chip_id=4, routers=[router(1)])])
        handshake = data["layouts"][MINIMAL_LAYOUT_ID]["regions"][1]
        handshake["address"] = 2000
        handshake["size"] = 16

        with self.assertRaisesRegex(ManifestError, "lies outside UNRESERVED"):
            self.load(data)

    def test_stream_regs_for_router(self):
        loaded = self.load(manifest([chip(0, is_local=True, physical_chip_id=4, routers=[router(1)])]))
        self.assertEqual(loaded.stream_regs_for_router(0, 0, 1), (22,))
        self.assertEqual(
            loaded.router_layout(0, 0, 1),
            loaded.layouts[MINIMAL_LAYOUT_ID],
        )

    def test_rejects_unknown_parent_and_duplicate_region_id(self):
        data = manifest([chip(0, is_local=True, physical_chip_id=4, routers=[router(1)])])
        regions = data["layouts"][MINIMAL_LAYOUT_ID]["regions"]
        regions[1]["parent"] = "missing.parent"
        with self.assertRaisesRegex(ManifestError, "parent 'missing.parent' is unknown"):
            self.load(data)

        data = manifest([chip(0, is_local=True, physical_chip_id=4, routers=[router(1)])])
        regions = data["layouts"][MINIMAL_LAYOUT_ID]["regions"]
        regions.append(dict(regions[1], id="lifecycle.handshake"))
        with self.assertRaisesRegex(ManifestError, "is duplicated"):
            self.load(data)

    def test_rejects_count_stride_mismatch(self):
        data = manifest([chip(0, is_local=True, physical_chip_id=4, routers=[router(1)])])
        handshake = data["layouts"][MINIMAL_LAYOUT_ID]["regions"][1]
        handshake["schema"] = "packet_ring"
        handshake["count"] = 2
        handshake["stride"] = 16
        handshake["size"] = 16
        with self.assertRaisesRegex(ManifestError, "count\\*stride must equal size"):
            self.load(data)

        padded = dict(handshake)
        padded["id"] = "credits.to_sender_ack"
        padded["parent"] = ""
        padded["schema"] = "u32_counter_array"
        padded["count"] = 2
        padded["stride"] = 4
        padded["size"] = 16
        data = manifest([chip(0, is_local=True, physical_chip_id=4, routers=[router(1)])])
        data["layouts"][MINIMAL_LAYOUT_ID]["regions"].append(padded)
        self.load(data)


if __name__ == "__main__":
    unittest.main()
