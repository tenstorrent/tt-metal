# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path

import jsonschema

from tt_metal.fabric.debug.visualizer.capture.manifest import load_manifest
from tt_metal.fabric.debug.visualizer.capture.peek import FABRIC_STREAM_IDS, CaptureError, peek_manifest
from tt_metal.fabric.debug.visualizer.capture.snapshot import build_snapshot
from tt_metal.fabric.debug.visualizer.capture.tests.test_manifest import required_blocks


SCHEMA_PATH = Path(__file__).parents[2] / "schema" / "fabric_debug_snapshot_schema.json"


class FakeLoc:
    def __init__(self, logical):
        self.logical = logical

    def to(self, system):
        if system == "logical":
            return (self.logical, "eth")
        raise KeyError(system)


class FakeDevice:
    def __init__(self, device_id, locations, arch="wormhole_b0"):
        self.id = device_id
        self._arch = arch
        self._locations = locations

    def get_block_locations(self, block_type):
        if block_type != "eth":
            return []
        return self._locations


def fixture_manifest():
    data = {
        "manifest_version": 1,
        "kind": "fabric_debug_manifest",
        "run": {
            "arch": "WORMHOLE_B0",
            "fabric_config": "FABRIC_2D",
            "fabric_type": "MESH",
            "host_rank": 0,
            "mpi_rank": 0,
            "world_size": 1,
            "written_at": "2026-09-15T00:00:00Z",
        },
        "meshes": [
            {
                "mesh_id": 0,
                "shape": [1, 2],
                "chips": [
                    {
                        "fabric_chip_id": 0,
                        "mesh_coord": [0, 0],
                        "physical_chip_id": 4,
                        "asic_id": None,
                        "is_local": True,
                        "master_router_chan": 1,
                        "routers": [
                            {
                                "eth_chan": 1,
                                "direction": "E",
                                "routing_plane": 0,
                                "link_class": "intramesh",
                                "logical_core": [0, 1],
                                "virtual_core": [18, 16],
                                "layout_id": "L0123456789abcdef",
                                "instance": {
                                    "handshake": 16,
                                    "sender_channels_per_vc": [1, 0, 0, 0],
                                    "receiver_channels_per_vc": [1, 0],
                                    "downstream_edm_mask_per_vc": [1, 0, 0, 0],
                                    "tensix_extension": False,
                                    "udm_mode": False,
                                },
                            }
                        ],
                    },
                    {
                        "fabric_chip_id": 1,
                        "mesh_coord": [0, 1],
                        "physical_chip_id": None,
                        "asic_id": None,
                        "is_local": False,
                        "master_router_chan": None,
                        "routers": [],
                    },
                ],
            }
        ],
        "links": [],
    }
    data.update(required_blocks())
    return data


class PeekTest(unittest.TestCase):
    def load_fixture(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.json"
            path.write_text(json.dumps(fixture_manifest()), encoding="utf-8")
            return load_manifest(path)

    def test_peeks_health_and_streams_on_local_router_only(self):
        manifest = self.load_fixture()
        locations = [FakeLoc((0, 0)), FakeLoc((0, 1))]
        devices = {4: FakeDevice(4, locations)}

        def read_u32(device, loc, address):
            if address == 0xFFB121B0:
                return 1
            if address == 0xFFB121F0:
                return 0x89ABCDEF
            if address == 0xFFB121F4:
                return 0x1
            return 8

        sample = peek_manifest(manifest, devices, read_u32)
        self.assertEqual(len(sample["routers"]), 1)
        router_sample = sample["routers"][0]
        self.assertTrue(router_sample["ok"])
        self.assertIsNone(router_sample["error"])
        self.assertEqual(router_sample["id"], {"mesh_id": 0, "chip_id": 0, "eth_chan": 1})
        self.assertEqual(router_sample["health"]["reset"], 1)
        self.assertEqual(router_sample["health"]["wall_clock"], (1 << 32) | 0x89ABCDEF)
        self.assertEqual(len(router_sample["streams"]), len(list(FABRIC_STREAM_IDS)))
        self.assertEqual(router_sample["streams"]["22"]["buf_space_available"], 8)

        snapshot = build_snapshot(manifest, [sample], captured_at="2026-09-14T22:00:00Z")
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator(
            schema,
            format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER,
        ).validate(snapshot)
        self.assertEqual(snapshot["manifest"]["run"]["mpi_rank"], 0)
        self.assertEqual(snapshot["samples"][0]["capture_time"], sample["capture_time"])
        self.assertEqual(snapshot["manifest"]["sha256"], manifest.sha256)
        self.assertNotIn("written_at", snapshot["manifest"]["run"])

    def test_records_error_when_device_is_missing(self):
        manifest = self.load_fixture()
        router_sample = peek_manifest(manifest, {}, lambda device, loc, address: 0)["routers"][0]
        self.assertFalse(router_sample["ok"])
        self.assertIn("not visible", router_sample["error"])
        self.assertEqual(router_sample["streams"], {})

    def test_records_error_on_logical_core_mismatch(self):
        manifest = self.load_fixture()
        devices = {4: FakeDevice(4, [FakeLoc((0, 0)), FakeLoc((9, 9))])}
        router_sample = peek_manifest(manifest, devices, lambda device, loc, address: 1)["routers"][0]
        self.assertFalse(router_sample["ok"])
        self.assertIn("logical core mismatch", router_sample["error"])

    def test_sample_time_is_the_peek_time_not_the_assembly_time(self):
        manifest = self.load_fixture()
        devices = {4: FakeDevice(4, [FakeLoc((0, 0)), FakeLoc((0, 1))])}
        sample = peek_manifest(manifest, devices, lambda device, loc, address: 0)
        sample["capture_time"] = "2020-01-01T00:00:00Z"

        snapshot = build_snapshot(manifest, [sample], captured_at="2026-09-14T22:00:00Z")
        self.assertEqual(snapshot["samples"][0]["capture_time"], "2020-01-01T00:00:00Z")
        self.assertEqual(snapshot["captured_at"], "2026-09-14T22:00:00Z")

    def test_snapshot_needs_at_least_one_sample(self):
        with self.assertRaises(ValueError):
            build_snapshot(self.load_fixture(), [])

    def test_unsupported_arch_is_an_error_rather_than_a_guess(self):
        manifest = self.load_fixture()
        manifest.run["arch"] = "MOONSHINE"
        devices = {4: FakeDevice(4, [FakeLoc((0, 0)), FakeLoc((0, 1))])}
        with self.assertRaises(CaptureError):
            peek_manifest(manifest, devices, lambda device, loc, address: 0)

    def test_partial_stream_failure_keeps_readable_values(self):
        manifest = self.load_fixture()
        devices = {4: FakeDevice(4, [FakeLoc((0, 0)), FakeLoc((0, 1))])}

        def read_u32(device, loc, address):
            if address == 0xFFB121B0:
                return 1
            if address in (0xFFB121F0, 0xFFB121F4):
                return 0
            # First overlay stream only.
            if address == 0xFFB40000 + (64 << 2):
                return 3
            return None

        router_sample = peek_manifest(manifest, devices, read_u32)["routers"][0]
        self.assertFalse(router_sample["ok"])
        self.assertEqual(router_sample["streams"]["0"]["buf_space_available"], 3)
        self.assertNotIn("1", router_sample["streams"])


if __name__ == "__main__":
    unittest.main()
