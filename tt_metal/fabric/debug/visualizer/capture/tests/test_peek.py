# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path

import jsonschema

from tt_metal.fabric.debug.visualizer.capture.manifest import load_manifest
from tt_metal.fabric.debug.visualizer.capture.peek import ALL_STREAM_IDS, CaptureError, peek_manifest
from fabric_erisc_utils import get_stream_reg_address
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
    def __init__(self, device_id, locations, arch="wormhole_b0", unique_id=0x1234):
        self.id = device_id
        self.unique_id = unique_id
        self._arch = arch
        self._locations = locations

    def get_block_locations(self, block_type):
        if block_type != "eth":
            return []
        return self._locations


class FakeContext:
    def __init__(self, devices):
        self.devices = devices
        self.device_by_unique_id = {device.unique_id: device for device in devices.values()}


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
                        "asic_id": "0x0000000000001234",
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
            if address == 1:
                return 0xA3B3C3D3
            return 8

        sample = peek_manifest(manifest, FakeContext(devices), read_u32)
        self.assertEqual(len(sample["routers"]), 1)
        router_sample = sample["routers"][0]
        self.assertEqual(router_sample["status"], "ok")
        self.assertIsNone(router_sample["error"])
        self.assertEqual(router_sample["id"], {"mesh_id": 0, "chip_id": 0, "eth_chan": 1})
        self.assertEqual(router_sample["asic_id"], "0x0000000000001234")
        self.assertTrue(router_sample["asic_id_matches_physical"])
        self.assertEqual(router_sample["health"]["reset"], 1)
        self.assertEqual(
            router_sample["health"]["reset_bits"],
            {"erisc0": False, "erisc1": None},
        )
        self.assertEqual(router_sample["health"]["wall_clock"], (1 << 32) | 0x89ABCDEF)
        self.assertEqual(router_sample["lifecycle"]["edm_status"], 0xA3B3C3D3)
        self.assertEqual(set(router_sample["streams"]["pre"]), {"22", "23"})
        self.assertEqual(router_sample["streams"]["pre"]["22"]["buf_space_available"], 8)
        self.assertEqual(router_sample["streams"]["post"], router_sample["streams"]["pre"])
        self.assertFalse(router_sample["streams"]["torn"])

        snapshot = build_snapshot(
            manifest,
            [sample],
            provenance={
                "ttexalens_version": "test",
                "tt_umd_version": "test",
                "hostname": "test-host",
                "owner_alive": None,
                "argv": ["capture"],
            },
            captured_at="2026-09-14T22:00:00Z",
            raw={
                "file": "snapshot.bin",
                "size": 0,
                "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            },
        )
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
        router_sample = peek_manifest(manifest, FakeContext({}), lambda device, loc, address: 0)["routers"][0]
        self.assertEqual(router_sample["status"], "unknown")
        self.assertIn("not visible", router_sample["error"])
        self.assertEqual(router_sample["streams"], {"pre": {}, "post": {}, "torn": False})

    def test_records_error_on_logical_core_mismatch(self):
        manifest = self.load_fixture()
        devices = {4: FakeDevice(4, [FakeLoc((0, 0)), FakeLoc((9, 9))])}
        router_sample = peek_manifest(manifest, FakeContext(devices), lambda device, loc, address: 1)["routers"][0]
        self.assertEqual(router_sample["status"], "unknown")
        self.assertIn("logical core mismatch", router_sample["error"])

    def test_asic_id_selects_device_and_cross_checks_physical_id(self):
        manifest = self.load_fixture()
        physical_device = FakeDevice(
            4,
            [FakeLoc((0, 0)), FakeLoc((9, 9))],
            unique_id=0xAAAA,
        )
        asic_device = FakeDevice(
            9,
            [FakeLoc((0, 0)), FakeLoc((0, 1))],
            unique_id=0x1234,
        )
        devices = {4: physical_device, 9: asic_device}
        read_device_ids = []

        def read_u32(device, loc, address):
            read_device_ids.append(device.id)
            if address == 1:
                return 0xA3B3C3D3
            return 0

        router_sample = peek_manifest(manifest, FakeContext(devices), read_u32)["routers"][0]
        self.assertEqual(router_sample["status"], "ok")
        self.assertFalse(router_sample["asic_id_matches_physical"])
        self.assertEqual(set(read_device_ids), {9})

    def test_sample_time_is_the_peek_time_not_the_assembly_time(self):
        manifest = self.load_fixture()
        devices = {4: FakeDevice(4, [FakeLoc((0, 0)), FakeLoc((0, 1))])}
        sample = peek_manifest(manifest, FakeContext(devices), lambda device, loc, address: 0)
        sample["capture_time"] = "2020-01-01T00:00:00Z"

        snapshot = build_snapshot(
            manifest,
            [sample],
            provenance={
                "ttexalens_version": "test",
                "tt_umd_version": "test",
                "hostname": "test-host",
                "owner_alive": False,
                "argv": ["capture"],
            },
            captured_at="2026-09-14T22:00:00Z",
        )
        self.assertEqual(snapshot["samples"][0]["capture_time"], "2020-01-01T00:00:00Z")
        self.assertEqual(snapshot["captured_at"], "2026-09-14T22:00:00Z")

    def test_snapshot_needs_at_least_one_sample(self):
        with self.assertRaises(ValueError):
            build_snapshot(self.load_fixture(), [], provenance={})

    def test_unsupported_arch_is_an_error_rather_than_a_guess(self):
        manifest = self.load_fixture()
        manifest.run["arch"] = "MOONSHINE"
        devices = {4: FakeDevice(4, [FakeLoc((0, 0)), FakeLoc((0, 1))])}
        with self.assertRaises(CaptureError):
            peek_manifest(manifest, FakeContext(devices), lambda device, loc, address: 0)

    def test_partial_stream_failure_keeps_readable_values(self):
        manifest = self.load_fixture()
        devices = {4: FakeDevice(4, [FakeLoc((0, 0)), FakeLoc((0, 1))])}
        stream_22 = get_stream_reg_address(22, "BUF_SPACE_AVAILABLE", "wormhole")
        stream_23 = get_stream_reg_address(23, "BUF_SPACE_AVAILABLE", "wormhole")

        def read_u32(device, loc, address):
            if address == 0xFFB121B0:
                return 1
            if address in (0xFFB121F0, 0xFFB121F4):
                return 0
            if address == 1:
                return 0xA3B3C3D3
            if address == stream_22:
                return 3
            if address == stream_23:
                return None
            return 0

        router_sample = peek_manifest(manifest, FakeContext(devices), read_u32)["routers"][0]
        self.assertEqual(router_sample["status"], "unreadable")
        self.assertEqual(router_sample["streams"]["pre"]["22"]["buf_space_available"], 3)
        self.assertNotIn("23", router_sample["streams"]["pre"])
        self.assertNotIn("0", router_sample["streams"]["pre"])

    def test_liveness_samples_are_fleet_wide_rounds(self):
        manifest = self.load_fixture()
        devices = {4: FakeDevice(4, [FakeLoc((0, 0)), FakeLoc((0, 1))])}
        timestamps = iter(
            [
                "2026-09-15T00:00:00Z",
                "2026-09-15T00:00:01Z",
                "2026-09-15T00:00:02Z",
                "2026-09-15T00:00:03Z",
            ]
        )
        heartbeat = 0
        sleeps = []

        def read_u32(device, loc, address):
            nonlocal heartbeat
            if address == 1:
                return 0xA3B3C3D3
            if address == 0x1F80:
                heartbeat += 1
                return 0xDCBA0000 | heartbeat
            return 0

        sample = peek_manifest(
            manifest,
            FakeContext(devices),
            read_u32,
            liveness_samples=3,
            liveness_interval=0.25,
            sleep=sleeps.append,
            timestamp=lambda: next(timestamps),
        )
        liveness = sample["routers"][0]["liveness"]
        self.assertEqual(
            [entry["t"] for entry in liveness],
            [
                "2026-09-15T00:00:01Z",
                "2026-09-15T00:00:02Z",
                "2026-09-15T00:00:03Z",
            ],
        )
        self.assertEqual(
            [entry["heartbeat"] for entry in liveness],
            [0xDCBA0001, 0xDCBA0002, 0xDCBA0003],
        )
        self.assertEqual(sleeps, [0.25, 0.25])

    def test_status_reset_and_unknown(self):
        manifest = self.load_fixture()
        context = FakeContext(
            {4: FakeDevice(4, [FakeLoc((0, 0)), FakeLoc((0, 1))])}
        )

        def reset_read(device, loc, address):
            if address == 0xFFB121B0:
                return 1 << 11
            if address == 1:
                return 0xA3B3C3D3
            return 0

        reset_sample = peek_manifest(manifest, context, reset_read)["routers"][0]
        self.assertEqual(reset_sample["status"], "reset")
        self.assertTrue(reset_sample["health"]["reset_bits"]["erisc0"])

        def unreadable_reset_read(device, loc, address):
            if address == get_stream_reg_address(22, "BUF_SPACE_AVAILABLE", "wormhole"):
                return None
            return reset_read(device, loc, address)

        unreadable_sample = peek_manifest(
            manifest,
            context,
            unreadable_reset_read,
        )["routers"][0]
        self.assertEqual(unreadable_sample["status"], "unreadable")

        unknown_sample = peek_manifest(
            manifest,
            context,
            lambda device, loc, address: 0,
        )["routers"][0]
        self.assertEqual(unknown_sample["status"], "unknown")
        self.assertIn("EDMStatus", unknown_sample["error"])

    def test_layout_streams_include_allocated_disabled_ids(self):
        manifest = self.load_fixture()
        devices = {4: FakeDevice(4, [FakeLoc((0, 0)), FakeLoc((0, 1))])}

        def read_u32(device, loc, address):
            if address == 1:
                return 0xA3B3C3D3
            return 0

        layout_sample = peek_manifest(manifest, FakeContext(devices), read_u32)["routers"][0]
        self.assertEqual(set(layout_sample["streams"]["pre"]), {"22", "23"})

        all_sample = peek_manifest(
            manifest,
            FakeContext(devices),
            read_u32,
            streams="all",
        )["routers"][0]
        self.assertEqual(
            set(all_sample["streams"]["pre"]),
            {str(stream_id) for stream_id in ALL_STREAM_IDS},
        )

    def test_torn_when_post_stream_values_move(self):
        manifest = self.load_fixture()
        devices = {4: FakeDevice(4, [FakeLoc((0, 0)), FakeLoc((0, 1))])}
        stream_22 = get_stream_reg_address(22, "BUF_SPACE_AVAILABLE", "wormhole")
        hits = {stream_22: 0}

        def read_u32(device, loc, address):
            if address == 1:
                return 0xA3B3C3D3
            if address == stream_22:
                hits[stream_22] += 1
                return 8 if hits[stream_22] == 1 else 7
            return 0

        router_sample = peek_manifest(manifest, FakeContext(devices), read_u32)["routers"][0]
        self.assertTrue(router_sample["streams"]["torn"])
        self.assertEqual(router_sample["status"], "torn")
        self.assertEqual(router_sample["streams"]["pre"]["22"]["buf_space_available"], 8)
        self.assertEqual(router_sample["streams"]["post"]["22"]["buf_space_available"], 7)

    def test_status_precedence_unreadable_beats_reset_beats_torn(self):
        manifest = self.load_fixture()
        devices = {4: FakeDevice(4, [FakeLoc((0, 0)), FakeLoc((0, 1))])}
        stream_22 = get_stream_reg_address(22, "BUF_SPACE_AVAILABLE", "wormhole")
        hits = {stream_22: 0}

        def moving_and_reset(device, loc, address):
            if address == 0xFFB121B0:
                return 1 << 11
            if address == 1:
                return 0xA3B3C3D3
            if address == stream_22:
                hits[stream_22] += 1
                return 8 if hits[stream_22] == 1 else 7
            return 0

        reset_over_torn = peek_manifest(
            manifest,
            FakeContext(devices),
            moving_and_reset,
        )["routers"][0]
        self.assertEqual(reset_over_torn["status"], "reset")
        self.assertTrue(reset_over_torn["streams"]["torn"])


if __name__ == "__main__":
    unittest.main()
