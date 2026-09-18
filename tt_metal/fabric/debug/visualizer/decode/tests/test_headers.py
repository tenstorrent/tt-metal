# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import struct
import tempfile
import unittest
from pathlib import Path

from tt_metal.fabric.debug.visualizer.decode.headers import (
    decode_1d_hops,
    decode_2d_path,
    decode_packet_header,
    noc_address,
    packet_header_shape,
)
from tt_metal.fabric.debug.visualizer.decode.inputs import discover_inputs
from tt_metal.fabric.debug.visualizer.decode.output import build_decoded
from tt_metal.fabric.debug.visualizer.decode.tests.fixtures import write_input


def context_2d(size=96):
    return {
        "is_2d_routing": True,
        "routing_2d_route_buffer_size": 36,
        "packet_header_size_bytes": size,
        "max_payload_size_bytes": 4352,
    }


def header(send_type: int) -> bytearray:
    payload = bytearray(96)
    struct.pack_into("<HBB", payload, 40, 64, send_type, 2)
    struct.pack_into("<HH4H", payload, 84, 3, 0, 1, 2, 3, 4)
    return payload


class HeaderDecoderTest(unittest.TestCase):
    def test_shape_selection(self):
        self.assertEqual(packet_header_shape({}, context_2d()), ("HybridMeshPacketHeaderT<36>", 96))
        self.assertEqual(
            packet_header_shape({"udm_mode": "ENABLED"}, {**context_2d(112)}),
            ("UDMHybridMeshPacketHeaderT<36>", 112),
        )
        self.assertEqual(
            packet_header_shape(
                {},
                {
                    "is_2d_routing": False,
                    "routing_1d_extension_words": 0,
                    "packet_header_size_bytes": 48,
                },
            ),
            ("LowLatencyPacketHeaderT<0>", 48),
        )

    def test_unicast_address_and_2d_routing(self):
        payload = header(0)
        raw_address = 0x12345 | (5 << 36) | (6 << 42)
        struct.pack_into("<Q", payload, 0, raw_address)
        payload[48:84] = bytes(range(36))
        decoded = decode_packet_header(
            payload,
            run={},
            context=context_2d(),
            mesh_ids={0},
            mesh_shape={"y": 2, "x": 4},
            mesh_coord={"y": 0, "x": 0},
        )

        self.assertTrue(decoded["plausible"])
        self.assertEqual(decoded["command"]["noc_address"], noc_address(raw_address))
        self.assertEqual(decoded["routing"]["destination"], {"mesh_id": 0, "chip_id": 3})
        self.assertEqual(decoded["routing"]["mcast"], {"E": 1, "W": 2, "N": 3, "S": 4})
        self.assertEqual(decoded["routing"]["path"], {"hops": ["W"], "complete": False})

    def test_each_command_union_shape(self):
        cases = {}
        payload = header(1)
        struct.pack_into("<QI", payload, 0, 1, 2)
        cases[1] = decode_packet_header(payload, run={}, context=context_2d(), mesh_ids={0})["command"]
        payload = header(2)
        struct.pack_into("<QIB", payload, 0, 1, 2, 1)
        cases[2] = decode_packet_header(payload, run={}, context=context_2d(), mesh_ids={0})["command"]
        payload = header(3)
        struct.pack_into("<QQIB", payload, 0, 1, 2, 3, 1)
        cases[3] = decode_packet_header(payload, run={}, context=context_2d(), mesh_ids={0})["command"]
        payload = header(4)
        struct.pack_into("<4Q3HBB", payload, 0, 1, 2, 3, 4, 5, 6, 7, 3, 0x15)
        cases[4] = decode_packet_header(payload, run={}, context=context_2d(), mesh_ids={0})["command"]
        payload = header(5)
        struct.pack_into("<IBBBB", payload, 0, 1, 2, 3, 4, 5)
        cases[5] = decode_packet_header(payload, run={}, context=context_2d(), mesh_ids={0})["command"]
        payload = header(6)
        struct.pack_into("<IIBBBB", payload, 0, 1, 2, 3, 4, 5, 6)
        cases[6] = decode_packet_header(payload, run={}, context=context_2d(), mesh_ids={0})["command"]
        payload = header(7)
        struct.pack_into("<Q", payload, 0, 1)
        cases[7] = decode_packet_header(payload, run={}, context=context_2d(), mesh_ids={0})["command"]
        payload = header(8)
        struct.pack_into("<4Q4B4B", payload, 0, 1, 2, 3, 4, 1, 1, 0, 0, 2, 2, 1, 0)
        cases[8] = decode_packet_header(payload, run={}, context=context_2d(), mesh_ids={0})["command"]

        self.assertEqual(cases[1]["value"], 2)
        self.assertTrue(cases[2]["flush"])
        self.assertEqual(cases[3]["semaphore_noc_address"]["raw"], 2)
        self.assertEqual(cases[4]["chunk_count"], 3)
        self.assertEqual(cases[5]["size_y"], 5)
        self.assertEqual(cases[6]["value"], 2)
        self.assertEqual(cases[7]["noc_address"]["raw"], 1)
        self.assertEqual(cases[8]["num_dests"], 2)

    def test_implausible_header_and_size_mismatch(self):
        payload = header(255)
        struct.pack_into("<H", payload, 40, 5000)
        struct.pack_into("<HH", payload, 84, 0, 9)
        decoded = decode_packet_header(payload, run={}, context=context_2d(), mesh_ids={0})
        self.assertFalse(decoded["plausible"])
        with self.assertRaisesRegex(ValueError, "computes to 96"):
            decode_packet_header(payload, run={}, context=context_2d(80), mesh_ids={0})

    def test_ring_integration_and_slots_none(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_input(root)
            inputs = discover_inputs([root])
            decoded = build_decoded(inputs)
            ring = decoded["routers"][0]["rings"][0]
            self.assertEqual(ring["id"], "sender.0.ring")
            self.assertEqual(len(ring["slots"]), 2)
            self.assertTrue(ring["slots"][0]["header"]["plausible"])
            self.assertFalse(ring["slots"][1]["header"]["plausible"])
            self.assertEqual(ring["slots"][0]["slot_state"], "unknown")
            # Shared fixture uses stride == header size, so no payload range.
            self.assertEqual(ring["slots"][0]["raw_ref"]["size"], 96)
            self.assertIsNone(ring["slots"][0]["payload_ref"])

            compact = build_decoded(inputs, slots="none")["routers"][0]["rings"][0]
            self.assertNotIn("slots", compact)

    def test_slot_payload_ref_points_past_header(self):
        from tt_metal.fabric.debug.visualizer.decode.rings import decode_rings

        blob = bytes(header(0)) + bytes(32)
        assert len(blob) == 128

        class StubDecoder:
            def raw_slice(self, _index, _ref):
                return blob

        router = {
            "capture": {"snapshot_index": 0},
            "regions": [
                {
                    "id": "sender.0.ring",
                    "schema": "packet_ring",
                    "status": "ok",
                    "error": None,
                    "count": 1,
                    "stride": 128,
                    "raw_ref": {"file": "snapshot_0.bin", "offset": 1000, "size": 128},
                },
                {
                    "id": "hal.routing_table",
                    "value": {"mesh_shape": {"y": 2, "x": 4}, "my_mesh_coord": {"y": 0, "x": 0}},
                },
            ],
        }
        (ring,) = decode_rings(
            router,
            StubDecoder(),
            run={},
            context={**context_2d(), "topology": "Mesh"},
            mesh_ids={0},
            slots="headers",
        )
        (slot,) = ring["slots"]
        self.assertEqual(
            slot["raw_ref"], {"file": "snapshot_0.bin", "offset": 1000, "size": 96}
        )
        self.assertEqual(
            slot["payload_ref"], {"file": "snapshot_0.bin", "offset": 1096, "size": 32}
        )

    def test_1d_hop_tape_unicast(self):
        decoded = decode_1d_hops(0x1AAA)
        self.assertEqual(
            decoded["hops"],
            ["forward", "forward", "forward", "forward", "forward", "forward", "write"],
        )
        self.assertEqual(decoded["kind"], "unicast")
        self.assertEqual(decoded["hops_remaining"], 7)

    def test_2d_path_walks_from_this_router(self):
        buffer = bytes([0b001000, 0, 0b000001, 0b100000])  # y0: S, y1 empty, x0: E, x1: local
        path = decode_2d_path(buffer, mesh_shape={"y": 2, "x": 2}, coord={"y": 0, "x": 0})
        self.assertEqual(path, {"hops": ["S", "E", "local"], "complete": True})

    def test_2d_path_stops_at_split(self):
        buffer = bytes([0b000101, 0, 0, 0])  # y0: E+N
        path = decode_2d_path(buffer, mesh_shape={"y": 2, "x": 2}, coord={"y": 0, "x": 0})
        self.assertEqual(path, {"hops": ["E+N"], "complete": False})

    def test_2d_path_empty_map_is_none(self):
        path = decode_2d_path(bytes(4), mesh_shape={"y": 2, "x": 2}, coord={"y": 0, "x": 0})
        self.assertEqual(path, {"hops": [], "complete": False})

    def test_2d_path_needs_coord(self):
        self.assertIsNone(decode_2d_path(bytes(4), mesh_shape={"y": 2, "x": 2}, coord=None))


if __name__ == "__main__":
    unittest.main()
