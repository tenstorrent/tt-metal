# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import struct
import unittest

from tt_metal.fabric.debug.visualizer.decode.structs import (
    decode_fabric_telemetry,
    decode_payload,
    decode_routing_l1_info,
    unpack_direction_table,
)


class StructDecoderTest(unittest.TestCase):
    def test_small_router_structs(self):
        enums = {"EDMStatus": {"READY": 7}}
        self.assertEqual(
            decode_payload("EDMStatus", struct.pack("<I", 7), enums=enums, region={}, mesh_count=1),
            {"raw": 7, "name": "READY"},
        )
        handshake = decode_payload(
            "handshake_info_t",
            struct.pack("<IHBx", 0xAA, 2, 3),
            enums={},
            region={},
            mesh_count=1,
        )
        self.assertEqual(handshake["neighbor_mesh_id"], 2)
        self.assertEqual(handshake["neighbor_device_id"], 3)

        location = bytearray(64)
        struct.pack_into("<I", location, 0, 0x100)
        struct.pack_into("<I", location, 16, 0x200)
        struct.pack_into("<HH", location, 32, 4, 5)
        struct.pack_into("<I", location, 48, 19)
        decoded = decode_payload(
            "EDMChannelWorkerLocationInfo",
            bytes(location),
            enums={},
            region={},
            mesh_count=1,
        )
        self.assertEqual(decoded["worker_xy"], {"x": 4, "y": 5})
        self.assertEqual(decoded["edm_read_counter"], 19)

        cursor = decode_payload(
            "SenderChannelProducerCursor",
            struct.pack("<4I", 17, 2, 0, 0),
            enums={},
            region={},
            mesh_count=1,
        )
        self.assertEqual(cursor, {"write_counter": 17, "write_index": 2})
        counters = decode_payload(
            "u32_counter_array",
            struct.pack("<4I", 1, 2, 3, 4),
            enums={},
            region={"count": 3},
            mesh_count=1,
        )
        self.assertEqual(counters["counters"], [1, 2, 3])

    def test_fabric_telemetry_offsets(self):
        payload = bytearray(160)
        struct.pack_into("<IHHBBBBI", payload, 0, 1, 2, 3, 4, 5, 6, 7, 8)
        struct.pack_into("<4Q", payload, 16, 10, 11, 12, 13)
        struct.pack_into("<4Q", payload, 48, 20, 21, 22, 23)
        struct.pack_into("<I", payload, 80, 1)
        struct.pack_into("<QQ", payload, 88, 30, 31)
        struct.pack_into("<I", payload, 104, 2)
        struct.pack_into("<QQ", payload, 112, 40, 41)
        struct.pack_into("<I7I", payload, 128, 99, *range(7))
        decoded = decode_fabric_telemetry(bytes(payload))

        self.assertEqual(decoded["static_info"]["neighbor_mesh_id"], 3)
        self.assertEqual(decoded["dynamic_info"]["tx_bandwidth"]["num_packets_sent"], 13)
        self.assertEqual(decoded["dynamic_info"]["erisc"][1]["rx_heartbeat"], 41)
        self.assertEqual(decoded["postcode"], 99)
        self.assertEqual(decoded["scratch"], list(range(7)))

    def test_routing_table_tail_and_direction_unpack(self):
        payload = bytearray(2704)
        struct.pack_into("<I", payload, 0, 1)
        struct.pack_into("<I", payload, 16, 0)
        struct.pack_into("<HH", payload, 32, 7, 8)
        payload[36] = 0b10001000  # compressed E, W, N
        payload[2700:2704] = bytes((3, 4, 1, 3))
        decoded = decode_routing_l1_info(
            bytes(payload),
            {"RouterCommand": {"RUN": 0}},
            mesh_count=1,
        )

        self.assertEqual(decoded["my_mesh_id"], 7)
        self.assertEqual(decoded["my_device_id"], 8)
        self.assertEqual(decoded["my_mesh_coord"], {"y": 3, "x": 4})
        self.assertEqual(decoded["mesh_shape"], {"y": 1, "x": 3})
        self.assertEqual(decoded["intra_mesh_directions"], ["E", "W", "N"])

    def test_direction_entries_spanning_bytes(self):
        # Eight 3-bit values packed exactly like direction_table_t::set_direction.
        values = list(range(7)) + [0]
        packed = 0
        for index, value in enumerate(values):
            packed |= value << (index * 3)
        payload = packed.to_bytes(3, "little")
        self.assertEqual(
            unpack_direction_table(payload, len(values)),
            ["E", "W", "N", "S", "Z", "INVALID_DIRECTION", "INVALID_ENTRY", "E"],
        )


if __name__ == "__main__":
    unittest.main()
