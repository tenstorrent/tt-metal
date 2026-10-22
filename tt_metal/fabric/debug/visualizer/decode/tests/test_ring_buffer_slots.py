# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Cases for decoding ring-buffer slots, on hand-written manifests and slot bytes. The headers mirror the fabric's
layouts, cut down to the members the slot rules read.

Run from the repository root: python3 -m unittest tt_metal.fabric.debug.visualizer.decode.tests.test_ring_buffer_slots
"""

import copy
import pathlib
import struct
import unittest

from tt_metal.fabric.debug.visualizer.common.manifest_lookup import ManifestError
from tt_metal.fabric.debug.visualizer.decode.inputs import Input
from tt_metal.fabric.debug.visualizer.decode.ring_buffer_slots import (
    EMPTY_RING_BUFFER_SLOT,
    Hop1D,
    SlotRoute1D,
    SlotRoute2D,
    get_ring_buffer_slots,
    get_slot_context,
)
from tt_metal.fabric.debug.visualizer.decode.route_context import ChipStep2D, ChipWalk2D, get_route_context
from tt_metal.fabric.debug.visualizer.decode.routers import DecodedRouter
from tt_metal.fabric.debug.visualizer.decode.routes import ChunkEncoding, Direction, ScatterChunk
from tt_metal.fabric.debug.visualizer.decode.routing_tables import RoutingTable
from tt_metal.fabric.debug.visualizer.decode.tests.test_routes import MANIFEST as ENCODINGS
from tt_metal.fabric.debug.visualizer.decode.tests.test_routes import D, E, N, S, W, line_router
from tt_metal.fabric.debug.visualizer.decode.tests.test_routing_tables import ACTION_VECTORS, chip
from tt_metal.fabric.debug.visualizer.decode.type_parser import get_types_from_manifest

# NocSendType values, and eth_chan_directions values.
WRITE, SCATTER = 0, 4
EAST, WEST = 0, 1

COMMAND_TYPES = {
    "NocCommandFields": {
        "kind": "union",
        "size": 8,
        "tag_schema": "enum:NocSendType",
        "members": [
            {
                "name": "unicast_write",
                "offset": 0,
                "size": 8,
                "schema": "struct:NocUnicastCommandHeader",
                "tag_value": "noc_unicast_write",
            },
            {
                "name": "unicast_scatter_write",
                "offset": 0,
                "size": 8,
                "schema": "struct:NocUnicastScatterCommandHeader",
                "tag_value": "noc_unicast_scatter_write",
            },
        ],
    },
    "NocUnicastCommandHeader": {
        "size": 8,
        "members": [{"name": "noc_address", "offset": 0, "size": 8, "schema": "u64"}],
    },
    "NocUnicastScatterCommandHeader": {
        "size": 8,
        "members": [
            {"name": "chunk_size", "offset": 0, "size": 6, "num_elements": 3, "size_per_element": 2, "schema": "u16"},
            {"name": "chunk_count", "offset": 6, "size": 1, "schema": "u8"},
            {
                "name": "chunk_encoding",
                "offset": 7,
                "size": 1,
                "num_elements": 4,
                "bits_per_entry": 2,
                "schema": "enum:NocScatterWriteChunkEncoding",
            },
        ],
    },
}
COMMON_MEMBERS = [
    {"name": "command_fields", "offset": 0, "size": 8, "schema": "union:NocCommandFields", "tag": "noc_send_type"},
    {"name": "payload_size_bytes", "offset": 8, "size": 2, "schema": "u16"},
    {"name": "noc_send_type", "offset": 10, "size": 1, "schema": "enum:NocSendType"},
    {"name": "src_ch_id", "offset": 11, "size": 1, "schema": "u8"},
]
HEADER_2D = {
    "size": 32,
    "members": COMMON_MEMBERS
    + [
        {"name": "route_buffer", "offset": 12, "size": 8, "num_elements": 8, "size_per_element": 1, "schema": "u8"},
        {"name": "dst_start_chip_id", "offset": 20, "size": 2, "schema": "u16"},
        {"name": "dst_start_mesh_id", "offset": 22, "size": 2, "schema": "u16"},
        {"name": "mcast_params", "offset": 24, "size": 8, "num_elements": 4, "size_per_element": 2, "schema": "u16"},
    ],
}
HEADER_1D = {
    "size": 20,
    "members": COMMON_MEMBERS
    + [{"name": "routing_fields", "offset": 12, "size": 8, "schema": "struct:RoutingFields1D"}],
}
# Routing fields with 1 extension word.
ROUTING_FIELDS_1D = {
    "size": 8,
    "members": [
        {"name": "value", "offset": 0, "size": 4, "schema": "u32"},
        {"name": "route_buffer", "offset": 4, "size": 4, "num_elements": 1, "size_per_element": 4, "schema": "u32"},
    ],
}
# A slot is the header, then the payload.
SLOT_2D, SLOT_1D = 48, 32


def manifest(is_2d, meshes, types, **context):
    header = "HybridMeshPacketHeader" if is_2d else "LowLatencyPacketHeader"
    return {
        **copy.deepcopy(ENCODINGS),
        "run": {"arch": "test_arch"},
        "fabric_context": {
            "is_2d_routing": is_2d,
            "packet_header": {"size": types[header]["size"], "schema": f"struct:{header}"},
            **context,
        },
        "vocabulary": {"z_port_roles": ["none", "intermesh_boundary", "express_chord"]},
        "archs": {"test_arch": {"types": copy.deepcopy({**COMMAND_TYPES, **types})}},
        "enums": {
            **copy.deepcopy(ENCODINGS["enums"]),
            "NocSendType": {"values": {"noc_unicast_write": WRITE, "noc_unicast_scatter_write": SCATTER}},
            "eth_chan_directions": {"values": {"east": 0, "west": 1, "north": 2, "south": 3, "z": 4, "count": 5}},
        },
        "meshes": meshes,
    }


def channel(slot_size, status="active", ingress=None):
    node = {
        "status": status,
        "ring_buffer": {"address": 0, "size": 2 * slot_size, "num_elements": 2, "size_per_element": slot_size},
        "fields": {},
    }
    if ingress is not None:
        node["fields"]["intermesh_ingress"] = {"kind": "flag", "value": ingress}
    return node


def router_2d(direction, peer, senders=None, receivers=None):
    return {
        "link": {"peer": peer},
        "fields": {"my_direction": {"kind": "enum", "value": direction, "schema": "enum:eth_chan_directions"}},
        "channels": {"senders": senders or {}, "receivers": receivers or {}},
    }


# The 2x3 mesh of test_routing_tables, with C2's east link leaving for mesh 1.
#   C0 C1 C2 -> M1/C0
#   C3 C4 C5
MANIFEST_2D = manifest(
    True,
    {
        "M0": {
            "shape": [2, 3],
            "torus": {"y": False, "x": False},
            "chips": {
                "C0": chip(
                    0,
                    0,
                    {
                        "E0": router_2d(
                            EAST,
                            "M0/C1/W0",
                            senders={"vc0": {"ch0": channel(SLOT_2D)}},
                            receivers={"vc0": {"ch0": channel(SLOT_2D, ingress=False)}},
                        )
                    },
                ),
                "C1": chip(
                    0,
                    1,
                    {"W0": router_2d(WEST, "M0/C0/E0", receivers={"vc0": {"ch0": channel(SLOT_2D, ingress=False)}})},
                ),
                "C2": chip(
                    0,
                    2,
                    {
                        "E0": router_2d(
                            EAST,
                            "M1/C0/W0",
                            senders={"vc0": {"ch0": channel(SLOT_2D)}},
                            receivers={
                                "vc0": {"ch0": channel(SLOT_2D, ingress=True)},
                                "vc1": {"ch0": channel(SLOT_2D, status="vc_not_serviced", ingress=True)},
                            },
                        )
                    },
                ),
                "C3": chip(1, 0),
                "C4": chip(1, 1),
                "C5": chip(1, 2),
            },
        },
        "M1": {"shape": [1, 1], "torus": {"y": False, "x": False}, "chips": {"C0": chip(0, 0)}},
    },
    {"HybridMeshPacketHeader": HEADER_2D},
)

# A 1D line of three chips, C0 - C1 - C2, with 1 extension word.
MANIFEST_1D = manifest(
    False,
    {
        "M0": {
            "shape": [1, 3],
            "torus": {"y": False, "x": False},
            "chips": {
                "C0": chip(
                    0,
                    0,
                    {
                        "E0": {
                            **line_router("M0/C1/W0"),
                            "channels": {"senders": {"vc0": {"ch0": channel(SLOT_1D)}}, "receivers": {}},
                        }
                    },
                ),
                "C1": chip(
                    0,
                    1,
                    {
                        "W0": {
                            **line_router("M0/C0/E0", "M0/C1/E0"),
                            "channels": {"senders": {}, "receivers": {"vc0": {"ch0": channel(SLOT_1D)}}},
                        },
                        "E0": line_router("M0/C2/W0", "M0/C1/W0"),
                    },
                ),
                "C2": chip(0, 2, {"W0": line_router("M0/C1/E0")}),
            },
        }
    },
    {"LowLatencyPacketHeader": HEADER_1D, "RoutingFields1D": ROUTING_FIELDS_1D},
    routing_1d_extension_words=1,
)

# Mesh 0's exit toward mesh 2 is chip 5.
EXIT_NODE_TABLE = [0xFF, 0xFF, 5]


def header_2d(route_map, dst_chip=0, dst_mesh=0, mcast=(0, 0, 0, 0), send_type=WRITE, command=bytes(8)):
    route_buffer = bytes(route_map).ljust(8, b"\0")
    return (
        command
        + struct.pack("<HBB", 16, send_type, 0)
        + route_buffer
        + struct.pack("<HH4H", dst_chip, dst_mesh, *mcast)
    )


def header_1d(value, extension_word=0, send_type=WRITE, command=bytes(8)):
    return command + struct.pack("<HBBII", 16, send_type, 0, value, extension_word)


def scatter_command(chunk_sizes, chunk_count, chunk_encoding):
    return struct.pack("<3HBB", *chunk_sizes, chunk_count, chunk_encoding)


def slots_of(manifest, router_path, rings, with_table=False):
    """
    The slots get_ring_buffer_slots gives for `router_path`, whose rings hold `rings`: the headers in each channel's slots,
    by channel path. With `with_table`, the router's routing table holds ACTION_VECTORS and EXIT_NODE_TABLE.
    """
    slot_size = SLOT_2D if manifest["fabric_context"]["is_2d_routing"] else SLOT_1D
    raw, regions = b"", {}
    for channel_path, headers in rings.items():
        regions[f"{channel_path}/ring_buffer"] = []
        for header in headers:
            regions[f"{channel_path}/ring_buffer"].append({"bytes": {"offset": len(raw), "size": slot_size}})
            raw += header.ljust(slot_size, b"\0")
    table = None
    if with_table:
        fields = {
            "route_table_2d": {"action_vectors": {"bytes": {"offset": len(raw), "size": len(ACTION_VECTORS)}}},
            "exit_node_table": EXIT_NODE_TABLE,
        }
        raw += ACTION_VECTORS
        table = RoutingTable(router_path.rsplit("/", 1)[0], [router_path], {}, fields, None, None, None)
    snapshot = Input(
        pathlib.Path("snapshot.json"), "0" * 64, {"routers": {}}, raw, pathlib.Path("manifest.json"), manifest, ()
    )
    types = get_types_from_manifest(manifest)
    ctx = get_slot_context(manifest, types, get_route_context(manifest))
    return get_ring_buffer_slots(snapshot, types, ctx, DecodedRouter(router_path, None, {}, regions, {}, {}), table)


SENDER, RECEIVER, NOT_SERVICED = "channels/senders/vc0/ch0", "channels/receivers/vc0/ch0", "channels/receivers/vc1/ch0"
# A map from C0 toward C2, as the worker builds it from C0's table: rows stay, then columns 0 and 1 go east.
TOWARD_C2 = [0, N, E, E, D]


def walk(*steps, delivers=(), exits=()):
    """A walk that takes `steps`, each "C<a> <direction> C<b>", delivering at `delivers` and leaving at `exits`."""
    chip_steps = []
    for step in steps:
        src, direction, dst = step.split()
        chip_steps.append(ChipStep2D(f"M0/{src}", Direction(direction), f"M0/{dst}"))
    return ChipWalk2D([f"M0/{key}" for key in delivers], chip_steps, [f"M0/{key}" for key in exits], [])


def only_slot(slots, channel_path):
    return slots[channel_path][0]


class EmptyTest(unittest.TestCase):
    def test_a_zero_header_is_empty(self):
        slots = slots_of(MANIFEST_2D, "M0/C0/E0", {SENDER: [bytes(0), header_2d(TOWARD_C2)]})
        self.assertEqual(slots[SENDER][0], EMPTY_RING_BUFFER_SLOT)
        self.assertFalse(slots[SENDER][1].empty)

    def test_a_channel_no_erisc_runs_has_no_slots(self):
        slots = slots_of(
            MANIFEST_2D, "M0/C2/E0", {RECEIVER: [bytes(0)], NOT_SERVICED: [header_2d(TOWARD_C2)]}, with_table=True
        )
        self.assertEqual(list(slots), [RECEIVER])


class Slots2DTest(unittest.TestCase):
    def test_a_sender_slot_starts_at_the_peer_entered_from_the_opposite_side(self):
        # C0's east router sends to C1's west router, so C1 reads the map first, as entered from the west.
        slot = only_slot(slots_of(MANIFEST_2D, "M0/C0/E0", {SENDER: [header_2d(TOWARD_C2)]}), SENDER)
        expected = SlotRoute2D("M0/C1", Direction.WEST, walk("C1 east C2", delivers=["C2"]), False, None)
        self.assertEqual((slot.route, slot.problems), (expected, []))
        self.assertEqual(slot.header["dst_start_chip_id"], 0)

    def test_the_same_packet_in_the_receiver_slot_it_lands_in(self):
        slot = only_slot(slots_of(MANIFEST_2D, "M0/C1/W0", {RECEIVER: [header_2d(TOWARD_C2)]}), RECEIVER)
        expected = SlotRoute2D("M0/C1", Direction.WEST, walk("C1 east C2", delivers=["C2"]), False, None)
        self.assertEqual(slot.route, expected)

    def test_a_sender_slot_leaving_the_mesh_isnt_followed(self):
        slot = only_slot(slots_of(MANIFEST_2D, "M0/C2/E0", {SENDER: [header_2d(TOWARD_C2, dst_mesh=1)]}), SENDER)
        self.assertEqual(slot.route, SlotRoute2D("M1/C0", Direction.WEST, None, False, None))


class Landing2DTest(unittest.TestCase):
    # C2's table toward C3, chip 3 at (1, 0): south first, then west twice.
    TOWARD_C3 = [S, 0, D, W, W]

    def landing(self, header, with_table=True):
        return only_slot(slots_of(MANIFEST_2D, "M0/C2/E0", {RECEIVER: [header]}, with_table), RECEIVER)

    def test_a_unicast_for_this_mesh_goes_toward_its_chip_before_and_after_the_rebuild(self):
        expected_walk = walk("C2 south C5", "C5 west C4", "C4 west C3", delivers=["C3"])
        # Before: the slot still holds the map of the mesh it came from.
        before = self.landing(header_2d([E, E, D], dst_chip=3))
        self.assertEqual(before.route, SlotRoute2D("M0/C2", None, expected_walk, True, False))
        # After: the router has rebuilt the map in place.
        after = self.landing(header_2d(self.TOWARD_C3, dst_chip=3))
        self.assertEqual(after.route, SlotRoute2D("M0/C2", None, expected_walk, True, True))

    def test_a_packet_for_another_mesh_goes_toward_this_meshs_exit(self):
        slot = self.landing(header_2d([E], dst_mesh=2))
        self.assertEqual(slot.route.walk, walk("C2 south C5", exits=["C5"]))

    def test_a_landing_that_cant_be_rebuilt_is_a_problem(self):
        cases = {
            "a multicast for this mesh": (header_2d([E], mcast=(1, 0, 0, 0)), True, "intermesh multicast isn't"),
            "no exit toward mesh 1": (header_2d([E], dst_mesh=1), True, "isn't in a 2x3 mesh"),
            "no routing table": (header_2d([E], dst_chip=3), False, "routing table wasn't captured"),
        }
        for name, (header, with_table, reason) in cases.items():
            with self.subTest(name):
                slot = self.landing(header, with_table)
                self.assertIsNone(slot.route)
                self.assertEqual(len(slot.problems), 1)
                self.assertIn(reason, slot.problems[0])


class Slots1DTest(unittest.TestCase):
    # Forward once, then write: 0b10 then 0b01, lowest hop first.
    FORWARD_THEN_WRITE = 0b0110

    def test_a_sender_slots_hop_0_is_read_across_the_link(self):
        slot = only_slot(slots_of(MANIFEST_1D, "M0/C0/E0", {SENDER: [header_1d(self.FORWARD_THEN_WRITE)]}), SENDER)
        self.assertEqual(slot.route, SlotRoute1D([Hop1D("forward_only", "M0/C1"), Hop1D("write_only", "M0/C2")]))

    def test_a_receiver_slots_hop_0_is_read_here(self):
        # The packet above, landed in C1's receiver and not yet forwarded.
        slot = only_slot(slots_of(MANIFEST_1D, "M0/C1/W0", {RECEIVER: [header_1d(self.FORWARD_THEN_WRITE)]}), RECEIVER)
        self.assertEqual(slot.route, SlotRoute1D([Hop1D("forward_only", "M0/C1"), Hop1D("write_only", "M0/C2")]))

    def test_a_route_continues_into_the_extension_word(self):
        # 16 forwards fill the first word, and the write is the extension word's first hop. The line ends at C2.
        slot = only_slot(slots_of(MANIFEST_1D, "M0/C1/W0", {RECEIVER: [header_1d(0xAAAAAAAA, 0x1)]}), RECEIVER)
        hops = slot.route.hops
        self.assertEqual(len(hops), 17)
        self.assertEqual(hops[-1], Hop1D("write_only", None))
        self.assertEqual([hop.chip for hop in hops[:3]], ["M0/C1", "M0/C2", None])

    def test_a_route_that_reaches_a_noop_is_a_problem(self):
        slot = only_slot(slots_of(MANIFEST_1D, "M0/C1/W0", {RECEIVER: [header_1d(0b10)]}), RECEIVER)
        self.assertIsNone(slot.route)
        self.assertEqual(slot.problems, ["no route: hop 1 is noop, which the router asserts on"])


class HeaderTest(unittest.TestCase):
    def test_a_scatter_writes_its_chunks(self):
        # Chunk 0 writes its 100 bytes, and chunk 1, the last, is a flushed increment of 7.
        command = scatter_command([100, 7, 0], 2, 0b11_01)
        header = header_2d(TOWARD_C2, send_type=SCATTER, command=command)
        slot = only_slot(slots_of(MANIFEST_2D, "M0/C0/E0", {SENDER: [header]}), SENDER)
        self.assertEqual(
            slot.scatter, [ScatterChunk(ChunkEncoding.UNICAST_WRITE, 100), ScatterChunk(ChunkEncoding.SEMINC_FLUSH, 7)]
        )
        self.assertEqual(slot.problems, [])

    def test_a_scatter_the_router_cant_carry_out_is_a_problem(self):
        header = header_2d(TOWARD_C2, send_type=SCATTER, command=scatter_command([1, 2, 3], 5, 0b01_01_01_01))
        slot = only_slot(slots_of(MANIFEST_2D, "M0/C0/E0", {SENDER: [header]}), SENDER)
        self.assertIsNone(slot.scatter)
        self.assertIn("can't carry out this scatter write", slot.problems[0])
        self.assertIsNotNone(slot.route)

    def test_an_unknown_send_type_is_a_problem_and_the_route_still_decodes(self):
        slot = only_slot(slots_of(MANIFEST_2D, "M0/C0/E0", {SENDER: [header_2d(TOWARD_C2, send_type=9)]}), SENDER)
        self.assertEqual(slot.problems, ["noc_send_type is 9, which names no send type"])
        self.assertEqual(slot.header["command_fields"]["member"], None)
        self.assertEqual(slot.route.walk.delivers, ["M0/C2"])


def renamed(manifest, type_name, old, new):
    manifest = copy.deepcopy(manifest)
    for member in manifest["archs"]["test_arch"]["types"][type_name]["members"]:
        if member["name"] == old:
            member["name"] = new
    return manifest


class ManifestTest(unittest.TestCase):
    def context(self, manifest):
        types = get_types_from_manifest(manifest)
        return get_slot_context(manifest, types, get_route_context(manifest))

    def test_a_renamed_route_member_is_named_even_with_every_slot_empty(self):
        cases = [
            (
                MANIFEST_2D,
                "HybridMeshPacketHeader",
                "dst_start_mesh_id",
                r"HybridMeshPacketHeader has no dst_start_mesh_id",
            ),
            (MANIFEST_1D, "RoutingFields1D", "route_buffer", r"routing_fields has no route_buffer"),
            (MANIFEST_2D, "NocUnicastScatterCommandHeader", "chunk_count", r"unicast_scatter_write has no chunk_count"),
        ]
        for manifest, type_name, old, named in cases:
            with self.subTest(old), self.assertRaisesRegex(ManifestError, named):
                self.context(renamed(manifest, type_name, old, f"{old}_v2"))

    def test_a_route_buffer_too_short_for_the_mesh_is_refused(self):
        manifest = copy.deepcopy(MANIFEST_2D)
        route_buffer = manifest["archs"]["test_arch"]["types"]["HybridMeshPacketHeader"]["members"][4]
        route_buffer.update(size=4, num_elements=4)
        with self.assertRaisesRegex(ManifestError, r"route_buffer holds 4 bytes; M0's 2x3 map needs 5"):
            self.context(manifest)

    def test_a_direction_the_rules_dont_know_is_refused(self):
        manifest = copy.deepcopy(MANIFEST_2D)
        manifest["meshes"]["M0"]["chips"]["C0"]["routers"]["E0"]["fields"]["my_direction"]["value"] = 5
        with self.assertRaisesRegex(ManifestError, r"my_direction is 5, a direction the 2D rules don't know"):
            slots_of(manifest, "M0/C0/E0", {SENDER: [bytes(0)]})


if __name__ == "__main__":
    unittest.main()
