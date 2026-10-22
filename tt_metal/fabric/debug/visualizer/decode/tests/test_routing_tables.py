# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Cases for storing each unique routing table once and decoding it, on hand-written manifests and snapshots. The
tables are worked out by hand from the host's encoders, apart from the corrupted ones.

Run from the repository root: python3 -m unittest tt_metal.fabric.debug.visualizer.decode.tests.test_routing_tables
"""

import copy
import pathlib
import struct
import unittest

from tt_metal.fabric.debug.visualizer.common.manifest_lookup import ManifestError
from tt_metal.fabric.debug.visualizer.decode.inputs import Input
from tt_metal.fabric.debug.visualizer.decode.route_context import ChipOdd2D, ChipStep2D, ChipWalk2D, get_route_context
from tt_metal.fabric.debug.visualizer.decode.routers import decode_router
from tt_metal.fabric.debug.visualizer.decode.routes import Direction, OddKind, TreeEdge2D
from tt_metal.fabric.debug.visualizer.decode.routing_tables import (
    PathEntry1D,
    PathTable1D,
    RouterTableRef,
    get_routing_table_store,
)
from tt_metal.fabric.debug.visualizer.decode.tests.test_routes import MANIFEST as ENCODINGS
from tt_metal.fabric.debug.visualizer.decode.tests.test_routes import line_router, tree_edge
from tt_metal.fabric.debug.visualizer.decode.type_parser import get_types_from_manifest

ADDRESS = 1000
RUNNING, PAUSED, RUN = 1, 2, 0


def table_types(table_member, table_type):
    """A routing table struct: the router's state, then the route table the run uses."""
    state_manager = {"name": "state_manager", "offset": 0, "size": 8, "schema": "struct:RouterStateManager"}
    return {
        "RouterStateManager": {
            "size": 8,
            "members": [
                {"name": "state", "offset": 0, "size": 4, "schema": "enum:RouterState"},
                {"name": "command", "offset": 4, "size": 4, "schema": "enum:RouterCommand"},
            ],
        },
        "routing_l1_info_t": {"size": 8 + table_type["size"], "members": [state_manager, table_member]},
        table_member["schema"][len("struct:") :]: table_type,
    }


def manifest(is_2d, shape, chips, types, **context):
    size = types["routing_l1_info_t"]["size"]
    return {
        **copy.deepcopy(ENCODINGS),
        "run": {"arch": "test_arch"},
        "fabric_context": {"is_2d_routing": is_2d, **context},
        "vocabulary": {"z_port_roles": ["none", "intermesh_boundary", "express_chord"]},
        "archs": {
            "test_arch": {
                "areas": {"routing_table": {"address": ADDRESS, "size": size, "schema": "struct:routing_l1_info_t"}},
                "stream_registers": {"registers": {}},
                "types": types,
            }
        },
        "enums": {
            **copy.deepcopy(ENCODINGS["enums"]),
            "RouterState": {"values": {"initializing": 0, "running": 1, "paused": 2}},
            "RouterCommand": {"values": {"run": 0, "pause": 1}},
        },
        "meshes": {"M0": {"shape": list(shape), "torus": {"y": False, "x": False}, "chips": chips}},
    }


def chip(y, x, routers=None):
    """A chip at (y, x). One with routers is local; the others are on another host."""
    if routers is None:
        return {"mesh_coord": [y, x], "is_local": False}
    return {"mesh_coord": [y, x], "is_local": True, "z_port_role": "none", "routers": routers}


# A 2x3 mesh, C0 to C5 in row order. Capture read C0's east and south routers and C1's west and east ones.
#   C0 C1 C2
#   C3 C4 C5
MANIFEST_2D = manifest(
    True,
    (2, 3),
    {
        "C0": chip(0, 0, {"E0": {"link": {"peer": "M0/C1/W0"}}, "S0": {"link": {"peer": None}}}),
        "C1": chip(0, 1, {"W0": {"link": {"peer": "M0/C0/E0"}}, "E0": {"link": {"peer": None}}}),
        "C2": chip(0, 2),
        "C3": chip(1, 0),
        "C4": chip(1, 1),
        "C5": chip(1, 2),
    },
    table_types(
        {"name": "route_table_2d", "offset": 8, "size": 16, "schema": "struct:route_table_2d_t"},
        {
            "size": 16,
            "members": [
                {"name": "action_vectors", "offset": 0, "size": 8, "schema": "bytes"},
                {"name": "mcast_trees", "offset": 8, "size": 8, "schema": "bytes"},
            ],
        },
    ),
)

# The 2x3 mesh's action vectors, as compute_2d_routing_action_vectors writes them: Y first, then X. A destination row's
# 2 codes take a byte, and so do a destination column's 3. Y codes: 1 north, 2 south. X codes: 1 east, 2 west.
ACTION_VECTORS = bytes(
    [
        0b01_00,  # toward row 0: row 0 stops, row 1 goes north
        0b00_10,  # toward row 1: row 0 goes south, row 1 stops
        0b10_10_00,  # toward column 0: column 0 stops, columns 1 and 2 go west
        0b10_00_01,  # toward column 1: column 0 goes east, column 1 stops, column 2 goes west
        0b00_01_01,  # toward column 2: columns 0 and 1 go east, column 2 stops
        0,
        0,
        0,
    ]
)
# C0's trees: the Y tree hangs row 1 south of row 0, and the X tree column 2 east of column 1, then column 1 east of 0.
TREES = tree_edge(1, 0, 2) + tree_edge(2, 1, 1) + tree_edge(1, 0, 1) + bytes(2)


def area_2d(state=RUNNING, action_vectors=ACTION_VECTORS):
    return struct.pack("<II", state, RUN) + action_vectors + TREES


# A 1D line of three chips, C0 - C1 - C2. Each receiver forwards through the router on its chip's other side.
MANIFEST_1D = manifest(
    False,
    (1, 3),
    {
        "C0": chip(0, 0, {"E0": line_router("M0/C1/W0")}),
        "C1": chip(0, 1, {"W0": line_router("M0/C0/E0", "M0/C1/E0"), "E0": line_router("M0/C2/W0", "M0/C1/W0")}),
        "C2": chip(0, 2, {"W0": line_router("M0/C1/E0")}),
    },
    table_types(
        {"name": "routing_path_table_1d", "offset": 8, "size": 40, "schema": "struct:intra_mesh_routing_path_t"},
        {"size": 40, "members": [{"name": "paths", "offset": 0, "size": 40, "schema": "bytes"}]},
    ),
    routing_1d_extension_words=1,
)

# With 1 extension word, each entry is 2 words. The host fills entries 1 and 2 (encode_1d_unicast) and leaves entry 0,
# the chip itself, and entry 4 zero. Entry 3 is corrupted: it forwards once, then reads a noop.
PATHS = struct.pack("<10I", 0, 0, 0x1, 0, 0b0110, 0, 0x2, 0, 0, 0)


def area_1d(paths=PATHS):
    return struct.pack("<II", RUNNING, RUN) + paths


def store(areas, manifest=MANIFEST_2D):
    """The store for a snapshot holding each router's area, in order. A router whose area is None wasn't read."""
    raw, entries = b"", {}
    for path, area in areas.items():
        entry = {"areas": {}, "regions": {}, "streams": {}, "error": None}
        if area is not None:
            entry["areas"]["routing_table"] = {"address": ADDRESS, "size": len(area), "offset": len(raw)}
            raw += area
        entries[path] = entry
    snapshot = Input(
        pathlib.Path("snapshot.json"), "0" * 64, {"routers": entries}, raw, pathlib.Path("manifest.json"), manifest, ()
    )
    types = get_types_from_manifest(manifest)
    decoded_routers = {path: decode_router(snapshot, types, path) for path in entries}
    return get_routing_table_store(snapshot, types, decoded_routers, get_route_context(manifest))


ROUTERS_2D = ("M0/C0/E0", "M0/C0/S0", "M0/C1/W0", "M0/C1/E0")


def all_four(**changed):
    """Every router of MANIFEST_2D, each holding the same area unless `changed` gives it another, by M0_C0_E0 and so on."""
    return {path: changed.get(path.replace("/", "_"), area_2d()) for path in ROUTERS_2D}


def walk(*steps, delivers):
    """A walk that takes `steps`, each "C<a> <direction> C<b>", and delivers at chip `delivers`."""
    chip_steps = []
    for step in steps:
        src, direction, dst = step.split()
        chip_steps.append(ChipStep2D(f"M0/{src}", Direction(direction), f"M0/{dst}"))
    return ChipWalk2D([f"M0/{delivers}"], chip_steps, [], [])


class StoringOnceTest(unittest.TestCase):
    def test_a_chip_whose_routers_agree_has_one_copy(self):
        tables = store(all_four())
        self.assertEqual(
            {key: copy.routers for key, copy in tables.copies.items()},
            {"M0/C0/E0": ["M0/C0/E0", "M0/C0/S0"], "M0/C1/W0": ["M0/C1/W0", "M0/C1/E0"]},
        )
        self.assertEqual(
            {path: table.table_key for path, table in tables.routers.items()},
            {"M0/C0/E0": "M0/C0/E0", "M0/C0/S0": "M0/C0/E0", "M0/C1/W0": "M0/C1/W0", "M0/C1/E0": "M0/C1/W0"},
        )

    def test_two_chips_never_share_a_copy(self):
        # All four areas are the same bytes, since this table holds nothing of its chip's own, yet each chip has a copy.
        self.assertEqual([copy.chip for copy in store(all_four()).copies.values()], ["M0/C0", "M0/C1"])

    def test_routers_that_differ_only_in_state_manager_share_a_copy(self):
        tables = store(all_four(M0_C0_S0=area_2d(state=PAUSED)))
        self.assertEqual(tables.copies["M0/C0/E0"].routers, ["M0/C0/E0", "M0/C0/S0"])
        self.assertEqual(
            tables.routers["M0/C0/S0"],
            RouterTableRef(
                "M0/C0/E0",
                {"state": {"raw": PAUSED, "name": "paused"}, "command": {"raw": RUN, "name": "run"}},
                {"bytes": {"offset": 24, "size": 24}},
            ),
        )
        self.assertEqual(tables.routers["M0/C0/E0"].state_manager["state"]["name"], "running")

    def test_a_router_with_a_corrupted_byte_has_its_own_copy(self):
        corrupted = bytearray(ACTION_VECTORS)
        corrupted[4] ^= 0x10
        tables = store(all_four(M0_C0_S0=area_2d(action_vectors=bytes(corrupted))))
        self.assertEqual(
            {key: copy.routers for key, copy in tables.copies.items()},
            {"M0/C0/E0": ["M0/C0/E0"], "M0/C0/S0": ["M0/C0/S0"], "M0/C1/W0": ["M0/C1/W0", "M0/C1/E0"]},
        )

    def test_a_copy_points_at_its_first_routers_bytes_and_leaves_state_manager_out(self):
        tables = store(all_four())
        copy_of_c1 = tables.copies["M0/C1/W0"]
        self.assertEqual(copy_of_c1.raw, {"bytes": {"offset": 48, "size": 24}})
        self.assertEqual(
            copy_of_c1.fields,
            {
                "route_table_2d": {
                    "action_vectors": {"bytes": {"offset": 56, "size": 8}},
                    "mcast_trees": {"bytes": {"offset": 64, "size": 8}},
                }
            },
        )

    def test_a_router_capture_did_not_read_has_no_routing(self):
        tables = store({"M0/C0/E0": None, "M0/C0/S0": area_2d()})
        self.assertEqual(list(tables.copies), ["M0/C0/S0"])
        self.assertEqual(list(tables.routers), ["M0/C0/S0"])

    def test_no_area_read_gives_an_empty_store(self):
        tables = store({"M0/C0/E0": None})
        self.assertEqual((tables.copies, tables.routers), ({}, {}))


class Routes2DTest(unittest.TestCase):
    def test_a_copy_has_a_route_toward_every_chip_of_its_mesh(self):
        routes = store(all_four()).copies["M0/C0/E0"].route_table_2d
        self.assertEqual(list(routes), [f"M0/C{index}" for index in range(6)])
        for dst, route in routes.items():
            with self.subTest(dst):
                self.assertEqual((route.delivers, route.exits, route.odd), ([dst], [], []))
        # Its own chip delivers where it starts, and the far corner is south first, then east.
        self.assertEqual(routes["M0/C0"], walk(delivers="C0"))
        self.assertEqual(routes["M0/C5"], walk("C0 south C3", "C3 east C4", "C4 east C5", delivers="C5"))

    def test_routes_start_at_the_copys_chip(self):
        routes = store(all_four()).copies["M0/C1/W0"].route_table_2d
        self.assertEqual(routes["M0/C3"], walk("C1 south C4", "C4 west C3", delivers="C3"))

    def test_a_corrupted_code_shows_as_odd(self):
        # Column 2's own code goes east, so a packet for column 2 is delivered there and also sent off the edge.
        corrupted = bytearray(ACTION_VECTORS)
        corrupted[4] = 0b01_01_01
        routes = store({"M0/C0/E0": area_2d(action_vectors=bytes(corrupted))}).copies["M0/C0/E0"].route_table_2d
        self.assertEqual(routes["M0/C2"].delivers, ["M0/C2"])
        self.assertEqual(routes["M0/C2"].odd, [ChipOdd2D("M0/C2", OddKind.OFF_EDGE, Direction.EAST)])
        self.assertEqual(routes["M0/C1"].odd, [])

    def test_trees(self):
        copy_of_c0 = store(all_four()).copies["M0/C0/E0"]
        self.assertEqual(copy_of_c0.mcast_trees.y, [TreeEdge2D(1, 0, Direction.SOUTH)])
        self.assertEqual(copy_of_c0.mcast_trees.x, [TreeEdge2D(2, 1, Direction.EAST), TreeEdge2D(1, 0, Direction.EAST)])
        self.assertIsNone(copy_of_c0.path_table_1d)


class PathTable1DTest(unittest.TestCase):
    def test_nonzero_entries_are_walked_and_zero_ones_counted(self):
        copy_of_c0 = store({"M0/C0/E0": area_1d()}, MANIFEST_1D).copies["M0/C0/E0"]
        self.assertEqual(
            copy_of_c0.path_table_1d,
            PathTable1D(
                [
                    PathEntry1D(1, ["write_only"], None),
                    PathEntry1D(2, ["forward_only", "write_only"], None),
                    PathEntry1D(3, None, "hop 1 is noop, which the router asserts on"),
                ],
                2,
            ),
        )
        self.assertEqual((copy_of_c0.route_table_2d, copy_of_c0.mcast_trees), (None, None))

    def test_two_chips_never_share_a_copy(self):
        areas = {"M0/C0/E0": area_1d(), "M0/C1/W0": area_1d(), "M0/C1/E0": area_1d(), "M0/C2/W0": area_1d()}
        tables = store(areas, MANIFEST_1D)
        self.assertEqual(list(tables.copies), ["M0/C0/E0", "M0/C1/W0", "M0/C2/W0"])


def renamed(manifest, struct_name, old, new):
    manifest = copy.deepcopy(manifest)
    for member in manifest["archs"]["test_arch"]["types"][struct_name]["members"]:
        if member["name"] == old:
            member["name"] = new
    return manifest


class ManifestTest(unittest.TestCase):
    def test_a_renamed_member_is_named(self):
        cases = [
            (MANIFEST_2D, "routing_l1_info_t", "state_manager", r"has no state_manager member"),
            (MANIFEST_2D, "routing_l1_info_t", "route_table_2d", r"routing_table has no route_table_2d member"),
            (MANIFEST_2D, "route_table_2d_t", "action_vectors", r"route_table_2d has no action_vectors member"),
            (MANIFEST_1D, "intra_mesh_routing_path_t", "paths", r"routing_path_table_1d has no paths member"),
        ]
        for manifest, struct_name, old, named in cases:
            area = area_2d() if manifest is MANIFEST_2D else area_1d()
            with self.subTest(old), self.assertRaisesRegex(ManifestError, named):
                store({"M0/C0/E0": area}, renamed(manifest, struct_name, old, f"{old}_v2"))

    def test_a_mesh_the_table_cannot_hold_is_refused(self):
        # Five bytes of action vectors hold the 2x3 mesh's codes; four don't.
        manifest = copy.deepcopy(MANIFEST_2D)
        action_vectors, mcast_trees = manifest["archs"]["test_arch"]["types"]["route_table_2d_t"]["members"]
        action_vectors["size"], mcast_trees["offset"], mcast_trees["size"] = 4, 4, 12
        with self.assertRaisesRegex(ManifestError, r"route_table_2d doesn't hold M0's 2x3 mesh"):
            store({"M0/C0/E0": area_2d()}, manifest)


if __name__ == "__main__":
    unittest.main()
