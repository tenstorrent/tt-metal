# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Hand-written cases for the route rules. Route words and maps are worked out by hand from the fabric's encoders,
apart from the broken maps the odd-case tests need.

Run from the repository root: python3 -m unittest tt_metal.fabric.debug.visualizer.decode.tests.test_routes
"""

import copy
import json
import struct
import unittest

from tt_metal.fabric.debug.visualizer.decode.routes import (
    ChunkEncoding,
    ManifestError,
    MeshView,
    Odd2D,
    OddKind,
    ScatterChunk,
    Step2D,
    TreeEdge2D,
    get_encodings_from_manifest,
    get_mcast_tree_edges_2d,
    get_mesh_view,
    get_path_table_entry_1d,
    get_route_map_from_table_2d,
    get_scatter_chunks,
    read_action_2d,
    read_packed,
    walk_1d,
    walk_2d,
)

# routing_encodings as make_routing_encodings_json writes it, and the one enum the rules read.
MANIFEST = {
    "routing_encodings": {
        "1d": {
            "bits_per_hop": 2,
            "hops_per_word": 16,
            "hop": {"noop": 0, "write_only": 1, "forward_only": 2, "write_and_forward": 3},
        },
        "2d": {
            "action_bits": {"east": 1, "west": 2, "north": 4, "south": 8, "z": 16, "local_deliver": 32},
            "table_entry_bits": 2,
            "table_entries_per_byte": 4,
            "y_code_action": [0, 4, 8, 16],
            "x_code_action": [0, 1, 2, 0],
            "mcast_tree_edge": {
                "bytes": 2,
                "child": {"shift": 0, "bits": 6},
                "parent": {"shift": 6, "bits": 6},
                "output": {"shift": 12, "bits": 2},
            },
        },
        "scatter": {"chunk_encoding_bits": 2, "max_chunks": 4},
    },
    "enums": {
        "NocScatterWriteChunkEncoding": {
            "values": {
                "chunk_encoding_nop": 0,
                "chunk_encoding_unicast_write": 1,
                "chunk_encoding_seminc_no_flush": 2,
                "chunk_encoding_seminc_flush": 3,
            }
        }
    },
}
ENC = get_encodings_from_manifest(MANIFEST)

# 2D action bytes.
E, W, N, S, Z, D = 1, 2, 4, 8, 16, 32


def hop_names(words):
    return [hop.name for hop in walk_1d(words, ENC.one_d)]


class GetEncodingsFromManifestTest(unittest.TestCase):
    def test_hops_are_indexed_by_value(self):
        self.assertEqual(
            [(hop.name, hop.writes, hop.forwards) for hop in ENC.one_d.hops],
            [
                ("noop", False, False),
                ("write_only", True, False),
                ("forward_only", False, True),
                ("write_and_forward", True, True),
            ],
        )

    def test_a_missing_value_is_named(self):
        manifest = copy.deepcopy(MANIFEST)
        del manifest["routing_encodings"]["1d"]["hop"]
        with self.assertRaisesRegex(ManifestError, r"routing_encodings\.1d\.hop"):
            get_encodings_from_manifest(manifest)

    def test_an_unknown_hop_name_is_refused(self):
        manifest = copy.deepcopy(MANIFEST)
        hop = manifest["routing_encodings"]["1d"]["hop"]
        hop["skip"] = hop.pop("noop")
        with self.assertRaisesRegex(ManifestError, "skip"):
            get_encodings_from_manifest(manifest)

    def test_a_2d_action_bit_shared_by_two_names_is_refused(self):
        manifest = copy.deepcopy(MANIFEST)
        manifest["routing_encodings"]["2d"]["action_bits"]["z"] = S
        with self.assertRaisesRegex(ManifestError, "action_bits"):
            get_encodings_from_manifest(manifest)

    def test_a_tree_edge_output_that_is_not_a_table_code_is_refused(self):
        manifest = copy.deepcopy(MANIFEST)
        manifest["routing_encodings"]["2d"]["mcast_tree_edge"]["output"]["bits"] = 3
        with self.assertRaisesRegex(ManifestError, r"mcast_tree_edge\.output"):
            get_encodings_from_manifest(manifest)

    def test_an_unknown_chunk_type_name_is_refused(self):
        manifest = copy.deepcopy(MANIFEST)
        types = manifest["enums"]["NocScatterWriteChunkEncoding"]["values"]
        types["chunk_encoding_atomic_add"] = types.pop("chunk_encoding_seminc_flush")
        with self.assertRaisesRegex(ManifestError, "chunk_encoding_atomic_add"):
            get_encodings_from_manifest(manifest)


class ReadPackedTest(unittest.TestCase):
    def test_entries_run_across_bytes(self):
        # 3-bit entries 5, 3, 7 are the run 0b111_011_101 = 0x1DD; entry 2 starts in byte 0 and ends in byte 1.
        self.assertEqual(read_packed(bytes([0xDD, 0x01]), 3, 3), [5, 3, 7])

    def test_too_few_bytes_are_refused(self):
        with self.assertRaises(ValueError):
            read_packed(bytes([0xDD]), 3, 3)


class Walk1DTest(unittest.TestCase):
    def test_unicast(self):
        # encode_1d_unicast's example: 3 hops is 0x1A.
        self.assertEqual(hop_names([0x1A]), ["forward_only", "forward_only", "write_only"])

    def test_unicast_into_an_extension_word(self):
        # 17 hops with 1 extension word: 16 forwards fill word 0, and the write is word 1's first field.
        self.assertEqual(hop_names([0xAAAAAAAA, 0x1]), ["forward_only"] * 16 + ["write_only"])

    def test_multicast(self):
        # encode_1d_multicast's example: start 3, range 2 is 0x7A.
        self.assertEqual(hop_names([0x7A]), ["forward_only", "forward_only", "write_and_forward", "write_only"])

    def test_sparse_multicast(self):
        # encode_1d_sparse_multicast's example: hop mask 0b1010 is 0x6E.
        self.assertEqual(hop_names([0x6E]), ["forward_only", "write_and_forward", "forward_only", "write_only"])

    def test_leftover_words_are_not_read(self):
        # Setting a sparse multicast writes only word 0, so the other words can hold an earlier route.
        self.assertEqual(hop_names([0x6E, 0xDEADBEEF]), hop_names([0x6E]))

    def test_a_route_to_self_is_refused(self):
        # encode_1d_unicast's self-route: 0 hops leaves every word 0, so hop 0 is noop.
        with self.assertRaisesRegex(ValueError, "hop 0 is noop"):
            hop_names([0, 0])

    def test_forwarding_past_the_last_word_is_refused(self):
        # 16 forwards use every field of the only word, so hop 16 reads the all-zero word after it.
        with self.assertRaisesRegex(ValueError, "hop 16 is noop"):
            hop_names([0xAAAAAAAA])


class GetPathTableEntry1DTest(unittest.TestCase):
    PATHS = struct.pack("<6I", 10, 11, 12, 13, 14, 15)

    def test_entries_are_one_plus_extension_words_apart(self):
        self.assertEqual(get_path_table_entry_1d(self.PATHS, 1, 0), [11])
        self.assertEqual(get_path_table_entry_1d(self.PATHS, 2, 1), [14, 15])

    def test_an_entry_past_the_table_is_refused(self):
        with self.assertRaises(ValueError):
            get_path_table_entry_1d(self.PATHS, 3, 1)


def router(peer):
    return {"link": {"peer": peer}}


# A 4x1 column whose Z express chords skip a row.
MESH_MANIFEST = {
    "vocabulary": {"z_port_roles": ["none", "intermesh_boundary", "express_chord"]},
    "meshes": {
        "M0": {
            "shape": [4, 1],
            "torus": {"y": True, "x": False},
            "chips": {
                # Both of C0's Z links reach C2.
                "C0": {
                    "mesh_coord": [0, 0],
                    "is_local": True,
                    "z_port_role": "express_chord",
                    "routers": {"S0": router("M0/C1/N0"), "Z0": router("M0/C2/Z0"), "Z1": router("M0/C2/Z1")},
                },
                # C1's Z port leads to another mesh.
                "C1": {
                    "mesh_coord": [1, 0],
                    "is_local": True,
                    "z_port_role": "intermesh_boundary",
                    "routers": {"Z0": router("M1/C0/Z0")},
                },
                # C2's Z links disagree: one reaches C0, the other is unconnected.
                "C2": {
                    "mesh_coord": [2, 0],
                    "is_local": True,
                    "z_port_role": "express_chord",
                    "routers": {"Z0": router("M0/C0/Z0"), "Z1": router(None)},
                },
                # C3 is on another host, so it has no routers.
                "C3": {"mesh_coord": [3, 0], "is_local": False},
            },
        },
    },
}


class GetMeshViewTest(unittest.TestCase):
    def test_shape_and_torus(self):
        view = get_mesh_view(MESH_MANIFEST, "M0")
        self.assertEqual((view.mesh_id, view.shape, view.torus), (0, (4, 1), (True, False)))

    def test_a_z_peer_is_known_where_an_express_chord_reaches_one_chip(self):
        self.assertEqual(get_mesh_view(MESH_MANIFEST, "M0").z_peers, {(0, 0): (2, 0)})

    def test_a_missing_mesh_is_named(self):
        with self.assertRaisesRegex(ManifestError, r"meshes\.M7"):
            get_mesh_view(MESH_MANIFEST, "M7")

    def test_a_z_port_role_the_rules_dont_know_is_refused_even_if_no_chip_has_it(self):
        manifest = copy.deepcopy(MESH_MANIFEST)
        manifest["vocabulary"]["z_port_roles"].append("x_express_chord")
        with self.assertRaisesRegex(ManifestError, r"vocabulary\.z_port_roles names .*'x_express_chord'"):
            get_mesh_view(manifest, "M0")

    def test_a_local_chip_the_z_rule_cannot_read_is_refused(self):
        cases = {
            "no z_port_role": (lambda chips: chips["C1"].pop("z_port_role"), r"chips\.C1\.z_port_role"),
            "an unknown z_port_role": (lambda chips: chips["C1"].update(z_port_role="chord"), r"'chord'"),
            "a Z link with no peer": (
                lambda chips: chips["C0"]["routers"]["Z0"]["link"].pop("peer"),
                r"chips\.C0\.routers\.Z0\.link\.peer",
            ),
        }
        for name, (edit, named) in cases.items():
            manifest = copy.deepcopy(MESH_MANIFEST)
            edit(manifest["meshes"]["M0"]["chips"])
            with self.subTest(name), self.assertRaisesRegex(ManifestError, named):
                get_mesh_view(manifest, "M0")


def load(manifest):
    """Runs both loaders, as check_manifest_routes does."""
    get_encodings_from_manifest(manifest)
    get_mesh_view(manifest, "M0")


def with_value(manifest, path, value):
    """A copy of `manifest` with the value at a dotted path replaced."""
    manifest = copy.deepcopy(manifest)
    *parents, key = path.split(".")
    node = manifest
    for parent in parents:
        node = node[parent]
    node[key] = value
    return manifest


class ManifestTypesTest(unittest.TestCase):
    BOTH = {**MANIFEST, **MESH_MANIFEST}

    def test_a_value_of_the_wrong_type_is_named(self):
        load(self.BOTH)
        c0 = "meshes.M0.chips.C0"
        cases = [
            ("routing_encodings.1d.hop", [0, 1, 2, 3], r"1d\.hop is an array; .* an object of integers"),
            ("routing_encodings.1d.bits_per_hop", True, r"bits_per_hop is a boolean; the route rules need an integer"),
            ("routing_encodings.2d.y_code_action", {"0": 0}, r"y_code_action is an object; .* an array of integers"),
            ("routing_encodings.2d.y_code_action", [0, 4, "8", 16], r"y_code_action\[2\] is a string"),
            ("routing_encodings.2d.mcast_tree_edge.bytes", 2.0, r"mcast_tree_edge\.bytes is a number"),
            ("meshes.M0.shape", {"y": 4, "x": 1}, r"meshes\.M0\.shape is an object; .* an array of integers"),
            ("meshes.M0.torus", [True, False], r"meshes\.M0\.torus is an array; the route rules need an object"),
            ("vocabulary.z_port_roles", "none", r"z_port_roles is a string; .* an array of strings"),
            (f"{c0}.mesh_coord", 0, r"C0\.mesh_coord is an integer; .* an array of integers"),
            (f"{c0}.is_local", "true", r"C0\.is_local is a string; the route rules need a boolean"),
            (f"{c0}.routers.Z0.link.peer", 2, r"Z0\.link\.peer is an integer; the route rules need a string or null"),
        ]
        for path, value, named in cases:
            with self.subTest(path=path, value=value), self.assertRaisesRegex(ManifestError, named):
                load(with_value(self.BOTH, path, value))


def mesh(y_size, x_size, torus=(False, False), z_peers=None):
    return MeshView(0, (y_size, x_size), torus, z_peers or {})


def walk(route_map, start, mesh_view, dst_mesh=0):
    return walk_2d(bytes(route_map), start, mesh_view, ENC.two_d, dst_mesh)


class ReadAction2DTest(unittest.TestCase):
    # A 2x3 mesh's map: row 0 says north and row 1 is empty; the columns say east, deliver, west.
    MAP = bytes([N, 0, E, D, W])

    def test_a_router_entered_from_the_east_or_west_reads_its_column(self):
        self.assertEqual(read_action_2d(self.MAP, 0, 1, 2, "east"), D)
        self.assertEqual(read_action_2d(self.MAP, 0, 1, 2, "west"), D)

    def test_any_other_router_reads_its_row_unless_the_row_is_empty(self):
        self.assertEqual(read_action_2d(self.MAP, 0, 1, 2, "north"), N)
        self.assertEqual(read_action_2d(self.MAP, 1, 1, 2, "north"), D)

    def test_the_start_of_a_route_reads_its_row_first(self):
        self.assertEqual(read_action_2d(self.MAP, 0, 1, 2, None), N)

    def test_a_misspelt_direction_is_refused(self):
        with self.assertRaises(ValueError):
            read_action_2d(self.MAP, 0, 1, 2, "esat")


class Walk2DTest(unittest.TestCase):
    def test_unicast(self):
        # The design doc's 3x3 example toward (2, 1): rows S, S, -, then columns E, deliver, W.
        result = walk([S, S, 0, E, D, W], (0, 0), mesh(3, 3))
        self.assertEqual(result.delivers, [(2, 1)])
        self.assertEqual(
            result.steps,
            [Step2D((0, 0), "south", (1, 0)), Step2D((1, 0), "south", (2, 0)), Step2D((2, 0), "east", (2, 1))],
        )
        self.assertEqual((result.exits, result.odd), ([], []))

    def test_multicast_follows_every_bit(self):
        # encode_2d_mcast_maps for one hop north and one south from (1, 1) of a 3x3 mesh, whose Y tree hangs rows 0
        # and 2 off row 1. The sender's own row isn't a target.
        result = walk([D, N | S, D, 0, D, 0], (1, 1), mesh(3, 3))
        self.assertEqual(result.delivers, [(0, 1), (2, 1)])
        self.assertEqual(result.odd, [])

    def test_multicast_rows_carry_the_column_moves(self):
        # encode_2d_mcast_maps for one hop south and two east from (0, 0) of a 2x3 mesh. Row 1 carries column 0's
        # east move, so (1, 2), entered from the west, must read its column's deliver, not row 1's east.
        result = walk([S, E | D, E | D, E | D, D], (0, 0), mesh(2, 3))
        self.assertEqual(result.delivers, [(1, 0), (1, 1), (1, 2)])
        self.assertEqual(result.odd, [])

    def test_a_step_wraps_on_a_torus(self):
        # West from column 0 of a 1x3 ring reaches column 2.
        result = walk([0, W, 0, D], (0, 0), mesh(1, 3, torus=(False, True)))
        self.assertEqual(result.steps, [Step2D((0, 0), "west", (0, 2))])
        self.assertEqual(result.delivers, [(0, 2)])

    def test_a_step_off_an_edge_that_does_not_wrap(self):
        result = walk([0, W, 0, D], (0, 0), mesh(1, 3))
        self.assertEqual((result.steps, result.delivers), ([], []))
        self.assertEqual(result.odd, [Odd2D((0, 0), "off_edge", "west")])

    def test_z_follows_the_known_peer(self):
        # Row 0 says Z, (0, 0)'s Z link reaches (2, 0), and column 0 says deliver.
        result = walk([Z, 0, 0, D, 0, 0], (0, 0), mesh(3, 3, z_peers={(0, 0): (2, 0)}))
        self.assertEqual(result.steps, [Step2D((0, 0), "z", (2, 0))])
        self.assertEqual(result.delivers, [(2, 0)])

    def test_z_with_no_known_peer(self):
        result = walk([Z, 0, 0, D, 0, 0], (0, 0), mesh(3, 3))
        self.assertEqual(result.odd, [Odd2D((0, 0), "no_z_peer", "z")])

    def test_a_chip_that_says_only_deliver_is_the_exit_to_another_mesh(self):
        # Toward (2, 2), the chip of mesh 0 that leads to mesh 1.
        route_map = [S, S, 0, 0, 0, D]
        result = walk(route_map, (0, 2), mesh(3, 3), dst_mesh=1)
        self.assertEqual((result.exits, result.delivers), ([(2, 2)], []))
        self.assertEqual(walk(route_map, (0, 2), mesh(3, 3), dst_mesh=0).delivers, [(2, 2)])

    def test_bits_the_router_ignores(self):
        # (0, 1) is entered from the west, and its column also says west.
        result = walk([0, E, W | D, 0], (0, 0), mesh(1, 3))
        self.assertEqual((result.delivers, result.odd), ([(0, 1)], [Odd2D((0, 1), "ignored_bit", "west")]))
        # (1, 1) is entered from the west, and its column says north.
        result = walk([0, 0, E, N | D], (1, 0), mesh(2, 2))
        self.assertEqual((result.delivers, result.odd), ([(1, 1)], [Odd2D((1, 1), "ignored_bit", "north")]))

    def test_an_empty_byte_or_unknown_bits(self):
        self.assertEqual(walk([0, 0, 0, 0], (0, 0), mesh(1, 3)).odd, [Odd2D((0, 0), "empty")])
        self.assertEqual(walk([0, 0x40 | D, 0, 0], (0, 0), mesh(1, 3)).odd, [Odd2D((0, 0), "unknown_bits")])

    def test_a_loop_stops(self):
        # Every column of a 1x3 ring says east, so the route goes round.
        result = walk([0, E, E, E], (0, 0), mesh(1, 3, torus=(False, True)))
        self.assertEqual(result.odd, [Odd2D((0, 1), "revisit", "west")])

    def test_a_kind_is_an_odd_kind_that_writes_to_json_as_its_name(self):
        (odd,) = walk([0, 0, 0, 0], (0, 0), mesh(1, 3)).odd
        self.assertIs(odd.kind, OddKind.EMPTY)
        self.assertEqual(json.dumps(odd.kind), '"empty"')


class GetRouteMapFromTable2DTest(unittest.TestCase):
    def test_the_design_doc_example(self):
        # 3x3 toward (2, 1). Destination row 2's codes: rows 0 and 1 go S, row 2 stops. Destination column 1's
        # codes: column 0 goes E, column 1 stops, column 2 goes W. Every other destination's byte is 0xFF.
        action_vectors = bytes([0xFF, 0xFF, 0b00_10_10, 0xFF, 0b10_00_01, 0xFF])
        route_map = get_route_map_from_table_2d(action_vectors, 3, 3, (2, 1), ENC.two_d)
        self.assertEqual(route_map, bytes([S, S, 0, E, D, W]))

    def test_each_destination_starts_on_a_new_byte(self):
        # 5x1 toward (1, 0): a destination row's 5 codes take 2 bytes. Row 0 goes S, row 1 stops, rows 2 to 4 go N.
        y_rows = [0xFF, 0xFF, 0b01_01_00_10, 0b01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]
        x_rows = [0]
        route_map = get_route_map_from_table_2d(bytes(y_rows + x_rows), 5, 1, (1, 0), ENC.two_d)
        self.assertEqual(route_map, bytes([S, 0, N, N, N, D]))

    def test_a_destination_outside_the_mesh_is_refused(self):
        with self.assertRaises(ValueError):
            get_route_map_from_table_2d(bytes(6), 3, 3, (3, 0), ENC.two_d)


def tree_edge(child, parent, code):
    return (child | parent << 6 | code << 12).to_bytes(2, "little")


class GetMcastTreeEdges2DTest(unittest.TestCase):
    def test_the_y_tree_comes_first_then_the_x_tree(self):
        # The trees of the chip at (1, 0) of a 4x3 mesh. The Y tree hangs row 3 off row 2, then rows 2 and 0 off row 1;
        # its first edge is fabric_common.h's example, 0x2083: child 3, parent 2, south. The X tree starts after the
        # Y tree's 3 edges, and hangs column 2 off column 1 and column 1 off column 0. Code 1 is north on Y, east on X.
        region = (0x2083).to_bytes(2, "little") + tree_edge(2, 1, 2) + tree_edge(0, 1, 1)
        region += tree_edge(2, 1, 1) + tree_edge(1, 0, 1)
        trees = get_mcast_tree_edges_2d(region, 4, 3, ENC.two_d)
        self.assertEqual(trees.y, [TreeEdge2D(3, 2, "south"), TreeEdge2D(2, 1, "south"), TreeEdge2D(0, 1, "north")])
        self.assertEqual(trees.x, [TreeEdge2D(2, 1, "east"), TreeEdge2D(1, 0, "east")])

    def test_an_axis_of_one_chip_has_no_edges(self):
        # A 1x3 mesh has no Y tree, so its X tree starts at byte 0.
        trees = get_mcast_tree_edges_2d(tree_edge(2, 1, 1) + tree_edge(1, 0, 1), 1, 3, ENC.two_d)
        self.assertEqual(trees.y, [])
        self.assertEqual(trees.x, [TreeEdge2D(2, 1, "east"), TreeEdge2D(1, 0, "east")])

    def test_code_3_is_z_on_y_and_nothing_on_x(self):
        trees = get_mcast_tree_edges_2d(tree_edge(1, 0, 3) + tree_edge(1, 0, 3), 2, 2, ENC.two_d)
        self.assertEqual((trees.y, trees.x), ([TreeEdge2D(1, 0, "z")], [TreeEdge2D(1, 0, None)]))

    def test_a_region_too_short_is_refused(self):
        with self.assertRaises(ValueError):
            get_mcast_tree_edges_2d(bytes(4), 2, 3, ENC.two_d)


WRITE, SEMINC, SEMINC_FLUSH = ChunkEncoding.UNICAST_WRITE, ChunkEncoding.SEMINC_NO_FLUSH, ChunkEncoding.SEMINC_FLUSH


class GetScatterChunksTest(unittest.TestCase):
    def test_two_writes_then_a_flushed_increment(self):
        # The fused write and increment: types 0b11_01_01, so chunk 2 adds its chunk_size slot after a flush.
        self.assertEqual(
            get_scatter_chunks(3, [64, 192, 1], 0b110101, 256, ENC.scatter),
            [ScatterChunk(WRITE, 64), ScatterChunk(WRITE, 192), ScatterChunk(SEMINC_FLUSH, 1)],
        )

    def test_the_last_write_gets_the_rest_of_the_payload(self):
        # A plain scatter write marks all four chunks as writes (0x55), whatever chunk_count is.
        self.assertEqual(
            get_scatter_chunks(2, [64, 0, 0], 0x55, 100, ENC.scatter),
            [ScatterChunk(WRITE, 64), ScatterChunk(WRITE, 36)],
        )
        self.assertEqual(
            [chunk.amount for chunk in get_scatter_chunks(4, [10, 20, 30], 0x55, 100, ENC.scatter)], [10, 20, 30, 40]
        )

    def test_only_the_last_chunk_type_is_read(self):
        # Chunk 0's type says nop, but the router writes chunk 0 without reading it.
        self.assertEqual(
            get_scatter_chunks(2, [16, 5, 0], 0b10_00, 16, ENC.scatter),
            [ScatterChunk(WRITE, 16), ScatterChunk(SEMINC, 5)],
        )

    def test_a_scatter_the_router_cannot_carry_out_is_refused(self):
        cases = {
            "one chunk": (1, [16, 0, 0], 0x55, 32),
            "five chunks": (5, [1, 2, 3], 0x55, 32),
            "a nop last": (2, [16, 0, 0], 0b00_01, 32),
            "an increment fourth, past the 3 chunk_size slots": (4, [1, 2, 3], 0b10_01_01_01, 6),
            "writes past the payload": (3, [16, 32, 0], 0x55, 40),
        }
        for name, args in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                get_scatter_chunks(*args, ENC.scatter)


if __name__ == "__main__":
    unittest.main()
