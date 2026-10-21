# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Cases for the path rule, on a hand-written router.

Run from the repository root: python3 -m unittest tt_metal.fabric.debug.visualizer.common.tests.test_router_paths
"""

import unittest

from tt_metal.fabric.debug.visualizer.common.manifest_lookup import ManifestError
from tt_metal.fabric.debug.visualizer.common.router_paths import (
    LeafKind,
    get_node,
    get_router,
    get_router_path,
    walk_router,
)

WHERE = "meshes.M0.chips.C0.routers.E1"
COUNTERS = {"address": 8192, "size": 8, "num_elements": 2, "size_per_element": 4, "schema": "u32"}
RING = {"address": 12288, "size": 64, "schema": "bytes", "extra": {"address": 1, "size": 1}}
CREDIT = {"array": "credit_counters/to_sender_completion", "index": 1}
FREE_SLOTS = {"kind": "stream", "stream_id": 0, "register": "buf_space_available", "schema": "u32"}
SPARE = {"address": 16384, "size": 4, "schema": "u32"}
ROUTER = {
    "identity": {"eth_chan": 1, "virtual_core": [25, 16]},
    "credit_counters": {"to_sender_completion": COUNTERS},
    "channels": {
        "senders": {
            "vc0": {
                "ch0": {
                    "ring_buffer": RING,
                    "credits": {"completed": CREDIT},
                    "fields": {
                        "free_slots": FREE_SLOTS,
                        "unallocated": None,
                        "is_injection_channel": {"kind": "flag", "value": False},
                    },
                }
            }
        }
    },
    "spares": [None, SPARE],
}


class WalkRouterTest(unittest.TestCase):
    def test_every_leaf_in_manifest_order(self):
        self.assertEqual(
            [(leaf.path, leaf.kind, leaf.node) for leaf in walk_router(ROUTER, WHERE)],
            [
                ("credit_counters/to_sender_completion", LeafKind.REGION, COUNTERS),
                ("channels/senders/vc0/ch0/ring_buffer", LeafKind.REGION, RING),
                ("channels/senders/vc0/ch0/credits/completed", LeafKind.ELEMENT, CREDIT),
                ("channels/senders/vc0/ch0/fields/free_slots", LeafKind.STREAM, FREE_SLOTS),
                ("spares/1", LeafKind.REGION, SPARE),
            ],
        )

    def test_where_is_each_leaf_s_place_in_the_manifest(self):
        wheres = {leaf.path: leaf.where for leaf in walk_router(ROUTER, WHERE)}
        self.assertEqual(
            wheres["channels/senders/vc0/ch0/fields/free_slots"], f"{WHERE}.channels.senders.vc0.ch0.fields.free_slots"
        )
        self.assertEqual(wheres["spares/1"], f"{WHERE}.spares[1]")


class GetNodeTest(unittest.TestCase):
    def test_a_path_finds_its_node(self):
        self.assertIs(get_node(ROUTER, "channels/senders/vc0/ch0/ring_buffer", WHERE), RING)
        self.assertIs(get_node(ROUTER, "spares/1", WHERE), SPARE)

    def test_each_leaf_s_path_finds_it(self):
        for leaf in walk_router(ROUTER, WHERE):
            with self.subTest(leaf.path):
                self.assertIs(get_node(ROUTER, leaf.path, WHERE), leaf.node)

    def test_a_path_to_nothing_is_named(self):
        for path, named in (
            ("channels/senders/vc1", r"E1\.channels\.senders has no 'vc1'"),
            ("spares/2", r"E1\.spares has no '2'"),
            ("spares/first", r"E1\.spares has no 'first'"),
            ("identity/eth_chan/0", r"E1\.identity\.eth_chan has no '0'"),
        ):
            with self.subTest(path), self.assertRaisesRegex(ManifestError, named):
                get_node(ROUTER, path, WHERE)


class GetRouterTest(unittest.TestCase):
    MANIFEST = {"meshes": {"M0": {"chips": {"C0": {"routers": {"E1": ROUTER}}}}}}

    def test_a_router_path_finds_its_router(self):
        self.assertEqual(get_router(self.MANIFEST, get_router_path("M0", "C0", "E1")), (ROUTER, WHERE))

    def test_a_router_that_isnt_there_is_named(self):
        with self.assertRaisesRegex(ManifestError, r"meshes\.M0\.chips\.C0\.routers\.W0"):
            get_router(self.MANIFEST, "M0/C0/W0")

    def test_something_that_isnt_a_router_path_is_refused(self):
        with self.assertRaisesRegex(ManifestError, "isn't a router path"):
            get_router(self.MANIFEST, "M0/C0")


if __name__ == "__main__":
    unittest.main()
