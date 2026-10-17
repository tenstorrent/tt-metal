# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Hand-written cases for the route rules. The route words are the fabric encoders' outputs, worked out by hand.

Run from the repository root: python3 -m unittest tt_metal.fabric.debug.visualizer.decode.tests.test_routes
"""

import copy
import struct
import unittest

from tt_metal.fabric.debug.visualizer.decode.routes import (
    EncodingsError,
    get_encodings_from_manifest,
    get_path_table_entry_1d,
    read_packed,
    walk_1d,
)

# routing_encodings as make_routing_encodings_json writes it.
MANIFEST = {
    "routing_encodings": {
        "1d": {
            "bits_per_hop": 2,
            "hops_per_word": 16,
            "hop": {"noop": 0, "write_only": 1, "forward_only": 2, "write_and_forward": 3},
        },
    },
}
ENC = get_encodings_from_manifest(MANIFEST)


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
        with self.assertRaisesRegex(EncodingsError, r"routing_encodings\.1d\.hop"):
            get_encodings_from_manifest(manifest)

    def test_an_unknown_hop_name_is_refused(self):
        manifest = copy.deepcopy(MANIFEST)
        hop = manifest["routing_encodings"]["1d"]["hop"]
        hop["skip"] = hop.pop("noop")
        with self.assertRaisesRegex(EncodingsError, "skip"):
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

    def test_route_to_self(self):
        # encode_1d_unicast's self-route: 0 hops leaves every word 0.
        self.assertEqual(hop_names([0, 0]), ["noop"])


class GetPathTableEntry1DTest(unittest.TestCase):
    PATHS = struct.pack("<6I", 10, 11, 12, 13, 14, 15)

    def test_entries_are_one_plus_extension_words_apart(self):
        self.assertEqual(get_path_table_entry_1d(self.PATHS, 1, 0), [11])
        self.assertEqual(get_path_table_entry_1d(self.PATHS, 2, 1), [14, 15])

    def test_an_entry_past_the_table_is_refused(self):
        with self.assertRaises(ValueError):
            get_path_table_entry_1d(self.PATHS, 3, 1)


if __name__ == "__main__":
    unittest.main()
