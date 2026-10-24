# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Hand-written cases for the type parser, on a small manifest that has every schema form.

Run from the repository root: python3 -m unittest tt_metal.fabric.debug.visualizer.decode.tests.test_type_parser
"""

import copy
import struct
import unittest

from tt_metal.fabric.debug.visualizer.decode.type_parser import (
    ManifestError,
    decode_struct,
    decode_value,
    get_types_from_manifest,
)

# The keys get_types_from_manifest reads, shaped as the manifest writer writes them.
MANIFEST = {
    "run": {"arch": "test_arch"},
    "archs": {
        "test_arch": {
            "types": {
                "Point": {
                    "size": 4,
                    "members": [
                        {"name": "x", "offset": 0, "size": 2, "schema": "u16"},
                        {"name": "y", "offset": 2, "size": 2, "schema": "i16"},
                    ],
                },
                # Every member form, in 24 bytes.
                "Record": {
                    "size": 24,
                    "members": [
                        {"name": "state", "offset": 0, "size": 1, "schema": "enum:State"},
                        {"name": "padding", "offset": 1, "size": 3, "schema": "pad"},
                        {"name": "where", "offset": 4, "size": 4, "schema": "struct:Point"},
                        {
                            "name": "trail",
                            "offset": 8,
                            "size": 8,
                            "num_elements": 2,
                            "size_per_element": 4,
                            "schema": "struct:Point",
                        },
                        {
                            "name": "codes",
                            "offset": 16,
                            "size": 2,
                            "num_elements": 5,
                            "bits_per_entry": 3,
                            "schema": "enum:State",
                        },
                        {"name": "blob", "offset": 18, "size": 2, "schema": "bytes"},
                        {
                            "name": "words",
                            "offset": 20,
                            "size": 4,
                            "num_elements": 2,
                            "size_per_element": 2,
                            "schema": "u16",
                        },
                    ],
                },
                # Two unions on one tag, which comes after the first, as in the UDM packet header.
                "Header": {
                    "size": 12,
                    "members": [
                        {"name": "command", "offset": 0, "size": 8, "schema": "union:Command", "tag": "kind"},
                        {"name": "kind", "offset": 8, "size": 1, "schema": "enum:Kind"},
                        {"name": "control", "offset": 9, "size": 2, "schema": "union:Control", "tag": "kind"},
                        {"name": "tail_padding", "offset": 11, "size": 1, "schema": "pad"},
                    ],
                },
                "Write": {"size": 8, "members": [{"name": "address", "offset": 0, "size": 8, "schema": "u64"}]},
                "Read": {"size": 4, "members": [{"name": "length", "offset": 0, "size": 4, "schema": "u32"}]},
                # No member for atomic, and no default.
                "Command": {
                    "kind": "union",
                    "size": 8,
                    "tag_schema": "enum:Kind",
                    "members": [
                        {"name": "write", "offset": 0, "size": 8, "schema": "struct:Write", "tag_value": "write"},
                        {"name": "read", "offset": 0, "size": 4, "schema": "struct:Read", "tag_value": "read"},
                    ],
                },
                # Read has its own member, and every other tag picks the default.
                "Control": {
                    "kind": "union",
                    "size": 2,
                    "tag_schema": "enum:Kind",
                    "default_member": "plain",
                    "members": [
                        {"name": "plain", "offset": 0, "size": 2, "schema": "u16"},
                        {"name": "read", "offset": 0, "size": 1, "schema": "u8", "tag_value": "read"},
                    ],
                },
            }
        }
    },
    "enums": {
        "State": {"values": {"idle": 0, "running": 1, "done": 2}},
        "Kind": {"values": {"write": 0, "read": 1, "atomic": 2}},
    },
}
TYPES = get_types_from_manifest(MANIFEST)

# Five 3-bit entries, entry 0 lowest: 2, 1, 2, 7, 0. Entry 2 straddles the two bytes, and 7 has no name.
CODES = 2 | 1 << 3 | 2 << 6 | 7 << 9
RECORD = struct.pack("<B3sHhHhHhH2sHH", 1, b"\xaa\xbb\xcc", 5, -3, 1, 2, 3, 4, CODES, b"\x01\x02", 0x1234, 0xFFFF)
ADDRESS = 0x1122334455667788


def header(kind):
    return struct.pack("<QBHB", ADDRESS, kind, 0xBEEF, 0)


def broken(change):
    """A copy of MANIFEST with `change(types, enums)` applied."""
    manifest = copy.deepcopy(MANIFEST)
    change(manifest["archs"]["test_arch"]["types"], manifest["enums"])
    return manifest


def members(types, name):
    return types[name]["members"]


class DecodeStructTest(unittest.TestCase):
    def test_every_member_form(self):
        self.assertEqual(
            decode_struct("Record", RECORD, 0, TYPES),
            {
                "state": {"raw": 1, "name": "running"},
                "where": {"x": 5, "y": -3},
                "trail": [{"x": 1, "y": 2}, {"x": 3, "y": 4}],
                "codes": [
                    {"raw": 2, "name": "done"},
                    {"raw": 1, "name": "running"},
                    {"raw": 2, "name": "done"},
                    {"raw": 7, "name": None},
                    {"raw": 0, "name": "idle"},
                ],
                "blob": {"bytes": {"offset": 18, "size": 2}},
                "words": [0x1234, 0xFFFF],
            },
        )

    def test_members_keep_their_order_without_pad(self):
        self.assertEqual(
            list(decode_struct("Record", RECORD, 0, TYPES)), ["state", "where", "trail", "codes", "blob", "words"]
        )

    def test_a_bytes_offset_is_into_the_whole_buffer(self):
        self.assertEqual(
            decode_struct("Record", b"\0" * 7 + RECORD, 7, TYPES)["blob"], {"bytes": {"offset": 25, "size": 2}}
        )

    def test_an_unknown_struct_is_named(self):
        with self.assertRaisesRegex(ManifestError, r"archs\.test_arch\.types has no struct 'Nope'"):
            decode_struct("Nope", RECORD, 0, TYPES)

    def test_bytes_past_the_end_are_refused(self):
        with self.assertRaisesRegex(ValueError, "past the 23 bytes"):
            decode_struct("Record", RECORD[:-1], 0, TYPES)


class TaggedUnionTest(unittest.TestCase):
    def test_a_tag_after_its_union_picks_the_member(self):
        self.assertEqual(
            decode_struct("Header", header(0), 0, TYPES),
            {
                "command": {"member": "write", "value": {"address": ADDRESS}},
                "kind": {"raw": 0, "name": "write"},
                "control": {"member": "plain", "value": 0xBEEF},
            },
        )

    def test_unions_keep_their_place_in_member_order(self):
        self.assertEqual(list(decode_struct("Header", header(0), 0, TYPES)), ["command", "kind", "control"])

    def test_a_member_reads_only_its_own_size(self):
        decoded = decode_struct("Header", header(1), 0, TYPES)
        self.assertEqual(decoded["command"], {"member": "read", "value": {"length": 0x55667788}})
        self.assertEqual(decoded["control"], {"member": "read", "value": 0xEF})

    def test_a_tag_with_no_member_picks_the_default_or_gives_the_bytes(self):
        decoded = decode_struct("Header", header(2), 0, TYPES)
        self.assertEqual(decoded["command"], {"member": None, "bytes": {"offset": 0, "size": 8}})
        self.assertEqual(decoded["control"], {"member": "plain", "value": 0xBEEF})

    def test_an_unnamed_tag_picks_the_default_or_gives_the_bytes(self):
        decoded = decode_struct("Header", header(7), 0, TYPES)
        self.assertEqual(decoded["kind"], {"raw": 7, "name": None})
        self.assertEqual(decoded["command"], {"member": None, "bytes": {"offset": 0, "size": 8}})
        self.assertEqual(decoded["control"], {"member": "plain", "value": 0xBEEF})


class DecodeValueTest(unittest.TestCase):
    def decode(self, node, raw, offset=0):
        return decode_value(node, raw, offset, TYPES, "region")

    def test_every_integer_width(self):
        for schema, data, expected in (
            ("u8", b"\x80", 128),
            ("i8", b"\x80", -128),
            ("u16", b"\x00\x80", 0x8000),
            ("i16", b"\x00\x80", -0x8000),
            ("u32", b"\xff\xff\xff\xff", 0xFFFFFFFF),
            ("i32", b"\xff\xff\xff\xff", -1),
            ("u64", b"\xff" * 8, 2**64 - 1),
            ("i64", b"\xff" * 8, -1),
        ):
            with self.subTest(schema):
                self.assertEqual(self.decode({"schema": schema, "size": len(data)}, data), expected)

    def test_an_enum_region_of_any_size(self):
        self.assertEqual(
            self.decode({"schema": "enum:State", "size": 4}, struct.pack("<I", 2)), {"raw": 2, "name": "done"}
        )

    def test_an_array_of_structs_at_an_offset(self):
        node = {"schema": "struct:Point", "size": 8, "num_elements": 2, "size_per_element": 4}
        self.assertEqual(self.decode(node, b"\0\0" + RECORD[8:16], 2), [{"x": 1, "y": 2}, {"x": 3, "y": 4}])

    def test_an_array_of_bytes_gives_each_element_s_offset(self):
        node = {"schema": "bytes", "size": 8, "num_elements": 2, "size_per_element": 4}
        self.assertEqual(
            self.decode(node, bytes(10), 2), [{"bytes": {"offset": 2, "size": 4}}, {"bytes": {"offset": 6, "size": 4}}]
        )

    def test_a_packed_table(self):
        node = {"schema": "enum:State", "size": 2, "num_elements": 5, "bits_per_entry": 3}
        self.assertEqual([entry["raw"] for entry in self.decode(node, RECORD[16:18])], [2, 1, 2, 7, 0])

    def test_pad_decodes_to_nothing(self):
        self.assertIsNone(self.decode({"schema": "pad", "size": 4}, bytes(4)))

    def test_a_tagged_union_outside_its_struct_is_refused(self):
        with self.assertRaisesRegex(ManifestError, "decodes only inside its struct"):
            self.decode({"schema": "union:Command", "size": 8, "tag": "kind"}, bytes(8))

    def test_a_union_with_no_tag_is_refused(self):
        with self.assertRaisesRegex(ManifestError, r"region is a union:Command with no tag"):
            self.decode({"schema": "union:Command", "size": 8}, bytes(8))

    def test_bytes_past_the_end_are_refused(self):
        with self.assertRaisesRegex(ValueError, r"region is bytes 2 to 6"):
            self.decode({"schema": "u32", "size": 4}, bytes(5), 2)


class GetTypesFromManifestTest(unittest.TestCase):
    def test_pad_members_can_share_a_name(self):
        def two_pads(types, enums):
            members(types, "Record")[1:2] = [
                {"name": "pad", "offset": 1, "size": 1, "schema": "pad"},
                {"name": "pad", "offset": 2, "size": 2, "schema": "pad"},
            ]

        types = get_types_from_manifest(broken(two_pads))
        self.assertEqual(decode_struct("Record", RECORD, 0, types), decode_struct("Record", RECORD, 0, TYPES))

    def test_every_check_names_what_it_refuses(self):
        def set_key(name, index, key, value):
            return lambda types, enums: members(types, name)[index].__setitem__(key, value)

        cases = (
            ("an unknown schema", set_key("Point", 0, "schema", "f16"), r"Point\.members\[0\]\.schema is 'f16'"),
            ("a struct types doesn't have", set_key("Record", 2, "schema", "struct:Nope"), "types has no struct Nope"),
            ("an enum enums doesn't have", set_key("Record", 0, "schema", "enum:Nope"), "enums has no Nope"),
            ("a union named as a struct", set_key("Header", 0, "schema", "struct:Command"), "has no struct Command"),
            ("an integer of the wrong size", set_key("Point", 0, "size", 4), "a u16 element is 2 bytes, not 4"),
            ("a struct of the wrong size", set_key("Record", 2, "size", 8), "a struct:Point element is 4 bytes, not 8"),
            (
                "a member past its struct's end",
                set_key("Point", 1, "offset", 3),
                r"Point\.members\[1\] is bytes 3 to 5",
            ),
            ("two members with one name", set_key("Point", 1, "name", "x"), "'x', which an earlier member has"),
            ("a count of zero", set_key("Record", 3, "num_elements", 0), r"Record\.members\[3\]\.num_elements is 0"),
            ("an array that isn't its size", set_key("Record", 3, "num_elements", 3), "3 elements of 4 bytes"),
            ("a packed table that doesn't fit", set_key("Record", 4, "num_elements", 6), "6 entries of 3 bits"),
            ("a packed table of integers", set_key("Record", 4, "schema", "u16"), "a packed table holds an enum:"),
            ("a tag on a plain member", set_key("Point", 0, "tag", "y"), "only a union: member has one"),
            ("a tag naming no member", set_key("Header", 0, "tag", "nope"), r"Header\.members\[0\]\.tag is 'nope'"),
            ("a tag of the wrong enum", set_key("Header", 1, "schema", "enum:State"), "holding one enum:Kind"),
            ("a union member not at 0", set_key("Command", 1, "offset", 4), r"Command\.members\[1\] is bytes 4 to 8"),
            ("an unknown tag value", set_key("Command", 1, "tag_value", "nope"), "enum:Kind doesn't name"),
            (
                "two members with one tag value",
                set_key("Command", 1, "tag_value", "write"),
                "'write', which an earlier",
            ),
            (
                "a union member with no tag",
                lambda types, enums: members(types, "Header")[0].pop("tag"),
                "union:Command with no tag",
            ),
            (
                "a default that isn't a member",
                lambda types, enums: types["Control"].__setitem__("default_member", "nope"),
                r"Control\.default_member is 'nope'",
            ),
            (
                "a tag schema that isn't an enum",
                lambda types, enums: types["Command"].__setitem__("tag_schema", "u8"),
                r"Command\.tag_schema is 'u8'",
            ),
            (
                "a type of an unknown kind",
                lambda types, enums: types["Point"].__setitem__("kind", "bitfield"),
                r"Point\.kind is 'bitfield'",
            ),
            (
                "one enum value with two names",
                lambda types, enums: enums["Kind"]["values"].__setitem__("also_write", 0),
                r"enums\.Kind\.values gives 0 to both 'write' and 'also_write'",
            ),
        )
        for name, change, named in cases:
            with self.subTest(name), self.assertRaisesRegex(ManifestError, named):
                get_types_from_manifest(broken(change))


if __name__ == "__main__":
    unittest.main()
