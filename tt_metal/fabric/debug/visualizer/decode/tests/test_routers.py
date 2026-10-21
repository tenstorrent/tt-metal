# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Cases for decoding one router's part of a snapshot, on a hand-written manifest and snapshot.

Run from the repository root: python3 -m unittest tt_metal.fabric.debug.visualizer.decode.tests.test_routers
"""

import copy
import pathlib
import struct
import unittest

from tt_metal.fabric.debug.visualizer.common.manifest_lookup import ManifestError
from tt_metal.fabric.debug.visualizer.decode.inputs import Input, InputError
from tt_metal.fabric.debug.visualizer.decode.routers import ElementValue, StreamValue, decode_router
from tt_metal.fabric.debug.visualizer.decode.type_parser import get_types_from_manifest

PATH = "M0/C0/E0"


def stream(stream_id, register, schema="u32"):
    return {"kind": "stream", "stream_id": stream_id, "register": register, "schema": schema}


MANIFEST = {
    "run": {"arch": "test_arch"},
    "archs": {
        "test_arch": {
            "areas": {"heartbeat": {"address": 100, "size": 4, "schema": "u32"}, "telemetry": None},
            "types": {
                "Pair": {
                    "size": 8,
                    "members": [
                        {"name": "a", "offset": 0, "size": 4, "schema": "u32"},
                        {"name": "b", "offset": 4, "size": 4, "schema": "u32"},
                    ],
                }
            },
        }
    },
    "enums": {"SyncState": {"values": {"idle": 0, "busy": 1}}},
    "meshes": {
        "M0": {
            "chips": {
                "C0": {
                    "routers": {
                        "E0": {
                            "credit_counters": {
                                "to_sender_completion": {
                                    "address": 200,
                                    "size": 8,
                                    "num_elements": 2,
                                    "size_per_element": 4,
                                    "schema": "u32",
                                }
                            },
                            "channels": {
                                "senders": {
                                    "vc0": {
                                        # Two references to one register, as streams 30 and 31 have.
                                        "ch0": {
                                            "credits": {
                                                "completed": {
                                                    "array": "credit_counters/to_sender_completion",
                                                    "index": 1,
                                                }
                                            },
                                            "fields": {
                                                "free_slots": stream(0, "buf_space_available"),
                                                "conn": {"address": 300, "size": 8, "schema": "struct:Pair"},
                                            },
                                        },
                                        "ch1": {"fields": {"free_slots": stream(0, "buf_space_available")}},
                                    }
                                }
                            },
                            "fields": {"sync": stream(30, "remote_src", "enum:SyncState")},
                        }
                    }
                }
            }
        }
    },
}

RAW = struct.pack("<I", 0xDCBA0001) + struct.pack("<II", 5, 7) + struct.pack("<II", 1, 2)
ENTRY = {
    "areas": {"heartbeat": {"address": 100, "size": 4, "offset": 0}},
    "regions": {
        "credit_counters/to_sender_completion": {"address": 200, "size": 8, "offset": 4},
        "channels/senders/vc0/ch0/fields/conn": {"address": 300, "size": 8, "offset": 12},
    },
    "streams": {
        "0": {"buf_space_available": {"before": 3, "after": 2}},
        "30": {"remote_src": {"before": 1, "after": 0}},
    },
    "error": None,
}


def decode(entry=ENTRY, manifest=MANIFEST, path=PATH):
    snapshot = {"routers": {PATH: entry}}
    loaded = Input(pathlib.Path("snapshot.json"), "0" * 64, snapshot, RAW, pathlib.Path("manifest.json"), manifest, ())
    return decode_router(loaded, get_types_from_manifest(manifest), path)


def changed(entry_change=None, manifest_change=None):
    """Copies of ENTRY and MANIFEST with the changes applied."""
    entry, manifest = copy.deepcopy(ENTRY), copy.deepcopy(MANIFEST)
    if entry_change is not None:
        entry_change(entry)
    if manifest_change is not None:
        manifest_change(manifest["meshes"]["M0"]["chips"]["C0"]["routers"]["E0"])
    return entry, manifest


class DecodeRouterTest(unittest.TestCase):
    def test_areas_and_regions_decode_at_their_offsets(self):
        decoded = decode()
        self.assertEqual(decoded.path, PATH)
        self.assertEqual(decoded.areas, {"heartbeat": 0xDCBA0001})
        self.assertEqual(
            decoded.regions,
            {"credit_counters/to_sender_completion": [5, 7], "channels/senders/vc0/ch0/fields/conn": {"a": 1, "b": 2}},
        )
        self.assertIsNone(decoded.error)

    def test_each_stream_reference_has_its_own_entry(self):
        free_slots = StreamValue(0, "buf_space_available", before=3, after=2)
        self.assertEqual(
            decode().streams,
            {
                "channels/senders/vc0/ch0/fields/free_slots": free_slots,
                "channels/senders/vc0/ch1/fields/free_slots": free_slots,
                "fields/sync": StreamValue(
                    30, "remote_src", before={"raw": 1, "name": "busy"}, after={"raw": 0, "name": "idle"}
                ),
            },
        )

    def test_an_element_reference_is_its_element_of_the_array(self):
        self.assertEqual(
            decode().elements,
            {"channels/senders/vc0/ch0/credits/completed": ElementValue("credit_counters/to_sender_completion", 1, 7)},
        )

    def test_a_router_capture_stopped_part_way_decodes_what_was_read(self):
        error = {"stage": "regions", "path": "channels/senders/vc0/ch0/fields/conn", "message": "timed out"}

        def stopped(entry):
            del entry["regions"]["credit_counters/to_sender_completion"]
            del entry["regions"]["channels/senders/vc0/ch0/fields/conn"]
            entry["streams"] = {"0": {"buf_space_available": {"before": 3}}}
            entry["error"] = error

        entry, manifest = changed(stopped)
        decoded = decode(entry, manifest)
        self.assertEqual(decoded.error, error)
        self.assertEqual(decoded.regions, {})
        self.assertEqual(
            decoded.streams["channels/senders/vc0/ch0/fields/free_slots"],
            StreamValue(0, "buf_space_available", before=3, after=None),
        )
        self.assertEqual(decoded.streams["fields/sync"], StreamValue(30, "remote_src", before=None, after=None))
        self.assertIsNone(decoded.elements["channels/senders/vc0/ch0/credits/completed"].value)

    def test_a_snapshot_that_disagrees_with_the_manifest_is_refused(self):
        def move_conn(router):
            router["channels"]["senders"]["vc0"]["ch0"]["fields"]["conn"]["address"] = 304

        for name, entry_change, manifest_change, named in (
            ("a copied address", None, move_conn, "fields.conn as .*'address': 300"),
            (
                "a region the router doesn't have",
                lambda entry: entry["regions"].__setitem__("leftover_l1", {"address": 0, "size": 4, "offset": 0}),
                None,
                "a region leftover_l1 in M0/C0/E0, which the manifest's router doesn't have",
            ),
            (
                "an area the manifest has as null",
                lambda entry: entry["areas"].__setitem__("telemetry", {"address": 0, "size": 4, "offset": 0}),
                None,
                r"areas\.telemetry as .* but the manifest has None",
            ),
        ):
            with self.subTest(name), self.assertRaisesRegex(InputError, named):
                decode(*changed(entry_change, manifest_change))

    def test_a_router_the_snapshot_doesnt_have_is_refused(self):
        with self.assertRaisesRegex(InputError, "has no router M0/C0/E1"):
            decode(path="M0/C0/E1")

    def test_an_element_reference_the_router_cant_resolve_is_refused(self):
        def credits(router):
            return router["channels"]["senders"]["vc0"]["ch0"]["credits"]["completed"]

        for name, manifest_change, named in (
            ("past its array", lambda router: credits(router).update(index=2), r"index is 2; .* has 2 elements"),
            ("into no region", lambda router: credits(router).update(array="credit_counters/nope"), "names no region"),
        ):
            with self.subTest(name), self.assertRaisesRegex(ManifestError, named):
                decode(*changed(manifest_change=manifest_change))


if __name__ == "__main__":
    unittest.main()
