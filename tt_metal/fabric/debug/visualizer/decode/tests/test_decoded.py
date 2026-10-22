# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Cases for the decoded file and decode's command line, from files written to a temporary directory.

Run from the repository root: python3 -m unittest tt_metal.fabric.debug.visualizer.decode.tests.test_decoded
"""

import contextlib
import datetime
import hashlib
import io
import json
import pathlib
import struct
import tempfile
import unittest

from tt_metal.fabric.debug.visualizer.decode.__main__ import main
from tt_metal.fabric.debug.visualizer.decode.decoded import decode_snapshot, get_decoded_name, write_decoded
from tt_metal.fabric.debug.visualizer.decode.inputs import Checkout, load_input
from tt_metal.fabric.debug.visualizer.decode.tests.test_ring_buffer_slots import COMMAND_TYPES, HEADER_2D, SLOT_2D
from tt_metal.fabric.debug.visualizer.decode.tests.test_ring_buffer_slots import MANIFEST_2D as SLOTS_MANIFEST
from tt_metal.fabric.debug.visualizer.decode.tests.test_ring_buffer_slots import header_2d
from tt_metal.fabric.debug.visualizer.decode.tests.test_routes import MANIFEST as ENCODINGS
from tt_metal.fabric.debug.visualizer.decode.tests.test_routes import D, E, tree_edge
from tt_metal.fabric.debug.visualizer.decode.tests.test_routing_tables import MANIFEST_2D as TABLES_MANIFEST

try:
    import jsonschema
except ImportError:  # the lab hosts have it, and check_decode needs it
    jsonschema = None

SHA = "a" * 40
DECODED_AT = datetime.datetime(2026, 10, 21, 12, 0, 0, tzinfo=datetime.timezone.utc)
SNAPSHOT_NAME = "fabric_snapshot_rank_1_of_1_20261021T120000Z.json"
SCHEMA_PATH = pathlib.Path(__file__).resolve().parents[1] / "fabric_decoded_schema.json"
SENDER, RECEIVER = "channels/senders/vc0/ch0", "channels/receivers/vc0/ch0"


def stream(stream_id):
    return {"stream_id": stream_id, "register": "buf_space_available", "schema": "u32"}


def ring(address):
    return {"address": address, "size": 2 * SLOT_2D, "num_elements": 2, "size_per_element": SLOT_2D, "schema": "bytes"}


# One router, C0's east router, in a 1x2 mesh whose other chip is on another host. Its sender ch0 is fed by nothing and
# holds one packet; its receiver ch0 is idle.
ROUTER = {
    "eriscs": {"erisc0": {"processor": "riscv_0", "physical_erisc": 0}},
    "link": {"peer": "M0/C1/W0"},
    "shape": {"vc0_bubble_flow_control": False},
    "intra_chip_downstream_edges": {},
    "fields": {
        "edm_status": {"address": 200, "size": 4, "schema": "u32"},
        "my_direction": {"kind": "enum", "value": 0, "schema": "enum:eth_chan_directions"},
        "speedy_vc0": {"kind": "flag", "value": False},
        "sender_credit_amortization": {"kind": "number", "value": 0},
    },
    "channels": {
        "senders": {
            "vc0": {
                "ch0": {
                    "status": "active",
                    "producer": None,
                    "ring_buffer": ring(400),
                    "credits": {"completed": stream(1)},
                    "fields": {"free_slots": stream(0)},
                }
            }
        },
        "receivers": {
            "vc0": {
                "ch0": {
                    "status": "active",
                    "ring_buffer": ring(500),
                    "fields": {"intermesh_ingress": {"kind": "flag", "value": False}, "pkts_sent": stream(2)},
                }
            }
        },
    },
}
MANIFEST = {
    "kind": "fabric_manifest",
    "run": {"arch": "test_arch", "git_sha": SHA, "git_dirty": False},
    "fabric_context": {
        "is_2d_routing": True,
        "packet_header": {"size": HEADER_2D["size"], "schema": "struct:HybridMeshPacketHeader"},
    },
    "routing_encodings": ENCODINGS["routing_encodings"],
    "vocabulary": {"z_port_roles": ["none", "intermesh_boundary", "express_chord"]},
    "archs": {
        "test_arch": {
            "areas": {
                "heartbeat": {
                    "address": 100,
                    "size": 4,
                    "schema": "u32",
                    "magic": 0xDCBA0000,
                    "magic_mask": 0xFFFF0000,
                },
                "routing_table": {"address": 300, "size": 24, "schema": "struct:routing_l1_info_t"},
            },
            "core_registers": [
                {"name": "soft_reset", "address": 0xFFB121B0, "size": 4, "schema": "u32", "reset_masks": [2048]},
                {"name": "wall_clock_low", "address": 0xFFB121F0, "size": 4, "schema": "u32"},
                {"name": "wall_clock_high", "address": 0xFFB121F8, "size": 4, "schema": "u32"},
            ],
            "stream_registers": {
                "base": 0xFFB40000,
                "stride": 0x1000,
                "count": 32,
                "registers": {
                    "buf_space_available": {"offset": 0x100, "bits": 17},
                    "remote_src": {"offset": 0, "bits": 24},
                },
            },
            "types": {
                **COMMAND_TYPES,
                "HybridMeshPacketHeader": HEADER_2D,
                **TABLES_MANIFEST["archs"]["test_arch"]["types"],
            },
        }
    },
    "enums": {
        **SLOTS_MANIFEST["enums"],
        **TABLES_MANIFEST["enums"],
        "ConnectionValue": {
            "values": {"unused_connection_value": 0, "open_connection_value": 1, "close_connection_request_value": 2}
        },
    },
    "meshes": {
        "M0": {
            "shape": [1, 2],
            "torus": {"y": False, "x": False},
            "credit_transport": {"vc0": {"backing": "stream_register", "reasons": []}},
            "chips": {
                "C0": {"mesh_coord": [0, 0], "is_local": True, "z_port_role": "none", "routers": {"E0": ROUTER}},
                "C1": {"mesh_coord": [0, 1], "is_local": False},
            },
        }
    },
}
# The 1x2 mesh's route table toward each column: toward column 0, column 1 goes west; toward column 1, column 0 goes
# east. Its X tree hangs column 1 east of column 0.
ACTION_VECTORS = bytes([0, 0b10_00, 0b00_01, 0, 0, 0, 0, 0])
ROUTING_TABLE = struct.pack("<II", 1, 0) + ACTION_VECTORS + tree_edge(1, 0, 1) + bytes(6)
# The sender's slot 0 holds a packet for C1, as C0's worker built its map: the row stays, column 0 goes east.
SENDER_RING = header_2d([0, E, D]).ljust(SLOT_2D, b"\0") + bytes(SLOT_2D)
RAW = struct.pack("<II", 0xDCBA0005, 7) + ROUTING_TABLE + SENDER_RING + bytes(2 * SLOT_2D)
REGISTERS = {"soft_reset": 0, "wall_clock_low": 1000, "wall_clock_high": 0}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_files(directory: pathlib.Path) -> tuple:
    """A manifest in generated/, and a snapshot of its one router and the snapshot's .bin in snapshots/."""
    manifest_path = directory / "generated" / "fabric_manifest_rank_1_of_1.json"
    snapshot_path = directory / "snapshots" / SNAPSHOT_NAME
    for path in (manifest_path, snapshot_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    manifest_bytes = json.dumps(MANIFEST).encode()
    manifest_path.write_bytes(manifest_bytes)
    snapshot_path.with_suffix(".bin").write_bytes(RAW)
    snapshot = {
        "kind": "fabric_snapshot",
        "snapshot_version": 1,
        "manifest": {"file": str(manifest_path), "sha256": sha256(manifest_bytes), "run": MANIFEST["run"]},
        "capture": {"host": "test"},
        "raw": {"file": snapshot_path.with_suffix(".bin").name, "size": len(RAW), "sha256": sha256(RAW)},
        "routers": {
            "M0/C0/E0": {
                "liveness": [{"t_ns": 1000, "heartbeat": 0xDCBA0001, "core_registers": REGISTERS}],
                "started_ns": 2000,
                "core_registers": {**REGISTERS, "wall_clock_low": 2000},
                "areas": {
                    "heartbeat": {"address": 100, "size": 4, "offset": 0},
                    "routing_table": {"address": 300, "size": 24, "offset": 8},
                },
                "regions": {
                    "fields/edm_status": {"address": 200, "size": 4, "offset": 4},
                    f"{SENDER}/ring_buffer": {"address": 400, "size": 2 * SLOT_2D, "offset": 32},
                    f"{RECEIVER}/ring_buffer": {"address": 500, "size": 2 * SLOT_2D, "offset": 32 + 2 * SLOT_2D},
                },
                # The sender has one free slot of two, and nothing has come back or arrived.
                "streams": {
                    str(stream_id): {"buf_space_available": {"before": free, "after": free}}
                    for stream_id, free in ((0, 1), (1, 0), (2, 0))
                },
                "error": None,
            }
        },
    }
    snapshot_path.write_text(json.dumps(snapshot))
    return manifest_path, snapshot_path


class TemporaryDirectoryTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = pathlib.Path(temporary.name)
        self.manifest_path, self.snapshot_path = write_files(self.directory)


class DecodeSnapshotTest(TemporaryDirectoryTest):
    def test_the_file_lists_its_source_and_decodes_each_router(self):
        snapshot = load_input(self.snapshot_path, Checkout(SHA, False))
        decoded = json.loads(json.dumps(decode_snapshot(snapshot, Checkout(SHA, False), DECODED_AT)))
        self.assertEqual(
            {key: decoded[key] for key in ("kind", "decoded_version", "decoded_at", "decoder")},
            {
                "kind": "fabric_decoded",
                "decoded_version": 1,
                "decoded_at": "2026-10-21T12:00:00Z",
                "decoder": {"git_sha": SHA, "git_dirty": False},
            },
        )
        self.assertEqual(
            decoded["sources"],
            [
                {
                    "snapshot": {"file": str(self.snapshot_path.resolve()), "sha256": snapshot.json_sha256},
                    "manifest": {
                        "file": str(self.manifest_path.resolve()),
                        "sha256": snapshot.json["manifest"]["sha256"],
                        "run": MANIFEST["run"],
                    },
                    "capture": {"host": "test"},
                    "warnings": [],
                }
            ],
        )
        router = decoded["routers"]["M0/C0/E0"]
        self.assertEqual(
            {key: router[key] for key in ("source", "error", "areas", "elements")},
            {"source": 0, "error": None, "areas": {"heartbeat": 0xDCBA0005}, "elements": {}},
        )
        self.assertEqual(router["regions"]["fields/edm_status"], 7)
        self.assertEqual(
            router["streams"][f"{SENDER}/fields/free_slots"],
            {
                "stream_id": 0,
                "register": "buf_space_available",
                "bits": 17,
                "before": 1,
                "after": 1,
                "before_word": 1,
                "after_word": 1,
            },
        )
        self.assertEqual(router["health"]["t_ns"], [1000, 2000])
        self.assertEqual(router["health"]["heartbeat"]["words"], [0xDCBA0001, 0xDCBA0005])
        self.assertEqual(router["health"]["wall_clock"]["mhz"], 1000.0)

    def decoded(self):
        snapshot = load_input(self.snapshot_path, Checkout(SHA, False))
        return json.loads(json.dumps(decode_snapshot(snapshot, Checkout(SHA, False), DECODED_AT)))

    def test_the_routing_table_is_stored_once_and_each_router_names_its_copy(self):
        decoded = self.decoded()
        table = decoded["routing_tables"]["M0/C0/E0"]
        self.assertEqual(
            (table["chip"], table["routers"], table["raw"]),
            ("M0/C0", ["M0/C0/E0"], {"bytes": {"offset": 8, "size": 24}}),
        )
        self.assertEqual(
            table["route_table_2d"]["M0/C1"],
            {
                "delivers": ["M0/C1"],
                "steps": [{"src": "M0/C0", "direction": "east", "dst": "M0/C1"}],
                "exits": [],
                "odd": [],
            },
        )
        self.assertEqual(table["mcast_trees"], {"y": [], "x": [{"child": 1, "parent": 0, "direction": "east"}]})
        self.assertIsNone(table["path_table_1d"])
        router = decoded["routers"]["M0/C0/E0"]
        self.assertNotIn("routing_table", router["areas"])
        self.assertEqual(
            router["routing"],
            {
                "table_key": "M0/C0/E0",
                "state_manager": {"state": {"raw": 1, "name": "running"}, "command": {"raw": 0, "name": "run"}},
                "raw": {"bytes": {"offset": 8, "size": 24}},
            },
        )

    def test_each_active_channel_has_its_counts_and_slots(self):
        channels = self.decoded()["routers"]["M0/C0/E0"]["channels"]
        self.assertEqual(list(channels), [SENDER, RECEIVER])
        sender, receiver = channels[SENDER], channels[RECEIVER]
        self.assertEqual((sender["kind"], sender["depth"], sender["producer"]), ("sender", 2, None))
        self.assertEqual(sender["unsent"], {"before": 1, "after": 1, "not_observable": None})
        self.assertEqual(sender["uncredited"]["not_observable"], "nothing feeds the channel")
        # C0's east router sends to C1, which reads the map entered from the west and delivers.
        packet, empty = sender["slots"]
        self.assertEqual(
            packet["route"],
            {
                "start": "M0/C1",
                "entered_from": "west",
                "walk": {"delivers": ["M0/C1"], "steps": [], "exits": [], "odd": []},
                "landing": False,
                "rebuilt": None,
            },
        )
        self.assertEqual((packet["empty"], packet["problems"]), (False, []))
        self.assertEqual(empty, {"empty": True, "header": None, "route": None, "scatter": None, "problems": []})
        self.assertEqual(receiver["kind"], "receiver")
        self.assertEqual(receiver["waiting"], {"before": 0, "after": 0, "not_observable": None})
        self.assertEqual([slot["empty"] for slot in receiver["slots"]], [True, True])

    @unittest.skipIf(jsonschema is None, "needs jsonschema")
    def test_the_file_matches_the_schema(self):
        validator = jsonschema.Draft202012Validator(json.loads(SCHEMA_PATH.read_text()))
        self.assertEqual([error.message for error in validator.iter_errors(self.decoded())], [])

    def test_without_a_checkout_there_is_no_decoder(self):
        decoded = decode_snapshot(load_input(self.snapshot_path, None), None, DECODED_AT)
        self.assertIsNone(decoded["decoder"])
        self.assertRegex(decoded["sources"][0]["warnings"][0], "decode isn't in a git checkout")


class FilesTest(TemporaryDirectoryTest):
    def test_a_decoded_file_is_named_after_its_snapshot(self):
        self.assertEqual(get_decoded_name(self.snapshot_path), "fabric_decoded_rank_1_of_1_20261021T120000Z.json")
        self.assertEqual(get_decoded_name(pathlib.Path("mine.json")), "fabric_decoded_mine.json")

    def test_writing_replaces_the_file_and_leaves_no_temporary(self):
        path = self.directory / "decoded.json"
        write_decoded({"first": 1}, path)
        write_decoded({"second": 2}, path)
        self.assertEqual(json.loads(path.read_text()), {"second": 2})
        self.assertEqual([child.name for child in self.directory.iterdir() if child.suffix == ".tmp"], [])


class MainTest(TemporaryDirectoryTest):
    def run_main(self, *args) -> tuple:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main([str(arg) for arg in args])
        return code, output.getvalue()

    def test_a_directory_of_snapshots_decodes_beside_them(self):
        code, output = self.run_main(self.snapshot_path.parent)
        decoded_path = self.snapshot_path.parent / "fabric_decoded_rank_1_of_1_20261021T120000Z.json"
        self.assertEqual(code, 0, output)
        self.assertIn(
            f"{decoded_path}: 1 routers, 0 with capture errors; 1 used ring buffer slots; 1 packets unsent in senders, "
            "0 waiting in receivers",
            output,
        )
        self.assertEqual(json.loads(decoded_path.read_text())["kind"], "fabric_decoded")

    def test_out_dir_takes_the_decoded_files(self):
        out_dir = self.directory / "out"
        code, output = self.run_main(self.snapshot_path, "-o", out_dir)
        self.assertEqual(code, 0, output)
        self.assertTrue((out_dir / "fabric_decoded_rank_1_of_1_20261021T120000Z.json").is_file())

    def test_a_moved_manifest_is_found_with_manifest(self):
        moved = self.directory / "elsewhere"
        moved.mkdir()
        self.manifest_path.rename(moved / self.manifest_path.name)
        code, output = self.run_main(self.snapshot_path, "--manifest", moved)
        self.assertEqual(code, 0, output)

    def test_a_refused_snapshot_fails_and_writes_nothing(self):
        self.manifest_path.write_text(json.dumps({**MANIFEST, "run": {**MANIFEST["run"], "arch": "other"}}))
        code, output = self.run_main(self.snapshot_path)
        self.assertEqual(code, 1)
        self.assertIn(f"Fabric decode: {self.snapshot_path} not decoded:", output)
        self.assertEqual(list(self.snapshot_path.parent.glob("fabric_decoded_*")), [])

    def test_a_directory_with_no_snapshots_fails(self):
        code, output = self.run_main(self.manifest_path.parent)
        self.assertEqual(code, 1)
        self.assertIn("no fabric_snapshot_*.json", output)


if __name__ == "__main__":
    unittest.main()
