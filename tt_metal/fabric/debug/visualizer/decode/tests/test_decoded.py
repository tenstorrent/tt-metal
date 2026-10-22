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

SHA = "a" * 40
DECODED_AT = datetime.datetime(2026, 10, 21, 12, 0, 0, tzinfo=datetime.timezone.utc)
SNAPSHOT_NAME = "fabric_snapshot_rank_1_of_1_20261021T120000Z.json"
MANIFEST = {
    "kind": "fabric_manifest",
    "run": {"arch": "test_arch", "git_sha": SHA, "git_dirty": False},
    "archs": {
        "test_arch": {
            "areas": {
                "heartbeat": {"address": 100, "size": 4, "schema": "u32", "magic": 0xDCBA0000, "magic_mask": 0xFFFF0000}
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
            "types": {},
        }
    },
    "enums": {},
    "meshes": {
        "M0": {
            "chips": {
                "C0": {
                    "routers": {
                        "E0": {
                            "eriscs": {"erisc0": {"processor": "riscv_0", "physical_erisc": 0}},
                            "fields": {
                                "edm_status": {"address": 200, "size": 4, "schema": "u32"},
                                "free_slots": {"stream_id": 0, "register": "buf_space_available", "schema": "u32"},
                            },
                        }
                    }
                }
            }
        }
    },
}
RAW = struct.pack("<II", 0xDCBA0005, 7)
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
                "areas": {"heartbeat": {"address": 100, "size": 4, "offset": 0}},
                "regions": {"fields/edm_status": {"address": 200, "size": 4, "offset": 4}},
                "streams": {"0": {"buf_space_available": {"before": 3, "after": 2}}},
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
            {key: value for key, value in router.items() if key != "health"},
            {
                "source": 0,
                "error": None,
                "areas": {"heartbeat": 0xDCBA0005},
                "regions": {"fields/edm_status": 7},
                "streams": {
                    "fields/free_slots": {
                        "stream_id": 0,
                        "register": "buf_space_available",
                        "bits": 17,
                        "before": 3,
                        "after": 2,
                        "before_word": 3,
                        "after_word": 2,
                    }
                },
                "elements": {},
            },
        )
        self.assertEqual(router["health"]["t_ns"], [1000, 2000])
        self.assertEqual(router["health"]["heartbeat"]["words"], [0xDCBA0001, 0xDCBA0005])
        self.assertEqual(router["health"]["wall_clock"]["mhz"], 1000.0)

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
        self.assertIn(f"{decoded_path}: 1 routers, 0 with capture errors", output)
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
