# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path

import jsonschema

from tt_metal.fabric.debug.visualizer.capture.snapshot import owner_alive


SCHEMA_PATH = (
    Path(__file__).parents[2] / "schema" / "fabric_debug_snapshot_schema.json"
)


def snapshot() -> dict:
    return {
        "snapshot_version": 1,
        "kind": "fabric_debug_snapshot",
        "captured_at": "2026-09-14T22:00:00Z",
        "manifest": {
            "path": "generated/fabric/fabric_debug_manifest_rank_1_of_2.json",
            "manifest_version": 1,
            "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "run": {
                "arch": "WORMHOLE_B0",
                "fabric_config": "FABRIC_2D",
                "host_rank": 0,
                "mpi_rank": 1,
                "world_size": 2,
            },
        },
        "provenance": {
            "ttexalens_version": "0.3.32",
            "tt_umd_version": "0.9.9",
            "hostname": "test-host",
            "owner_alive": None,
            "argv": ["capture", "--manifest", "manifest.json"],
        },
        "raw": {
            "file": "fabric_debug_snapshot_rank_1_of_2.bin",
            "size": 1024,
            "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        },
        "samples": [
            {
                "capture_time": "2026-09-14T22:00:00Z",
                "routers": [
                    {
                        "id": {"mesh_id": 0, "chip_id": 1, "eth_chan": 8},
                        "physical_chip_id": 0,
                        "asic_id": "0x0000000000001234",
                        "asic_id_matches_physical": True,
                        "status": "ok",
                        "error": None,
                        "health": {
                            "reset": 1,
                            "reset_bits": {"erisc0": False, "erisc1": None},
                            "wall_clock": 123456789,
                        },
                        "lifecycle": {
                            "edm_status": 0xA3B3C3D3,
                            "termination_signal": 0,
                            "go_signal": 0x80,
                        },
                        "liveness": [
                            {
                                "t": "2026-09-14T22:00:00Z",
                                "heartbeat": 0xDCBA0001,
                                "wall_clock": 123456000,
                            }
                        ],
                        "streams": {
                            "pre": {
                                "0": {"buf_space_available": 0},
                                "22": {"buf_space_available": 8},
                            },
                            "post": {
                                "0": {"buf_space_available": 0},
                                "22": {"buf_space_available": 8},
                            },
                            "torn": False,
                        },
                        "blobs": {
                            "unreserved": {
                                "address": 98304,
                                "size": 1024,
                                "offset": 0,
                                "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                                "status": "ok",
                                "error": None,
                            }
                        },
                        "identity": {
                            "my_mesh_id": 0,
                            "my_device_id": 1,
                            "matches_manifest": True,
                        },
                    }
                ],
            }
        ],
    }


class SnapshotSchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(cls.schema)
        cls.validator = jsonschema.Draft202012Validator(
            cls.schema,
            format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER,
        )

    def test_valid_snapshot(self):
        self.validator.validate(snapshot())

    def test_sample_requires_descriptive_capture_time(self):
        data = snapshot()
        sample = data["samples"][0]
        sample["t"] = sample.pop("capture_time")

        errors = list(self.validator.iter_errors(data))
        self.assertTrue(any("'capture_time' is a required property" in error.message for error in errors))

    def test_sha256_is_required(self):
        data = snapshot()
        del data["manifest"]["sha256"]
        errors = list(self.validator.iter_errors(data))
        self.assertTrue(any("'sha256' is a required property" in error.message for error in errors))

    def test_mpi_rank_is_required(self):
        data = snapshot()
        del data["manifest"]["run"]["mpi_rank"]
        errors = list(self.validator.iter_errors(data))
        self.assertTrue(any("'mpi_rank' is a required property" in error.message for error in errors))

    def test_owner_alive_detects_another_process_device_fd(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            proc_root = Path(temp_dir)
            fd_dir = proc_root / "999999" / "fd"
            fd_dir.mkdir(parents=True)
            (fd_dir / "3").symlink_to("/dev/tenstorrent/0")
            self.assertTrue(owner_alive(proc_root))

    def test_owner_alive_is_false_when_no_process_has_device_open(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertFalse(owner_alive(Path(temp_dir)))


if __name__ == "__main__":
    unittest.main()
