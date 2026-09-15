# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import unittest
from pathlib import Path

import jsonschema


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
        "samples": [
            {
                "capture_time": "2026-09-14T22:00:00Z",
                "routers": [
                    {
                        "id": {"mesh_id": 0, "chip_id": 1, "eth_chan": 8},
                        "physical_chip_id": 0,
                        "ok": True,
                        "error": None,
                        "health": {"reset": 1, "wall_clock": 123456789},
                        "streams": {
                            "0": {"buf_space_available": 0},
                            "22": {"buf_space_available": 8},
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


if __name__ == "__main__":
    unittest.main()
