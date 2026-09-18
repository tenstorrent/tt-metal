# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path

import jsonschema

from tt_metal.fabric.debug.visualizer.decode.inputs import discover_inputs
from tt_metal.fabric.debug.visualizer.decode.output import build_decoded
from tt_metal.fabric.debug.visualizer.decode.tests.fixtures import write_input

SCHEMA_PATH = Path(__file__).parents[2] / "schema" / "fabric_debug_decoded_schema.json"


class DecodedSchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(cls.schema)
        cls.validator = jsonschema.Draft202012Validator(
            cls.schema,
            format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER,
        )

    def test_slice_a_output_validates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_input(root)
            decoded = build_decoded(
                discover_inputs([root]),
                generated_at="2026-09-15T00:00:01Z",
            )

            self.validator.validate(decoded)
            self.assertEqual(decoded["fabric_context"]["packet_header_type"], "HybridMeshPacketHeaderT<36>")
            self.assertEqual(decoded["coverage"]["ok"], 1)

    def test_missing_rank_output_validates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_input(root / "r0", rank=0, local_chip=0, all_chip_ids=(0, 1))
            _, snapshot, raw = write_input(
                root / "r1", rank=1, local_chip=1, all_chip_ids=(0, 1)
            )
            snapshot.unlink()
            raw.unlink()
            decoded = build_decoded(discover_inputs([root / "r0", root / "r1"]))

            self.validator.validate(decoded)
            self.assertIsNone(decoded["inputs"][1]["snapshot"])

    def test_raw_files_union_two_ranks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_input(root / "r0", rank=0, local_chip=0, all_chip_ids=(0, 1))
            write_input(root / "r1", rank=1, local_chip=1, all_chip_ids=(0, 1))
            decoded = build_decoded(discover_inputs([root / "r0", root / "r1"]))

            self.validator.validate(decoded)
            table = decoded["raw_files"]
            self.assertEqual(len(table), 2)
            for item, entry in zip(decoded["inputs"], table):
                raw = item["raw"]
                self.assertEqual(entry["file"], raw["file"])
                self.assertEqual(entry["size"], raw["size"])
                self.assertEqual(entry["sha256"], raw["sha256"])

    def test_raw_files_skips_rank_without_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_input(root / "r0", rank=0, local_chip=0, all_chip_ids=(0, 1))
            _, snapshot, raw = write_input(
                root / "r1", rank=1, local_chip=1, all_chip_ids=(0, 1)
            )
            snapshot.unlink()
            raw.unlink()
            decoded = build_decoded(discover_inputs([root / "r0", root / "r1"]))

            self.validator.validate(decoded)
            self.assertEqual(len(decoded["raw_files"]), 1)


if __name__ == "__main__":
    unittest.main()
