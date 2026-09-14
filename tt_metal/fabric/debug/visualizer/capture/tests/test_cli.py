# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import jsonschema

from tt_metal.fabric.debug.visualizer.capture.cli import capture, parse_args
from tt_metal.fabric.debug.visualizer.capture.tests.test_peek import FakeDevice, FakeLoc, fixture_manifest

SCHEMA_PATH = Path(__file__).parents[2] / "schema" / "fabric_debug_snapshot_schema.json"


class CliTest(unittest.TestCase):
    def test_manifest_and_output_are_required(self):
        with self.assertRaises(SystemExit):
            parse_args(["-o", "snapshot.json"])
        with self.assertRaises(SystemExit):
            parse_args(["--manifest", "manifest.json"])

    def test_capture_writes_schema_valid_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = temp_path / "manifest.json"
            output_path = temp_path / "snapshot.json"
            manifest_path.write_text(json.dumps(fixture_manifest()), encoding="utf-8")

            device = FakeDevice(4, [FakeLoc((0, 0)), FakeLoc((0, 1))])
            context = SimpleNamespace(devices={4: device})

            snapshot = capture(
                manifest_path,
                output_path,
                init_ttexalens=lambda: context,
                read_from_device=lambda location, address, device_id, size, live_context: 0,
            )

            self.assertTrue(output_path.is_file())
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), snapshot)
            self.assertEqual(len(snapshot["samples"][0]["routers"]), 1)

            schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
            jsonschema.Draft202012Validator(
                schema,
                format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER,
            ).validate(snapshot)


if __name__ == "__main__":
    unittest.main()
