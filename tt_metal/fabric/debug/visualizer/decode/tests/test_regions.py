# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path

from tt_metal.fabric.debug.visualizer.decode.inputs import discover_inputs
from tt_metal.fabric.debug.visualizer.decode.output import build_decoded
from tt_metal.fabric.debug.visualizer.decode.tests.fixtures import write_input


def by_id(decoded):
    return {region["id"]: region for region in decoded["routers"][0]["regions"]}


class RegionDecoderTest(unittest.TestCase):
    def test_slices_unreserved_fixed_and_stream_regions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_input(root)
            decoded = build_decoded(discover_inputs([root]), expert_raw=True)
            regions = by_id(decoded)

            self.assertEqual(regions["lifecycle.handshake"]["value"]["neighbor_mesh_id"], 7)
            self.assertEqual(regions["lifecycle.edm_status"]["value"]["name"], "READY_FOR_TRAFFIC")
            self.assertEqual(regions["sender.0.control.conn_info"]["value"]["worker_xy"], {"x": 3, "y": 4})
            self.assertEqual(regions["sender.0.control.cursor"]["value"]["write_index"], 1)
            self.assertEqual(regions["credits.counters"]["value"]["counters"], [10, 20, 30, 40])
            self.assertEqual(regions["credits.sender.0.free_slots"]["value"]["post"], 2)
            self.assertEqual(regions["hal.telemetry"]["value"]["postcode"], 0xCAFE)
            self.assertEqual(regions["hal.routing_table"]["value"]["my_device_id"], 0)
            self.assertEqual(regions["future.schema"]["status"], "unsupported")
            self.assertIsNone(regions["lifecycle"]["raw_ref"])
            self.assertIn("raw_hex", regions["lifecycle.handshake"])

    def test_blob_failure_propagates_and_missing_raw_is_not_captured(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _, snapshot, raw = write_input(root)
            data = json.loads(snapshot.read_text())
            data["samples"][0]["routers"][0]["blobs"]["fabric_telemetry"].update(
                status="unreadable", error="read failed"
            )
            snapshot.write_text(json.dumps(data))
            decoded = build_decoded(discover_inputs([root]))
            telemetry = by_id(decoded)["hal.telemetry"]
            self.assertEqual(telemetry["status"], "unreadable")
            self.assertEqual(telemetry["error"], "read failed")

            raw.unlink()
            decoded = build_decoded(discover_inputs([root]))
            self.assertEqual(by_id(decoded)["lifecycle.handshake"]["status"], "not_captured")
            self.assertEqual(
                by_id(decoded)["credits.sender.0.free_slots"]["status"],
                "ok",
            )

    def test_per_stream_torn_and_reset_precedence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _, snapshot, _ = write_input(root, status="reset")
            data = json.loads(snapshot.read_text())
            data["samples"][0]["routers"][0]["streams"]["post"]["22"]["buf_space_available"] = 3
            snapshot.write_text(json.dumps(data))
            decoded = build_decoded(discover_inputs([root]))
            regions = by_id(decoded)
            self.assertEqual(regions["credits.sender.0.free_slots"]["status"], "torn")
            self.assertEqual(regions["lifecycle.handshake"]["status"], "reset")


if __name__ == "__main__":
    unittest.main()
