# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path

from tt_metal.fabric.debug.visualizer.decode.inputs import DecodeError, discover_inputs
from tt_metal.fabric.debug.visualizer.decode.merge import merge_inputs
from tt_metal.fabric.debug.visualizer.decode.tests.fixtures import write_input


class DecodeMergeTest(unittest.TestCase):
    def test_two_ranks_have_one_owner_per_router(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_input(root / "r0", rank=0, local_chip=0, all_chip_ids=(0, 1))
            write_input(root / "r1", rank=1, local_chip=1, all_chip_ids=(0, 1))
            merged = merge_inputs(discover_inputs([root / "r0", root / "r1"]))

            self.assertEqual(merged["coverage"]["routers_total"], 2)
            self.assertEqual(merged["coverage"]["captured"], 2)
            self.assertEqual(merged["coverage"]["ok"], 2)
            self.assertEqual(
                [(router["id"]["chip_id"], router["capture"]["snapshot_index"]) for router in merged["routers"]],
                [(0, 0), (1, 1)],
            )

    def test_manifest_without_snapshot_becomes_coverage_hole(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_input(root / "r0", rank=0, local_chip=0, all_chip_ids=(0, 1))
            _, snapshot, raw = write_input(
                root / "r1", rank=1, local_chip=1, all_chip_ids=(0, 1)
            )
            snapshot.unlink()
            raw.unlink()
            merged = merge_inputs(discover_inputs([root / "r0", root / "r1"]))

            self.assertEqual(merged["coverage"]["captured"], 1)
            self.assertEqual(merged["coverage"]["not_captured"], 1)
            self.assertEqual(merged["routers"][1]["capture"]["status"], "not_captured")

    def test_two_local_snapshots_for_same_router_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_input(root / "a", rank=0, local_chip=0)
            write_input(root / "b", rank=1, local_chip=0)
            with self.assertRaisesRegex(DecodeError, "owned by more than one snapshot"):
                merge_inputs(discover_inputs([root / "a", root / "b"]))


if __name__ == "__main__":
    unittest.main()
