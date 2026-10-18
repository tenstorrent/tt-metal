# SPDX-License-Identifier: Apache-2.0
"""CPU source-contract checks for the isolated register-resident experiment."""

import ast
import inspect
import textwrap
import unittest

from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_gdn_resident_candidate as candidate


class ResidentSourceTests(unittest.TestCase):
    def test_launcher_preserves_precision_and_shape_except_explicit_changes(self):
        source = candidate.LAUNCHER_SOURCE
        for new, old in (
            ("def resident_recurrence(", "def _rm_recurrence("),
            ("jobs = batch * 96", "jobs = batch * 48"),
            ("reader = READER", "reader = _RM_QK_READER"),
            ("kernel_source=COMPUTE,", "kernel_source=_BROADCAST_DECODE_COMPUTE,"),
            ("*job, length, self._half_column_poison])", "*job, length])"),
            ("dst_full_sync_en=True", "dst_full_sync_en=False"),
            ("[job[1], length]", "[job[1] * length]"),
        ):
            self.assertEqual(source.count(new), 1)
            source = source.replace(new, old)
        baseline = textwrap.dedent(inspect.getsource(candidate.baseline._rm_recurrence))
        self.assertEqual(ast.dump(ast.parse(source)), ast.dump(ast.parse(baseline)))

    def test_reader_keeps_validated_face_mapping_and_io(self):
        source = candidate.READER
        for new, old in (
            ("if(token==0) {\n        cb_reserve_back(0,4);", "cb_reserve_back(0,4);"),
            ("cb_push_back(0,4);\n        }", "cb_push_back(0,4);"),
            ("if(token+1==length) cb_wait_front(12,4);", "cb_wait_front(12,4);"),
            ("if(token+1==length) cb_pop_front(12,4);", "cb_pop_front(12,4);"),
        ):
            self.assertEqual(source.count(new), 1)
            source = source.replace(new, old)
        self.assertEqual(source, candidate.qualified.READER)

    def test_production_namespaces_unchanged(self):
        self.assertIsNot(candidate.RECURRENCE.__globals__, candidate.baseline._rm_recurrence.__globals__)
        self.assertIs(candidate.baseline._rm_recurrence.__globals__["_RM_QK_READER"], candidate.baseline._RM_QK_READER)
        self.assertEqual(candidate.qualified.RECURRENCE.__globals__["COMPUTE"], candidate.qualified.COMPUTE)

    def test_resident_protocol_requires_full_sync_and_no_clear(self):
        source = candidate.BODY_PATH.read_text()
        self.assertIn("static_assert(DST_SYNC_MODE == DstSync::SyncFull)", source)
        self.assertIn("static_assert(compute_kernel_lib::DEST_AUTO_LIMIT == 8)", source)
        self.assertIn("_llk_pack_dest_section_done_<DstSync::SyncFull, true, false>()", source)
        self.assertNotIn("tile_regs_release();", source)
        self.assertNotIn("half_reduce_product<", source)
        self.assertNotIn("sfpu_mac", source)
        self.assertNotIn("SFPMAD", source)
        self.assertEqual(source.count("pack_tile("), 2)
        self.assertIn("half_add(6, i, i)", source)  # product + previous state, never swapped/FMA
        self.assertIn("half_sub(5, 4, 4)", source)  # V - prediction
        self.assertIn("half_mul(4, 6, 4)", source)  # delta * beta
        self.assertEqual(source.count("half_add(4, 5, 4)"), 2)  # same K-fold order twice


if __name__ == "__main__":
    unittest.main()
