# SPDX-License-Identifier: Apache-2.0
"""CPU-only guards for the isolated delta/broadcast pruning experiment."""
import re
import unittest

from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_gdn_resident_tail_candidate as candidate


def tokens(text):
    return re.sub(r"\s+", "", re.sub(r"//[^\n]*", "", text))


def function(text, name):
    start = text.index("inline void " + name)
    opening = text.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


class ResidentTailTests(unittest.TestCase):
    def test_original_launcher_reader_and_namespace_are_unchanged(self):
        self.assertEqual(candidate.LAUNCHER_SOURCE, candidate.resident.LAUNCHER_SOURCE)
        self.assertEqual(candidate.READER, candidate.resident.READER)
        self.assertIsNot(candidate.RECURRENCE.__globals__, candidate.resident.RECURRENCE.__globals__)
        self.assertEqual(candidate.resident.RECURRENCE.__globals__["COMPUTE"], candidate.resident.COMPUTE)

    def test_only_three_consumer_substitutions(self):
        text = candidate.COMPUTE.replace(candidate.BODY_PATH.read_text() + "\n", "", 1)
        for new, old in (
            ("resident_row_delta_sub(5, 4, 4)", "half_sub(5, 4, 4)"),
            ("resident_row_delta_mul(4, 6, 4)", "half_mul(4, 6, 4)"),
            ("resident_left_mul_bcast_row(6, 4)", "sfpu_mul_bcast_row(6, 4)"),
        ):
            self.assertEqual(text.count(new), 1)
            text = text.replace(new, old)
        self.assertEqual(text, candidate.resident.COMPUTE)

    def test_native_broadcast_only_omits_unused_data_memory_operations(self):
        native = (
            candidate.REPO / "tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_binary_bcast.h"
        ).read_text()
        actual = candidate.BODY_PATH.read_text()
        for old, new in (
            ("_process_row_bcast_data_band_", "_resident_left_row_band_"),
            ("_calculate_sfpu_binary_bcast_row_full_tile_", "_resident_left_row_tile_"),
        ):
            expected = (
                function(native, old)
                .replace(old, new)
                .replace("_process_row_bcast_data_band_", "_resident_left_row_band_")
            )
            for op, base in (("LOAD", "data"), ("STORE", "out")):
                for i in (2, 3):
                    expected = expected.replace(
                        f"    TT_SFP{op}(p_sfpu::LREG{i}, IM, ADDR_MOD_7, {base}_slot_base + COL_GROUP_OFFSETS[{i}]);\n",
                        "",
                    )
            self.assertEqual(tokens(function(actual, new)), tokens(expected))
        self.assertIn("(ckernel::BinaryOp::MUL),", actual)  # other template branches never instantiated

    def test_two_iterations_cover_both_eight_column_groups(self):
        text = tokens(candidate.BODY_PATH.read_text())
        self.assertIn("(APPROX,ckernel::BinaryOp::SUB,2,DST_ACCUM_MODE,ckernel::DstRoundingMode::Default)", text)
        self.assertIn("(APPROX,ckernel::BinaryOp::MUL,2,DST_ACCUM_MODE)", text)
        self.assertEqual(text.count("a,b,out,VectorMode::RC_custom"), 2)
        self.assertEqual(candidate.recurrence("resident_control"), candidate.resident.RECURRENCE)
        self.assertEqual(candidate.recurrence("current_rm"), candidate.baseline._rm_recurrence)


if __name__ == "__main__":
    unittest.main()
