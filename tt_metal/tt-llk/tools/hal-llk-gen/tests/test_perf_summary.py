# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Offline test of python/perf_summary.py on synthetic perf CSV files (needs pandas)."""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))

try:
    import pandas  # noqa: F401

    HAVE_PANDAS = True
except ImportError:
    HAVE_PANDAS = False

COLS = "formats.input_A,formats.input_B,formats.output,dest_acc,mathop,math_fidelity,marker"
RUN_TYPES = ["L1_TO_L1", "UNPACK_ISOLATE", "MATH_ISOLATE", "PACK_ISOLATE"]


def write_csv(root, label, scale):
    d = os.path.join(root, label, "perf_data", "perf_eltwise_binary")
    os.makedirs(d)
    header = COLS + "," + ",".join(f"mean({rt})" for rt in RUN_TYPES)
    rows = [header]
    for fmt, base in (("Float16_b", 30.0), ("Bfp8_b", 40.0)):
        for marker in ("TILE_LOOP", "KERNEL"):
            vals = ",".join(
                str(base * scale * (10 if marker == "KERNEL" else 1)) for _ in RUN_TYPES
            )
            rows.append(
                f"DataFormat.{fmt},DataFormat.{fmt},DataFormat.{fmt},DestAccumulation.No,Elwadd,LoFi,{marker},{vals}"
            )
    with open(os.path.join(d, "perf_eltwise_binary.post.csv"), "w") as f:
        f.write("\n".join(rows) + "\n")


@unittest.skipUnless(HAVE_PANDAS, "pandas not installed")
class PerfSummaryTest(unittest.TestCase):
    def test_ratios(self):
        import perf_summary

        with tempfile.TemporaryDirectory() as root:
            for i in (1, 2, 3):
                write_csv(root, f"orig_r{i}", 1.0)
                write_csv(root, f"hal_r{i}", 1.5 if i != 2 else 1.4)
            res = perf_summary.summarize(root, 3)
            self.assertEqual(res["l1"]["n"], 2)
            self.assertAlmostEqual(res["l1"]["ratio_median"], 1.5)
            self.assertEqual(res["l1"]["faster_variants"], 0)
            self.assertEqual(
                res["l1_cov"]["n"], 2
            )  # both formats are functionally checked formats
            self.assertIn("kern", res)
            self.assertIn("| L1->L1 per tile | 2 |", perf_summary.markdown(res))

    def test_missing_data(self):
        import perf_summary

        with tempfile.TemporaryDirectory() as root:
            write_csv(root, "orig_r1", 1.0)
            with self.assertRaises(SystemExit):
                perf_summary.summarize(root, 1)


if __name__ == "__main__":
    unittest.main()
