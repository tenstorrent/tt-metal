# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for controlled geometry and native-row evidence validation."""

import csv
import tempfile
import unittest
from pathlib import Path

from models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_dram_readers import (
    geometry,
    reader_order,
    summarize_csv,
)


class ReaderProbeTests(unittest.TestCase):
    def test_common_padding_and_storage(self):
        cases = (
            (2816, 8192, 11, "bfloat8_b", 9216, 8),
            (2816, 9216, 11, "bfloat8_b", 9216, 8),
            (4096, 2816, 16, "bfloat8_b", 3072, 8),
            (8192, 2816, 16, "bfloat8_b", 3072, 8),
            (2816, 4224, 11, "bfloat4_b", 4608, 8),
            (2112, 2816, 11, "bfloat4_b", 3072, 6),
        )
        for k, n, block, dtype, physical_n, cores in cases:
            result = geometry(k, n, block, 8, 11, dtype)
            self.assertEqual(result["physical_MKN"], [32, k, physical_n])
            self.assertEqual(result["input_storage_cores"], cores)
            for r in (1, 2, 3):
                self.assertEqual(result["per_bank_N_tiles"] % r, 0)
                self.assertEqual(result["readers"][str(r)]["per_reader_N_tiles"] * r, result["per_bank_N_tiles"])
        with self.assertRaises(ValueError):
            geometry(2816, 8192, 3, 8, 11, "bfloat8_b")

    def test_complete_balanced_orders(self):
        orders = [reader_order(i) for i in range(6)]
        self.assertEqual(len({tuple(order) for order in orders}), 6)
        for position in range(3):
            self.assertEqual(sorted(order[position] for order in orders), [1, 1, 2, 2, 3, 3])

    def test_csv_requires_actual_dtype_and_complete_single_op_windows(self):
        fields = [
            "OP CODE",
            "OP TYPE",
            "ATTRIBUTES",
            "MATH FIDELITY",
            "INPUT_0_DATATYPE",
            "INPUT_1_DATATYPE",
            "OUTPUT_0_DATATYPE",
            "CORE COUNT",
            "DEVICE KERNEL DURATION [ns]",
        ]
        marker = "DRAM_READER_output_k16_r2_round0"
        rows = [
            {"OP CODE": marker, "OP TYPE": "signpost"},
            *[
                dict(
                    zip(
                        fields,
                        [
                            "MatmulDeviceOperation",
                            "tt_dnn_device",
                            "num_workers_per_dram_bank=2",
                            "LoFi",
                            "BFLOAT16",
                            "BFLOAT8_B",
                            "FLOAT32",
                            "24",
                            str(value),
                        ],
                    )
                )
                for value in (10000, 12000)
            ],
            {"OP CODE": marker + "_END", "OP TYPE": "signpost"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ops.csv"

            def run():
                with path.open("w") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
                report = dict(
                    replays_per_sample=2,
                    declared_dram_peak_GBps=512,
                    cases=[
                        dict(
                            status="measured_host_device_pending",
                            legal_readers=[2],
                            input_dtype="bfloat16",
                            weight_dtype="bfloat8_b",
                            output_dtype="float32",
                            compute={"math_fidelity": "LoFi"},
                            geometry={"physical_weight_bytes": 22000, "logical_weight_bytes": 11000},
                            samples=[dict(signpost=marker, readers=2)],
                        )
                    ],
                )
                return summarize_csv(path, report)

            report = run()
            summary = report["cases"][0]["device_summary"]["2"]
            self.assertEqual(summary["median_us"], 11)
            self.assertEqual(summary["physical_weight_GBps"], 2)
            rows[1]["INPUT_0_DATATYPE"] = "FLOAT32"
            with self.assertRaisesRegex(ValueError, "INPUT_0_DATATYPE"):
                run()
            rows[1]["INPUT_0_DATATYPE"] = "BFLOAT16"
            rows.pop(2)
            with self.assertRaisesRegex(ValueError, "exactly one native matmul"):
                run()


if __name__ == "__main__":
    unittest.main()
