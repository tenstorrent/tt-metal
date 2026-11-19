# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Small CPU fixtures; no TTNN import, device or real-capture conversion."""

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tests.profile_acceptance import (
    PHASE_COMPONENTS,
    SESSION,
    TRACE,
    UID_FIELDS,
    capture_health,
    fidelities,
    load_program_metadata,
    validate_executions,
    validate_metadata,
)
from models.autoports.qwen_qwen3_6_35b_a3b.tests.profile_full_model_components import acceptance


def fixture():
    traces = {"model": 11, "sampling": 12}
    windows, programs = {}, {}
    for phase, names in PHASE_COMPONENTS.items():
        rows = []
        for name in names:
            for rank in range(4):
                for step in range(16):
                    for operation in range(2):
                        row = {
                            "OP TYPE": "tt_dnn_device",
                            "OP CODE": "FixtureCompute" if operation == 0 else "FixtureMovement",
                            "DEVICE ID": str(rank),
                            "GLOBAL CALL COUNT": str(traces[name] * 100 + rank * 2 + operation),
                            TRACE: str(traces[name]),
                            SESSION: str(1 + step + (0 if phase == "token_out" else 16)),
                            "PROGRAM HASH": str(operation),
                            "DEVICE KERNEL DURATION [ns]": "100",
                        }
                        tensor = {
                            "shape": {axis: "32[32]" for axis in "WZYX"},
                            "dtype": "BFLOAT16",
                            "layout": "TILE",
                            "storage_type": {
                                "device_id": rank,
                                "memory_config": {"buffer_type": "DRAM", "memory_layout": "INTERLEAVED"},
                            },
                        }
                        for prefix in ("INPUT_0", "OUTPUT_0"):
                            row.update({prefix + "_" + axis + "_PAD[LOGICAL]": "32[32]" for axis in "WZYX"})
                            row.update(
                                {
                                    prefix + "_DATATYPE": "BFLOAT16",
                                    prefix + "_LAYOUT": "TILE",
                                    prefix + "_MEMORY": f"DEV_{rank}_DRAM_INTERLEAVED",
                                }
                            )
                        index_tensor = copy.deepcopy(tensor)
                        index_tensor["dtype"] = "UINT32"
                        row.update(
                            {
                                key.replace("INPUT_0", "INPUT_1"): ("UINT32" if key.endswith("DATATYPE") else value)
                                for key, value in list(row.items())
                                if key.startswith("INPUT_0")
                            }
                        )
                        programs[rank, operation] = {
                            "device_id": rank,
                            "op_hash": operation,
                            "op_code": row["OP CODE"],
                            "input_tensors": [tensor, index_tensor],
                            "output_tensors": [tensor],
                            "kernel_info": {
                                "compute_kernels": (
                                    [
                                        {"source": "first.cpp", "name": "first", "math_fidelity": "HiFi2"},
                                        {"source": "second.cpp", "name": "second", "math_fidelity": "LoFi"},
                                    ]
                                    if operation == 0
                                    else []
                                ),
                                "datamovement_kernels": [{"source": "reader.cpp", "name": "reader"}],
                            },
                        }
                        rows.append(row)
        windows[phase] = rows
    native = [dict(row) for rows in windows.values() for row in rows]
    return windows, native, traces, programs


class ProfileAcceptanceTest(unittest.TestCase):
    def setUp(self):
        self.windows, self.native, self.traces, self.programs = fixture()
        self.rows = [row for rows in self.windows.values() for row in rows]

    def test_complete(self):
        checked = validate_executions(self.windows, self.native, self.traces)
        self.assertEqual(checked["native_execution_count"], 512)
        inventory = validate_metadata(self.rows, self.programs)
        self.assertEqual(len(inventory), 8)
        self.assertEqual(json.loads(fidelities(self.rows[0], self.programs)), ["HiFi2", "LoFi"])
        self.assertEqual(json.loads(fidelities(self.rows[1], self.programs)), [])
        self.assertEqual(len(inventory["0:0"]["kernel_info"]["compute_kernels"]), 2)
        self.assertEqual([tensor["dtype"] for tensor in inventory["0:0"]["input_tensors"]], ["BFLOAT16", "UINT32"])

    def test_missing_same_operator_every_replay(self):
        self.windows = {
            phase: [row for row in rows if row["PROGRAM HASH"] != "1"] for phase, rows in self.windows.items()
        }
        with self.assertRaisesRegex(ValueError, "missing converted"):
            validate_executions(self.windows, self.native, self.traces)

    def test_native_missing(self):
        with self.assertRaisesRegex(ValueError, "absent native"):
            validate_executions(self.windows, self.native[1:], self.traces)

    def test_native_duplicate(self):
        with self.assertRaisesRegex(ValueError, "Duplicate native"):
            validate_executions(self.windows, self.native + self.native[:1], self.traces)

    def test_converted_duplicate(self):
        self.windows["token_out"].append(self.windows["token_out"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate converted"):
            validate_executions(self.windows, self.native, self.traces)

    def test_rank_session_mismatch(self):
        for row in self.windows["token_out"]:
            if row["DEVICE ID"] == "3":
                row[SESSION] = str(int(row[SESSION]) + 100)
        with self.assertRaisesRegex(ValueError, "rank session mismatch"):
            validate_executions(self.windows, self.native, self.traces)

    def test_metadata_missing_and_truncated(self):
        for mutate in (
            lambda data: data.pop("kernel_info"),
            lambda data: data["input_tensors"][0].pop("dtype"),
            lambda data: data.update(input_tensors_original_count=3),
            lambda data: data["kernel_info"]["compute_kernels"][0].pop("math_fidelity"),
        ):
            programs = copy.deepcopy(self.programs)
            mutate(programs[0, 0])
            with self.assertRaisesRegex(ValueError, "Unsupported/incomplete"):
                validate_metadata(self.rows, programs)

    def test_missing_converted_dtype_or_shape(self):
        for field in ("INPUT_0_DATATYPE", "OUTPUT_0_X_PAD[LOGICAL]"):
            rows = copy.deepcopy(self.rows)
            rows[0].pop(field)
            with self.assertRaisesRegex(ValueError, "Missing/mismatched converted"):
                validate_metadata(rows, self.programs)

    def test_no_compute_needs_explicit_movement_inventory(self):
        self.programs[0, 1]["kernel_info"]["datamovement_kernels"] = []
        with self.assertRaisesRegex(ValueError, "missing native kernel inventory"):
            validate_metadata(self.rows, self.programs)

    def test_capture_and_postprocessing_warning_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "capture.log"):
                capture_health(root)
            (root / "capture.log").write_text("Successful fixture\n")
            capture_health(root)
            for name, warning in (
                ("capture.log", "Instrumentation failure: Too many source locations."),
                ("capture.log", "Profiler DRAM buffers were full, markers were dropped!"),
                ("postprocess.log", "Skipping cached op reference with no prior data"),
                ("postprocess.log", "Skipping op with malformed JSON"),
            ):
                (root / "capture.log").write_text("Successful fixture\n")
                (root / name).write_text(warning)
                with self.assertRaisesRegex(ValueError, "Incomplete profiler"):
                    capture_health(root)
                (root / name).unlink()

    def test_integrated_native_metadata_and_execution_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw/.logs"
            raw.mkdir(parents=True)
            (root / "capture.log").write_text("Synthetic CPU fixture; no device evidence\n")
            with (raw / "cpp_device_perf_report.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=UID_FIELDS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self.native)
            messages = raw / "tracy_ops_data.csv"
            with messages.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["MessageName"], delimiter=";", quotechar="`")
                writer.writeheader()
                for data in self.programs.values():
                    writer.writerow({"MessageName": "TT_DNN_DEVICE_OP: fixture ->\n" + json.dumps(data)})
            timeline = []
            for phase, rows in self.windows.items():
                timeline.append({"OP TYPE": "signpost", "OP CODE": "PERF_" + phase.upper()})
                timeline.extend(rows)
                timeline.append({"OP TYPE": "signpost", "OP CODE": "PERF_" + phase.upper() + "_END"})
            coverage, programs, inventory = acceptance(root, {"trace_ids": self.traces}, timeline)
            self.assertEqual(coverage["native_execution_count"], 512)
            self.assertEqual(programs, self.programs)
            self.assertEqual(inventory["0:0"]["kernel_info"], self.programs[0, 0]["kernel_info"])
            with self.assertRaisesRegex(ValueError, "Missing native program metadata"):
                load_program_metadata(messages, {(0, 999)})
            messages.write_text("MessageName\n`TT_DNN_DEVICE_OP: fixture ->\n{bad}`\n")
            with self.assertRaisesRegex(ValueError, "Malformed native"):
                load_program_metadata(messages, {(0, 0)})


if __name__ == "__main__":
    unittest.main()
