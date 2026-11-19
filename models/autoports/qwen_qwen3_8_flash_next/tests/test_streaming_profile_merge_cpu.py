# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only selection/failure controls plus actual tt-perf-report integration.

Run as a module, not through repository pytest/conftest hardware setup.
The timings below are explicit synthetic plumbing fixtures, not measurements.
"""

import copy
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from models.autoports.qwen_qwen3_8_flash_next.tests.report_streaming_performance import merge_rows


def fixture(mode="decode", executions=3):
    def marker(name):
        return {"OP TYPE": "signpost", "OP CODE": name}

    def op(raw):
        return {
            "OP TYPE": "tt_dnn_device",
            "OP CODE": "UnaryDeviceOperation",
            "GLOBAL CALL COUNT": raw << 10,
            "DEVICE ID": "0",
            "CORE COUNT": "1",
            "DEVICE ARCH": "blackhole",
            "AVAILABLE WORKER CORE COUNT": "120",
            "INPUT_0_MEMORY": "",
            "ATTRIBUTES": "",
            "HOST START TS": "1",
            "HOST END TS": "2",
            "HOST DURATION [ns]": "1",
        }

    templates = [op(10), op(11)]
    if mode == "decode":
        host = [
            marker("STREAMING_CAPTURE_BEGIN"),
            *templates,
            marker("STREAMING_CAPTURE_END"),
            marker("PERF_DECODE"),
            marker("PERF_DECODE_END"),
        ]
    else:
        host = [marker("PERF_PREFILL"), *templates, marker("PERF_PREFILL_END")]
    device = []
    for raw in (10, 11):
        for execution in range(executions if mode == "decode" else 1):
            start = 1000 + execution * 1000 + (raw - 10) * 100
            device.append(
                {
                    "DEVICE ID": "0",
                    "GLOBAL CALL COUNT": raw,
                    "EXECUTION": execution,
                    "CORE COUNT": "1",
                    "DEVICE KERNEL START CYCLE": start,
                    "DEVICE KERNEL END CYCLE": start + 20,
                    "DEVICE KERNEL DURATION [ns]": str(100 + execution * 10 + raw),
                }
            )
    profile = {
        "mode": mode,
        "streaming_profile": True,
        "runtime_audit": True,
        "finite": True,
        "streaming_selection": {
            "mode": mode,
            "begin": 10,
            "end": 12,
            "trace_replay_calls": ["unmeasured_warmup", "measured_final"] if mode == "decode" else [],
        },
    }
    return (
        host,
        device,
        profile,
        "[streaming profiler] active\nzone CSV: wrote 100 rows (0 dropped bytes)\n[streaming profiler] capture: 20 frames, 1.0 MB from 1 device(s); FIFO high-water marks 1%\n",
    )


class StreamingMergeTests(unittest.TestCase):
    def test_decode_last_measured_execution(self):
        rows, proof = merge_rows(*fixture())
        self.assertEqual(proof["selected_execution"], 2)
        self.assertEqual([row["DEVICE KERNEL DURATION [ns]"] for row in rows[1:-1]], ["130", "131"])
        self.assertEqual([row["GLOBAL CALL COUNT"] for row in rows[1:-1]], [10240, 11264])
        self.assertTrue(all(row["OP TO OP LATENCY [ns]"] == "" for row in rows[1:-1]))
        self.assertTrue(all(row["HOST DURATION [ns]"] == "" for row in rows[1:-1]))

    def test_capture_without_execution_and_eager_prefill(self):
        self.assertEqual(merge_rows(*fixture(executions=2))[1]["selected_execution"], 1)
        self.assertEqual(merge_rows(*fixture("prefill"))[1]["selected_execution"], 0)

    def test_missing_duplicate_and_reordered_evidence_rejected(self):
        host, device, profile, log = fixture()
        variants = []
        variants.append((host, device[:-1], profile, log))
        variants.append((host, [*device, device[0]], profile, log))
        variants.append((host[:-1], device, profile, log))
        reversed_time = copy.deepcopy(device)
        reversed_time[1]["DEVICE KERNEL START CYCLE"] = 1001
        variants.append((host, reversed_time, profile, log))
        wrong_calls = copy.deepcopy(profile)
        wrong_calls["streaming_selection"]["trace_replay_calls"].append("unmeasured_after_measurement")
        variants.append((host, device, wrong_calls, log))
        variants.append(
            (host, device, profile, log + "consumer ops-csv dropped64bytes".replace("dropped64", "dropped 64 "))
        )
        variants.append((host, device, profile, log + "consumer ops-csv: 4 order regressions"))
        variants.append((host, device, profile, log + 'consumer "ops-csv" missed 64 bytes of frames'))
        variants.append(
            (
                host,
                device,
                profile,
                log
                + "Device 0: 16 words on 1 lanes were published after the relay's last sweep and are not in the capture",
            )
        )
        variants.append((host, device, profile, log.split("[streaming profiler] capture:")[0]))
        fallback_host = copy.deepcopy(host)
        fallback_host.insert(2, {"OP TYPE": "tt_dnn_host", "GLOBAL CALL COUNT": (1 << 31) | 123})
        variants.append((fallback_host, device, profile, log))
        eager_decode = copy.deepcopy(host)
        eager_decode.insert(-1, copy.deepcopy(host[1]))
        variants.append((eager_decode, device, profile, log))
        for variant in variants:
            with self.subTest(index=variants.index(variant)), self.assertRaises(ValueError):
                merge_rows(*variant)

    def test_real_host_export_parser_preserves_markers_and_ids(self):
        from tracy.common import (
            TRACY_OPS_DATA_FILE_NAME,
            TRACY_OPS_TIMES_FILE_NAME,
            generate_logs_folder,
            generate_reports_folder,
        )

        from models.autoports.qwen_qwen3_8_flash_next.tests.profile_streaming_capture import process_host_only

        host, device, profile, log = fixture()
        with tempfile.TemporaryDirectory(prefix="streaming-host-export-cpu-") as directory:
            root = Path(directory)
            logs = generate_logs_folder(root)
            logs.mkdir(parents=True)
            messages, times = [], []
            for index, row in enumerate(host):
                timestamp = (index + 1) * 100
                if row["OP TYPE"] == "signpost":
                    message = f"TT_SIGNPOST: {row['OP CODE']}"
                else:
                    opid = row["GLOBAL CALL COUNT"]
                    message = "TT_DNN_DEVICE_OP ->\n" + json.dumps(
                        {
                            "op_code": row["OP CODE"],
                            "op_type": row["OP TYPE"],
                            "global_call_count": opid,
                            "device_id": 0,
                            "attributes": {},
                            "input_tensors": [],
                            "output_tensors": [],
                        }
                    )
                    times.append(
                        {
                            "name": "TT_DNN_DEVICE_OP",
                            "zone_text": f"id:{opid}",
                            "special_parent_text": "",
                            "ns_since_start": timestamp,
                            "exec_time_ns": 1,
                        }
                    )
                messages.append({"MessageName": message, "total_ns": timestamp})
            with (logs / TRACY_OPS_DATA_FILE_NAME).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(messages[0]), delimiter=";", quotechar="`")
                writer.writeheader()
                writer.writerows(messages)
            with (logs / TRACY_OPS_TIMES_FILE_NAME).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(times[0]))
                writer.writeheader()
                writer.writerows(times)
            process_host_only(root, None, False, device_only=False)
            exported = list(generate_reports_folder(root).glob("*.csv"))
            self.assertEqual(len(exported), 1)
            with exported[0].open() as stream:
                actual_host = list(csv.DictReader(stream))
            result, proof = merge_rows(actual_host, device, profile, log)
            self.assertEqual(
                [row["OP CODE"] for row in result],
                ["PERF_DECODE", "UnaryDeviceOperation", "UnaryDeviceOperation", "PERF_DECODE_END"],
            )
            self.assertEqual(proof["raw_id_order"], [10, 11])
            self.assertEqual(proof["selected_execution"], 2)
        self.assertNotIn("ttnn", sys.modules)

    def test_actual_report_pipeline_keeps_unavailable_gaps_null(self):
        host, device, profile, log = fixture()
        with tempfile.TemporaryDirectory(prefix="streaming-report-cpu-") as directory:
            root = Path(directory)
            for name, rows in (("host.csv", host), ("streaming.csv", device)):
                with (root / name).open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
                    writer.writeheader()
                    writer.writerows(rows)
            (root / "profile.json").write_text(json.dumps(profile))
            (root / "capture.log").write_text(log)
            command = [
                sys.executable,
                "-m",
                "models.autoports.qwen_qwen3_8_flash_next.tests.report_streaming_performance",
                "--host-csv",
                str(root / "host.csv"),
                "--streaming-csv",
                str(root / "streaming.csv"),
                "--profile-json",
                str(root / "profile.json"),
                "--capture-log",
                str(root / "capture.log"),
                "--output-dir",
                str(root / "report"),
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            console = root / "report/decode_perf_report.console.log"
            self.assertEqual(
                completed.returncode,
                0,
                completed.stdout + completed.stderr + "\n" + (console.read_text() if console.exists() else ""),
            )
            summary = json.loads((root / "report/latency_summary.json").read_text())["decode"]
            self.assertEqual(summary["ops"], 2)
            self.assertAlmostEqual(summary["device_kernel_sum_ms"], 0.000261)
            self.assertIsNone(summary["op_gap_sum_ms"])
            self.assertIsNone(summary["device_kernel_plus_gap_ms"])
            self.assertTrue((root / "report/decode_perf_report.txt").stat().st_size > 0)
            self.assertTrue((root / "report/decode_perf_report.csv.xz").exists())
            changed_profile = root / "different_profile.json"
            changed_profile.write_text(json.dumps({**profile, "layer": 99}))
            rejected = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "models.autoports.qwen_qwen3_8_flash_next.tests.report_performance",
                    "--source-csv",
                    str(root / "report/streaming_selected_ops.csv"),
                    "--output-dir",
                    str(root / "wrong-report"),
                    "--profile-json",
                    str(changed_profile),
                    "--program-support-count",
                    "0",
                    "--mode",
                    "decode",
                    "--streaming-merge-provenance",
                    str(root / "report/streaming_merge.json"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("does not match profile JSON", rejected.stderr)
        self.assertNotIn("ttnn", sys.modules)


if __name__ == "__main__":
    unittest.main()
