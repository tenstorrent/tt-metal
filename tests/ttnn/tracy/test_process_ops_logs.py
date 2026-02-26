#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import csv
from pathlib import Path

from tracy import process_ops_logs


# class for mocking creation of npe data
class _FakeNpeResult:
    def __init__(self, noc_util, mcast_noc_util, dram_bw_util, cong_impact):
        self.overall_avg_link_util = noc_util
        self.overall_avg_mcast_write_link_util = mcast_noc_util
        self.dram_bw_util = dram_bw_util
        self._cong_impact = cong_impact

    def getCongestionImpact(self):
        return self._cong_impact


class _FakeNpeDatapoint:
    def __init__(self, result):
        self.result = result


class _FakeNpeStats:
    def __init__(self, op_to_result):
        self._op_to_result = op_to_result

    def getDatapointByID(self, op_id):
        result = self._op_to_result.get(op_id)
        if result is None:
            return None
        return _FakeNpeDatapoint(result)


def test_append_device_data_populates_multicast_noc_util(monkeypatch, tmp_path):
    ops = {
        1: {
            "global_call_count": 1,
            "device_id": 0,
        }
    }
    trace_replays = {}

    fake_stats = _FakeNpeStats(
        {
            1: _FakeNpeResult(
                noc_util=91.24,
                mcast_noc_util=44.44,
                dram_bw_util=38.88,
                cong_impact=12.345,
            )
        }
    )
    monkeypatch.setattr(process_ops_logs, "analyzeNoCTraces", lambda _log_folder: fake_stats)

    process_ops_logs.append_device_data(
        ops=ops,
        traceReplays=trace_replays,
        logFolder=tmp_path,
        analyze_noc_traces=True,
        device_analysis_types=[],
    )

    assert ops[1]["NOC UTIL (%)"] == 91.2
    assert ops[1]["MULTICAST NOC UTIL (%)"] == 44.4
    assert ops[1]["DRAM BW UTIL (%)"] == 38.9
    assert ops[1]["NPE CONG IMPACT (%)"] == 12.35


def test_generate_reports_writes_multicast_noc_util_column(tmp_path):
    log_folder = tmp_path / "logs"
    report_folder = tmp_path / "reports"
    log_folder.mkdir(parents=True, exist_ok=True)

    ops = {
        1: {
            "global_call_count": 1,
            "device_id": 0,
            "host_time": {"ns_since_start": 10, "exec_time_ns": 20},
            "metal_trace_id": None,
            "input_tensors": [],
            "output_tensors": [],
            "NOC UTIL (%)": 50.0,
            "MULTICAST NOC UTIL (%)": 25.0,
            "DRAM BW UTIL (%)": 75.0,
            "NPE CONG IMPACT (%)": 1.25,
        }
    }

    process_ops_logs.generate_reports(
        ops=ops,
        deviceOps={},
        traceOps={},
        signposts={},
        logFolder=log_folder,
        outputFolder=report_folder,
        date=False,
        nameAppend=None,
    )

    report_csv = Path(report_folder) / "ops_perf_results.csv"
    assert report_csv.is_file()

    with report_csv.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        row = next(reader)
        assert "MULTICAST NOC UTIL (%)" in reader.fieldnames
        assert row["MULTICAST NOC UTIL (%)"] == "25.0"
