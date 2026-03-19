#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import csv
from pathlib import Path

import pytest

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


@pytest.mark.skip(reason="Missing mock for device log file; needs fix to properly stub _enrich_ops_from_device_logs")
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


def _make_device_op_time(run_host_id, timestamp=100, analysis=None):
    """Build a minimal device op timing entry used by _enrich_ops_from_device_logs."""
    return {
        "timeseries": [
            (
                {"run_host_id": run_host_id, "zone_name": "FW"},
                timestamp,
                {},
                "BRISC",
                (0, 0),
            )
        ],
        "analysis": analysis or {},
    }


def _make_dispatch_op(run_host_id, analysis=None):
    """Build a minimal dispatch op entry."""
    return {
        "timeseries": [
            (
                {"meta_data": str({"workers_runtime_id": run_host_id})},
                50,
                {},
                "BRISC",
                (0, 0),
            )
        ],
        "analysis": analysis or {"dispatch_dur": {"series": [], "stats": {}}},
    }


def _stub_device_data(device_id, device_ops, dispatch_ops):
    """Return a fake import_log_run_stats result for a single device."""
    return {
        "deviceInfo": {"freq": 1200, "max_compute_cores": 64},
        "devices": {
            device_id: {
                "cores": {
                    "DEVICE": {
                        "riscs": {
                            "TENSIX": {
                                "ops": device_ops,
                                "dispatch_ops": dispatch_ops,
                            }
                        }
                    }
                }
            }
        },
    }


def test_enrich_device_logs_skips_unmatched_device_ops_in_trace_replay(monkeypatch, tmp_path):
    """Dispatch profiling with trace replay may produce device ops whose
    run_host_id has no matching host op (e.g. internal trace replay dispatch
    entries).  Before the fix this raised an AssertionError; after the fix these
    ops are silently skipped and the report is generated successfully."""

    device_id = 0
    matched_id = 1
    unmatched_id = 999

    host_ops_by_device = {
        device_id: [
            {"global_call_count": matched_id, "metal_trace_id": 42},
        ]
    }

    device_ops = [
        _make_device_op_time(matched_id, timestamp=100),
        _make_device_op_time(unmatched_id, timestamp=200),
    ]
    dispatch_ops = [
        _make_dispatch_op(matched_id),
    ]

    trace_replays = {device_id: {42: [1000]}}

    fake_data = _stub_device_data(device_id, device_ops, dispatch_ops)
    monkeypatch.setattr(process_ops_logs, "import_log_run_stats", lambda _setup: fake_data)

    (tmp_path / "profile_log_device.csv").touch()

    result = process_ops_logs._enrich_ops_from_device_logs(host_ops_by_device, tmp_path, [], trace_replays)

    assert len(result[device_id]) == 1
    assert result[device_id][0]["global_call_count"] == matched_id


def test_enrich_device_logs_ignores_leftover_dispatch_ops_in_trace_replay(monkeypatch, tmp_path):
    """When profiling dispatch with trace replay, some dispatch ops may not
    match any device op (trace replay dispatch entries).  Before the fix this
    hit 'Unrecognized dispatch OPs' assertion; now it is logged and ignored."""

    device_id = 0
    op_id = 1
    orphan_dispatch_id = 888

    host_ops_by_device = {
        device_id: [
            {"global_call_count": op_id, "metal_trace_id": 7},
        ]
    }

    device_ops = [
        _make_device_op_time(op_id, timestamp=100),
    ]
    dispatch_ops = [
        _make_dispatch_op(op_id),
        _make_dispatch_op(orphan_dispatch_id),
    ]

    trace_replays = {device_id: {7: [2000]}}

    fake_data = _stub_device_data(device_id, device_ops, dispatch_ops)
    monkeypatch.setattr(process_ops_logs, "import_log_run_stats", lambda _setup: fake_data)

    (tmp_path / "profile_log_device.csv").touch()

    result = process_ops_logs._enrich_ops_from_device_logs(host_ops_by_device, tmp_path, [], trace_replays)

    assert len(result[device_id]) == 1
    assert result[device_id][0]["global_call_count"] == op_id


def test_enrich_device_logs_still_asserts_on_unrecognized_dispatch_without_trace(monkeypatch, tmp_path):
    """Without trace replays, leftover dispatch ops should still assert to
    catch real mismatches (the original behaviour is preserved)."""

    device_id = 0
    op_id = 1
    orphan_dispatch_id = 888

    host_ops_by_device = {
        device_id: [
            {"global_call_count": op_id},
        ]
    }

    device_ops = [
        _make_device_op_time(op_id, timestamp=100),
    ]
    dispatch_ops = [
        _make_dispatch_op(op_id),
        _make_dispatch_op(orphan_dispatch_id),
    ]

    fake_data = _stub_device_data(device_id, device_ops, dispatch_ops)
    monkeypatch.setattr(process_ops_logs, "import_log_run_stats", lambda _setup: fake_data)

    (tmp_path / "profile_log_device.csv").touch()

    with pytest.raises(AssertionError, match="Unrecognized dispatch OPs"):
        process_ops_logs._enrich_ops_from_device_logs(host_ops_by_device, tmp_path, [], None)
