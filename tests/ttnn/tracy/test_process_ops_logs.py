#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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


def test_generate_reports_writes_sub_device_id_column(tmp_path):
    log_folder = tmp_path / "logs"
    report_folder = tmp_path / "reports"
    log_folder.mkdir(parents=True, exist_ok=True)

    device_log = log_folder / "profile_log_device.csv"
    device_log.write_text(
        "\n".join(
            [
                "ARCH: wormhole_b0, CHIP_FREQ[MHz]: 1000, Max Compute Cores: 64",
                "PCIe slot,core_x,core_y,RISC processor type,timer_id,time[cycles since reset],data,run host ID,trace id,trace id counter,zone name,type,source line,source file,meta data",
                '0,0,0,BRISC,1,100,0,42,,,BRISC-FW,ZONE_START,1,k.cpp,{"sub_device_id":1;"sub_device_manager_id":7}',
            ]
        )
    )

    ops = {
        42: {
            "global_call_count": 42,
            "device_id": 0,
            "host_time": {"ns_since_start": 10, "exec_time_ns": 20},
            "metal_trace_id": None,
            "input_tensors": [],
            "output_tensors": [],
        }
    }

    sub_device_lookup = process_ops_logs.build_sub_device_id_lookup_from_device_csv(device_log)
    host_ops_by_device = {0: [ops[42].copy()]}
    process_ops_logs.attach_sub_device_ids_to_ops(host_ops_by_device, sub_device_lookup)
    ops[42]["sub_device_id"] = host_ops_by_device[0][0]["sub_device_id"]

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
        assert "SUB DEVICE ID" in reader.fieldnames
        assert row["SUB DEVICE ID"] == "1"
        assert "SUB DEVICE MANAGER ID" not in reader.fieldnames


def test_get_op_sub_device_lookup_key_prefers_device_perf_row():
    op = {
        "global_call_count": 1,
        "device_id": 0,
        "metal_trace_id": None,
        "_device_perf_row": {
            "GLOBAL CALL COUNT": 2048,
            "DEVICE ID": 0,
            "METAL TRACE ID": "",
            "METAL TRACE REPLAY SESSION ID": "",
        },
    }
    assert process_ops_logs.get_op_sub_device_lookup_key(op, 0) == (0, 2048, -1, -1)


def test_build_sub_device_id_lookup_ignores_manager_id_only_rows(tmp_path):
    device_log = tmp_path / "profile_log_device.csv"
    device_log.write_text(
        "\n".join(
            [
                "ARCH: wormhole_b0, CHIP_FREQ[MHz]: 1000, Max Compute Cores: 64",
                "PCIe slot,core_x,core_y,RISC processor type,timer_id,time[cycles since reset],data,run host ID,trace id,trace id counter,zone name,type,source line,source file,meta data",
                '0,0,0,BRISC,1,100,0,42,0,1,BRISC-FW,ZONE_START,1,k.cpp,{"sub_device_id":0;"sub_device_manager_id":7}',
            ]
        )
    )

    lookup = process_ops_logs.build_sub_device_id_lookup_from_device_csv(device_log)
    assert lookup[(0, 42, 0, 1)] == 0


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


def test_attach_counter_derived_bw_uses_real_peaks(tmp_path):
    """End-to-end for the counter-derived BW% path: a synthetic noc-trace + fabric_link_bw
    sidecar must yield NoC BW% against the AICLK-derived per-port peak and ETH BW% against
    the trained per-link speed -- both computed from real peaks, not hardcoded constants."""
    import json

    # 50 GB/s/link, as the device sidecar reports on a 400G BH mesh.
    (tmp_path / "fabric_link_bw_0.json").write_text(json.dumps({"device_id": 0, "per_link_gb_s": 50.0}))
    # op 1: pure NoC, one port, 21.6 MB in 1 ms -> 21.6 GB/s == 50% of the 43.2 GB/s port peak.
    # op 2: one fabric link, 25 MB in 1 ms -> 25 GB/s == 50% of the 50 GB/s link peak.
    (tmp_path / "noc_trace_0.json").write_text(
        json.dumps(
            [
                {"run_host_id": 1, "noc": "NOC_0", "num_bytes": 21_600_000, "sx": 0, "sy": 0, "type": "WRITE"},
                {
                    "run_host_id": 2,
                    "noc": "NOC_0",
                    "num_bytes": 25_000_000,
                    "sx": 0,
                    "sy": 0,
                    "dx": 1,
                    "dy": 9,
                    "fabric_send": {"eth_chan": 0},
                },
            ]
        )
    )

    # _device_perf_row cycles -> a measured AICLK of 1350 MHz (1.35 Mcycle over 1 ms), so the NoC
    # port peak is derived from the real clock: 1350 MHz x 32 B/cycle = 43.2 GB/s. This mirrors the
    # op["_device_perf_row"] the C++ fast path attaches (CSV-stage keys are not populated yet).
    def _op(gid):
        return {
            "global_call_count": gid,
            "_device_perf_row": {
                "DEVICE FW DURATION [ns]": 1_000_000,
                "DEVICE FW START CYCLE": 0,
                "DEVICE FW END CYCLE": 1_350_000,
            },
        }

    ops = [_op(1), _op(2)]
    touched = process_ops_logs._attach_counter_derived_bw(iter(ops), tmp_path)
    assert touched == 2
    assert ops[0]["NOC BW UTIL FROM COUNTERS (%)"] == pytest.approx(50.0, abs=1.0)
    assert "ETH BW UTIL FROM COUNTERS (%)" not in ops[0]  # no fabric traffic -> no lie
    assert ops[1]["ETH BW UTIL FROM COUNTERS (%)"] == pytest.approx(50.0, abs=1.0)


def test_generate_reports_writes_noc_counter_bytes_column(tmp_path):
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
            "NOC BYTES FROM COUNTERS": 402653184,
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
    with report_csv.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        row = next(reader)
        assert "NOC BYTES FROM COUNTERS" in reader.fieldnames
        assert row["NOC BYTES FROM COUNTERS"] == "402653184"
