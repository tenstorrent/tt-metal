#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import csv
import json
from pathlib import Path

import pytest

from tracy import process_ops_logs


def _write_tracy_op_logs(log_folder, messages, op_ids):
    with (log_folder / "tracy_ops_data.csv").open("w", newline="") as data_file:
        writer = csv.writer(data_file, delimiter=";", quotechar="`", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["MessageName", "total_ns"])
        writer.writerows(messages)

    with (log_folder / "tracy_ops_times.csv").open("w", newline="") as times_file:
        writer = csv.writer(times_file)
        writer.writerow(["name", "zone_text", "exec_time_ns", "special_parent_text"])
        for op_id in op_ids:
            writer.writerow(["TT_DNN_DEVICE_OP", f"id:{op_id}", 1, ""])


def _tensor_metadata(x_shape):
    return {
        "storage_type": {"device_id": 0, "memory_config": {"buffer_type": "DRAM", "memory_layout": "INTERLEAVED"}},
        "shape": {"W": "1[1]", "Z": "1[1]", "Y": "1[1]", "X": x_shape},
        "layout": "TILE",
        "dtype": "BFLOAT16",
    }


def test_import_tracy_op_logs_uses_cached_invocation_tensor_metadata(tmp_path):
    first_op = {
        "global_call_count": 41,
        "op_code": "reshape",
        "attributes": {"axis": "-1"},
        "input_tensors": [_tensor_metadata("32[32]")],
        "output_tensors": [_tensor_metadata("32[32]")],
        "op_type": "tt_dnn_device",
        "device_id": 0,
        "op_hash": 777,
        "program_cache_hit": False,
        "kernel_info": {"compute_kernels": [{"source": "reshape.cpp"}]},
    }
    cached_metadata = {
        "input_tensors": [_tensor_metadata("64[64]")],
        "output_tensors": [_tensor_metadata("64[64]")],
    }
    _write_tracy_op_logs(
        tmp_path,
        [
            (f'TT_DNN_DEVICE_OP: "reshape", 777, 0, false, 41 ->\n{json.dumps(first_op)}', 100),
            ("TT_METAL_TRACE_BEGIN: 0, 9", 150),
            (
                f'TT_DNN_DEVICE_OP: "reshape", 777, 0, true, 42 ->\n{json.dumps(cached_metadata)}',
                200,
            ),
        ],
        op_ids=(41, 42),
    )

    ops, _, _ = process_ops_logs.import_tracy_op_logs(tmp_path)

    assert ops[41]["input_tensors"][0]["shape"]["X"] == "32[32]"
    assert ops[42]["input_tensors"][0]["shape"]["X"] == "64[64]"
    assert ops[42]["output_tensors"][0]["shape"]["X"] == "64[64]"
    assert ops[42]["global_call_count"] == 42
    assert ops[42]["program_cache_hit"] is True
    assert ops[41]["metal_trace_id"] is None
    assert ops[42]["metal_trace_id"] == 9
    assert ops[42]["attributes"] == first_op["attributes"]
    assert ops[42]["kernel_info"] == first_op["kernel_info"]


def test_import_tracy_op_logs_accepts_legacy_cached_message(tmp_path):
    first_op = {
        "global_call_count": 41,
        "op_code": "reshape",
        "input_tensors": [_tensor_metadata("32[32]")],
        "output_tensors": [],
        "device_id": 0,
        "op_hash": 777,
    }
    _write_tracy_op_logs(
        tmp_path,
        [
            (f'TT_DNN_DEVICE_OP: "reshape", 777, 0, false, 41 ->\n{json.dumps(first_op)}', 100),
            ('TT_DNN_DEVICE_OP: "reshape", 777, 0, true, 43', 300),
        ],
        op_ids=(41, 43),
    )

    ops, _, _ = process_ops_logs.import_tracy_op_logs(tmp_path)

    assert ops[43]["input_tensors"][0]["shape"]["X"] == "32[32]"
    assert ops[43]["global_call_count"] == 43
    assert ops[43]["program_cache_hit"] is True
    assert ops[43]["metal_trace_id"] is None


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


def _host_op(op_id, trace_id):
    return {
        "global_call_count": op_id,
        "device_id": 0,
        "metal_trace_id": trace_id,
        "host_time": {"ns_since_start": op_id, "exec_time_ns": 1},
    }


def _perf_row(op_id, trace_id, session_id):
    return {
        "GLOBAL CALL COUNT": op_id,
        "METAL TRACE ID": trace_id,
        "METAL TRACE REPLAY SESSION ID": session_id,
        "CORE COUNT": 8,
    }


def test_enrich_ops_from_perf_csv_leaves_out_ops_of_a_never_replayed_trace():
    # Trace 0 was replayed once (host marker + device rows); trace 1 was only captured.
    host_ops = {0: [_host_op(10, 0), _host_op(11, 0), _host_op(20, 1), _host_op(21, 1)]}
    device_rows = {0: {(10, 0, 1): _perf_row(10, 0, 1), (11, 0, 1): _perf_row(11, 0, 1)}}
    trace_replays = {0: {0: [12345]}}

    enriched = process_ops_logs._enrich_ops_from_perf_csv(host_ops, device_rows, trace_replays)

    assert [op["global_call_count"] for op in enriched[0]] == [10, 11]
    assert all(op["tracy_time"] == 12345 for op in enriched[0])


def test_enrich_ops_from_perf_csv_still_asserts_on_a_replayed_trace_missing_one_row():
    host_ops = {0: [_host_op(10, 0), _host_op(11, 0)]}
    device_rows = {0: {(10, 0, 1): _perf_row(10, 0, 1)}}
    trace_replays = {0: {0: [12345]}}

    with pytest.raises(AssertionError, match="Op 11 not present"):  # allow-pytest.raises: runs with --noconftest
        process_ops_logs._enrich_ops_from_perf_csv(host_ops, device_rows, trace_replays)


def test_enrich_ops_from_perf_csv_still_asserts_when_a_host_replayed_trace_has_no_device_rows():
    # The host replayed trace 1 (REPLAY marker present) but the device report has no row for it:
    # that is lost device data, not a never-replayed trace, so it must not be dropped silently.
    host_ops = {0: [_host_op(10, 0), _host_op(20, 1)]}
    device_rows = {0: {(10, 0, 1): _perf_row(10, 0, 1)}}
    trace_replays = {0: {0: [12345], 1: [23456]}}

    with pytest.raises(AssertionError, match="host replayed this trace"):  # allow-pytest.raises: runs with --noconftest
        process_ops_logs._enrich_ops_from_perf_csv(host_ops, device_rows, trace_replays)


def test_enrich_ops_from_perf_csv_accepts_a_device_whose_only_trace_was_never_replayed():
    # Device 1 only captured trace 0 and never replayed it, so the report has no entry for device 1 at all.
    host_ops = {0: [_host_op(10, None)], 1: [_host_op(20, 0), _host_op(21, 0)]}
    device_rows = {0: {(10, None, None): _perf_row(10, None, None)}}
    trace_replays = {}

    enriched = process_ops_logs._enrich_ops_from_perf_csv(host_ops, device_rows, trace_replays)

    assert [op["global_call_count"] for op in enriched[0]] == [10]
    assert enriched[1] == []


def test_enrich_ops_from_perf_csv_still_asserts_on_a_device_missing_from_the_report():
    # Device 1 ran a non-trace op but the report has no entry for it: that is lost device data.
    host_ops = {0: [_host_op(10, None)], 1: [_host_op(20, None)]}
    device_rows = {0: {(10, None, None): _perf_row(10, None, None)}}

    with pytest.raises(AssertionError, match="no rows at all"):  # allow-pytest.raises: runs with --noconftest
        process_ops_logs._enrich_ops_from_perf_csv(host_ops, device_rows, None)
