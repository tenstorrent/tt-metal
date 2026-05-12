# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Pure-Python unit tests for ttnn.op_perf (issue #36650).

These tests validate the profiling data structures, aggregation logic,
bottleneck detection, and export formats without requiring TT hardware.
"""

import json
import os
import importlib.util
import sys
import tempfile

import pytest

# Import op_perf directly (bypass ttnn.__init__ which requires C++ backend)
_op_perf_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "..", "ttnn", "ttnn", "op_perf.py"
)
_op_perf_path = os.path.normpath(_op_perf_path)
_spec = importlib.util.spec_from_file_location("ttnn.op_perf", _op_perf_path)
_op_perf = importlib.util.module_from_spec(_spec)
sys.modules["ttnn.op_perf"] = _op_perf
_spec.loader.exec_module(_op_perf)

OpRecord = _op_perf.OpRecord
AggregatedOp = _op_perf.AggregatedOp
BottleneckHint = _op_perf.BottleneckHint
OpPerfReport = _op_perf.OpPerfReport
PerfTrace = _op_perf.PerfTrace
_is_transfer_op = _op_perf._is_transfer_op


# ── Fixtures ─────────────────────────────────────────────────────────────


def _make_records():
    """Create a representative set of op records for testing."""
    return [
        OpRecord("ttnn.matmul", 5000.0, [(32, 1024)], ["bfloat16"], [(32, 1024)], 0),
        OpRecord("ttnn.matmul", 4800.0, [(32, 1024)], ["bfloat16"], [(32, 1024)], 1),
        OpRecord("ttnn.matmul", 4900.0, [(32, 1024)], ["bfloat16"], [(32, 1024)], 2),
        OpRecord("ttnn.add", 200.0, [(32, 1024)], ["bfloat16"], [(32, 1024)], 3),
        OpRecord("ttnn.add", 180.0, [(32, 1024)], ["bfloat16"], [(32, 1024)], 4),
        OpRecord("ttnn.layer_norm", 800.0, [(32, 1024)], ["bfloat16"], [(32, 1024)], 5),
        OpRecord("ttnn.from_torch", 1500.0, [(32, 1024)], ["float32"], [(32, 1024)], 6, is_transfer=True),
        OpRecord("ttnn.to_device", 1200.0, [(32, 1024)], ["bfloat16"], [(32, 1024)], 7, is_transfer=True),
        OpRecord("ttnn.gelu", 300.0, [(32, 1024)], ["bfloat16"], [(32, 1024)], 8),
        OpRecord("ttnn.softmax", 400.0, [(32, 1024)], ["bfloat16"], [(32, 1024)], 9),
    ]


def _make_report():
    records = _make_records()
    total = sum(r.duration_us for r in records)
    return OpPerfReport(records=records, total_time_us=total)


# ── OpRecord Tests ───────────────────────────────────────────────────────


def test_op_record_duration_ms():
    r = OpRecord("ttnn.matmul", 5000.0, [], [], [], 0)
    assert r.duration_ms == 5.0


def test_op_record_fields():
    r = OpRecord("ttnn.add", 123.4, [(32,)], ["bfloat16"], [(32,)], 7, is_transfer=False)
    assert r.op_name == "ttnn.add"
    assert r.duration_us == 123.4
    assert r.input_shapes == [(32,)]
    assert r.input_dtypes == ["bfloat16"]
    assert r.output_shapes == [(32,)]
    assert r.call_index == 7
    assert r.is_transfer is False


# ── Transfer Classification Tests ────────────────────────────────────────


def test_transfer_op_from_torch():
    assert _is_transfer_op("ttnn.from_torch") is True


def test_transfer_op_to_device():
    assert _is_transfer_op("ttnn.to_device") is True


def test_transfer_op_to_layout():
    assert _is_transfer_op("ttnn.to_layout") is True


def test_transfer_op_matmul_not_transfer():
    assert _is_transfer_op("ttnn.matmul") is False


def test_transfer_op_gelu_not_transfer():
    assert _is_transfer_op("ttnn.gelu") is False


def test_transfer_op_case_insensitive():
    assert _is_transfer_op("ttnn.FROM_TORCH") is True


def test_transfer_op_clone():
    assert _is_transfer_op("ttnn.clone") is True


# ── Aggregation Tests ────────────────────────────────────────────────────


def test_aggregation_count():
    report = _make_report()
    agg = report.aggregated
    names = [a.op_name for a in agg]
    assert "ttnn.matmul" in names
    assert "ttnn.add" in names
    assert "ttnn.from_torch" in names


def test_aggregation_sorted_by_total():
    report = _make_report()
    agg = report.aggregated
    assert agg[0].op_name == "ttnn.matmul"
    assert agg[0].total_us == pytest.approx(14700.0)


def test_aggregation_count_field():
    report = _make_report()
    matmul_agg = [a for a in report.aggregated if a.op_name == "ttnn.matmul"][0]
    assert matmul_agg.count == 3


def test_aggregation_mean():
    report = _make_report()
    matmul_agg = [a for a in report.aggregated if a.op_name == "ttnn.matmul"][0]
    assert matmul_agg.mean_us == pytest.approx(4900.0)


def test_aggregation_min_max():
    report = _make_report()
    matmul_agg = [a for a in report.aggregated if a.op_name == "ttnn.matmul"][0]
    assert matmul_agg.min_us == 4800.0
    assert matmul_agg.max_us == 5000.0


def test_aggregation_pct_of_total():
    report = _make_report()
    total_pct = sum(a.pct_of_total for a in report.aggregated)
    assert total_pct == pytest.approx(100.0, abs=0.1)


# ── Top Ops Tests ────────────────────────────────────────────────────────


def test_top_ops_default():
    report = _make_report()
    top = report.top_ops()
    assert len(top) <= 10
    assert top[0].op_name == "ttnn.matmul"


def test_top_ops_limited():
    report = _make_report()
    top = report.top_ops(n=3)
    assert len(top) == 3


# ── Transfer vs Compute Time ─────────────────────────────────────────────


def test_transfer_time():
    report = _make_report()
    assert report.transfer_time_us == pytest.approx(2700.0)


def test_compute_time():
    report = _make_report()
    expected_compute = report.total_time_us - 2700.0
    assert report.compute_time_us == pytest.approx(expected_compute)


def test_transfer_pct():
    report = _make_report()
    assert 0 < report.transfer_pct < 100


# ── Bottleneck Hints Tests ───────────────────────────────────────────────


def test_dominant_op_hint():
    """matmul is >50% of total -- should trigger 'dominates runtime' hint."""
    records = [
        OpRecord("ttnn.matmul", 9000.0, [], [], [], 0),
        OpRecord("ttnn.add", 1000.0, [], [], [], 1),
    ]
    report = OpPerfReport(records=records, total_time_us=10000.0)
    hints = report.bottleneck_hints
    high_hints = [h for h in hints if h.severity == "high" and "dominates" in h.message]
    assert len(high_hints) >= 1


def test_transfer_overhead_hint():
    """Transfers >20% -- should trigger transfer hint."""
    records = [
        OpRecord("ttnn.from_torch", 3000.0, [], [], [], 0, is_transfer=True),
        OpRecord("ttnn.to_device", 2000.0, [], [], [], 1, is_transfer=True),
        OpRecord("ttnn.matmul", 5000.0, [], [], [], 2),
    ]
    report = OpPerfReport(records=records, total_time_us=10000.0)
    hints = report.bottleneck_hints
    transfer_hints = [h for h in hints if "transfer" in h.message.lower()]
    assert len(transfer_hints) >= 1


def test_no_hints_balanced():
    """Balanced workload -- should produce fewer hints."""
    records = [
        OpRecord(f"ttnn.op_{i}", 100.0, [], [], [], i) for i in range(10)
    ]
    report = OpPerfReport(records=records, total_time_us=1000.0)
    high_hints = [h for h in report.bottleneck_hints if h.severity == "high"]
    assert len(high_hints) == 0


# ── Summary Output Tests ────────────────────────────────────────────────


def test_summary_contains_header():
    report = _make_report()
    text = report.summary()
    assert "TTNN Op Performance Summary" in text


def test_summary_contains_total_time():
    report = _make_report()
    text = report.summary()
    assert "Total profiled time" in text


def test_summary_contains_top_ops():
    report = _make_report()
    text = report.summary()
    assert "ttnn.matmul" in text


def test_summary_contains_transfer_info():
    report = _make_report()
    text = report.summary()
    assert "Transfer time" in text


def test_str_calls_summary():
    report = _make_report()
    assert str(report) == report.summary()


def test_repr():
    report = _make_report()
    r = repr(report)
    assert "OpPerfReport" in r
    assert "ops=10" in r


# ── JSON Export Tests ────────────────────────────────────────────────────


def test_to_json_valid():
    report = _make_report()
    text = report.to_json()
    data = json.loads(text)
    assert "summary" in data
    assert "top_ops" in data
    assert "records" in data
    assert "bottleneck_hints" in data


def test_to_json_summary_fields():
    report = _make_report()
    data = json.loads(report.to_json())
    s = data["summary"]
    assert s["total_ops"] == 10
    assert s["total_time_us"] > 0
    assert s["transfer_pct"] > 0


def test_to_json_records_match():
    report = _make_report()
    data = json.loads(report.to_json())
    assert len(data["records"]) == len(report.records)


def test_to_json_file(tmp_path):
    report = _make_report()
    out = str(tmp_path / "report.json")
    report.to_json(out)
    assert os.path.exists(out)
    data = json.loads(open(out).read())
    assert data["summary"]["total_ops"] == 10


# ── CSV Export Tests ─────────────────────────────────────────────────────


def test_to_csv_file(tmp_path):
    report = _make_report()
    out = str(tmp_path / "report.csv")
    report.to_csv(out)
    assert os.path.exists(out)
    with open(out) as f:
        lines = f.readlines()
    # Header + 10 records
    assert len(lines) == 11


def test_to_csv_header(tmp_path):
    report = _make_report()
    out = str(tmp_path / "report.csv")
    report.to_csv(out)
    with open(out) as f:
        header = f.readline().strip()
    assert "call_index" in header
    assert "op_name" in header
    assert "duration_us" in header


# ── PerfTrace Initialization Tests ───────────────────────────────────────


def test_perf_trace_init():
    trace = PerfTrace()
    assert trace._records == []
    assert trace._call_counter == 0


def test_perf_trace_report_empty():
    trace = PerfTrace()
    report = trace.report()
    assert len(report.records) == 0
    assert report.total_time_us == 0


# ── Edge Cases ───────────────────────────────────────────────────────────


def test_empty_report():
    report = OpPerfReport(records=[], total_time_us=0.0)
    assert len(report.aggregated) == 0
    assert len(report.top_ops()) == 0
    assert report.transfer_pct == 0.0
    assert report.summary() is not None


def test_single_op_report():
    records = [OpRecord("ttnn.matmul", 1000.0, [], [], [], 0)]
    report = OpPerfReport(records=records, total_time_us=1000.0)
    assert len(report.aggregated) == 1
    assert report.aggregated[0].pct_of_total == pytest.approx(100.0)


def test_aggregated_op_total_ms():
    agg = AggregatedOp("ttnn.matmul", 3, 3000.0, 900.0, 1100.0, 1000.0, 50.0)
    assert agg.total_ms == 3.0


def test_bottleneck_hint_fields():
    h = BottleneckHint("high", "test message", "ttnn.matmul", "test detail")
    assert h.severity == "high"
    assert h.message == "test message"
    assert h.op_name == "ttnn.matmul"
    assert h.detail == "test detail"
