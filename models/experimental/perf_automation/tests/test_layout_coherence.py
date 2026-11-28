# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Inc-6 — layout visibility + coherence lever (no hardware)."""

import csv

from agent.opclass import is_layout_conversion
from agent.tracy_tool import build_buckets, stack_report


def test_is_layout_conversion_classifier():
    assert is_layout_conversion("TilizeDeviceOperation")
    assert is_layout_conversion("Untilize 1024 x 6144")
    assert is_layout_conversion("TilizeWithValPaddingDeviceOperation")
    assert not is_layout_conversion("MatmulDeviceOperation", "TILE", "TILE")
    assert not is_layout_conversion("LayerNormDeviceOperation", "TILE", "TILE")
    assert is_layout_conversion("ReshapeViewDeviceOperation", "ROW_MAJOR", "TILE")
    assert not is_layout_conversion("MatmulDeviceOperation", "", "")


def _write_csvs(tmp_path, rows):
    """rows: list of (gcc, op_code, device_us, in_layout, out_layout)."""
    report = tmp_path / "report.csv"
    raw = tmp_path / "raw.csv"
    with open(report, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["OP Code", "Global Call Count", "Device Time", "Cores", "Bound", "Op-to-Op Gap"]
        )
        w.writeheader()
        for gcc, op, us, _il, _ol in rows:
            w.writerow(
                {
                    "OP Code": op,
                    "Global Call Count": gcc,
                    "Device Time": us,
                    "Cores": 64,
                    "Bound": "",
                    "Op-to-Op Gap": "",
                }
            )
    with open(raw, "w", newline="") as f:
        cols = [
            "OP CODE",
            "GLOBAL CALL COUNT",
            "INPUT_0_LAYOUT",
            "OUTPUT_0_LAYOUT",
            "MATH FIDELITY",
            "INPUT_0_MEMORY",
            "ATTRIBUTES",
        ]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for gcc, op, _us, il, ol in rows:
            w.writerow(
                {
                    "OP CODE": op,
                    "GLOBAL CALL COUNT": gcc,
                    "INPUT_0_LAYOUT": il,
                    "OUTPUT_0_LAYOUT": ol,
                    "MATH FIDELITY": "HiFi4",
                    "INPUT_0_MEMORY": "DRAM_INTERLEAVED",
                    "ATTRIBUTES": "",
                }
            )
    return report, raw


def test_build_buckets_attributes_layout_churn(tmp_path):
    rows = [
        (1, "MatmulDeviceOperation", 10000.0, "TILE", "TILE"),
        (2, "MatmulDeviceOperation", 10000.0, "TILE", "TILE"),
        (3, "TilizeDeviceOperation", 5000.0, "ROW_MAJOR", "TILE"),
        (4, "TilizeDeviceOperation", 5000.0, "ROW_MAJOR", "TILE"),
        (5, "UntilizeDeviceOperation", 5000.0, "TILE", "ROW_MAJOR"),
    ]
    report, raw = _write_csvs(tmp_path, rows)
    buckets = build_buckets(report, raw)
    by_id = {b["id"]: b for b in buckets}
    assert by_id["datamove"]["layout_churn_count"] == 3
    assert abs(by_id["datamove"]["layout_churn_ms"] - 15.0) < 1e-6
    assert by_id["matmul"]["layout_churn_count"] == 0
    assert by_id["matmul"]["layout_churn_ms"] == 0.0


def test_stack_report_surfaces_churn():
    buckets = [
        {
            "id": "datamove",
            "device_ms": 15.0,
            "pct": 60.0,
            "count": 3,
            "tags": {"op_class": "datamove"},
            "layout_churn_ms": 15.0,
            "layout_churn_count": 3,
        },
        {
            "id": "matmul",
            "device_ms": 10.0,
            "pct": 40.0,
            "count": 2,
            "tags": {"op_class": "matmul"},
            "layout_churn_ms": 0.0,
            "layout_churn_count": 0,
        },
    ]
    report = stack_report(buckets, {"count": 3, "device_ms": 15.0, "pct_device": 60.0})
    assert "layout-churn 3×" in report
    assert "layout coherence:" in report and "#layout-coherence" in report


def test_router_indexes_coherence_lever():
    from agent.router import build_index, route

    idx = build_index()
    hits = route(idx, {"op_class": "datamove", "rank": "time"})
    assert any(e["id"] == "layout-coherence" for e in hits)
