# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import json
import sqlite3
import sys
from pathlib import Path

# Add local ttnn to path for graph_report module without importing ttnn runtime.
_local_ttnn_path = str(Path(__file__).parent.parent.parent.parent.parent / "ttnn" / "ttnn")
sys.path.insert(0, _local_ttnn_path)

import graph_report


def _make_report(graph, devices=None, per_operation_buffers=None, python_io=None, metadata=None):
    md = dict(metadata) if metadata is not None else {}
    md.setdefault("rank", 0)
    report = {"version": 1, "graph": graph, "devices": devices or [], "metadata": md}
    if per_operation_buffers is not None:
        report["per_operation_buffers"] = per_operation_buffers
    if python_io is not None:
        report["python_io"] = python_io
    return report


def _import_to_db(report_dict, tmp_path):
    report_path = tmp_path / "report.json"
    with open(report_path, "w") as f:
        json.dump(report_dict, f)
    db_path = graph_report.import_report(report_path, tmp_path / "output")
    conn = sqlite3.connect(db_path)
    return conn, conn.cursor()


def test_device_tensors_keep_same_address_on_different_devices(tmp_path):
    """Importer must keep distinct device placements even when addresses collide."""
    mock_graph = [
        {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [1, 3]},
        {
            "counter": 1,
            "node_type": "function_start",
            "params": {"name": "ttnn::all_gather", "inputs": "1"},
            "connections": [],
            "input_tensors": [],
        },
        {
            "counter": 2,
            "node_type": "tensor",
            "params": {
                "tensor_id": "101",
                "shape": "[1, 32, 32]",
                "dtype": "bfloat16",
                "layout": "TILE",
                "device_id": "0",
                "address": "12345678",
                "buffer_type": "1",
                "device_tensors": '[{"device_id": 0, "address": 5555}, {"device_id": 1, "address": 5555}]',
            },
            "connections": [],
        },
        {"counter": 3, "node_type": "capture_end", "params": {}, "connections": []},
    ]

    report = _make_report(mock_graph)
    conn, cursor = _import_to_db(report, tmp_path)

    cursor.execute("SELECT tensor_id, device_id, address FROM device_tensors ORDER BY device_id")
    rows = cursor.fetchall()

    assert rows == [(101, 0, 5555), (101, 1, 5555)]
    conn.close()


def test_device_tensors_still_dedup_exact_duplicate_quads(tmp_path):
    """Importer should still collapse exact duplicate (tensor_id, device_id, address, rank) rows."""
    mock_graph = [
        {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [1, 3]},
        {
            "counter": 1,
            "node_type": "function_start",
            "params": {"name": "ttnn::all_gather", "inputs": "1"},
            "connections": [],
            "input_tensors": [],
        },
        {
            "counter": 2,
            "node_type": "tensor",
            "params": {
                "tensor_id": "102",
                "shape": "[1, 32, 32]",
                "dtype": "bfloat16",
                "layout": "TILE",
                "device_id": "0",
                "address": "12345678",
                "buffer_type": "1",
                "device_tensors": '[{"device_id": 0, "address": 7777}, {"device_id": 0, "address": 7777}]',
            },
            "connections": [],
        },
        {"counter": 3, "node_type": "capture_end", "params": {}, "connections": []},
    ]

    report = _make_report(mock_graph)
    conn, cursor = _import_to_db(report, tmp_path)

    cursor.execute("SELECT tensor_id, device_id, address FROM device_tensors")
    rows = cursor.fetchall()

    assert rows == [(102, 0, 7777)]
    conn.close()
