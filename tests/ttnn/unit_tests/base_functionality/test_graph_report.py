# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the C++-based graph capture and offline import workflow.

This tests the decoupled workflow:
1. C++ captures graph to JSON file
2. Offline import to SQLite for visualization
"""

import json
import sqlite3
import sys
from pathlib import Path

import pytest
import torch

# Add local ttnn to path for graph_report module (before importing ttnn)
_local_ttnn_path = str(Path(__file__).parent.parent.parent.parent.parent / "ttnn" / "ttnn")
sys.path.insert(0, _local_ttnn_path)

# Import graph_report directly (doesn't need ttnn._ttnn)
import graph_report

# Now import ttnn for device tests
import ttnn
from models.common.utility_functions import is_wormhole_b0


@pytest.fixture
def tmp_report_dir(tmp_path):
    """Create a temporary directory for reports."""
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    return report_dir


def _make_report(
    graph,
    devices=None,
    per_operation_buffers=None,
    buffer_pages_by_address=None,
    python_io=None,
    metadata=None,
):
    """Build a complete JSON report dict matching C++ output format."""
    report = {"version": 1, "graph": graph, "devices": devices or [], "metadata": metadata or {}}
    if per_operation_buffers is not None:
        report["per_operation_buffers"] = per_operation_buffers
    if buffer_pages_by_address is not None:
        report["buffer_pages_by_address"] = buffer_pages_by_address
    if python_io is not None:
        report["python_io"] = python_io
    return report


def _import_to_db(report_dict, tmp_path):
    """Write report JSON, import via import_report(), return (connection, cursor)."""
    report_path = tmp_path / "report.json"
    with open(report_path, "w") as f:
        json.dump(report_dict, f)
    db_path = graph_report.import_report(report_path, tmp_path / "output")
    conn = sqlite3.connect(db_path)
    return conn, conn.cursor()


class TestImportGraphUnit:
    """Pure unit tests for import_graph function - no device required."""

    def test_output_tensors_extracted_from_function_end(self, tmp_path):
        """Test that output tensors are extracted from function_end connections."""
        mock_graph = [
            {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [1, 5]},
            {
                "counter": 1,
                "node_type": "tensor",
                "params": {"tensor_id": "42", "shape": "[1,1,32,32]"},
                "connections": [],
            },
            {
                "counter": 2,
                "node_type": "function_start",
                "params": {"name": "ttnn::relu", "inputs": "1"},
                "connections": [],
                "input_tensors": [1],
            },
            {
                "counter": 3,
                "node_type": "tensor",
                "params": {"tensor_id": "101", "shape": "[1,1,32,32]"},
                "connections": [],
            },
            {
                "counter": 4,
                "node_type": "function_end",
                "params": {"name": "ttnn::relu"},
                "connections": [3],
                "duration_ns": 1000,
            },
            {"counter": 5, "node_type": "capture_end", "params": {}, "connections": []},
        ]

        report = _make_report(mock_graph)
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute("SELECT operation_id, output_index, tensor_id FROM output_tensors")
        output_rows = cursor.fetchall()

        assert len(output_rows) == 1, f"Expected 1 output tensor, got {len(output_rows)}"
        assert str(output_rows[0][2]) == "101", f"Expected tensor_id '101', got {output_rows[0][2]}"

        # input_tensors[1] points to node counter 1 (tensor with tensor_id="42")
        cursor.execute("SELECT operation_id, input_index, tensor_id FROM input_tensors")
        input_rows = cursor.fetchall()
        assert len(input_rows) == 1, f"Expected 1 input tensor, got {len(input_rows)}"
        assert input_rows[0][2] == 42, f"Expected tensor_id 42 (resolved from node 1), got {input_rows[0][2]}"

        conn.close()

    def test_multiple_output_tensors(self, tmp_path):
        """Test function_end with multiple output tensor connections."""
        mock_graph = [
            {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [1, 5]},
            {
                "counter": 1,
                "node_type": "function_start",
                "params": {"name": "ttnn::split", "inputs": "1"},
                "connections": [],
                "input_tensors": [],
            },
            {
                "counter": 2,
                "node_type": "tensor",
                "params": {"tensor_id": "201", "shape": "[1,1,16,32]"},
                "connections": [],
            },
            {
                "counter": 3,
                "node_type": "tensor",
                "params": {"tensor_id": "202", "shape": "[1,1,16,32]"},
                "connections": [],
            },
            {
                "counter": 4,
                "node_type": "function_end",
                "params": {"name": "ttnn::split"},
                "connections": [2, 3],
                "duration_ns": 2000,
            },
            {"counter": 5, "node_type": "capture_end", "params": {}, "connections": []},
        ]

        report = _make_report(mock_graph)
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute("SELECT tensor_id FROM output_tensors ORDER BY output_index")
        output_rows = cursor.fetchall()

        assert len(output_rows) == 2, f"Expected 2 output tensors, got {len(output_rows)}"
        assert str(output_rows[0][0]) == "201"
        assert str(output_rows[1][0]) == "202"

        conn.close()

    def test_stack_traces_imported(self, tmp_path):
        """Test that stack traces are imported when present."""
        mock_graph = [
            {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [1, 3]},
            {
                "counter": 1,
                "node_type": "function_start",
                "params": {"name": "ttnn::relu", "inputs": "1"},
                "connections": [],
                "input_tensors": [],
                "stack_trace": ["frame1", "frame2", "frame3"],
            },
            {
                "counter": 2,
                "node_type": "function_end",
                "params": {"name": "ttnn::relu"},
                "connections": [],
                "duration_ns": 1000,
            },
            {"counter": 3, "node_type": "capture_end", "params": {}, "connections": []},
        ]

        report = _make_report(mock_graph)
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute("SELECT operation_id, stack_trace FROM stack_traces")
        rows = cursor.fetchall()

        assert len(rows) == 1
        assert rows[0][0] == 1
        assert "frame1" in rows[0][1]

        conn.close()

    def test_error_nodes_imported(self, tmp_path):
        """Test that error nodes are imported."""
        mock_graph = [
            {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [1, 3]},
            {
                "counter": 1,
                "node_type": "function_start",
                "params": {"name": "ttnn::bad_op", "inputs": "1"},
                "connections": [2],
                "input_tensors": [],
            },
            {
                "counter": 2,
                "node_type": "error",
                "params": {
                    "error_type": "exception",
                    "error_message": "Something went wrong",
                    "error_operation": "ttnn::bad_op",
                },
                "connections": [],
            },
            {"counter": 3, "node_type": "capture_end", "params": {}, "connections": []},
        ]

        report = _make_report(mock_graph)
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute("SELECT error_type, error_message, error_operation FROM errors")
        rows = cursor.fetchall()

        assert len(rows) == 1
        assert rows[0][0] == "exception"
        assert rows[0][1] == "Something went wrong"
        assert rows[0][2] == "ttnn::bad_op"

        conn.close()

    def test_operation_arguments_imported(self, tmp_path):
        """Test that operation arguments are imported from function_start."""
        mock_graph = [
            {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [1, 3]},
            {
                "counter": 1,
                "node_type": "function_start",
                "params": {"name": "ttnn::add", "inputs": "2"},
                "connections": [],
                "input_tensors": [],
                "arguments": ["tensor_a", "tensor_b", "alpha=1.0"],
            },
            {
                "counter": 2,
                "node_type": "function_end",
                "params": {"name": "ttnn::add"},
                "connections": [],
                "duration_ns": 5000,
            },
            {"counter": 3, "node_type": "capture_end", "params": {}, "connections": []},
        ]

        report = _make_report(mock_graph)
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute("SELECT name, value FROM operation_arguments ORDER BY name")
        rows = cursor.fetchall()

        assert len(rows) == 3
        # arg_0, arg_1, arg_2
        values = [r[1] for r in rows]
        assert "tensor_a" in values
        assert "tensor_b" in values
        assert "alpha=1.0" in values

        conn.close()

    def test_buffers_imported(self, tmp_path):
        """Test that buffer allocations are imported."""
        mock_graph = [
            {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [1, 4]},
            {
                "counter": 1,
                "node_type": "function_start",
                "params": {"name": "ttnn::relu", "inputs": "1"},
                "connections": [],
                "input_tensors": [],
            },
            {
                "counter": 2,
                "node_type": "buffer_allocate",
                "params": {"device_id": "0", "address": "12345", "size": "4096", "type": "L1", "layout": "INTERLEAVED"},
                "connections": [],
            },
            {
                "counter": 3,
                "node_type": "function_end",
                "params": {"name": "ttnn::relu"},
                "connections": [],
                "duration_ns": 1000,
            },
            {"counter": 4, "node_type": "capture_end", "params": {}, "connections": []},
        ]

        report = _make_report(mock_graph)
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute("SELECT operation_id, device_id, address, max_size_per_bank, buffer_type FROM buffers")
        rows = cursor.fetchall()

        assert len(rows) == 1
        assert rows[0][0] == 1  # operation_id
        assert rows[0][1] == 0  # device_id
        assert rows[0][2] == 12345  # address
        assert rows[0][3] == 4096  # size
        assert rows[0][4] == 1  # buffer_type (1=L1)

        conn.close()

    def test_buffer_type_mapping_all_types(self, tmp_path):
        """All buffer types (DRAM, L1, L1_SMALL, SYSTEM_MEMORY, TRACE) map to correct integers."""
        type_map = {"DRAM": 0, "L1": 1, "SYSTEM_MEMORY": 2, "L1_SMALL": 3, "TRACE": 4}
        nodes = [
            {
                "counter": 0,
                "node_type": "capture_start",
                "params": {},
                "connections": list(range(1, 1 + 3 * len(type_map), 3)),
            },
        ]
        counter = 1
        for type_name in type_map:
            nodes.append(
                {
                    "counter": counter,
                    "node_type": "function_start",
                    "params": {"name": f"ttnn::op_{type_name}", "inputs": "0"},
                    "connections": [],
                    "input_tensors": [],
                }
            )
            nodes.append(
                {
                    "counter": counter + 1,
                    "node_type": "buffer_allocate",
                    "params": {
                        "device_id": "0",
                        "address": str(counter * 10000),
                        "size": "4096",
                        "page_size": "1024",
                        "type": type_name,
                        "layout": "INTERLEAVED",
                    },
                    "connections": [],
                }
            )
            nodes.append(
                {
                    "counter": counter + 2,
                    "node_type": "function_end",
                    "params": {"name": f"ttnn::op_{type_name}"},
                    "connections": [],
                    "duration_ns": 100,
                }
            )
            counter += 3
        nodes.append({"counter": counter, "node_type": "capture_end", "params": {}, "connections": []})

        report = _make_report(nodes)
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute("SELECT address, buffer_type FROM buffers ORDER BY address")
        rows = cursor.fetchall()
        addr_to_type = {addr: bt for addr, bt in rows}

        for i, (type_name, expected_int) in enumerate(type_map.items()):
            addr = (1 + i * 3) * 10000
            actual_val = addr_to_type.get(addr)
            assert (
                actual_val == expected_int
            ), f"Buffer type '{type_name}' at addr {addr}: expected {expected_int}, got {actual_val}"

        conn.close()

    def test_l1_small_buffer_type_not_collapsed_to_dram(self, tmp_path):
        """Regression: L1_SMALL must map to 3, not 0 (DRAM)."""
        mock_graph = [
            {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [1]},
            {
                "counter": 1,
                "node_type": "function_start",
                "params": {"name": "ttnn::relu", "inputs": "0"},
                "connections": [],
                "input_tensors": [],
            },
            {
                "counter": 2,
                "node_type": "buffer_allocate",
                "params": {
                    "device_id": "0",
                    "address": "99999",
                    "size": "2048",
                    "page_size": "512",
                    "type": "L1_SMALL",
                    "layout": "INTERLEAVED",
                },
                "connections": [],
            },
            {
                "counter": 3,
                "node_type": "function_end",
                "params": {"name": "ttnn::relu"},
                "connections": [],
                "duration_ns": 100,
            },
            {"counter": 4, "node_type": "capture_end", "params": {}, "connections": []},
        ]

        report = _make_report(mock_graph)
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute("SELECT buffer_type FROM buffers WHERE address=99999")
        (bt,) = cursor.fetchone()
        assert bt == 3, f"buffer_type should be 3 (L1_SMALL), got {bt}"

        conn.close()

    def test_mixed_buffer_types_cumulative(self, tmp_path):
        """Cumulative snapshots preserve correct buffer_type for mixed DRAM/L1/L1_SMALL."""
        mock_graph = [
            {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [1, 5]},
            # Op 1: allocates DRAM + L1_SMALL
            {
                "counter": 1,
                "node_type": "function_start",
                "params": {"name": "ttnn::op_a", "inputs": "0"},
                "connections": [],
                "input_tensors": [],
            },
            {
                "counter": 2,
                "node_type": "buffer_allocate",
                "params": {
                    "device_id": "0",
                    "address": "1000",
                    "size": "8192",
                    "page_size": "1024",
                    "type": "DRAM",
                    "layout": "INTERLEAVED",
                },
                "connections": [],
            },
            {
                "counter": 3,
                "node_type": "buffer_allocate",
                "params": {
                    "device_id": "0",
                    "address": "2000",
                    "size": "512",
                    "page_size": "256",
                    "type": "L1_SMALL",
                    "layout": "INTERLEAVED",
                },
                "connections": [],
            },
            {
                "counter": 4,
                "node_type": "function_end",
                "params": {"name": "ttnn::op_a"},
                "connections": [],
                "duration_ns": 100,
            },
            # Op 2: allocates L1 (cumulative snapshot should have all 3)
            {
                "counter": 5,
                "node_type": "function_start",
                "params": {"name": "ttnn::op_b", "inputs": "0"},
                "connections": [],
                "input_tensors": [],
            },
            {
                "counter": 6,
                "node_type": "buffer_allocate",
                "params": {
                    "device_id": "0",
                    "address": "3000",
                    "size": "4096",
                    "page_size": "512",
                    "type": "L1",
                    "layout": "INTERLEAVED",
                },
                "connections": [],
            },
            {
                "counter": 7,
                "node_type": "function_end",
                "params": {"name": "ttnn::op_b"},
                "connections": [],
                "duration_ns": 200,
            },
            {"counter": 8, "node_type": "capture_end", "params": {}, "connections": []},
        ]

        report = _make_report(mock_graph)
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute("SELECT address, buffer_type FROM buffers WHERE operation_id=2 ORDER BY address")
        rows = cursor.fetchall()
        assert len(rows) == 3, f"Op 2 should have 3 cumulative buffers, got {len(rows)}"
        type_by_addr = {addr: bt for addr, bt in rows}
        assert type_by_addr[1000] == 0, f"DRAM buffer should be 0, got {type_by_addr[1000]}"
        assert type_by_addr[2000] == 3, f"L1_SMALL buffer should be 3, got {type_by_addr[2000]}"
        assert type_by_addr[3000] == 1, f"L1 buffer should be 1, got {type_by_addr[3000]}"

        conn.close()

    def test_buffers_cumulative_per_operation(self, tmp_path):
        """Buffers must be cumulative: each operation includes all currently-allocated buffers."""
        mock_graph = [
            {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [1, 10]},
            {
                "counter": 1,
                "node_type": "function_start",
                "params": {"name": "op_A", "inputs": "0"},
                "connections": [],
                "input_tensors": [],
            },
            {
                "counter": 2,
                "node_type": "buffer_allocate",
                "params": {
                    "device_id": "0",
                    "address": "1000",
                    "size": "4096",
                    "type": "DRAM",
                    "layout": "INTERLEAVED",
                },
                "connections": [],
            },
            {
                "counter": 3,
                "node_type": "function_end",
                "params": {"name": "op_A"},
                "connections": [],
                "duration_ns": 100,
            },
            {
                "counter": 4,
                "node_type": "function_start",
                "params": {"name": "op_B", "inputs": "0"},
                "connections": [],
                "input_tensors": [],
            },
            {
                "counter": 5,
                "node_type": "buffer_allocate",
                "params": {
                    "device_id": "0",
                    "address": "2000",
                    "size": "8192",
                    "type": "DRAM",
                    "layout": "INTERLEAVED",
                },
                "connections": [],
            },
            {
                "counter": 6,
                "node_type": "function_end",
                "params": {"name": "op_B"},
                "connections": [],
                "duration_ns": 200,
            },
            {
                "counter": 7,
                "node_type": "function_start",
                "params": {"name": "op_C", "inputs": "0"},
                "connections": [],
                "input_tensors": [],
            },
            {
                "counter": 8,
                "node_type": "buffer_allocate",
                "params": {"device_id": "0", "address": "3000", "size": "2048", "type": "L1", "layout": "INTERLEAVED"},
                "connections": [],
            },
            {
                "counter": 9,
                "node_type": "function_end",
                "params": {"name": "op_C"},
                "connections": [],
                "duration_ns": 300,
            },
            {"counter": 10, "node_type": "capture_end", "params": {}, "connections": []},
        ]

        report = _make_report(mock_graph)
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute("SELECT operation_id, address FROM buffers ORDER BY operation_id, address")
        rows = cursor.fetchall()

        op_1_buffers = [r[1] for r in rows if r[0] == 1]
        op_2_buffers = [r[1] for r in rows if r[0] == 2]
        op_3_buffers = [r[1] for r in rows if r[0] == 3]

        assert op_1_buffers == [1000], f"op_A should have 1 buffer, got {op_1_buffers}"
        assert op_2_buffers == [1000, 2000], f"op_B should have 2 cumulative buffers, got {op_2_buffers}"
        assert op_3_buffers == [1000, 2000, 3000], f"op_C should have 3 cumulative buffers, got {op_3_buffers}"

        assert len(rows) == 6  # 1 + 2 + 3
        conn.close()

    def test_device_tensors_imported(self, tmp_path):
        """Test that device_tensors are imported for multi-device tensors."""
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
                    "tensor_id": "100",
                    "shape": "[1, 32, 32]",
                    "dtype": "bfloat16",
                    "layout": "TILE",
                    "device_id": "0",
                    "address": "12345678",
                    "buffer_type": "L1",
                    "device_tensors": '[{"device_id": 0, "address": 12345678}, {"device_id": 1, "address": 22345678}, {"device_id": 2, "address": 32345678}, {"device_id": 3, "address": 42345678}]',
                },
                "connections": [],
            },
            {"counter": 3, "node_type": "capture_end", "params": {}, "connections": []},
        ]

        report = _make_report(mock_graph)
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute("SELECT tensor_id, device_id, address FROM device_tensors ORDER BY device_id")
        rows = cursor.fetchall()

        assert len(rows) == 4, f"Expected 4 device tensor entries, got {len(rows)}"
        assert rows[0] == (100, 0, 12345678)
        assert rows[1] == (100, 1, 22345678)
        assert rows[2] == (100, 2, 32345678)
        assert rows[3] == (100, 3, 42345678)

        conn.close()

    def test_device_tensors_prefers_mesh_device_id(self, tmp_path):
        """Test that mesh_device_id is used over device_id when present in device_tensors JSON."""
        mock_graph = [
            {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [1, 3]},
            {
                "counter": 1,
                "node_type": "function_start",
                "params": {"name": "ttnn::to_device", "inputs": "1"},
                "connections": [],
                "input_tensors": [],
            },
            {
                "counter": 2,
                "node_type": "tensor",
                "params": {
                    "tensor_id": "50",
                    "shape": "[1024, 1024]",
                    "dtype": "bfloat16",
                    "layout": "TILE",
                    "device_id": "1",
                    "address": "1920032",
                    "buffer_type": "DRAM",
                    "device_tensors": '[{"device_id": 0, "mesh_device_id": 1, "address": 1920032}]',
                },
                "connections": [],
            },
            {"counter": 3, "node_type": "capture_end", "params": {}, "connections": []},
        ]

        report = _make_report(mock_graph)
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute("SELECT tensor_id, device_id, address FROM device_tensors")
        rows = cursor.fetchall()

        assert len(rows) == 1
        assert rows[0] == (50, 1, 1920032), f"Should use mesh_device_id (1) not physical device_id (0). Got {rows[0]}"
        conn.close()

    def test_full_tensor_info_imported(self, tmp_path):
        """Test that full tensor info (dtype, layout, memory_config, device_id, address, buffer_type) is imported."""
        mock_graph = [
            {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [1, 3]},
            {
                "counter": 1,
                "node_type": "function_start",
                "params": {"name": "ttnn::relu", "inputs": "1"},
                "connections": [],
                "input_tensors": [],
            },
            {
                "counter": 2,
                "node_type": "tensor",
                "params": {
                    "tensor_id": "100",
                    "shape": "[1, 32, 32]",
                    "dtype": "bfloat16",
                    "layout": "TILE",
                    "memory_config": "MemoryConfig(DRAM, INTERLEAVED)",
                    "device_id": "0",
                    "address": "12345678",
                    "buffer_type": "BufferType::DRAM",
                    "buffer_type_value": "0",
                    "size": "2048",
                },
                "connections": [],
            },
            {"counter": 3, "node_type": "capture_end", "params": {}, "connections": []},
        ]

        report = _make_report(mock_graph)
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute(
            "SELECT tensor_id, shape, dtype, layout, memory_config, device_id, address, buffer_type, size FROM tensors"
        )
        rows = cursor.fetchall()

        assert len(rows) == 1
        row = rows[0]
        assert str(row[0]) == "100"  # tensor_id
        assert row[1] == "[1, 32, 32]"  # shape
        assert row[2] == "bfloat16"  # dtype
        assert row[3] == "TILE"  # layout
        assert "DRAM" in row[4]  # memory_config
        assert row[5] == 0  # device_id
        assert row[6] == 12345678  # address
        assert row[7] == 0  # buffer_type (0=DRAM)
        assert row[8] == 2048  # size in bytes

        conn.close()

    def test_edges_populated_from_captured_graph(self, tmp_path):
        """Test that edges are extracted from captured_graph subgraphs."""
        mock_graph = [
            {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [1, 4]},
            {
                "counter": 1,
                "node_type": "function_start",
                "params": {"name": "ttnn::relu", "inputs": "1"},
                "connections": [2],
                "input_tensors": [],
            },
            {
                "counter": 2,
                "node_type": "tensor",
                "params": {"tensor_id": "100", "shape": "[1,32,32]"},
                "connections": [3],
            },
            {
                "counter": 3,
                "node_type": "function_end",
                "params": {"name": "ttnn::relu"},
                "connections": [4],
                "duration_ns": 1000,
            },
            {"counter": 4, "node_type": "capture_end", "params": {}, "connections": []},
        ]

        report = _make_report(mock_graph)
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute("SELECT COUNT(*) FROM edges")
        assert cursor.fetchone()[0] > 0

        conn.close()

    def test_buffer_pages_imported(self, tmp_path):
        """Test that buffer pages are imported when present in the report."""
        graph = [
            {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [1]},
            {"counter": 1, "node_type": "capture_end", "params": {}, "connections": []},
        ]
        buffer_pages = [
            {
                "device_id": 0,
                "address": 12345678,
                "core_y": 1,
                "core_x": 2,
                "bank_id": 5,
                "page_index": 0,
                "page_address": 12345678,
                "page_size": 1024,
                "buffer_type": 1,
            },
            {
                "device_id": 0,
                "address": 12345678,
                "core_y": 1,
                "core_x": 3,
                "bank_id": 6,
                "page_index": 1,
                "page_address": 12346702,
                "page_size": 1024,
                "buffer_type": 1,
            },
        ]
        report = _make_report(graph)
        report["buffer_pages"] = buffer_pages
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute(
            "SELECT device_id, address, core_y, core_x, bank_id, page_index, page_address, page_size, buffer_type FROM buffer_pages ORDER BY page_index"
        )
        rows = cursor.fetchall()

        assert len(rows) == 2
        # First page
        assert rows[0][0] == 0  # device_id
        assert rows[0][2] == 1  # core_y
        assert rows[0][3] == 2  # core_x
        assert rows[0][4] == 5  # bank_id
        assert rows[0][5] == 0  # page_index
        assert rows[0][7] == 1024  # page_size
        assert rows[0][8] == 1  # buffer_type (L1)
        # Second page
        assert rows[1][3] == 3  # core_x
        assert rows[1][5] == 1  # page_index

    def test_per_operation_buffer_pages_imported(self, tmp_path):
        """Test that per-operation buffer page snapshots are imported with correct operation_id mapping."""
        graph = [
            {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [1, 5], "stacking_level": 0},
            {
                "counter": 1,
                "node_type": "function_start",
                "params": {"name": "ttnn.add", "num_inputs": "0"},
                "connections": [2],
                "input_tensors": [],
                "stacking_level": 1,
                "arguments": [],
            },
            {
                "counter": 2,
                "node_type": "function_end",
                "params": {"name": "ttnn.add"},
                "connections": [],
                "stacking_level": 1,
                "duration_ns": 100,
            },
            {
                "counter": 3,
                "node_type": "function_start",
                "params": {"name": "ttnn.mul", "num_inputs": "0"},
                "connections": [4],
                "input_tensors": [],
                "stacking_level": 1,
                "arguments": [],
            },
            {
                "counter": 4,
                "node_type": "function_end",
                "params": {"name": "ttnn.mul"},
                "connections": [],
                "stacking_level": 1,
                "duration_ns": 200,
            },
            {"counter": 5, "node_type": "capture_end", "params": {}, "connections": [], "stacking_level": 0},
        ]
        report = _make_report(graph)
        report["buffer_pages"] = [
            {
                "device_id": 0,
                "address": 999,
                "core_y": 0,
                "core_x": 0,
                "bank_id": 0,
                "page_index": 0,
                "page_address": 999,
                "page_size": 64,
                "buffer_type": 1,
            },
        ]
        report["per_operation_buffer_pages"] = {
            "1": [
                {
                    "device_id": 0,
                    "address": 1000,
                    "core_y": 1,
                    "core_x": 2,
                    "bank_id": 10,
                    "page_index": 0,
                    "page_address": 1000,
                    "page_size": 128,
                    "buffer_type": 1,
                },
                {
                    "device_id": 0,
                    "address": 1000,
                    "core_y": 1,
                    "core_x": 3,
                    "bank_id": 11,
                    "page_index": 1,
                    "page_address": 1128,
                    "page_size": 128,
                    "buffer_type": 1,
                },
            ],
            "3": [
                {
                    "device_id": 0,
                    "address": 2000,
                    "core_y": 2,
                    "core_x": 4,
                    "bank_id": 20,
                    "page_index": 0,
                    "page_address": 2000,
                    "page_size": 256,
                    "buffer_type": 1,
                },
            ],
        }

        conn, cursor = _import_to_db(report, tmp_path)

        # Flat buffer_pages are preferred over per-op snapshots
        cursor.execute("SELECT COUNT(*) FROM buffer_pages")
        total = cursor.fetchone()[0]
        assert total == 1, f"Expected 1 flat buffer page, got {total}"

        # Flat pages all share the same base operation_id
        cursor.execute("SELECT DISTINCT operation_id FROM buffer_pages ORDER BY operation_id")
        op_ids = [r[0] for r in cursor.fetchall()]
        assert len(op_ids) == 1, f"Expected 1 distinct operation_id for flat pages, got {len(op_ids)}"

        # Verify the flat page data was imported correctly
        cursor.execute("SELECT address, core_y, core_x, bank_id, page_size FROM buffer_pages ORDER BY page_index")
        pages = cursor.fetchall()
        assert len(pages) == 1
        assert pages[0] == (999, 0, 0, 0, 64)

        conn.close()

    def test_flat_buffer_pages_fallback(self, tmp_path):
        """Test that flat buffer_pages are used when per_operation_buffer_pages is absent."""
        graph = [
            {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [1], "stacking_level": 0},
            {"counter": 1, "node_type": "capture_end", "params": {}, "connections": [], "stacking_level": 0},
        ]
        report = _make_report(graph)
        report["buffer_pages"] = [
            {
                "device_id": 0,
                "address": 5000,
                "core_y": 0,
                "core_x": 0,
                "bank_id": 0,
                "page_index": 0,
                "page_address": 5000,
                "page_size": 512,
                "buffer_type": 1,
            },
            {
                "device_id": 0,
                "address": 5000,
                "core_y": 0,
                "core_x": 1,
                "bank_id": 1,
                "page_index": 1,
                "page_address": 5512,
                "page_size": 512,
                "buffer_type": 1,
            },
        ]
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute("SELECT COUNT(*) FROM buffer_pages")
        total = cursor.fetchone()[0]
        assert total == 2, f"Flat fallback should import 2 pages, got {total}"

        cursor.execute("SELECT page_size FROM buffer_pages ORDER BY page_index")
        sizes = [r[0] for r in cursor.fetchall()]
        assert sizes == [512, 512]

        conn.close()

    def test_cluster_mesh_descriptors_saved(self, tmp_path):
        """Test that cluster and mesh descriptors are saved during import."""
        graph = [
            {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [1]},
            {"counter": 1, "node_type": "capture_end", "params": {}, "connections": []},
        ]
        report = _make_report(graph)
        report["cluster_descriptor"] = "# Mock cluster descriptor YAML\ndevices:\n  - id: 0\n    type: wormhole\n"
        report["mesh_coordinate_mapping"] = "# physical_chip_mesh_coordinate_mapping_1_of_1.yaml\nchips:\n  0: [0, 0]\n"

        report_path = tmp_path / "report.json"
        with open(report_path, "w") as f:
            json.dump(report, f)
        output_dir = tmp_path / "output"
        graph_report.import_report(report_path, output_dir)

        # Check cluster descriptor was saved
        cluster_path = output_dir / "cluster_descriptor.yaml"
        assert cluster_path.exists(), "cluster_descriptor.yaml should be created"
        with open(cluster_path) as f:
            content = f.read()
        assert "wormhole" in content

        # Check mesh coordinate mapping was saved
        mesh_path = output_dir / "physical_chip_mesh_coordinate_mapping_1_of_1.yaml"
        assert mesh_path.exists(), "physical_chip_mesh_coordinate_mapping_1_of_1.yaml should be created"
        with open(mesh_path) as f:
            content = f.read()
        assert "chips" in content
        assert "0: [0, 0]" in content


class TestInputTensorResolution:
    """Regression tests: input_tensors must resolve node counters to real tensor_ids."""

    def test_input_tensor_resolves_node_counter_to_tensor_id(self, tmp_path):
        """
        Regression: input_tensors field contains node counter values, not tensor_ids.
        The import must look up the node and extract tensor_id from its params.
        """
        mock_graph = [
            {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [1, 5]},
            {
                "counter": 1,
                "node_type": "tensor",
                "params": {"tensor_id": "77", "shape": "[1,1,32,32]"},
                "connections": [],
            },
            {
                "counter": 2,
                "node_type": "function_start",
                "params": {"name": "ttnn::relu", "inputs": "1"},
                "connections": [],
                "input_tensors": [1],
            },
            {
                "counter": 3,
                "node_type": "function_end",
                "params": {"name": "ttnn::relu"},
                "connections": [],
                "duration_ns": 500,
            },
            {"counter": 4, "node_type": "capture_end", "params": {}, "connections": []},
        ]

        report = _make_report(mock_graph)
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute("SELECT tensor_id FROM input_tensors")
        rows = cursor.fetchall()
        assert len(rows) == 1
        assert (
            rows[0][0] == 77
        ), f"input_tensors should store the real tensor_id (77), not the node counter (1). Got {rows[0][0]}"
        conn.close()

    def test_input_tensor_id_exists_in_tensors_table(self, tmp_path):
        """
        Regression: all tensor_ids referenced in input_tensors must exist in the tensors table.
        The old bug stored node counter values which didn't match any tensor row.
        """
        mock_graph = [
            {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [2, 6]},
            {
                "counter": 1,
                "node_type": "tensor",
                "params": {"tensor_id": "10", "shape": "[1024,1024]"},
                "connections": [],
            },
            {
                "counter": 2,
                "node_type": "function_start",
                "params": {"name": "ttnn::matmul", "inputs": "1"},
                "connections": [],
                "input_tensors": [1],
            },
            {
                "counter": 3,
                "node_type": "tensor",
                "params": {"tensor_id": "20", "shape": "[1024,1024]"},
                "connections": [],
            },
            {
                "counter": 4,
                "node_type": "function_end",
                "params": {"name": "ttnn::matmul"},
                "connections": [3],
                "duration_ns": 1000,
            },
            {"counter": 5, "node_type": "capture_end", "params": {}, "connections": []},
        ]

        report = _make_report(mock_graph)
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute("SELECT tensor_id FROM tensors")
        tensor_ids = {row[0] for row in cursor.fetchall()}

        cursor.execute("SELECT tensor_id FROM input_tensors")
        for row in cursor.fetchall():
            assert row[0] in tensor_ids or str(row[0]) in tensor_ids, (
                f"input_tensors references tensor_id={row[0]} which doesn't exist in tensors table. "
                f"Available: {tensor_ids}"
            )
        conn.close()

    def test_multiple_input_tensors_all_resolved(self, tmp_path):
        """Test that all input tensors for an operation are correctly resolved."""
        mock_graph = [
            {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [1, 8]},
            {
                "counter": 1,
                "node_type": "tensor",
                "params": {"tensor_id": "5", "shape": "[1024,1024]"},
                "connections": [],
            },
            {
                "counter": 2,
                "node_type": "tensor",
                "params": {"tensor_id": "8", "shape": "[1024,1024]"},
                "connections": [],
            },
            {
                "counter": 3,
                "node_type": "tensor",
                "params": {"tensor_id": "11", "shape": "[1,1024]"},
                "connections": [],
            },
            {
                "counter": 4,
                "node_type": "function_start",
                "params": {"name": "ttnn::linear", "inputs": "3"},
                "connections": [],
                "input_tensors": [1, 2, 3],
            },
            {
                "counter": 5,
                "node_type": "tensor",
                "params": {"tensor_id": "20", "shape": "[1024,1024]"},
                "connections": [],
            },
            {
                "counter": 6,
                "node_type": "function_end",
                "params": {"name": "ttnn::linear"},
                "connections": [5],
                "duration_ns": 5000,
            },
            {"counter": 7, "node_type": "capture_end", "params": {}, "connections": []},
        ]

        report = _make_report(mock_graph)
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute("SELECT input_index, tensor_id FROM input_tensors ORDER BY input_index")
        rows = cursor.fetchall()

        assert len(rows) == 3, f"Expected 3 input tensors, got {len(rows)}"
        assert rows[0] == (0, 5), f"Input 0 should be tensor_id 5 (node 1), got {rows[0]}"
        assert rows[1] == (1, 8), f"Input 1 should be tensor_id 8 (node 2), got {rows[1]}"
        assert rows[2] == (2, 11), f"Input 2 should be tensor_id 11 (node 3), got {rows[2]}"
        conn.close()

    def test_invalid_node_counter_skipped(self, tmp_path):
        """Input tensor references to non-existent nodes are silently skipped."""
        mock_graph = [
            {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [1, 4]},
            {
                "counter": 1,
                "node_type": "function_start",
                "params": {"name": "ttnn::relu", "inputs": "1"},
                "connections": [],
                "input_tensors": [999],
            },
            {
                "counter": 2,
                "node_type": "function_end",
                "params": {"name": "ttnn::relu"},
                "connections": [],
                "duration_ns": 100,
            },
            {"counter": 3, "node_type": "capture_end", "params": {}, "connections": []},
        ]

        report = _make_report(mock_graph)
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute("SELECT * FROM input_tensors")
        assert cursor.fetchall() == [], "Invalid node counter should not produce input_tensor rows"
        conn.close()


class TestImportValidation:
    """Tests for the referential integrity validation in the importer."""

    def _make_db(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        graph_report.create_database_schema(cursor)
        return conn, cursor

    def test_clean_import_produces_no_warnings(self, tmp_path):
        """A well-formed graph should produce zero integrity warnings."""
        mock_graph = [
            {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [1, 5]},
            {
                "counter": 1,
                "node_type": "tensor",
                "params": {"tensor_id": "10", "shape": "[32,32]"},
                "connections": [],
            },
            {
                "counter": 2,
                "node_type": "function_start",
                "params": {"name": "ttnn::relu", "inputs": "1"},
                "connections": [],
                "input_tensors": [1],
            },
            {
                "counter": 3,
                "node_type": "tensor",
                "params": {"tensor_id": "20", "shape": "[32,32]"},
                "connections": [],
            },
            {
                "counter": 4,
                "node_type": "function_end",
                "params": {"name": "ttnn::relu"},
                "connections": [3],
                "duration_ns": 100,
            },
            {"counter": 5, "node_type": "capture_end", "params": {}, "connections": []},
        ]

        conn, cursor = self._make_db(tmp_path)
        stats = graph_report.import_graph(cursor, mock_graph)
        conn.commit()

        assert stats.get("warnings", []) == [], f"Clean import should have no warnings, got: {stats['warnings']}"
        conn.close()

    def test_dangling_input_tensor_produces_warning(self, tmp_path):
        """
        Simulate the old bug: manually insert an input_tensor row pointing to
        a nonexistent tensor_id and verify the validator catches it.
        """
        mock_graph = [
            {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [1, 3]},
            {
                "counter": 1,
                "node_type": "function_start",
                "params": {"name": "ttnn::relu", "inputs": "1"},
                "connections": [],
                "input_tensors": [],
            },
            {
                "counter": 2,
                "node_type": "function_end",
                "params": {"name": "ttnn::relu"},
                "connections": [],
                "duration_ns": 100,
            },
            {"counter": 3, "node_type": "capture_end", "params": {}, "connections": []},
        ]

        conn, cursor = self._make_db(tmp_path)
        graph_report.import_graph(cursor, mock_graph)
        conn.commit()

        # Manually insert a dangling reference (simulating the old bug)
        cursor.execute("INSERT INTO input_tensors VALUES (0, 0, 999)")
        conn.commit()

        # Run validation directly
        warnings = graph_report._validate_graph_integrity(
            operations_batch=[(0, "ttnn::relu", 0.0)],
            tensors_batch=[],
            input_tensors_batch=[(0, 0, 999)],
            output_tensors_batch=[],
            operation_arguments_batch=[],
            device_tensors_batch=[],
            buffers_batch=[],
            devices=None,
        )
        assert len(warnings) >= 1
        assert any("tensor_id=999" in w for w in warnings), f"Should warn about dangling tensor_id=999, got: {warnings}"
        conn.close()


class TestBufferMaxSizePerBank:
    """Regression tests: max_size_per_bank must be per-bank, not total buffer size."""

    @staticmethod
    def _make_buffer_graph(size, page_size, buf_type="DRAM", layout="INTERLEAVED", num_cores=0):
        return [
            {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [1, 4]},
            {
                "counter": 1,
                "node_type": "function_start",
                "params": {"name": "ttnn::to_device", "inputs": "1"},
                "connections": [],
                "input_tensors": [],
            },
            {
                "counter": 2,
                "node_type": "buffer_allocate",
                "params": {
                    "device_id": "0",
                    "address": "1000",
                    "size": str(size),
                    "page_size": str(page_size),
                    "num_cores": str(num_cores),
                    "type": buf_type,
                    "layout": layout,
                },
                "connections": [],
            },
            {
                "counter": 3,
                "node_type": "function_end",
                "params": {"name": "ttnn::to_device"},
                "connections": [],
                "duration_ns": 100,
            },
            {"counter": 4, "node_type": "capture_end", "params": {}, "connections": []},
        ]

    def test_dram_interleaved_per_bank_size(self, tmp_path):
        """
        Regression: a 1024x1024 bf16 TILE tensor is 2,097,152 bytes total.
        With 12 DRAM channels: ceil(1024 pages / 12 banks) * 2048 page_size = 176,128 per bank.
        """
        devices = [{"device_id": 0, "num_dram_channels": 12, "l1_num_banks": 64}]
        graph = self._make_buffer_graph(size=2097152, page_size=2048, buf_type="DRAM")

        report = _make_report(graph, devices=devices)
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute("SELECT max_size_per_bank FROM buffers")
        actual = cursor.fetchone()[0]
        expected = 86 * 2048  # ceil(1024/12) * page_size = 176128
        assert actual == expected, f"max_size_per_bank should be {expected} (per-bank), got {actual}"
        conn.close()

    def test_l1_interleaved_per_bank_size(self, tmp_path):
        """L1 interleaved buffers should use l1_num_banks for per-bank computation."""
        devices = [{"device_id": 0, "num_dram_channels": 12, "l1_num_banks": 64}]
        graph = self._make_buffer_graph(size=65536, page_size=2048, buf_type="L1")

        report = _make_report(graph, devices=devices)
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute("SELECT max_size_per_bank FROM buffers")
        actual = cursor.fetchone()[0]
        expected = 2048  # 32 pages, ceil(32/64) = 1 page/bank, 1 * 2048
        assert actual == expected, f"L1 per-bank should be {expected}, got {actual}"
        conn.close()

    def test_small_dram_buffer_per_bank(self, tmp_path):
        """
        Regression from the real bug: bias tensor buffer (26,624 bytes).
        ceil(13 pages / 12 banks) * 2048 = 4096 per bank.
        """
        devices = [{"device_id": 0, "num_dram_channels": 12, "l1_num_banks": 64}]
        graph = self._make_buffer_graph(size=26624, page_size=2048, buf_type="DRAM")

        report = _make_report(graph, devices=devices)
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute("SELECT max_size_per_bank FROM buffers")
        actual = cursor.fetchone()[0]
        expected = 2 * 2048  # ceil(13/12) * 2048 = 4096
        assert actual == expected, f"Bias buffer per-bank should be {expected}, got {actual}"
        conn.close()

    def test_no_device_info_falls_back_to_total_size(self, tmp_path):
        """Without device info, max_size_per_bank falls back to total buffer size."""
        graph = self._make_buffer_graph(size=4096, page_size=2048, buf_type="DRAM")

        report = _make_report(graph)
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute("SELECT max_size_per_bank FROM buffers")
        assert cursor.fetchone()[0] == 4096, "Without device info, should fall back to total size"
        conn.close()

    def test_sharded_buffer_per_core_size(self, tmp_path):
        """Sharded buffers compute per-bank from num_cores."""
        devices = [{"device_id": 0, "num_dram_channels": 12, "l1_num_banks": 64}]
        graph = self._make_buffer_graph(size=32768, page_size=2048, buf_type="L1", layout="HEIGHT_SHARDED", num_cores=8)

        report = _make_report(graph, devices=devices)
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute("SELECT max_size_per_bank FROM buffers")
        assert cursor.fetchone()[0] == 4096, f"Sharded per-core should be 4096"
        conn.close()


class TestLinearModelImport:
    """
    End-to-end regression test using the exact graph structure from the team's test_linear model:
        a = ttnn.ones([1024, 1024], layout=ttnn.TILE_LAYOUT, device=device)
        b = ttnn.ones([1024, 1024], layout=ttnn.TILE_LAYOUT, device=device)
        c = ttnn.ones([1, 1024], layout=ttnn.TILE_LAYOUT, device=device)
        d = ttnn.linear(a, b, bias=c)

    Verifies the import produces output consistent with the known-good (correct) database.
    """

    MOCK_DEVICE_INFO = [
        {
            "device_id": 1,
            "num_dram_channels": 12,
            "l1_num_banks": 64,
            "num_y_cores": 8,
            "num_x_cores": 8,
            "num_y_compute_cores": 8,
            "num_x_compute_cores": 8,
            "worker_l1_size": 1499136,
            "l1_bank_size": 1395424,
            "address_at_first_l1_bank": 0,
            "address_at_first_l1_cb_buffer": 103712,
            "num_banks_per_storage_core": 1,
            "num_compute_cores": 64,
            "num_storage_cores": 0,
            "total_l1_memory": 95944704,
            "total_l1_for_tensors": 0,
            "total_l1_for_interleaved_buffers": 89307136,
            "total_l1_for_sharded_buffers": 89307136,
            "cb_limit": 1395424,
        }
    ]

    MOCK_LINEAR_GRAPH = [
        {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [1, 7, 13, 19]},
        # --- ttnn.ones #1 → ttnn.to_device ---
        {
            "counter": 1,
            "node_type": "function_start",
            "params": {"name": "ttnn.to_device", "inputs": "1"},
            "connections": [3, 4, 5],
            "input_tensors": [2],
        },
        {
            "counter": 2,
            "node_type": "tensor",
            "params": {
                "tensor_id": "0",
                "shape": "Shape([1024, 1024])",
                "dtype": "DataType::BFLOAT16",
                "layout": "Layout::TILE",
            },
            "connections": [1],
        },
        {"counter": 3, "node_type": "buffer", "params": {}, "connections": [6, 6]},
        {
            "counter": 4,
            "node_type": "buffer_allocate",
            "params": {
                "device_id": "1",
                "address": "1920032",
                "size": "2097152",
                "page_size": "2048",
                "num_cores": "0",
                "type": "DRAM",
                "layout": "INTERLEAVED",
            },
            "connections": [3],
        },
        {
            "counter": 5,
            "node_type": "function_end",
            "params": {"name": "ttnn.to_device"},
            "connections": [6, 7],
            "duration_ns": 2784790,
        },
        {
            "counter": 6,
            "node_type": "tensor",
            "params": {
                "tensor_id": "1",
                "shape": "Shape([1024, 1024])",
                "dtype": "DataType::BFLOAT16",
                "layout": "Layout::TILE",
                "memory_config": "MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)",
                "device_id": "1",
                "address": "1920032",
                "buffer_type": "DRAM",
                "device_tensors": '[{"device_id": 0, "mesh_device_id": 1, "address": 1920032}]',
            },
            "connections": [19],
        },
        # --- ttnn.ones #2 → ttnn.to_device ---
        {
            "counter": 7,
            "node_type": "function_start",
            "params": {"name": "ttnn.to_device", "inputs": "1"},
            "connections": [9, 10, 11],
            "input_tensors": [8],
        },
        {
            "counter": 8,
            "node_type": "tensor",
            "params": {
                "tensor_id": "2",
                "shape": "Shape([1024, 1024])",
                "dtype": "DataType::BFLOAT16",
                "layout": "Layout::TILE",
            },
            "connections": [7],
        },
        {"counter": 9, "node_type": "buffer", "params": {}, "connections": [12, 12]},
        {
            "counter": 10,
            "node_type": "buffer_allocate",
            "params": {
                "device_id": "1",
                "address": "2096160",
                "size": "2097152",
                "page_size": "2048",
                "num_cores": "0",
                "type": "DRAM",
                "layout": "INTERLEAVED",
            },
            "connections": [9],
        },
        {
            "counter": 11,
            "node_type": "function_end",
            "params": {"name": "ttnn.to_device"},
            "connections": [12, 13],
            "duration_ns": 252479,
        },
        {
            "counter": 12,
            "node_type": "tensor",
            "params": {
                "tensor_id": "3",
                "shape": "Shape([1024, 1024])",
                "dtype": "DataType::BFLOAT16",
                "layout": "Layout::TILE",
                "memory_config": "MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)",
                "device_id": "1",
                "address": "2096160",
                "buffer_type": "DRAM",
                "device_tensors": '[{"device_id": 0, "mesh_device_id": 1, "address": 2096160}]',
            },
            "connections": [19],
        },
        # --- ttnn.ones #3 → ttnn.to_device ---
        {
            "counter": 13,
            "node_type": "function_start",
            "params": {"name": "ttnn.to_device", "inputs": "1"},
            "connections": [15, 16, 17],
            "input_tensors": [14],
        },
        {
            "counter": 14,
            "node_type": "tensor",
            "params": {
                "tensor_id": "4",
                "shape": "Shape([1, 1024])",
                "dtype": "DataType::BFLOAT16",
                "layout": "Layout::TILE",
            },
            "connections": [13],
        },
        {"counter": 15, "node_type": "buffer", "params": {}, "connections": [18, 18]},
        {
            "counter": 16,
            "node_type": "buffer_allocate",
            "params": {
                "device_id": "1",
                "address": "2272288",
                "size": "65536",
                "page_size": "2048",
                "num_cores": "0",
                "type": "DRAM",
                "layout": "INTERLEAVED",
            },
            "connections": [15],
        },
        {
            "counter": 17,
            "node_type": "function_end",
            "params": {"name": "ttnn.to_device"},
            "connections": [18, 19],
            "duration_ns": 31240,
        },
        {
            "counter": 18,
            "node_type": "tensor",
            "params": {
                "tensor_id": "5",
                "shape": "Shape([1, 1024])",
                "dtype": "DataType::BFLOAT16",
                "layout": "Layout::TILE",
                "memory_config": "MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)",
                "device_id": "1",
                "address": "2272288",
                "buffer_type": "DRAM",
                "device_tensors": '[{"device_id": 0, "mesh_device_id": 1, "address": 2272288}]',
            },
            "connections": [19],
        },
        # --- ttnn.linear → MatmulDeviceOperation ---
        {
            "counter": 19,
            "node_type": "function_start",
            "params": {"name": "MatmulDeviceOperation", "inputs": "3"},
            "connections": [20, 25, 26, 27, 28, 29, 30, 31, 32],
            "input_tensors": [6, 12, 18],
        },
        {
            "counter": 20,
            "node_type": "function_start",
            "params": {"name": "tt::tt_metal::create_device_tensor"},
            "connections": [21, 22, 23],
            "input_tensors": [],
        },
        {"counter": 21, "node_type": "buffer", "params": {}, "connections": [24, 24]},
        {
            "counter": 22,
            "node_type": "buffer_allocate",
            "params": {
                "device_id": "1",
                "address": "2278432",
                "size": "2097152",
                "page_size": "2048",
                "num_cores": "0",
                "type": "DRAM",
                "layout": "INTERLEAVED",
            },
            "connections": [21],
        },
        {
            "counter": 23,
            "node_type": "function_end",
            "params": {"name": "tt::tt_metal::create_device_tensor"},
            "connections": [24],
            "duration_ns": 14929,
        },
        {
            "counter": 24,
            "node_type": "tensor",
            "params": {
                "tensor_id": "7",
                "shape": "Shape([1024, 1024])",
                "dtype": "DataType::BFLOAT16",
                "layout": "Layout::TILE",
                "memory_config": "MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)",
                "device_id": "1",
                "address": "2278432",
                "buffer_type": "DRAM",
                "device_tensors": '[{"device_id": 0, "mesh_device_id": 1, "address": 2278432}]',
            },
            "connections": [],
        },
        {"counter": 25, "node_type": "circular_buffer_deallocate_all", "params": {}, "connections": [19]},
        {"counter": 26, "node_type": "circular_buffer_allocate", "params": {}, "connections": []},
        {"counter": 27, "node_type": "circular_buffer_allocate", "params": {}, "connections": []},
        {"counter": 28, "node_type": "circular_buffer_allocate", "params": {}, "connections": []},
        {"counter": 29, "node_type": "circular_buffer_allocate", "params": {}, "connections": []},
        {"counter": 30, "node_type": "buffer", "params": {}, "connections": []},
        {
            "counter": 31,
            "node_type": "buffer_allocate",
            "params": {
                "device_id": "1",
                "address": "1073737728",
                "size": "26624",
                "page_size": "2048",
                "num_cores": "0",
                "type": "DRAM",
                "layout": "INTERLEAVED",
            },
            "connections": [30],
        },
        {
            "counter": 32,
            "node_type": "function_end",
            "params": {"name": "MatmulDeviceOperation"},
            "connections": [24, 33],
            "duration_ns": 13232532,
        },
        {"counter": 33, "node_type": "capture_end", "params": {}, "connections": []},
    ]

    def _import_linear_graph(self, tmp_path):
        report = _make_report(self.MOCK_LINEAR_GRAPH, devices=self.MOCK_DEVICE_INFO)
        return _import_to_db(report, tmp_path)

    def test_operations_count_and_names(self, tmp_path):
        """The linear model should produce 5 operations matching the C++ trace."""
        conn, cursor = self._import_linear_graph(tmp_path)

        cursor.execute("SELECT operation_id, name FROM operations ORDER BY operation_id")
        rows = cursor.fetchall()

        assert len(rows) == 4, f"Expected 4 ops (nested child ops filtered), got {len(rows)}"
        assert rows[0][1] == "ttnn.to_device"
        assert rows[1][1] == "ttnn.to_device"
        assert rows[2][1] == "ttnn.to_device"
        assert rows[3][1] == "MatmulDeviceOperation"
        conn.close()

    def test_all_input_tensors_reference_valid_tensor_ids(self, tmp_path):
        """
        Regression: the old bug stored node counters (2, 8, 14, 6, 12, 18) as tensor_ids
        in input_tensors. After the fix, they must resolve to real tensor_ids that exist
        in the tensors table.
        """
        conn, cursor = self._import_linear_graph(tmp_path)

        cursor.execute("SELECT tensor_id FROM tensors")
        valid_tensor_ids = {row[0] for row in cursor.fetchall()}
        valid_tensor_ids_int = {int(t) if isinstance(t, str) else t for t in valid_tensor_ids}

        cursor.execute(
            "SELECT operation_id, input_index, tensor_id FROM input_tensors ORDER BY operation_id, input_index"
        )
        input_rows = cursor.fetchall()

        for op_id, idx, tid in input_rows:
            tid_int = int(tid) if isinstance(tid, str) else tid
            assert tid_int in valid_tensor_ids_int, (
                f"input_tensors({op_id}, {idx}) references tensor_id={tid} which doesn't exist in tensors table. "
                f"This is likely the old bug where node counters were stored instead of real tensor_ids. "
                f"Valid tensor_ids: {sorted(valid_tensor_ids_int)}"
            )
        conn.close()

    def test_matmul_input_tensors_are_device_tensors(self, tmp_path):
        """
        The MatmulDeviceOperation's input_tensors field has [6, 12, 18] (node counters
        for the three device tensors). After resolution, these should become tensor_ids
        1, 3, 5 — the device-side tensors from the three to_device operations.
        """
        conn, cursor = self._import_linear_graph(tmp_path)

        cursor.execute("SELECT input_index, tensor_id FROM input_tensors WHERE operation_id = 4 ORDER BY input_index")
        rows = cursor.fetchall()

        assert len(rows) == 3, f"MatmulDeviceOperation should have 3 inputs, got {len(rows)}"
        resolved_ids = [int(r[1]) if isinstance(r[1], str) else r[1] for r in rows]
        assert resolved_ids == [1, 3, 5], (
            f"Matmul inputs should be tensor_ids [1, 3, 5] (device tensors), "
            f"not node counters [6, 12, 18]. Got {resolved_ids}"
        )
        conn.close()

    def test_to_device_input_tensors_are_host_tensors(self, tmp_path):
        """Each ttnn.to_device should take one host tensor as input."""
        conn, cursor = self._import_linear_graph(tmp_path)

        cursor.execute("SELECT operation_id, tensor_id FROM input_tensors WHERE operation_id < 4 ORDER BY operation_id")
        rows = cursor.fetchall()

        assert len(rows) == 3
        resolved_ids = [int(r[1]) if isinstance(r[1], str) else r[1] for r in rows]
        assert resolved_ids == [0, 2, 4], f"to_device inputs should be host tensor_ids [0, 2, 4], got {resolved_ids}"
        conn.close()

    def test_output_tensors_match_expected(self, tmp_path):
        """Verify output tensor associations match the graph structure."""
        conn, cursor = self._import_linear_graph(tmp_path)

        cursor.execute(
            "SELECT operation_id, output_index, tensor_id FROM output_tensors ORDER BY operation_id, output_index"
        )
        rows = cursor.fetchall()

        output_map = {r[0]: int(r[2]) if isinstance(r[2], str) else r[2] for r in rows}
        assert output_map[1] == 1, "to_device #1 output should be tensor 1"
        assert output_map[2] == 3, "to_device #2 output should be tensor 3"
        assert output_map[3] == 5, "to_device #3 output should be tensor 5"
        assert output_map[4] == 7, "MatmulDeviceOperation output should be tensor 7"
        conn.close()

    def test_buffer_sizes_are_per_bank_not_total(self, tmp_path):
        """
        Regression: the correct DB has per-bank sizes (176128, 6144, 4096).
        The old bug stored total sizes (2097152, 65536, 26624).
        """
        conn, cursor = self._import_linear_graph(tmp_path)

        cursor.execute("SELECT address, max_size_per_bank FROM buffers ORDER BY address")
        rows = cursor.fetchall()
        buf_by_addr = {r[0]: r[1] for r in rows}

        assert (
            buf_by_addr[1920032] == 176128
        ), f"1024x1024 bf16 DRAM buffer should be 176128/bank, got {buf_by_addr[1920032]}"
        assert (
            buf_by_addr[2096160] == 176128
        ), f"1024x1024 bf16 DRAM buffer should be 176128/bank, got {buf_by_addr[2096160]}"
        assert buf_by_addr[2272288] == 6144, f"1x1024 bf16 DRAM buffer should be 6144/bank, got {buf_by_addr[2272288]}"
        assert (
            buf_by_addr[2278432] == 176128
        ), f"output 1024x1024 DRAM buffer should be 176128/bank, got {buf_by_addr[2278432]}"
        assert (
            buf_by_addr[1073737728] == 4096
        ), f"matmul intermediate DRAM buffer should be 4096/bank, got {buf_by_addr[1073737728]}"
        conn.close()

    def test_buffers_are_cumulative_per_operation(self, tmp_path):
        """
        Regression: the correct DB has cumulative buffer snapshots per operation.
        op 1: 1 buffer, op 2: 2 buffers, op 3: 3 buffers, op 4 (matmul): 5 buffers.
        Nested create_device_tensor is filtered, its buffers still tracked.
        """
        conn, cursor = self._import_linear_graph(tmp_path)

        cursor.execute("SELECT operation_id, COUNT(*) FROM buffers GROUP BY operation_id ORDER BY operation_id")
        rows = cursor.fetchall()

        expected_counts = {1: 1, 2: 2, 3: 3, 4: 5}
        actual_counts = {r[0]: r[1] for r in rows}

        assert actual_counts == expected_counts, (
            f"Buffer counts should be cumulative per operation. " f"Expected {expected_counts}, got {actual_counts}"
        )
        conn.close()

    def test_device_tensors_have_correct_addresses_and_device_ids(self, tmp_path):
        """Device tensor addresses and device_ids should match the correct DB (mesh_device_id=1)."""
        conn, cursor = self._import_linear_graph(tmp_path)

        cursor.execute("SELECT tensor_id, device_id, address FROM device_tensors ORDER BY CAST(tensor_id AS INTEGER)")
        rows = cursor.fetchall()

        expected = [
            (1, 0, 1920032),
            (3, 0, 2096160),
            (5, 0, 2272288),
            (7, 0, 2278432),
        ]
        assert len(rows) == len(expected), f"Expected {len(expected)} device_tensors, got {len(rows)}"
        for row, (exp_tid, exp_dev, exp_addr) in zip(rows, expected):
            tid = int(row[0]) if isinstance(row[0], str) else row[0]
            assert tid == exp_tid, f"Expected tensor_id {exp_tid}, got {tid}"
            assert row[1] == exp_dev, f"Tensor {tid}: expected device_id {exp_dev}, got {row[1]}"
            assert row[2] == exp_addr, f"Tensor {tid}: expected address {exp_addr}, got {row[2]}"
        conn.close()

    def test_tensor_count_and_shapes(self, tmp_path):
        """7 unique tensors (3 host + 4 device, tensor_id 6 is skipped) with correct shapes."""
        conn, cursor = self._import_linear_graph(tmp_path)

        cursor.execute("SELECT tensor_id, shape FROM tensors ORDER BY CAST(tensor_id AS INTEGER)")
        rows = cursor.fetchall()

        assert len(rows) == 7, f"Expected 7 tensors (ids 0-5,7), got {len(rows)}"

        shapes = {int(r[0]) if isinstance(r[0], str) else r[0]: r[1] for r in rows}
        for tid in [0, 1, 2, 3, 7]:
            assert "1024, 1024" in str(shapes[tid]), f"Tensor {tid} should be 1024x1024, got {shapes[tid]}"
        for tid in [4, 5]:
            assert "1, 1024" in str(shapes[tid]), f"Tensor {tid} should be 1x1024, got {shapes[tid]}"
        conn.close()


class TestGraphCaptureToFile:
    """Tests for end_graph_capture_to_file API."""

    def test_basic_capture_to_file(self, device, tmp_report_dir):
        """Test basic graph capture to JSON file."""
        report_path = tmp_report_dir / "report.json"

        torch_input = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16)
        tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        _ = ttnn.relu(tt_input)
        captured_graph = ttnn.graph.end_graph_capture_to_file(report_path)

        # Verify file was created
        assert report_path.exists(), "Report file should be created"

        # Verify returned graph matches file contents
        with open(report_path) as f:
            report = json.load(f)

        assert "graph" in report
        assert "version" in report
        assert report["graph"] == captured_graph

    def test_report_contains_device_info(self, device, tmp_report_dir):
        """Test that report contains device information."""
        report_path = tmp_report_dir / "report.json"

        torch_input = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16)
        tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        _ = ttnn.relu(tt_input)
        _ = ttnn.graph.end_graph_capture_to_file(report_path)

        with open(report_path) as f:
            report = json.load(f)

        assert "devices" in report
        assert len(report["devices"]) > 0

        # Check device has expected fields
        dev = report["devices"][0]
        assert "device_id" in dev
        assert "arch" in dev

    def test_report_contains_duration(self, device, tmp_report_dir):
        """Test that function_end nodes contain duration."""
        report_path = tmp_report_dir / "report.json"

        torch_input = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16)
        tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        _ = ttnn.relu(tt_input)
        captured_graph = ttnn.graph.end_graph_capture_to_file(report_path)

        # Find function_end nodes
        function_ends = [n for n in captured_graph if n.get("node_type") == "function_end"]
        assert len(function_ends) > 0, "Should have function_end nodes"

        # At least one should have duration
        has_duration = any("duration_ns" in n for n in function_ends)
        assert has_duration, "function_end nodes should have duration_ns"


class TestDurationExtraction:
    """Tests for duration extraction utilities."""

    def test_extract_total_duration(self, device):
        """Test extracting total capture duration."""
        torch_input = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16)
        tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        _ = ttnn.relu(tt_input)
        captured_graph = ttnn.graph.end_graph_capture()

        duration = graph_report.extract_total_duration_from_graph(captured_graph)
        assert isinstance(duration, float)
        assert duration > 0, "relu should have a positive duration"

    def test_extract_operation_durations(self, device):
        """Test extracting per-operation durations."""
        torch_input = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16)
        tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        _ = ttnn.relu(tt_input)
        _ = ttnn.add(tt_input, tt_input)
        captured_graph = ttnn.graph.end_graph_capture()

        durations = graph_report.extract_operation_durations(captured_graph)
        assert isinstance(durations, dict)
        assert len(durations) >= 2, f"Should have at least 2 operation durations (relu, add), got {len(durations)}"
        for name, dur in durations.items():
            assert dur > 0, f"Duration for {name} should be positive"


class TestGraphReportImport:
    """Tests for the offline import functionality."""

    def test_import_creates_database(self, device, tmp_report_dir):
        """Test that import creates SQLite database."""
        report_path = tmp_report_dir / "report.json"
        db_dir = tmp_report_dir / "db"

        torch_input = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16)
        tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        _ = ttnn.relu(tt_input)
        _ = ttnn.graph.end_graph_capture_to_file(report_path)

        db_path = graph_report.import_report(report_path, db_dir)

        assert db_path.exists()
        assert db_path.name == "db.sqlite"

    def test_import_populates_tables(self, device, tmp_report_dir):
        """Test that import populates all expected tables."""
        report_path = tmp_report_dir / "report.json"
        db_dir = tmp_report_dir / "db"

        torch_input = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16)
        tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        _ = ttnn.relu(tt_input)
        _ = ttnn.graph.end_graph_capture_to_file(report_path)

        db_path = graph_report.import_report(report_path, db_dir)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM devices")
        assert cursor.fetchone()[0] >= 1, "devices table should have at least 1 device"

        cursor.execute("SELECT COUNT(*) FROM operations")
        op_count = cursor.fetchone()[0]
        assert op_count >= 1, f"operations table should have at least 1 operation (relu), got {op_count}"

        cursor.execute("SELECT COUNT(*) FROM captured_graph")
        cg_count = cursor.fetchone()[0]
        assert cg_count >= 1, f"captured_graph should have at least 1 entry, got {cg_count}"

        cursor.execute("SELECT COUNT(*) FROM tensors")
        assert cursor.fetchone()[0] >= 1, "tensors table should have at least 1 tensor"

        conn.close()


class TestReportVersion:
    """Tests for report version handling."""

    def test_version_constant_exposed(self):
        """Test that REPORT_VERSION is exposed."""
        assert hasattr(ttnn.graph, "REPORT_VERSION")
        assert isinstance(ttnn.graph.REPORT_VERSION, int)
        assert ttnn.graph.REPORT_VERSION >= 1

    def test_report_has_matching_version(self, device, tmp_report_dir):
        """Test that generated reports have correct version."""
        report_path = tmp_report_dir / "report.json"

        torch_input = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16)
        tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        _ = ttnn.relu(tt_input)
        _ = ttnn.graph.end_graph_capture_to_file(report_path)

        with open(report_path) as f:
            report = json.load(f)

        assert report["version"] == ttnn.graph.REPORT_VERSION


class TestStackTraces:
    """Tests for C++ stack trace capture."""

    def test_stack_traces_enabled_by_default(self):
        """Test that stack traces are enabled by default."""
        assert ttnn.graph.is_stack_trace_enabled()

    def test_enable_disable_stack_traces(self):
        """Test enabling and disabling stack traces."""
        assert ttnn.graph.is_stack_trace_enabled()

        ttnn.graph.disable_stack_traces()
        assert not ttnn.graph.is_stack_trace_enabled()

        ttnn.graph.enable_stack_traces()
        assert ttnn.graph.is_stack_trace_enabled()

    def test_stack_traces_captured_by_default(self, device):
        """Test that stack traces are captured in function_start nodes by default."""
        assert ttnn.graph.is_stack_trace_enabled()

        torch_input = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16)
        tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        _ = ttnn.relu(tt_input)
        captured_graph = ttnn.graph.end_graph_capture()

        function_starts = [n for n in captured_graph if n["node_type"] == "function_start"]
        assert len(function_starts) > 0, "Should have at least one function_start node"

        has_stack_trace = any("stack_trace" in n for n in function_starts)
        assert has_stack_trace, "Stack traces should be present by default"
        for node in function_starts:
            if "stack_trace" in node:
                assert isinstance(node["stack_trace"], list)

    def test_no_stack_traces_when_disabled(self, device):
        """Test that no stack traces are captured when disabled."""
        ttnn.graph.disable_stack_traces()
        try:
            torch_input = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16)
            tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
            _ = ttnn.relu(tt_input)
            captured_graph = ttnn.graph.end_graph_capture()

            for node in captured_graph:
                assert "stack_trace" not in node, f"Node {node['node_type']} should not have stack_trace"
        finally:
            ttnn.graph.enable_stack_traces()


class TestResNet50Patterns:
    """
    Tests for patterns observed in the ResNet50 reference database:
    - Host weight tensors used as inputs to conv2d operations
    - Deallocate operations (no output tensors)
    - Multiple buffer types and layouts (DRAM, L1, L1_SMALL)
    - Large number of cumulative buffers per operation
    - Operations with multiple input/output tensors
    """

    MOCK_RESNET_GRAPH = [
        # capture_start
        {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [1, 7, 13, 19, 25]},
        # Op 1: ttnn.to_device (host tensor -> device tensor)
        {
            "counter": 1,
            "node_type": "function_start",
            "params": {"name": "ttnn.to_device", "inputs": "1"},
            "connections": [],
            "input_tensors": [2],
            "stack_trace": ["frame1"],
        },
        {
            "counter": 2,
            "node_type": "tensor",
            "params": {
                "tensor_id": "0",
                "shape": "Shape([64, 16, 4, 4])",
                "dtype": "DataType::FLOAT32",
                "layout": "Layout::ROW_MAJOR",
            },
            "connections": [],
        },
        {
            "counter": 3,
            "node_type": "buffer_allocate",
            "params": {"size": "65536", "type": "DRAM", "layout": "INTERLEAVED", "device_id": "0", "address": "1000"},
            "connections": [],
        },
        {
            "counter": 4,
            "node_type": "tensor",
            "params": {
                "tensor_id": "1",
                "shape": "Shape([64, 16, 4, 4])",
                "dtype": "DataType::BFLOAT16",
                "layout": "Layout::TILE",
                "device_id": "0",
                "address": "1000",
                "buffer_type": "DRAM",
                "memory_config": "MemoryConfig(DRAM)",
            },
            "connections": [],
        },
        {
            "counter": 5,
            "node_type": "function_end",
            "params": {"name": "ttnn.to_device"},
            "connections": [4],
            "duration_ns": 100,
        },
        # Op 2: conv2d (host weight + device activation -> device output)
        {
            "counter": 6,
            "node_type": "tensor",
            "params": {
                "tensor_id": "10",
                "shape": "Shape([64, 64, 3, 3])",
                "dtype": "DataType::FLOAT32",
                "layout": "Layout::ROW_MAJOR",
            },
            "connections": [],
        },
        {
            "counter": 7,
            "node_type": "function_start",
            "params": {"name": "ttnn::conv2d", "inputs": "3"},
            "connections": [],
            "input_tensors": [4, 6],
            "arguments": ["activation_tensor", "weight_tensor", "bias_tensor"],
            "stack_trace": ["frame2"],
        },
        {
            "counter": 8,
            "node_type": "buffer_allocate",
            "params": {"size": "176128", "type": "DRAM", "layout": "INTERLEAVED", "device_id": "0", "address": "2000"},
            "connections": [],
        },
        {
            "counter": 9,
            "node_type": "buffer_allocate",
            "params": {
                "size": "4096",
                "type": "L1",
                "layout": "HEIGHT_SHARDED",
                "device_id": "0",
                "address": "5000",
                "num_cores": "64",
            },
            "connections": [],
        },
        {
            "counter": 10,
            "node_type": "tensor",
            "params": {
                "tensor_id": "2",
                "shape": "Shape([1, 1, 3136, 64])",
                "dtype": "DataType::BFLOAT16",
                "layout": "Layout::TILE",
                "device_id": "0",
                "address": "2000",
                "buffer_type": "DRAM",
                "memory_config": "MemoryConfig(DRAM)",
            },
            "connections": [],
        },
        {
            "counter": 11,
            "node_type": "function_end",
            "params": {"name": "ttnn::conv2d"},
            "connections": [10],
            "duration_ns": 5000,
        },
        # Op 3: add_ (two device tensors -> device output)
        {
            "counter": 12,
            "node_type": "tensor",
            "params": {
                "tensor_id": "3",
                "shape": "Shape([1, 1, 3136, 64])",
                "dtype": "DataType::BFLOAT16",
                "layout": "Layout::TILE",
                "device_id": "0",
                "address": "3000",
                "buffer_type": "L1",
                "memory_config": "MemoryConfig(L1)",
            },
            "connections": [],
        },
        {
            "counter": 13,
            "node_type": "function_start",
            "params": {"name": "ttnn::add_", "inputs": "2"},
            "connections": [],
            "input_tensors": [10, 12],
            "stack_trace": ["frame3"],
        },
        {
            "counter": 14,
            "node_type": "buffer_allocate",
            "params": {"size": "8192", "type": "L1", "layout": "INTERLEAVED", "device_id": "0", "address": "4000"},
            "connections": [],
        },
        {
            "counter": 15,
            "node_type": "tensor",
            "params": {
                "tensor_id": "4",
                "shape": "Shape([1, 1, 3136, 64])",
                "dtype": "DataType::BFLOAT16",
                "layout": "Layout::TILE",
                "device_id": "0",
                "address": "4000",
                "buffer_type": "L1",
                "memory_config": "MemoryConfig(L1)",
            },
            "connections": [],
        },
        {
            "counter": 16,
            "node_type": "function_end",
            "params": {"name": "ttnn::add_"},
            "connections": [15],
            "duration_ns": 200,
        },
        # Deallocate: C++ only emits buffer_deallocate (function tracking is not wrapped).
        # The importer synthesizes an operation from this node.
        # connections points to the buffer node (counter 8) which connects to tensor (counter 10).
        {
            "counter": 17,
            "node_type": "buffer_deallocate",
            "params": {"address": "2000", "size": "176128", "type": "DRAM", "device_id": "0"},
            "connections": [8],
        },
        # capture_end
        {"counter": 25, "node_type": "capture_end", "params": {}, "connections": []},
    ]

    MOCK_DEVICE_INFO = [
        {
            "device_id": 0,
            "num_dram_channels": 12,
            "l1_num_banks": 64,
            "num_y_cores": 8,
            "num_x_cores": 8,
            "num_y_compute_cores": 8,
            "num_x_compute_cores": 8,
            "num_storage_cores": 0,
            "worker_l1_size": 1499136,
            "l1_bank_size": 23424,
        }
    ]

    def _import_resnet(self, tmp_path):
        report = _make_report(self.MOCK_RESNET_GRAPH, devices=self.MOCK_DEVICE_INFO)
        return _import_to_db(report, tmp_path)

    def test_host_weight_tensors_kept_in_compatible_mode(self, tmp_path):
        """Host weight tensors referenced as inputs must be preserved in compatible mode."""
        conn, cursor = self._import_resnet(tmp_path)

        cursor.execute("SELECT tensor_id, device_id FROM tensors WHERE device_id IS NULL")
        host_tensors = cursor.fetchall()
        # tensor_id 0 (input to to_device) + tensor_id 10 (conv2d weight)
        assert (
            len(host_tensors) == 2
        ), f"Expected 2 host tensors (to_device input + conv weight), got {len(host_tensors)}"

        # Verify each host tensor is referenced in input_tensors
        for tid, _ in host_tensors:
            tid_int = int(tid) if isinstance(tid, str) else tid
            cursor.execute("SELECT COUNT(*) FROM input_tensors WHERE tensor_id = ?", (tid_int,))
            refs = cursor.fetchone()[0]
            assert refs > 0, f"Host tensor {tid} should be referenced in input_tensors"
        conn.close()

    def test_deallocate_ops_have_no_output_tensors(self, tmp_path):
        """Deallocate operations should have input tensors but no output tensors."""
        conn, cursor = self._import_resnet(tmp_path)

        cursor.execute("SELECT operation_id, name FROM operations WHERE name LIKE '%deallocate%'")
        dealloc_ops = cursor.fetchall()
        assert len(dealloc_ops) == 1, f"Expected 1 deallocate op, got {len(dealloc_ops)}"

        op_id = dealloc_ops[0][0]

        cursor.execute("SELECT COUNT(*) FROM input_tensors WHERE operation_id = ?", (op_id,))
        input_count = cursor.fetchone()[0]
        assert input_count == 1, f"Deallocate should have 1 input, got {input_count}"

        cursor.execute("SELECT COUNT(*) FROM output_tensors WHERE operation_id = ?", (op_id,))
        output_count = cursor.fetchone()[0]
        assert output_count == 0, f"Deallocate should have 0 outputs, got {output_count}"
        conn.close()

    def test_conv2d_has_host_and_device_inputs(self, tmp_path):
        """Conv2d should accept both host weight tensors and device activation tensors as inputs."""
        conn, cursor = self._import_resnet(tmp_path)

        cursor.execute("SELECT operation_id FROM operations WHERE name = 'ttnn::conv2d'")
        conv_op = cursor.fetchone()
        assert conv_op is not None, "Should have a conv2d operation"

        cursor.execute(
            """
            SELECT t.tensor_id, t.device_id
            FROM input_tensors it JOIN tensors t ON it.tensor_id = t.tensor_id
            WHERE it.operation_id = ?
            ORDER BY it.input_index
        """,
            (conv_op[0],),
        )
        inputs = cursor.fetchall()

        device_inputs = [r for r in inputs if r[1] is not None]
        host_inputs = [r for r in inputs if r[1] is None]
        assert len(device_inputs) >= 1, f"Conv2d should have at least 1 device input, got {len(device_inputs)}"
        assert len(host_inputs) >= 1, f"Conv2d should have at least 1 host weight input, got {len(host_inputs)}"
        conn.close()

    def test_multiple_buffer_types(self, tmp_path):
        """Buffers should support DRAM and L1 buffer types."""
        conn, cursor = self._import_resnet(tmp_path)

        cursor.execute("SELECT DISTINCT buffer_type FROM buffers ORDER BY buffer_type")
        type_values = {r[0] for r in cursor.fetchall()}
        assert len(type_values) >= 2, f"Should have at least DRAM and L1 buffer types, got {type_values}"
        assert 0 in type_values, "Missing buffer_type 0 (DRAM)"
        assert 1 in type_values, "Missing buffer_type 1 (L1)"
        conn.close()

    def test_cumulative_buffers_reflect_allocations_and_deallocations(self, tmp_path):
        """
        Buffer counts should grow with allocations and shrink with deallocations.
        Op 1 (to_device): 1 DRAM buffer allocated -> 1 total
        Op 2 (conv2d): 2 more buffers (DRAM + L1) -> 3 total
        Op 3 (add_): 1 more L1 buffer -> 4 total
        Op 4 (deallocate): 1 DRAM buffer deallocated before op -> 3 total
        """
        conn, cursor = self._import_resnet(tmp_path)

        cursor.execute(
            """
            SELECT operation_id, COUNT(*) FROM buffers
            GROUP BY operation_id ORDER BY operation_id
        """
        )
        rows = cursor.fetchall()
        counts_by_op = {r[0]: r[1] for r in rows}
        assert len(counts_by_op) >= 3, f"Should have at least 3 ops with buffers, got {len(counts_by_op)}"

        op_ids = sorted(counts_by_op.keys())
        assert counts_by_op[op_ids[0]] == 1, f"First op should have 1 buffer, got {counts_by_op[op_ids[0]]}"
        assert counts_by_op[op_ids[1]] == 3, f"Second op should have 3 buffers, got {counts_by_op[op_ids[1]]}"
        assert counts_by_op[op_ids[2]] == 4, f"Third op should have 4 buffers, got {counts_by_op[op_ids[2]]}"
        if len(op_ids) >= 4:
            assert (
                counts_by_op[op_ids[3]] == 3
            ), f"Fourth op (post-dealloc) should have 3 buffers, got {counts_by_op[op_ids[3]]}"
        conn.close()

    def test_stack_traces_for_real_operations(self, tmp_path):
        """Operations with function_start should have stack traces. Synthesized deallocate ops may not."""
        conn, cursor = self._import_resnet(tmp_path)

        cursor.execute("SELECT COUNT(*) FROM operations WHERE name != 'ttnn::deallocate'")
        real_op_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM stack_traces")
        st_count = cursor.fetchone()[0]
        assert st_count == real_op_count, f"Expected {real_op_count} stack traces, got {st_count}"
        conn.close()

    def test_captured_graph_per_operation(self, tmp_path):
        """Each operation should have a captured_graph entry."""
        conn, cursor = self._import_resnet(tmp_path)

        cursor.execute("SELECT COUNT(*) FROM operations")
        op_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM captured_graph")
        cg_count = cursor.fetchone()[0]
        assert cg_count == op_count, f"Expected {op_count} captured_graph rows, got {cg_count}"
        conn.close()

    def test_operation_ids_start_at_one(self, tmp_path):
        """Operation IDs should start at 1 (1-based indexing)."""
        conn, cursor = self._import_resnet(tmp_path)

        cursor.execute("SELECT MIN(operation_id), MAX(operation_id), COUNT(*) FROM operations")
        min_id, max_id, count = cursor.fetchone()
        assert min_id == 1, f"First operation_id should be 1, got {min_id}"
        conn.close()

    def test_all_input_output_tensors_reference_valid_ids(self, tmp_path):
        """All tensor references in input/output tables should exist in the tensors table."""
        conn, cursor = self._import_resnet(tmp_path)

        cursor.execute("SELECT tensor_id FROM tensors")
        valid_ids = {int(r[0]) if isinstance(r[0], str) else r[0] for r in cursor.fetchall()}

        cursor.execute("SELECT operation_id, tensor_id FROM input_tensors")
        for op_id, tid in cursor.fetchall():
            tid_int = int(tid) if isinstance(tid, str) else tid
            assert tid_int in valid_ids, f"input_tensors op={op_id} references tensor {tid} not in tensors table"

        cursor.execute("SELECT operation_id, tensor_id FROM output_tensors")
        for op_id, tid in cursor.fetchall():
            tid_int = int(tid) if isinstance(tid, str) else tid
            assert tid_int in valid_ids, f"output_tensors op={op_id} references tensor {tid} not in tensors table"
        conn.close()


@pytest.mark.skipif(not is_wormhole_b0(), reason="Requires Wormhole B0")
class TestLinearModelE2E:
    """
    End-to-end test: run ttnn.ones + ttnn.linear on real hardware,
    capture graph to JSON, import to SQLite, and validate structural properties.
    """

    def test_linear_model_structural_properties(self, device, tmp_path):
        report_path = tmp_path / "linear_report.json"

        with ttnn.manage_config("enable_fast_runtime_mode", False), ttnn.manage_config(
            "enable_logging", True
        ), ttnn.manage_config("enable_graph_report", True):
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
            a = ttnn.ones([1024, 1024], layout=ttnn.TILE_LAYOUT, device=device)
            b = ttnn.ones([1024, 1024], layout=ttnn.TILE_LAYOUT, device=device)
            c = ttnn.ones([1, 1024], layout=ttnn.TILE_LAYOUT, device=device)
            ttnn.linear(a, b, bias=c)
            ttnn.graph.end_graph_capture_to_file(report_path)

        assert report_path.exists(), "Report JSON should be created"

        output_dir = tmp_path / "output"
        db_path = graph_report.import_report(report_path, output_dir)
        assert db_path.exists(), "Imported DB should be created"

        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        c.execute("SELECT COUNT(*) FROM operations")
        op_count = c.fetchone()[0]
        assert op_count >= 4, f"Expected at least 4 operations, got {op_count}"

        c.execute("SELECT COUNT(*) FROM tensors WHERE device_id IS NOT NULL")
        device_count = c.fetchone()[0]
        assert device_count >= 4, f"Expected at least 4 device tensors, got {device_count}"

        c.execute("SELECT COUNT(*) FROM input_tensors")
        input_count = c.fetchone()[0]
        assert input_count >= 3, f"Expected at least 3 input_tensor rows, got {input_count}"

        c.execute("SELECT COUNT(*) FROM output_tensors")
        output_count = c.fetchone()[0]
        assert output_count >= 4, f"Expected at least 4 output_tensor rows, got {output_count}"

        c.execute(
            """
            SELECT COUNT(*) FROM input_tensors it
            LEFT JOIN tensors t ON it.tensor_id = t.tensor_id
            WHERE t.tensor_id IS NULL
        """
        )
        dangling = c.fetchone()[0]
        assert dangling == 0, f"{dangling} dangling input_tensors references"

        c.execute(
            """
            SELECT COUNT(*) FROM output_tensors ot
            LEFT JOIN tensors t ON ot.tensor_id = t.tensor_id
            WHERE t.tensor_id IS NULL
        """
        )
        dangling = c.fetchone()[0]
        assert dangling == 0, f"{dangling} dangling output_tensors references"

        c.execute("SELECT COUNT(*) FROM buffers")
        buf_count = c.fetchone()[0]
        assert buf_count >= 5, f"Expected at least 5 buffer rows, got {buf_count}"

        c.execute("SELECT max_size_per_bank FROM buffers")
        for (size,) in c.fetchall():
            assert size < 10_000_000, f"max_size_per_bank={size} looks like a total size, not per-bank"

        c.execute("SELECT COUNT(*) FROM captured_graph")
        cg_count = c.fetchone()[0]
        assert cg_count == op_count, f"Expected {op_count} captured_graph rows, got {cg_count}"

        c.execute("SELECT COUNT(*) FROM stack_traces")
        st_count = c.fetchone()[0]
        assert st_count == op_count, f"Expected {op_count} stack_trace rows, got {st_count}"

        conn.close()


@pytest.fixture
def imagenet_label_dict():
    import ast

    path = Path("models/sample_data/imagenet_class_labels.txt")
    assert path.exists(), f"ImageNet labels not found at {path}"
    with open(path, "r") as f:
        return ast.literal_eval(f.read())


@pytest.mark.skipif(not is_wormhole_b0(), reason="Requires Wormhole B0")
@pytest.mark.timeout(600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_loc",
    ((16, "models/demos/vision/classification/resnet50/ttnn_resnet/demo/images/"),),
)
def test_resnet50_e2e_graph_capture(
    mesh_device, batch_size, input_loc, imagenet_label_dict, model_location_generator, tmp_path
):
    """Run ResNet50 inference, capture graph, import to DB, and validate structural properties."""
    import shutil

    from models.demos.vision.classification.resnet50.ttnn_resnet.demo.demo import run_resnet_inference

    report_path = tmp_path / "resnet50_report.json"

    with ttnn.manage_config("enable_fast_runtime_mode", False), ttnn.manage_config(
        "enable_detailed_buffer_report", True
    ):
        ttnn.graph.enable_buffer_pages()
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        run_resnet_inference(batch_size, input_loc, imagenet_label_dict, mesh_device, model_location_generator)
        ttnn.graph.end_graph_capture_to_file(report_path)
        ttnn.graph.disable_buffer_pages()

    assert report_path.exists(), "Report JSON should be created"

    output_dir = tmp_path / "output"
    db_path = graph_report.import_report(report_path, output_dir)
    assert db_path.exists(), "Imported DB should be created"

    shutil.copy2(db_path, "/tmp/db.sqlite")

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM operations")
    op_count = c.fetchone()[0]
    assert op_count >= 300, f"Expected at least 300 operations, got {op_count}"

    c.execute("SELECT name, COUNT(*) FROM operations GROUP BY name ORDER BY COUNT(*) DESC")
    op_dist = dict(c.fetchall())
    assert op_dist.get("ttnn.conv2d", 0) == 159, f"Expected 159 conv2d ops, got {op_dist.get('ttnn.conv2d', 0)}"
    assert (
        op_dist.get("ttnn.deallocate", 0) == 51
    ), f"Expected 51 deallocate ops, got {op_dist.get('ttnn.deallocate', 0)}"
    assert op_dist.get("ttnn.add_", 0) == 48, f"Expected 48 add_ ops, got {op_dist.get('ttnn.add_', 0)}"

    c.execute("SELECT COUNT(*) FROM tensors WHERE device_id IS NULL")
    host_count = c.fetchone()[0]
    assert host_count > 0, "Expected host tensors (weights)"

    c.execute("SELECT COUNT(*) FROM tensors WHERE device_id IS NOT NULL")
    device_count = c.fetchone()[0]
    assert device_count > 0, "Expected device tensors"

    c.execute("SELECT COUNT(*) FROM input_tensors")
    input_count = c.fetchone()[0]
    assert input_count >= 600, f"Expected at least 600 input_tensor rows, got {input_count}"

    c.execute("SELECT COUNT(*) FROM output_tensors")
    output_count = c.fetchone()[0]
    assert output_count >= 300, f"Expected at least 300 output_tensor rows, got {output_count}"

    c.execute(
        """
        SELECT COUNT(*) FROM input_tensors it
        LEFT JOIN tensors t ON it.tensor_id = t.tensor_id
        WHERE t.tensor_id IS NULL
    """
    )
    dangling = c.fetchone()[0]
    assert dangling == 0, f"{dangling} dangling input_tensors references"

    c.execute(
        """
        SELECT COUNT(*) FROM output_tensors ot
        LEFT JOIN tensors t ON ot.tensor_id = t.tensor_id
        WHERE t.tensor_id IS NULL
    """
    )
    dangling = c.fetchone()[0]
    assert dangling == 0, f"{dangling} dangling output_tensors references"

    c.execute(
        """
        SELECT COUNT(*) FROM operations o
        WHERE o.name = 'ttnn.deallocate'
        AND EXISTS (SELECT 1 FROM input_tensors it WHERE it.operation_id = o.operation_id)
        AND NOT EXISTS (SELECT 1 FROM output_tensors ot WHERE ot.operation_id = o.operation_id)
    """
    )
    dealloc_correct = c.fetchone()[0]
    assert dealloc_correct == 51, f"Expected 51 deallocate ops with input+no-output, got {dealloc_correct}"

    c.execute("SELECT COUNT(*) FROM buffers")
    buf_count = c.fetchone()[0]
    assert buf_count > 0, "Expected non-zero buffers"

    c.execute("SELECT COUNT(*) FROM captured_graph")
    cg_count = c.fetchone()[0]
    assert cg_count == op_count, f"Expected {op_count} captured_graph rows, got {cg_count}"

    c.execute("SELECT COUNT(*) FROM stack_traces")
    st_count = c.fetchone()[0]
    assert st_count == op_count, f"Expected {op_count} stack_trace rows, got {st_count}"

    conn.close()


# ---------------------------------------------------------------------------
# Unit tests for Python-level graph helpers (graph.py)
# ---------------------------------------------------------------------------
class TestSafeArgStr:
    """Tests for ttnn.graph._safe_arg_str - safe argument stringification."""

    def test_plain_int(self):
        from ttnn.graph import _safe_arg_str

        assert _safe_arg_str(42) == "42"

    def test_plain_string(self):
        from ttnn.graph import _safe_arg_str

        assert _safe_arg_str("hello") == "hello"

    def test_plain_float(self):
        from ttnn.graph import _safe_arg_str

        assert _safe_arg_str(3.14) == "3.14"

    def test_none(self):
        from ttnn.graph import _safe_arg_str

        assert _safe_arg_str(None) == "None"

    def test_bool(self):
        from ttnn.graph import _safe_arg_str

        assert _safe_arg_str(True) == "True"

    def test_torch_tensor(self):
        from ttnn.graph import _safe_arg_str

        t = torch.randn(2, 3)
        result = _safe_arg_str(t)
        assert "torch.Tensor" in result
        assert "[2, 3]" in result
        assert "float32" in result

    def test_torch_tensor_does_not_print_data(self):
        from ttnn.graph import _safe_arg_str

        t = torch.randn(100, 100)
        result = _safe_arg_str(t)
        assert len(result) < 200, "Should be compact, not the full tensor data"

    def test_list_not_recursed(self):
        from ttnn.graph import _safe_arg_str

        result = _safe_arg_str([1, 2, 3])
        assert result == "[1, 2, 3]"

    def test_enum_value(self):
        from ttnn.graph import _safe_arg_str

        result = _safe_arg_str(ttnn.TILE_LAYOUT)
        assert "TILE" in result


class TestRecordPythonOperation:
    """Tests for ttnn.graph.record_python_operation."""

    def setup_method(self):
        import ttnn.graph as g

        g._python_io_data = []
        g.enable_python_stack_traces()

    def test_records_kwargs(self):
        import ttnn.graph as g

        g.record_python_operation("ttnn.relu", (), {"memory_config": "DRAM"})
        assert len(g._python_io_data) == 1
        entry = g._python_io_data[0]
        assert entry["name"] == "ttnn.relu"
        assert entry["arguments"]["memory_config"] == "DRAM"

    def test_records_positional_args(self):
        import ttnn.graph as g

        g.record_python_operation("ttnn.add", (10, 20), {})
        entry = g._python_io_data[0]
        assert entry["arguments"]["0"] == "10"
        assert entry["arguments"]["1"] == "20"

    def test_mixed_args_and_kwargs(self):
        import ttnn.graph as g

        g.record_python_operation("ttnn.linear", ("pos0",), {"bias": "bias_val"})
        entry = g._python_io_data[0]
        assert entry["arguments"]["0"] == "pos0"
        assert entry["arguments"]["bias"] == "bias_val"

    def test_torch_tensor_arg_compact(self):
        import ttnn.graph as g

        t = torch.randn(4, 8)
        g.record_python_operation("ttnn.from_torch", (t,), {})
        entry = g._python_io_data[0]
        val = entry["arguments"]["0"]
        assert "torch.Tensor" in val
        assert "[4, 8]" in val

    def test_multiple_records_append(self):
        import ttnn.graph as g

        g.record_python_operation("op1", (), {})
        g.record_python_operation("op2", (), {})
        g.record_python_operation("op3", (), {})
        assert len(g._python_io_data) == 3
        assert [e["name"] for e in g._python_io_data] == ["op1", "op2", "op3"]

    def test_empty_args(self):
        import ttnn.graph as g

        g.record_python_operation("ttnn.deallocate", (), {})
        entry = g._python_io_data[0]
        assert entry["arguments"] == {}

    def test_python_stack_trace_captured_when_enabled(self):
        import ttnn.graph as g

        g.enable_python_stack_traces()
        try:
            g.record_python_operation("ttnn.relu", (), {"x": "val"})
            entry = g._python_io_data[0]
            assert "python_stack_trace" in entry
            assert len(entry["python_stack_trace"]) > 0
            joined = "\n".join(entry["python_stack_trace"])
            assert "test_graph_report.py" in joined
        finally:
            g.disable_python_stack_traces()

    def test_python_stack_trace_absent_when_disabled(self):
        import ttnn.graph as g

        g.disable_python_stack_traces()
        g.record_python_operation("ttnn.relu", (), {})
        entry = g._python_io_data[0]
        assert "python_stack_trace" not in entry

    def test_python_stack_trace_filters_internals(self):
        import ttnn.graph as g

        g.enable_python_stack_traces()
        try:
            g.record_python_operation("ttnn.add", (1,), {})
            entry = g._python_io_data[0]
            joined = "\n".join(entry["python_stack_trace"])
            assert "graph.py" not in joined, "Internal graph.py frames should be filtered"
            assert "decorators.py" not in joined, "Internal decorators.py frames should be filtered"
        finally:
            g.disable_python_stack_traces()

    def test_enable_disable_toggle(self):
        import ttnn.graph as g

        assert g.is_python_stack_trace_enabled()
        g.disable_python_stack_traces()
        assert not g.is_python_stack_trace_enabled()
        g.enable_python_stack_traces()
        assert g.is_python_stack_trace_enabled()


class TestStoreCapturedGraph:
    """Tests for ttnn.graph.store_captured_graph."""

    def setup_method(self):
        import ttnn.graph as g

        g._python_io_data = []

    def test_attaches_to_last_entry(self):
        import ttnn.graph as g

        g._python_io_data.append({"name": "op1"})
        g._python_io_data.append({"name": "op2"})
        mock_cg = [{"node_type": "capture_start"}]
        g.store_captured_graph(mock_cg)
        assert "captured_graph" not in g._python_io_data[0]
        assert g._python_io_data[1]["captured_graph"] == mock_cg

    def test_noop_on_empty_list(self):
        import ttnn.graph as g

        g.store_captured_graph([{"node_type": "capture_start"}])
        assert len(g._python_io_data) == 0

    def test_overwrites_existing(self):
        import ttnn.graph as g

        g._python_io_data.append({"name": "op1", "captured_graph": "old"})
        g.store_captured_graph("new")
        assert g._python_io_data[0]["captured_graph"] == "new"


class TestBeginGraphCaptureClearing:
    """Tests for begin_graph_capture clearing behavior."""

    def test_clears_python_io_when_not_active(self):
        import ttnn.graph as g

        g._python_io_data = [{"name": "stale"}]
        g.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        assert g._python_io_data == []
        ttnn.graph.end_graph_capture()

    def test_preserves_python_io_when_active(self):
        import ttnn.graph as g

        g.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        g._python_io_data = [{"name": "keep_me"}]
        g.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        assert len(g._python_io_data) == 1
        assert g._python_io_data[0]["name"] == "keep_me"
        ttnn.graph.end_graph_capture()
        ttnn.graph.end_graph_capture()


# ---------------------------------------------------------------------------
# Unit tests for graph_report.py import_graph - new features
# ---------------------------------------------------------------------------
class TestPythonIOArgumentImport:
    """Tests that python_io arguments are preferred over C++ arguments."""

    def test_python_io_arguments_used_when_present(self, tmp_path):
        """When python_io has arguments, they should be used instead of C++ ones."""
        mock_graph = [
            {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [1, 3]},
            {
                "counter": 1,
                "node_type": "function_start",
                "params": {"name": "ttnn.relu"},
                "connections": [],
                "input_tensors": [],
                "arguments": ["cpp_arg0", "cpp_arg1"],
            },
            {
                "counter": 2,
                "node_type": "function_end",
                "params": {"name": "ttnn.relu"},
                "connections": [],
                "duration_ns": 100,
            },
            {"counter": 3, "node_type": "capture_end", "params": {}, "connections": []},
        ]
        python_io = [
            {
                "name": "ttnn.relu",
                "arguments": {"0": "ttnn.Tensor(shape=[1,32], dtype=bfloat16)", "memory_config": "DRAM"},
                "input_tensor_ids": [],
            }
        ]

        report = _make_report(mock_graph, python_io=python_io)
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute("SELECT name, value FROM operation_arguments ORDER BY name")
        rows = cursor.fetchall()
        values = {r[0]: r[1] for r in rows}

        assert "memory_config" in values, "Named kwarg should appear"
        assert values["memory_config"] == "DRAM"
        assert values["0"] == "ttnn.Tensor(shape=[1,32], dtype=bfloat16)"
        assert "cpp_arg0" not in [r[1] for r in rows], "C++ args should be overridden"
        conn.close()

    def test_cpp_arguments_fallback_when_no_python_io(self, tmp_path):
        """When no python_io is provided, C++ arguments should be used."""
        mock_graph = [
            {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [1, 3]},
            {
                "counter": 1,
                "node_type": "function_start",
                "params": {"name": "ttnn.relu"},
                "connections": [],
                "input_tensors": [],
                "arguments": ["tensor_a", "alpha=1.0"],
            },
            {
                "counter": 2,
                "node_type": "function_end",
                "params": {"name": "ttnn.relu"},
                "connections": [],
                "duration_ns": 100,
            },
            {"counter": 3, "node_type": "capture_end", "params": {}, "connections": []},
        ]

        report = _make_report(mock_graph)
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute("SELECT name, value FROM operation_arguments ORDER BY name")
        rows = cursor.fetchall()
        assert len(rows) == 2
        assert "tensor_a" in [r[1] for r in rows]
        conn.close()


class TestPerOpCapturedGraphImport:
    """Tests that per-op captured_graph from python_io is used directly."""

    def test_per_op_captured_graph_used_when_available(self, tmp_path):
        """captured_graph from python_io should be stored as-is."""
        per_op_graph = [
            {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [1]},
            {
                "counter": 1,
                "node_type": "function_start",
                "params": {"name": "InnerOp"},
                "connections": [2],
            },
            {
                "counter": 2,
                "node_type": "function_end",
                "params": {"name": "InnerOp"},
                "connections": [3],
            },
            {"counter": 3, "node_type": "capture_end", "params": {}, "connections": []},
        ]
        mock_graph = [
            {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [1, 3]},
            {
                "counter": 1,
                "node_type": "function_start",
                "params": {"name": "ttnn.relu"},
                "connections": [],
                "input_tensors": [],
            },
            {
                "counter": 2,
                "node_type": "function_end",
                "params": {"name": "ttnn.relu"},
                "connections": [],
                "duration_ns": 100,
            },
            {"counter": 3, "node_type": "capture_end", "params": {}, "connections": []},
        ]
        python_io = [
            {
                "name": "ttnn.relu",
                "arguments": {},
                "input_tensor_ids": [],
                "captured_graph": per_op_graph,
            }
        ]

        report = _make_report(mock_graph, python_io=python_io)
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute("SELECT captured_graph FROM captured_graph")
        rows = cursor.fetchall()
        assert len(rows) == 1
        stored = json.loads(rows[0][0])
        assert stored == per_op_graph, "Per-op graph should be stored exactly as provided"
        conn.close()

    def test_fallback_extraction_when_no_per_op_graph(self, tmp_path):
        """Without captured_graph in python_io, importer should extract from global graph."""
        mock_graph = [
            {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [1, 3]},
            {
                "counter": 1,
                "node_type": "function_start",
                "params": {"name": "ttnn.relu"},
                "connections": [3],
                "input_tensors": [],
            },
            {
                "counter": 2,
                "node_type": "function_end",
                "params": {"name": "ttnn.relu"},
                "connections": [],
                "duration_ns": 100,
            },
            {"counter": 3, "node_type": "capture_end", "params": {}, "connections": []},
        ]
        python_io = [{"name": "ttnn.relu", "arguments": {}, "input_tensor_ids": []}]

        report = _make_report(mock_graph, python_io=python_io)
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute("SELECT captured_graph FROM captured_graph")
        rows = cursor.fetchall()
        assert len(rows) == 1
        stored = json.loads(rows[0][0])
        assert stored[0]["node_type"] == "capture_start"
        assert stored[-1]["node_type"] == "capture_end"
        conn.close()


class TestFromTorchFiltering:
    """Tests for the smart from_torch filtering (leading block only)."""

    def _make_graph(self, op_names):
        """Helper: build a simple graph with given operation names."""
        nodes = [{"counter": 0, "node_type": "capture_start", "params": {}, "connections": []}]
        c = 1
        for name in op_names:
            nodes.append(
                {
                    "counter": c,
                    "node_type": "function_start",
                    "params": {"name": name},
                    "connections": [],
                    "input_tensors": [],
                }
            )
            c += 1
            nodes.append(
                {
                    "counter": c,
                    "node_type": "function_end",
                    "params": {"name": name},
                    "connections": [],
                    "duration_ns": 100,
                }
            )
            c += 1
        nodes.append({"counter": c, "node_type": "capture_end", "params": {}, "connections": []})
        return nodes

    def test_leading_from_torch_filtered(self, tmp_path):
        """from_torch before any compute op should be filtered out."""
        graph = self._make_graph(["ttnn.from_torch", "ttnn.from_torch", "ttnn.conv2d"])

        report = _make_report(graph)
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute("SELECT name FROM operations ORDER BY operation_id")
        ops = [r[0] for r in cursor.fetchall()]
        assert "ttnn.from_torch" not in ops, "Leading from_torch should be filtered"
        assert "ttnn.conv2d" in ops
        conn.close()

    def test_from_torch_after_compute_kept(self, tmp_path):
        """from_torch after a compute op should be kept."""
        graph = self._make_graph(["ttnn.conv2d", "ttnn.from_torch"])

        report = _make_report(graph)
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute("SELECT name FROM operations ORDER BY operation_id")
        ops = [r[0] for r in cursor.fetchall()]
        assert ops == ["ttnn.conv2d", "ttnn.from_torch"]
        conn.close()

    def test_mixed_leading_block(self, tmp_path):
        """from_torch interleaved with filtered ops before compute should all be filtered."""
        graph = self._make_graph(
            [
                "ttnn.from_torch",
                "Tensor::to_device",
                "ttnn.from_torch",
                "ttnn.conv2d",
                "ttnn.from_torch",
            ]
        )

        report = _make_report(graph)
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute("SELECT name FROM operations ORDER BY operation_id")
        ops = [r[0] for r in cursor.fetchall()]
        assert ops == ["ttnn.conv2d", "ttnn.from_torch"]
        conn.close()


class TestFilteredOpPrefixes:
    """Tests that _FILTERED_OP_PREFIXES operations are transparent."""

    @staticmethod
    def _make_flat_graph(op_names):
        nodes = [{"counter": 0, "node_type": "capture_start", "params": {}, "connections": []}]
        c = 1
        for name in op_names:
            nodes.append(
                {
                    "counter": c,
                    "node_type": "function_start",
                    "params": {"name": name},
                    "connections": [],
                    "input_tensors": [],
                }
            )
            c += 1
            nodes.append(
                {
                    "counter": c,
                    "node_type": "function_end",
                    "params": {"name": name},
                    "connections": [],
                    "duration_ns": 100,
                }
            )
            c += 1
        nodes.append({"counter": c, "node_type": "capture_end", "params": {}, "connections": []})
        return nodes

    @pytest.mark.parametrize(
        "filtered_op",
        [
            "Tensor::deallocate",
            "Tensor::to_device",
            "Tensor::reshape",
            "tt::tt_metal::to_dtype",
            "ttnn::convert_python_tensor_to_tt_tensor",
            "tt::tt_metal::detail::convert_tt_tensor_to_framework_tensor",
        ],
    )
    def test_filtered_op_prefix_excluded(self, tmp_path, filtered_op):
        """Each _FILTERED_OP_PREFIXES entry should be excluded from operations."""
        graph = self._make_flat_graph([filtered_op, "ttnn.relu"])
        report = _make_report(graph)
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute("SELECT name FROM operations")
        ops = [r[0] for r in cursor.fetchall()]
        assert filtered_op not in ops, f"{filtered_op} should be filtered"
        assert "ttnn.relu" in ops, "Non-filtered op should remain"
        conn.close()


class TestPythonIONameMatching:
    """Tests that python_io records are matched to graph nodes by name."""

    def test_multiple_ops_same_name_matched_in_order(self, tmp_path):
        """Two ops with the same name should consume python_io records in order."""
        mock_graph = [
            {"counter": 0, "node_type": "capture_start", "params": {}, "connections": []},
            {
                "counter": 1,
                "node_type": "function_start",
                "params": {"name": "ttnn.relu"},
                "connections": [],
                "input_tensors": [],
            },
            {
                "counter": 2,
                "node_type": "function_end",
                "params": {"name": "ttnn.relu"},
                "connections": [],
                "duration_ns": 100,
            },
            {
                "counter": 3,
                "node_type": "function_start",
                "params": {"name": "ttnn.relu"},
                "connections": [],
                "input_tensors": [],
            },
            {
                "counter": 4,
                "node_type": "function_end",
                "params": {"name": "ttnn.relu"},
                "connections": [],
                "duration_ns": 200,
            },
            {"counter": 5, "node_type": "capture_end", "params": {}, "connections": []},
        ]
        python_io = [
            {"name": "ttnn.relu", "arguments": {"alpha": "1.0"}, "input_tensor_ids": []},
            {"name": "ttnn.relu", "arguments": {"alpha": "2.0"}, "input_tensor_ids": []},
        ]

        report = _make_report(mock_graph, python_io=python_io)
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute(
            """
            SELECT oa.value FROM operation_arguments oa
            JOIN operations o ON oa.operation_id = o.operation_id
            WHERE oa.name = 'alpha'
            ORDER BY o.operation_id
        """
        )
        values = [r[0] for r in cursor.fetchall()]
        assert values == ["1.0", "2.0"], f"Expected ordered matching, got {values}"
        conn.close()

    def test_unmatched_python_io_ignored(self, tmp_path):
        """python_io records for non-existent ops should be silently ignored."""
        mock_graph = [
            {"counter": 0, "node_type": "capture_start", "params": {}, "connections": []},
            {
                "counter": 1,
                "node_type": "function_start",
                "params": {"name": "ttnn.relu"},
                "connections": [],
                "input_tensors": [],
            },
            {
                "counter": 2,
                "node_type": "function_end",
                "params": {"name": "ttnn.relu"},
                "connections": [],
                "duration_ns": 100,
            },
            {"counter": 3, "node_type": "capture_end", "params": {}, "connections": []},
        ]
        python_io = [
            {"name": "ttnn.relu", "arguments": {"x": "tensor"}, "input_tensor_ids": []},
            {"name": "ttnn.nonexistent", "arguments": {"y": "other"}, "input_tensor_ids": []},
        ]

        report = _make_report(mock_graph, python_io=python_io)
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute("SELECT COUNT(*) FROM operations")
        assert cursor.fetchone()[0] == 1
        conn.close()


class TestCapturedGraphFallbackExtraction:
    """Tests for the global-graph extraction fallback (when no per-op captured_graph)."""

    def test_extracted_graph_has_sequential_counters(self, tmp_path):
        """Fallback extraction should renumber counters to be sequential."""
        mock_graph = [
            {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [100]},
            {
                "counter": 100,
                "node_type": "function_start",
                "params": {"name": "ttnn.relu"},
                "connections": [102],
                "input_tensors": [],
            },
            {"counter": 101, "node_type": "tensor", "params": {"tensor_id": "1"}, "connections": [100]},
            {
                "counter": 102,
                "node_type": "function_end",
                "params": {"name": "ttnn.relu"},
                "connections": [103],
                "duration_ns": 100,
            },
            {"counter": 103, "node_type": "tensor", "params": {"tensor_id": "2"}, "connections": []},
            {"counter": 999, "node_type": "capture_end", "params": {}, "connections": []},
        ]

        report = _make_report(mock_graph)
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute("SELECT captured_graph FROM captured_graph")
        rows = cursor.fetchall()
        assert len(rows) == 1
        cg = json.loads(rows[0][0])
        counters = [n["counter"] for n in cg]
        assert counters == list(range(len(cg))), f"Counters should be sequential: {counters}"
        assert cg[0]["node_type"] == "capture_start"
        assert cg[-1]["node_type"] == "capture_end"
        conn.close()

    def test_extracted_graph_connections_remapped(self, tmp_path):
        """Connections in extracted graph should reference local counters."""
        mock_graph = [
            {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [50]},
            {
                "counter": 50,
                "node_type": "function_start",
                "params": {"name": "ttnn.relu"},
                "connections": [52],
                "input_tensors": [51],
            },
            {"counter": 51, "node_type": "tensor", "params": {"tensor_id": "1"}, "connections": [50]},
            {
                "counter": 52,
                "node_type": "function_end",
                "params": {"name": "ttnn.relu"},
                "connections": [],
                "duration_ns": 100,
            },
            {"counter": 999, "node_type": "capture_end", "params": {}, "connections": []},
        ]

        report = _make_report(mock_graph)
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute("SELECT captured_graph FROM captured_graph")
        cg = json.loads(cursor.fetchone()[0])
        max_counter = max(n["counter"] for n in cg)
        for node in cg:
            for conn_id in node.get("connections", []):
                assert conn_id <= max_counter, (
                    f"Connection {conn_id} exceeds max counter {max_counter} "
                    f"in node {node['counter']} ({node['node_type']})"
                )
        conn.close()

    def test_parent_wrapper_included_in_captured_graph(self, tmp_path):
        """The parent function_start/end should appear in the extracted subgraph."""
        mock_graph = [
            {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [1]},
            {
                "counter": 1,
                "node_type": "function_start",
                "params": {"name": "ttnn.relu"},
                "connections": [2, 4],
                "input_tensors": [],
            },
            {
                "counter": 2,
                "node_type": "function_start",
                "params": {"name": "InnerOp"},
                "connections": [3],
                "input_tensors": [],
            },
            {
                "counter": 3,
                "node_type": "function_end",
                "params": {"name": "InnerOp"},
                "connections": [],
                "duration_ns": 50,
            },
            {
                "counter": 4,
                "node_type": "function_end",
                "params": {"name": "ttnn.relu"},
                "connections": [],
                "duration_ns": 100,
            },
            {"counter": 5, "node_type": "capture_end", "params": {}, "connections": []},
        ]

        report = _make_report(mock_graph)
        conn, cursor = _import_to_db(report, tmp_path)

        cursor.execute("SELECT captured_graph FROM captured_graph")
        cg = json.loads(cursor.fetchone()[0])
        fn_names = [
            n.get("params", {}).get("name", "") for n in cg if n["node_type"] in ("function_start", "function_end")
        ]
        assert "ttnn.relu" in fn_names, f"Parent wrapper should be included in captured_graph, found: {fn_names}"
        conn.close()


class TestVersionedBufferPages:
    """Tests that versioned buffer_pages_by_address with alloc_counter works."""

    def test_reallocation_picks_correct_snapshot(self, tmp_path):
        """When an address is re-allocated, each op should get the snapshot active at its time."""
        mock_report = {
            "version": 1,
            "graph": [
                {"counter": 0, "node_type": "capture_start", "params": {}, "connections": []},
                {
                    "counter": 1,
                    "node_type": "function_start",
                    "params": {"name": "ttnn.fold"},
                    "connections": [],
                    "input_tensors": [],
                },
                {
                    "counter": 10,
                    "node_type": "function_end",
                    "params": {"name": "ttnn.fold"},
                    "connections": [],
                    "duration_ns": 100,
                },
                {
                    "counter": 11,
                    "node_type": "function_start",
                    "params": {"name": "ttnn.max_pool2d"},
                    "connections": [],
                    "input_tensors": [],
                },
                {
                    "counter": 20,
                    "node_type": "function_end",
                    "params": {"name": "ttnn.max_pool2d"},
                    "connections": [],
                    "duration_ns": 200,
                },
                {"counter": 21, "node_type": "capture_end", "params": {}, "connections": []},
            ],
            "devices": [],
            "metadata": {},
            "per_operation_buffers": {
                "1": [{"address": 5000, "size": 448, "type": "L1"}],
                "11": [{"address": 5000, "size": 128, "type": "L1"}],
            },
            "buffer_pages_by_address": {
                "5000": [
                    {
                        "alloc_counter": 3,
                        "pages": [
                            {
                                "device_id": 0,
                                "address": 5000,
                                "core_y": 0,
                                "core_x": 0,
                                "bank_id": 0,
                                "page_index": i,
                                "page_address": 5000 + i * 448,
                                "page_size": 448,
                                "buffer_type": 1,
                            }
                            for i in range(4)
                        ],
                    },
                    {
                        "alloc_counter": 15,
                        "pages": [
                            {
                                "device_id": 0,
                                "address": 5000,
                                "core_y": 0,
                                "core_x": 0,
                                "bank_id": 0,
                                "page_index": i,
                                "page_address": 5000 + i * 128,
                                "page_size": 128,
                                "buffer_type": 1,
                            }
                            for i in range(8)
                        ],
                    },
                ],
            },
        }

        report_path = tmp_path / "report.json"
        with open(report_path, "w") as f:
            json.dump(mock_report, f)

        output_dir = tmp_path / "output"
        db_path = graph_report.import_report(report_path, output_dir)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT o.name, COUNT(*) FROM buffer_pages bp "
            "JOIN operations o ON bp.operation_id = o.operation_id "
            "GROUP BY o.name ORDER BY o.name"
        )
        results = {name: count for name, count in cursor.fetchall()}

        assert (
            results.get("ttnn.fold") == 4
        ), f"fold should use alloc_counter=3 snapshot (4 pages), got {results.get('ttnn.fold')}"
        assert (
            results.get("ttnn.max_pool2d") == 8
        ), f"max_pool2d should use alloc_counter=15 snapshot (8 pages), got {results.get('ttnn.max_pool2d')}"
        conn.close()

    def test_simple_format_still_works(self, tmp_path):
        """Non-versioned buffer_pages_by_address (flat list) should still import."""
        mock_report = {
            "version": 1,
            "graph": [
                {"counter": 0, "node_type": "capture_start", "params": {}, "connections": []},
                {
                    "counter": 1,
                    "node_type": "function_start",
                    "params": {"name": "ttnn.add"},
                    "connections": [],
                    "input_tensors": [],
                },
                {
                    "counter": 2,
                    "node_type": "function_end",
                    "params": {"name": "ttnn.add"},
                    "connections": [],
                    "duration_ns": 100,
                },
                {"counter": 3, "node_type": "capture_end", "params": {}, "connections": []},
            ],
            "devices": [],
            "metadata": {},
            "per_operation_buffers": {
                "1": [{"address": 1000, "size": 64, "type": "L1"}],
            },
            "buffer_pages_by_address": {
                "1000": [
                    {
                        "device_id": 0,
                        "address": 1000,
                        "core_y": 0,
                        "core_x": 0,
                        "bank_id": 0,
                        "page_index": 0,
                        "page_address": 1000,
                        "page_size": 64,
                        "buffer_type": 1,
                    },
                ],
            },
        }

        report_path = tmp_path / "report.json"
        with open(report_path, "w") as f:
            json.dump(mock_report, f)

        output_dir = tmp_path / "output"
        db_path = graph_report.import_report(report_path, output_dir)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM buffer_pages")
        assert cursor.fetchone()[0] == 1
        conn.close()


class TestFastOperationGraphTracking:
    """Tests that FastOperation emits track_function_start/end during graph capture."""

    def test_python_op_names_in_graph(self, device, tmp_path):
        """FastOperation.__call__ should produce Python-level function_start/end nodes."""
        torch_input = torch.randn(1, 32)
        tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

        report_path = str(tmp_path / "report.json")
        ttnn.graph.begin_graph_capture()
        with ttnn.manage_config("enable_fast_runtime_mode", True):
            output = ttnn.add(tt_input, tt_input)
        ttnn.graph.end_graph_capture_to_file(report_path)

        with open(report_path) as f:
            report = json.load(f)

        fn_names = [
            n["params"]["name"]
            for n in report["graph"]
            if n.get("node_type") == "function_start" and "name" in n.get("params", {})
        ]
        assert "ttnn.add" in fn_names, f"Expected 'ttnn.add' in function_start names, got: {fn_names}"

    def test_python_io_records_tensor_ids(self, device, tmp_path):
        """FastOperation should record input_tensor_ids and output_tensor_ids in python_io."""
        report_path = str(tmp_path / "report.json")

        with ttnn.manage_config("enable_fast_runtime_mode", True):
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
            a = ttnn.ones([1, 32], layout=ttnn.TILE_LAYOUT, device=device)
            b = ttnn.add(a, a)
            ttnn.graph.end_graph_capture_to_file(report_path)

        report_path = Path(report_path)
        sidecar_path = report_path.with_suffix(".python_io.json")
        assert sidecar_path.exists(), f"python_io sidecar file not found at {sidecar_path}"

        with open(sidecar_path) as f:
            py_io = json.load(f)

        names = [r["name"] for r in py_io]
        assert "ttnn.ones" in names, f"Expected ttnn.ones in python_io, got {names}"
        assert "ttnn.add" in names, f"Expected ttnn.add in python_io, got {names}"

        add_record = next(r for r in py_io if r["name"] == "ttnn.add")
        assert len(add_record.get("input_tensor_ids", [])) > 0, "ttnn.add should have input_tensor_ids"
        assert len(add_record.get("output_tensor_ids", [])) > 0, "ttnn.add should have output_tensor_ids"

    def test_tensor_connectivity_across_operations(self, device, tmp_path):
        """Output tensor ID of op A should appear as input tensor ID of op B."""
        report_path = tmp_path / "report.json"

        with ttnn.manage_config("enable_fast_runtime_mode", True):
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
            a = ttnn.ones([1, 32], layout=ttnn.TILE_LAYOUT, device=device)
            b = ttnn.add(a, a)
            ttnn.graph.end_graph_capture_to_file(report_path)

        db_path = graph_report.import_report(report_path, tmp_path / "output")
        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        c.execute(
            """SELECT COUNT(DISTINCT ot.tensor_id)
               FROM output_tensors ot
               INNER JOIN input_tensors it ON ot.tensor_id = it.tensor_id"""
        )
        connected = c.fetchone()[0]
        assert connected >= 1, f"Expected at least 1 connected tensor ID, got {connected}"
        conn.close()


class TestPyIdToCppTensorReconciliation:
    """Tests that Python-assigned output tensor IDs get proper tensor entries
    even when they don't appear in any C++ graph tensor node."""

    def test_missing_pyid_created_from_cpp_tensor_node(self, tmp_path):
        """When python_io output_tensor_ids references an ID not in the graph,
        the import should create a tensor entry by cloning from the function_end
        node's connected C++ tensor."""
        graph = [
            {
                "counter": 0,
                "node_type": "capture_start",
                "params": {},
                "connections": [1],
                "arguments": [],
                "input_tensors": [],
                "stacking_level": 0,
            },
            {
                "counter": 1,
                "node_type": "function_start",
                "params": {"name": "ttnn.relu"},
                "connections": [2, 5],
                "arguments": [],
                "input_tensors": [3],
                "stacking_level": 0,
            },
            {
                "counter": 2,
                "node_type": "function_start",
                "params": {"name": "ReluOp"},
                "connections": [4],
                "arguments": [],
                "input_tensors": [3],
                "stacking_level": 1,
            },
            {
                "counter": 3,
                "node_type": "tensor",
                "params": {
                    "tensor_id": "100",
                    "shape": "[1,32]",
                    "dtype": "BFLOAT16",
                    "layout": "TILE",
                    "memory_config": "",
                    "device_id": 0,
                    "address": 1000,
                },
                "connections": [2],
                "arguments": [],
                "input_tensors": [],
                "stacking_level": 0,
            },
            {
                "counter": 4,
                "node_type": "function_end",
                "params": {"name": "ReluOp"},
                "connections": [6],
                "arguments": [],
                "input_tensors": [],
                "stacking_level": 1,
            },
            {
                "counter": 5,
                "node_type": "function_end",
                "params": {"name": "ttnn.relu"},
                "connections": [6],
                "arguments": [],
                "input_tensors": [],
                "stacking_level": 0,
            },
            {
                "counter": 6,
                "node_type": "tensor",
                "params": {
                    "tensor_id": "200",
                    "shape": "[1,32]",
                    "dtype": "BFLOAT16",
                    "layout": "TILE",
                    "memory_config": "",
                    "device_id": 0,
                    "address": 2000,
                },
                "connections": [],
                "arguments": [],
                "input_tensors": [],
                "stacking_level": 0,
            },
            {
                "counter": 7,
                "node_type": "capture_end",
                "params": {},
                "connections": [],
                "arguments": [],
                "input_tensors": [],
                "stacking_level": 0,
            },
        ]
        python_io = [
            {"name": "ttnn.relu", "arguments": {}, "input_tensor_ids": [100], "output_tensor_ids": [999]},
        ]
        report = _make_report(graph, python_io=python_io)
        conn, c = _import_to_db(report, tmp_path)

        c.execute("SELECT tensor_id FROM output_tensors")
        out_ids = {r[0] for r in c.fetchall()}
        assert 999 in out_ids, f"Python-assigned output ID 999 should be in output_tensors, got {out_ids}"

        c.execute("SELECT tensor_id FROM tensors WHERE tensor_id = 999")
        row = c.fetchone()
        assert row is not None, "Reconciliation should create a tensor entry for Python-assigned ID 999"
        conn.close()


class TestPythonStackTraceImport:
    """Tests that Python stack traces from python_io are stored in the stack_traces table."""

    def test_python_stack_trace_stored_alone(self, tmp_path):
        """Python stack trace stored when no C++ stack trace exists."""
        graph = [
            {
                "counter": 0,
                "node_type": "capture_start",
                "params": {},
                "connections": [1],
                "arguments": [],
                "input_tensors": [],
                "stacking_level": 0,
            },
            {
                "counter": 1,
                "node_type": "function_start",
                "params": {"name": "ttnn::add", "inputs": 2},
                "connections": [],
                "arguments": [],
                "input_tensors": [],
                "stacking_level": 0,
            },
            {
                "counter": 2,
                "node_type": "function_end",
                "params": {"name": "ttnn::add"},
                "connections": [],
                "arguments": [],
                "input_tensors": [],
                "stacking_level": 0,
            },
            {
                "counter": 3,
                "node_type": "capture_end",
                "params": {},
                "connections": [],
                "arguments": [],
                "input_tensors": [],
                "stacking_level": 0,
            },
        ]
        python_io = [
            {
                "name": "ttnn::add",
                "arguments": {"a": "tensor_a", "b": "tensor_b"},
                "input_tensor_ids": [],
                "python_stack_trace": [
                    '  File "my_model.py", line 42, in forward\n    out = ttnn.add(x, y)',
                    '  File "train.py", line 10, in main\n    model.forward(batch)',
                ],
            }
        ]
        report = _make_report(graph, python_io=python_io)
        conn, c = _import_to_db(report, tmp_path)
        c.execute("SELECT stack_trace FROM stack_traces WHERE operation_id = 1")
        row = c.fetchone()
        assert row is not None, "Stack trace should be stored"
        assert "my_model.py" in row[0]
        assert "train.py" in row[0]
        assert "--- C++ ---" not in row[0], "No C++ separator when only Python trace"
        conn.close()

    def test_python_stack_trace_replaces_cpp(self, tmp_path):
        """Python stack trace replaces C++ trace for Python-level ops."""
        graph = [
            {
                "counter": 0,
                "node_type": "capture_start",
                "params": {},
                "connections": [1],
                "arguments": [],
                "input_tensors": [],
                "stacking_level": 0,
            },
            {
                "counter": 1,
                "node_type": "function_start",
                "params": {"name": "ttnn::add", "inputs": 2},
                "connections": [],
                "arguments": [],
                "input_tensors": [],
                "stacking_level": 0,
                "stack_trace": ["lib.so(ttnn::add+0x123) [0x7f]", "lib.so(caller+0x456) [0x7f]"],
            },
            {
                "counter": 2,
                "node_type": "function_end",
                "params": {"name": "ttnn::add"},
                "connections": [],
                "arguments": [],
                "input_tensors": [],
                "stacking_level": 0,
            },
            {
                "counter": 3,
                "node_type": "capture_end",
                "params": {},
                "connections": [],
                "arguments": [],
                "input_tensors": [],
                "stacking_level": 0,
            },
        ]
        python_io = [
            {
                "name": "ttnn::add",
                "arguments": {},
                "input_tensor_ids": [],
                "python_stack_trace": ['  File "model.py", line 5, in run\n    ttnn.add(a, b)'],
            }
        ]
        report = _make_report(graph, python_io=python_io)
        conn, c = _import_to_db(report, tmp_path)
        c.execute("SELECT stack_trace FROM stack_traces WHERE operation_id = 1")
        row = c.fetchone()
        assert row is not None
        trace = row[0]
        assert "model.py" in trace, "Python trace should be stored"
        assert "lib.so" not in trace, "C++ trace should be replaced, not kept"
        conn.close()

    def test_no_python_stack_trace_uses_cpp_only(self, tmp_path):
        """When python_io has no stack trace, C++ trace used as-is."""
        graph = [
            {
                "counter": 0,
                "node_type": "capture_start",
                "params": {},
                "connections": [1],
                "arguments": [],
                "input_tensors": [],
                "stacking_level": 0,
            },
            {
                "counter": 1,
                "node_type": "function_start",
                "params": {"name": "ttnn::add", "inputs": 2},
                "connections": [],
                "arguments": [],
                "input_tensors": [],
                "stacking_level": 0,
                "stack_trace": ["lib.so(ttnn::add+0x123) [0x7f]"],
            },
            {
                "counter": 2,
                "node_type": "function_end",
                "params": {"name": "ttnn::add"},
                "connections": [],
                "arguments": [],
                "input_tensors": [],
                "stacking_level": 0,
            },
            {
                "counter": 3,
                "node_type": "capture_end",
                "params": {},
                "connections": [],
                "arguments": [],
                "input_tensors": [],
                "stacking_level": 0,
            },
        ]
        python_io = [{"name": "ttnn::add", "arguments": {}, "input_tensor_ids": []}]
        report = _make_report(graph, python_io=python_io)
        conn, c = _import_to_db(report, tmp_path)
        c.execute("SELECT stack_trace FROM stack_traces WHERE operation_id = 1")
        row = c.fetchone()
        assert row is not None
        assert "lib.so" in row[0]
        assert "--- C++ ---" not in row[0]
        conn.close()
