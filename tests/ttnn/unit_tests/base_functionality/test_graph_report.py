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


@pytest.fixture
def tmp_report_dir(tmp_path):
    """Create a temporary directory for reports."""
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    return report_dir


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

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        graph_report.create_database_schema(cursor)

        graph_report.import_graph(cursor, mock_graph, base_operation_id=0)
        conn.commit()

        # Check output_tensors was populated
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

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        graph_report.create_database_schema(cursor)

        graph_report.import_graph(cursor, mock_graph, base_operation_id=0)
        conn.commit()

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

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        graph_report.create_database_schema(cursor)

        stats = graph_report.import_graph(cursor, mock_graph, base_operation_id=0)
        conn.commit()

        cursor.execute("SELECT operation_name, stack_trace FROM stack_traces")
        rows = cursor.fetchall()

        assert len(rows) == 1
        assert rows[0][0] == "ttnn::relu"
        assert "frame1" in rows[0][1]
        assert stats.get("stack_traces", 0) == 1

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

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        graph_report.create_database_schema(cursor)

        stats = graph_report.import_graph(cursor, mock_graph, base_operation_id=0)
        conn.commit()

        cursor.execute("SELECT error_type, error_message, error_operation FROM errors")
        rows = cursor.fetchall()

        assert len(rows) == 1
        assert rows[0][0] == "exception"
        assert rows[0][1] == "Something went wrong"
        assert rows[0][2] == "ttnn::bad_op"
        assert stats.get("errors", 0) == 1

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

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        graph_report.create_database_schema(cursor)

        graph_report.import_graph(cursor, mock_graph, base_operation_id=0)
        conn.commit()

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

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        graph_report.create_database_schema(cursor)

        stats = graph_report.import_graph(cursor, mock_graph, base_operation_id=0)
        conn.commit()

        cursor.execute("SELECT device_id, address, max_size_per_bank, buffer_type FROM buffers")
        rows = cursor.fetchall()

        assert len(rows) == 1
        assert rows[0][0] == 0  # device_id
        assert rows[0][1] == 12345  # address
        assert rows[0][2] == 4096  # size
        assert rows[0][3] == 1  # buffer_type (1=L1)
        assert stats["buffers"] == 1

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
                    # Multi-device tensor spanning 4 devices
                    "device_tensors": '[{"device_id": 0, "address": 12345678}, {"device_id": 1, "address": 22345678}, {"device_id": 2, "address": 32345678}, {"device_id": 3, "address": 42345678}]',
                },
                "connections": [],
            },
            {"counter": 3, "node_type": "capture_end", "params": {}, "connections": []},
        ]

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        graph_report.create_database_schema(cursor)

        stats = graph_report.import_graph(cursor, mock_graph, base_operation_id=0)
        conn.commit()

        # Check device_tensors table
        cursor.execute("SELECT tensor_id, device_id, address FROM device_tensors ORDER BY device_id")
        rows = cursor.fetchall()

        assert len(rows) == 4, f"Expected 4 device tensor entries, got {len(rows)}"
        assert rows[0] == (100, 0, 12345678)
        assert rows[1] == (100, 1, 22345678)
        assert rows[2] == (100, 2, 32345678)
        assert rows[3] == (100, 3, 42345678)

        assert stats.get("device_tensors", 0) == 4

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
                    "buffer_type": "DRAM",
                },
                "connections": [],
            },
            {"counter": 3, "node_type": "capture_end", "params": {}, "connections": []},
        ]

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        graph_report.create_database_schema(cursor)

        graph_report.import_graph(cursor, mock_graph, base_operation_id=0)
        conn.commit()

        cursor.execute(
            "SELECT tensor_id, shape, dtype, layout, memory_config, device_id, address, buffer_type FROM tensors"
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

        conn.close()

    def test_edges_extracted_from_connections(self, tmp_path):
        """Test that edges are extracted from the connections field."""
        # Graph with clear connection structure:
        # 0 (capture_start) -> [1, 4]
        # 1 (function_start) -> [2]
        # 2 (tensor) -> [3]
        # 3 (function_end) -> [4]
        # 4 (capture_end) -> []
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

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        graph_report.create_database_schema(cursor)

        stats = graph_report.import_graph(cursor, mock_graph, base_operation_id=0)
        conn.commit()

        cursor.execute(
            "SELECT source_unique_id, sink_unique_id, source_output_index, sink_input_index FROM edges ORDER BY source_unique_id, source_output_index"
        )
        rows = cursor.fetchall()

        # Expected edges:
        # 0 -> 1 (output_index=0)
        # 0 -> 4 (output_index=1)
        # 1 -> 2 (output_index=0)
        # 2 -> 3 (output_index=0)
        # 3 -> 4 (output_index=0)
        assert len(rows) == 5, f"Expected 5 edges, got {len(rows)}"

        # Check specific edges
        assert (0, 1, 0) == (rows[0][0], rows[0][1], rows[0][2])  # 0 -> 1
        assert (0, 4, 1) == (rows[1][0], rows[1][1], rows[1][2])  # 0 -> 4
        assert (1, 2, 0) == (rows[2][0], rows[2][1], rows[2][2])  # 1 -> 2
        assert (2, 3, 0) == (rows[3][0], rows[3][1], rows[3][2])  # 2 -> 3
        assert (3, 4, 0) == (rows[4][0], rows[4][1], rows[4][2])  # 3 -> 4

        assert stats["edges"] == 5

        conn.close()

    def test_buffer_pages_imported(self, tmp_path):
        """Test that buffer pages are imported when present in the report."""
        mock_report = {
            "version": 1,
            "graph": [
                {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [1]},
                {"counter": 1, "node_type": "capture_end", "params": {}, "connections": []},
            ],
            "devices": [],
            "metadata": {},
            "buffer_pages": [
                {
                    "device_id": 0,
                    "address": 12345678,
                    "core_y": 1,
                    "core_x": 2,
                    "bank_id": 5,
                    "page_index": 0,
                    "page_address": 12345678,
                    "page_size": 1024,
                    "buffer_type": 1,  # L1
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
                    "buffer_type": 1,  # L1
                },
            ],
        }

        # Write mock report
        report_path = tmp_path / "report.json"
        with open(report_path, "w") as f:
            json.dump(mock_report, f)

        # Import
        output_dir = tmp_path / "output"
        db_path = graph_report.import_report(report_path, output_dir)

        # Verify buffer pages were imported
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT device_id, address, core_y, core_x, bank_id, page_index, page_address, page_size, buffer_type FROM buffer_pages ORDER BY page_index"
        )
        rows = cursor.fetchall()
        conn.close()

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

    def test_cluster_mesh_descriptors_saved(self, tmp_path):
        """Test that cluster and mesh descriptors are saved during import."""
        # Create a mock report with cluster and mesh coordinate mapping
        mock_report = {
            "version": 1,
            "graph": [
                {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [1]},
                {"counter": 1, "node_type": "capture_end", "params": {}, "connections": []},
            ],
            "devices": [],
            "metadata": {},
            "cluster_descriptor": "# Mock cluster descriptor YAML\ndevices:\n  - id: 0\n    type: wormhole\n",
            "mesh_coordinate_mapping": "# physical_chip_mesh_coordinate_mapping_1_of_1.yaml\nchips:\n  0: [0, 0]\n",
        }

        # Write mock report
        report_path = tmp_path / "report.json"
        with open(report_path, "w") as f:
            json.dump(mock_report, f)

        # Import
        output_dir = tmp_path / "output"
        graph_report.import_report(report_path, output_dir)

        # Check cluster descriptor was saved
        cluster_path = output_dir / "cluster_descriptor.yaml"
        assert cluster_path.exists(), "cluster_descriptor.yaml should be created"
        with open(cluster_path) as f:
            content = f.read()
        assert "wormhole" in content

        # Check mesh coordinate mapping was saved
        mesh_path = output_dir / "mesh_coordinate_mapping.yaml"
        assert mesh_path.exists(), "mesh_coordinate_mapping.yaml should be created"
        with open(mesh_path) as f:
            content = f.read()
        assert "chips" in content
        assert "0: [0, 0]" in content


class TestInputTensorResolution:
    """Regression tests: input_tensors must resolve node counters to real tensor_ids."""

    def _make_db(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        graph_report.create_database_schema(cursor)
        return conn, cursor

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

        conn, cursor = self._make_db(tmp_path)
        graph_report.import_graph(cursor, mock_graph)
        conn.commit()

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

        conn, cursor = self._make_db(tmp_path)
        graph_report.import_graph(cursor, mock_graph)
        conn.commit()

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

        conn, cursor = self._make_db(tmp_path)
        graph_report.import_graph(cursor, mock_graph)
        conn.commit()

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

        conn, cursor = self._make_db(tmp_path)
        graph_report.import_graph(cursor, mock_graph)
        conn.commit()

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

    def test_linear_model_import_has_no_warnings(self, tmp_path):
        """The full test_linear graph import should produce zero integrity warnings."""
        conn, cursor = self._make_db(tmp_path)
        graph_report.import_devices(cursor, TestLinearModelImport.MOCK_DEVICE_INFO)
        stats = graph_report.import_graph(
            cursor,
            TestLinearModelImport.MOCK_LINEAR_GRAPH,
            base_operation_id=0,
            devices=TestLinearModelImport.MOCK_DEVICE_INFO,
        )
        conn.commit()

        assert (
            stats.get("warnings", []) == []
        ), f"test_linear import should have no warnings after fixes, got: {stats['warnings']}"
        conn.close()


class TestBufferMaxSizePerBank:
    """Regression tests: max_size_per_bank must be per-bank, not total buffer size."""

    def _make_db(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        graph_report.create_database_schema(cursor)
        return conn, cursor

    def _make_buffer_graph(self, size, page_size, buf_type="DRAM", layout="INTERLEAVED", num_cores=0):
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
        The old bug stored 2,097,152 (total size) instead.
        """
        devices = [{"device_id": 0, "num_dram_channels": 12, "l1_num_banks": 64}]
        graph = self._make_buffer_graph(size=2097152, page_size=2048, buf_type="DRAM")

        conn, cursor = self._make_db(tmp_path)
        graph_report.import_graph(cursor, graph, devices=devices)
        conn.commit()

        cursor.execute("SELECT max_size_per_bank FROM buffers")
        rows = cursor.fetchall()
        assert len(rows) == 1
        actual = rows[0][0]
        expected = 86 * 2048  # ceil(1024/12) * page_size = 176128
        assert (
            actual == expected
        ), f"max_size_per_bank should be {expected} (per-bank), not {2097152} (total size). Got {actual}"
        conn.close()

    def test_l1_interleaved_per_bank_size(self, tmp_path):
        """L1 interleaved buffers should use l1_num_banks for per-bank computation."""
        devices = [{"device_id": 0, "num_dram_channels": 12, "l1_num_banks": 64}]
        graph = self._make_buffer_graph(size=65536, page_size=2048, buf_type="L1")

        conn, cursor = self._make_db(tmp_path)
        graph_report.import_graph(cursor, graph, devices=devices)
        conn.commit()

        cursor.execute("SELECT max_size_per_bank FROM buffers")
        rows = cursor.fetchall()
        assert len(rows) == 1
        actual = rows[0][0]
        # 65536 / 2048 = 32 pages, ceil(32/64) = 1 page per bank, 1 * 2048 = 2048
        expected = 2048
        assert actual == expected, f"L1 per-bank should be {expected}, got {actual}"
        conn.close()

    def test_small_dram_buffer_per_bank(self, tmp_path):
        """
        Regression from the real bug: bias tensor buffer (26,624 bytes).
        ceil(13 pages / 12 banks) * 2048 = 4096 per bank.
        """
        devices = [{"device_id": 0, "num_dram_channels": 12, "l1_num_banks": 64}]
        graph = self._make_buffer_graph(size=26624, page_size=2048, buf_type="DRAM")

        conn, cursor = self._make_db(tmp_path)
        graph_report.import_graph(cursor, graph, devices=devices)
        conn.commit()

        cursor.execute("SELECT max_size_per_bank FROM buffers")
        rows = cursor.fetchall()
        actual = rows[0][0]
        expected = 2 * 2048  # ceil(13/12) * 2048 = 4096
        assert actual == expected, f"Bias buffer per-bank should be {expected}, not {26624} (total). Got {actual}"
        conn.close()

    def test_no_device_info_falls_back_to_total_size(self, tmp_path):
        """Without device info, max_size_per_bank falls back to total buffer size."""
        graph = self._make_buffer_graph(size=4096, page_size=2048, buf_type="DRAM")

        conn, cursor = self._make_db(tmp_path)
        graph_report.import_graph(cursor, graph, devices=None)
        conn.commit()

        cursor.execute("SELECT max_size_per_bank FROM buffers")
        rows = cursor.fetchall()
        assert rows[0][0] == 4096, "Without device info, should fall back to total size"
        conn.close()

    def test_sharded_buffer_per_core_size(self, tmp_path):
        """Sharded buffers compute per-bank from num_cores."""
        devices = [{"device_id": 0, "num_dram_channels": 12, "l1_num_banks": 64}]
        # 32768 bytes, 2048 page_size, 8 cores -> 16 pages total, ceil(16/8)=2 pages/core, 2*2048=4096
        graph = self._make_buffer_graph(size=32768, page_size=2048, buf_type="L1", layout="HEIGHT_SHARDED", num_cores=8)

        conn, cursor = self._make_db(tmp_path)
        graph_report.import_graph(cursor, graph, devices=devices)
        conn.commit()

        cursor.execute("SELECT max_size_per_bank FROM buffers")
        rows = cursor.fetchall()
        assert rows[0][0] == 4096, f"Sharded per-core should be 4096, got {rows[0][0]}"
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
        # --- ttnn.ones #1 → Tensor::to_device ---
        {
            "counter": 1,
            "node_type": "function_start",
            "params": {"name": "Tensor::to_device", "inputs": "1"},
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
            "params": {"name": "Tensor::to_device"},
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
                "device_tensors": '[{"device_id": 1, "address": 1920032}]',
            },
            "connections": [19],
        },
        # --- ttnn.ones #2 → Tensor::to_device ---
        {
            "counter": 7,
            "node_type": "function_start",
            "params": {"name": "Tensor::to_device", "inputs": "1"},
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
            "params": {"name": "Tensor::to_device"},
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
                "device_tensors": '[{"device_id": 1, "address": 2096160}]',
            },
            "connections": [19],
        },
        # --- ttnn.ones #3 → Tensor::to_device ---
        {
            "counter": 13,
            "node_type": "function_start",
            "params": {"name": "Tensor::to_device", "inputs": "1"},
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
            "params": {"name": "Tensor::to_device"},
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
                "device_tensors": '[{"device_id": 1, "address": 2272288}]',
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
                "device_tensors": '[{"device_id": 1, "address": 2278432}]',
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
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        graph_report.create_database_schema(cursor)
        graph_report.import_devices(cursor, self.MOCK_DEVICE_INFO)
        graph_report.import_graph(cursor, self.MOCK_LINEAR_GRAPH, base_operation_id=0, devices=self.MOCK_DEVICE_INFO)
        conn.commit()
        return conn, cursor

    def test_operations_count_and_names(self, tmp_path):
        """The linear model should produce 5 operations matching the C++ trace."""
        conn, cursor = self._import_linear_graph(tmp_path)

        cursor.execute("SELECT operation_id, name FROM operations ORDER BY operation_id")
        rows = cursor.fetchall()

        assert len(rows) == 5
        assert rows[0][1] == "Tensor::to_device"
        assert rows[1][1] == "Tensor::to_device"
        assert rows[2][1] == "Tensor::to_device"
        assert rows[3][1] == "tt::tt_metal::create_device_tensor"
        assert rows[4][1] == "MatmulDeviceOperation"
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
        """Each Tensor::to_device should take one host tensor as input."""
        conn, cursor = self._import_linear_graph(tmp_path)

        cursor.execute("SELECT operation_id, tensor_id FROM input_tensors WHERE operation_id < 3 ORDER BY operation_id")
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
        assert output_map[0] == 1, "to_device #1 output should be tensor 1"
        assert output_map[1] == 3, "to_device #2 output should be tensor 3"
        assert output_map[2] == 5, "to_device #3 output should be tensor 5"
        assert output_map[3] == 7, "create_device_tensor output should be tensor 7"
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

    def test_device_tensors_have_correct_addresses(self, tmp_path):
        """Device tensor addresses should match the correct DB."""
        conn, cursor = self._import_linear_graph(tmp_path)

        cursor.execute("SELECT tensor_id, device_id, address FROM device_tensors ORDER BY CAST(tensor_id AS INTEGER)")
        rows = cursor.fetchall()

        expected = [
            (1, 1920032),
            (3, 2096160),
            (5, 2272288),
            (7, 2278432),
        ]
        assert len(rows) == len(expected), f"Expected {len(expected)} device_tensors, got {len(rows)}"
        for row, (exp_tid, exp_addr) in zip(rows, expected):
            tid = int(row[0]) if isinstance(row[0], str) else row[0]
            assert tid == exp_tid, f"Expected tensor_id {exp_tid}, got {tid}"
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

    def test_full_import_report_flow(self, tmp_path):
        """Test through the full import_report entry point with the linear model."""
        report = {
            "version": graph_report.SUPPORTED_REPORT_VERSION,
            "graph": self.MOCK_LINEAR_GRAPH,
            "devices": self.MOCK_DEVICE_INFO,
            "metadata": {"model": "test_linear"},
        }

        report_path = tmp_path / "linear_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f)

        output_dir = tmp_path / "output"
        db_path = graph_report.import_report(report_path, output_dir)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM operations")
        assert cursor.fetchone()[0] == 5

        cursor.execute("SELECT COUNT(*) FROM input_tensors")
        input_count = cursor.fetchone()[0]
        assert input_count == 6, f"Expected 6 input_tensor rows (3 to_device + 3 matmul), got {input_count}"

        cursor.execute("SELECT tensor_id FROM tensors")
        valid_ids = {int(r[0]) if isinstance(r[0], str) else r[0] for r in cursor.fetchall()}
        cursor.execute("SELECT tensor_id FROM input_tensors")
        for row in cursor.fetchall():
            tid = int(row[0]) if isinstance(row[0], str) else row[0]
            assert tid in valid_ids, f"Dangling input_tensors reference: {tid} not in {sorted(valid_ids)}"

        cursor.execute("SELECT max_size_per_bank FROM buffers WHERE address = 1920032")
        assert cursor.fetchone()[0] == 176128, "Per-bank size should be computed, not total"

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
        assert duration >= 0

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
        assert len(durations) > 0
        for name, dur in durations.items():
            assert dur >= 0, f"Duration for {name} should be non-negative"


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

        # Check devices table
        cursor.execute("SELECT COUNT(*) FROM devices")
        assert cursor.fetchone()[0] > 0, "devices table should have data"

        # Check operations table
        cursor.execute("SELECT COUNT(*) FROM operations")
        assert cursor.fetchone()[0] >= 0, "operations table should exist"

        # Check captured_graph table
        cursor.execute("SELECT COUNT(*) FROM captured_graph")
        assert cursor.fetchone()[0] > 0, "captured_graph table should have data"

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

    def test_stack_traces_disabled_by_default(self):
        """Test that stack traces are disabled by default."""
        assert not ttnn.graph.is_stack_trace_enabled()

    def test_enable_disable_stack_traces(self):
        """Test enabling and disabling stack traces."""
        assert not ttnn.graph.is_stack_trace_enabled()

        ttnn.graph.enable_stack_traces()
        assert ttnn.graph.is_stack_trace_enabled()

        ttnn.graph.disable_stack_traces()
        assert not ttnn.graph.is_stack_trace_enabled()

    def test_stack_traces_captured_when_enabled(self, device):
        """Test that stack traces are captured in function_start nodes when enabled."""
        ttnn.graph.enable_stack_traces()
        try:
            torch_input = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16)
            tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
            _ = ttnn.relu(tt_input)
            captured_graph = ttnn.graph.end_graph_capture()

            function_starts = [n for n in captured_graph if n["node_type"] == "function_start"]
            assert len(function_starts) > 0, "Should have at least one function_start node"

            # At least one should have a stack trace (on Linux/macOS)
            has_stack_trace = any("stack_trace" in n for n in function_starts)
            if has_stack_trace:
                for node in function_starts:
                    if "stack_trace" in node:
                        assert isinstance(node["stack_trace"], list)
        finally:
            ttnn.graph.disable_stack_traces()

    def test_no_stack_traces_when_disabled(self, device):
        """Test that no stack traces are captured when disabled."""
        ttnn.graph.disable_stack_traces()

        torch_input = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16)
        tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        _ = ttnn.relu(tt_input)
        captured_graph = ttnn.graph.end_graph_capture()

        for node in captured_graph:
            assert "stack_trace" not in node, f"Node {node['node_type']} should not have stack_trace"
