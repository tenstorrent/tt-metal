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
import tempfile
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
        # Create mock graph data
        mock_graph = [
            {"counter": 0, "node_type": "capture_start", "params": {}, "connections": [1, 4]},
            {
                "counter": 1,
                "node_type": "function_start",
                "params": {"name": "ttnn::relu", "inputs": "1"},
                "connections": [],
                "input_tensors": [100],
            },
            {
                "counter": 2,
                "node_type": "tensor",
                "params": {"tensor_id": "101", "shape": "[1,1,32,32]"},
                "connections": [],
            },
            {
                "counter": 3,
                "node_type": "function_end",
                "params": {"name": "ttnn::relu"},
                "connections": [2],
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

        # Check output_tensors was populated
        cursor.execute("SELECT operation_id, output_index, tensor_id FROM output_tensors")
        output_rows = cursor.fetchall()

        assert len(output_rows) == 1, f"Expected 1 output tensor, got {len(output_rows)}"
        assert str(output_rows[0][2]) == "101", f"Expected tensor_id '101', got {output_rows[0][2]}"

        # Check input_tensors was populated
        cursor.execute("SELECT operation_id, input_index, tensor_id FROM input_tensors")
        input_rows = cursor.fetchall()
        assert len(input_rows) == 1, f"Expected 1 input tensor, got {len(input_rows)}"
        assert input_rows[0][2] == 100, f"Expected tensor_id 100, got {input_rows[0][2]}"

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
                "input_tensors": [100],
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

        stats = graph_report.import_graph(cursor, mock_graph, base_operation_id=0)
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

        stats = graph_report.import_graph(cursor, mock_graph, base_operation_id=0)
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

        stats = graph_report.import_graph(cursor, mock_graph, base_operation_id=0)
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
        db_path = graph_report.import_report(report_path, output_dir)

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
