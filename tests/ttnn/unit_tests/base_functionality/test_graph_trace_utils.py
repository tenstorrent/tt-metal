# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


def test_extract_peak_L1_memory_usage():
    """Test peak L1 memory extraction from graph trace"""
    with ttnn.manage_device(device_id=0) as device:
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NO_DISPATCH)
        input_tensor = ttnn.from_torch(
            torch.rand(1, 1, 32, 32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        output = ttnn.relu(input_tensor)
        captured_graph = ttnn.graph.end_graph_capture()

        peak_l1 = ttnn.graph.extract_peak_L1_memory_usage(captured_graph)
        assert isinstance(peak_l1, int)
        assert peak_l1 >= 0


def test_count_intermediate_and_output_tensors():
    """Test tensor counting from graph trace"""
    with ttnn.manage_device(device_id=0) as device:
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NO_DISPATCH)
        input_tensor = ttnn.from_torch(
            torch.rand(1, 1, 32, 32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        output = ttnn.add(input_tensor, input_tensor)
        captured_graph = ttnn.graph.end_graph_capture()

        intermediate, output_count = ttnn.graph.count_intermediate_and_output_tensors(captured_graph)
        assert isinstance(intermediate, int)
        assert isinstance(output_count, int)
        assert intermediate >= 0
        assert output_count >= 1


def test_extract_output_info():
    """Test output tensor info extraction"""
    with ttnn.manage_device(device_id=0) as device:
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NO_DISPATCH)
        input_tensor = ttnn.from_torch(
            torch.rand(1, 1, 32, 32, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        output = ttnn.relu(input_tensor)
        captured_graph = ttnn.graph.end_graph_capture()

        output_info = ttnn.graph.extract_output_info(captured_graph)
        assert isinstance(output_info, list)
        assert len(output_info) >= 1

        info = output_info[0]
        assert hasattr(info, "shape")
        assert hasattr(info, "size")
        assert hasattr(info, "type")
        assert info.size > 0


def test_extract_circular_buffers_peak_size_per_core():
    """Test CB peak size extraction"""
    with ttnn.manage_device(device_id=0) as device:
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NO_DISPATCH)
        input_tensor = ttnn.from_torch(
            torch.rand(1, 1, 32, 32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        output = ttnn.relu(input_tensor)
        captured_graph = ttnn.graph.end_graph_capture()

        cb_peak = ttnn.graph.extract_circular_buffers_peak_size_per_core(captured_graph)
        assert isinstance(cb_peak, int)
        assert cb_peak >= 0


def test_extract_l1_buffer_allocation_peak_size_per_core():
    """Test L1 buffer peak per core with device grid"""
    with ttnn.manage_device(device_id=0) as device:
        grid_size = device.compute_with_storage_grid_size()
        cores = grid_size.x * grid_size.y

        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NO_DISPATCH)
        input_tensor = ttnn.from_torch(
            torch.rand(1, 1, 32, 32, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        output = ttnn.relu(input_tensor)
        captured_graph = ttnn.graph.end_graph_capture()

        l1_peak_per_core = ttnn.graph.extract_l1_buffer_allocation_peak_size_per_core(captured_graph, cores)
        assert isinstance(l1_peak_per_core, int)
        assert l1_peak_per_core >= 0


def test_empty_trace():
    """Test functions handle empty traces gracefully"""
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NO_DISPATCH)
    captured_graph = ttnn.graph.end_graph_capture()

    # Should handle empty trace without crashing
    peak_l1 = ttnn.graph.extract_peak_L1_memory_usage(captured_graph)
    assert peak_l1 == 0
