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


def test_peak_memory_with_broadcast():
    """Test peak L1 memory with broadcast operation - guaranteed non-zero"""
    with ttnn.manage_device(device_id=0) as device:
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NO_DISPATCH)

        # Different shapes force broadcast and intermediate allocation
        input_a = ttnn.from_torch(
            torch.rand(4, 1, 32, 32, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        input_b = ttnn.from_torch(
            torch.rand(1, 1, 32, 32, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        output = ttnn.add(input_a, input_b)
        captured_graph = ttnn.graph.end_graph_capture()

        peak_l1 = ttnn.graph.extract_peak_L1_memory_usage(captured_graph)
        assert isinstance(peak_l1, int)
        assert peak_l1 > 0, f"Expected non-zero peak L1 with broadcast, got {peak_l1}"
        # From C++ tests, different broadcast configs show different peaks:
        # (1,3,32,32)+(1,3,32,32) = ~30KB, (4,3,32,32)+(1,3,32,32) = ~67KB
        assert peak_l1 > 20000, f"Expected >20KB for broadcast operation, got {peak_l1:,} bytes"


def test_peak_memory_larger_tensors():
    """Test peak L1 memory with larger tensors"""
    with ttnn.manage_device(device_id=0) as device:
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NO_DISPATCH)

        # Larger tensors ensure buffer allocations
        input_a = ttnn.from_torch(
            torch.rand(1, 1, 128, 128, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        input_b = ttnn.from_torch(
            torch.rand(1, 1, 128, 128, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        output = ttnn.add(input_a, input_b)
        captured_graph = ttnn.graph.end_graph_capture()

        peak_l1 = ttnn.graph.extract_peak_L1_memory_usage(captured_graph)
        assert isinstance(peak_l1, int)
        assert peak_l1 > 0, f"Expected non-zero peak L1 with larger tensors, got {peak_l1}"
        # Larger tensors should have significant memory usage
        assert peak_l1 > 100000, f"Expected >100KB for 128x128 tensors, got {peak_l1:,} bytes"


def test_peak_memory_chained_operations():
    """Test peak L1 memory with multiple chained operations"""
    with ttnn.manage_device(device_id=0) as device:
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NO_DISPATCH)

        x = ttnn.from_torch(
            torch.rand(1, 1, 64, 64, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Chain multiple operations to accumulate memory allocations
        x = ttnn.relu(x)
        x = ttnn.add(x, x)
        x = ttnn.multiply(x, x)

        captured_graph = ttnn.graph.end_graph_capture()

        peak_l1 = ttnn.graph.extract_peak_L1_memory_usage(captured_graph)
        assert isinstance(peak_l1, int)
        assert peak_l1 > 0, f"Expected non-zero peak L1 with chained ops, got {peak_l1}"


def test_circular_buffers_with_operations():
    """Test CB peak size with actual operations that allocate CBs"""
    with ttnn.manage_device(device_id=0) as device:
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NO_DISPATCH)

        input_a = ttnn.from_torch(
            torch.rand(1, 1, 64, 64, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        input_b = ttnn.from_torch(
            torch.rand(1, 1, 64, 64, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Binary operation should allocate circular buffers
        output = ttnn.add(input_a, input_b)
        captured_graph = ttnn.graph.end_graph_capture()

        cb_peak = ttnn.graph.extract_circular_buffers_peak_size_per_core(captured_graph)
        assert isinstance(cb_peak, int)
        assert cb_peak >= 0
        # Binary operations typically allocate CBs (expected 3*4096 = 12288 from C++ tests)
        if cb_peak > 0:
            assert cb_peak >= 4096, f"Expected CB allocation of at least 4KB, got {cb_peak}"


def test_output_info_with_multiple_outputs():
    """Test extract_output_info with operations producing clear outputs"""
    with ttnn.manage_device(device_id=0) as device:
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NO_DISPATCH)

        input_tensor = ttnn.from_torch(
            torch.rand(2, 1, 64, 64, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Operation that produces output
        output = ttnn.add(input_tensor, input_tensor)
        captured_graph = ttnn.graph.end_graph_capture()

        output_info = ttnn.graph.extract_output_info(captured_graph)
        assert isinstance(output_info, list)
        assert len(output_info) >= 1

        # Verify TensorInfo has expected properties
        for info in output_info:
            assert hasattr(info, "shape")
            assert hasattr(info, "size")
            assert hasattr(info, "type")
            assert info.size > 0
            # Check that shape is reasonable
            assert len(info.shape) == 4, f"Expected 4D shape, got {info.shape}"
