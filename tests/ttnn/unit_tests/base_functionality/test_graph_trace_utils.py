# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import warnings

import torch
import ttnn


def test_extract_peak_L1_memory_usage():
    """Test peak L1 memory extraction from graph trace"""
    with ttnn.manage_device(device_id=0) as device:
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NO_DISPATCH)
        input_tensor = ttnn.from_torch(
            torch.rand(1, 1, 32, 32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        ttnn.relu(input_tensor)
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
        ttnn.add(input_tensor, input_tensor)
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
        ttnn.relu(input_tensor)
        captured_graph = ttnn.graph.end_graph_capture()

        output_info = ttnn.graph.extract_output_info(captured_graph)
        assert isinstance(output_info, list)
        assert len(output_info) >= 1

        info = output_info[0]
        assert hasattr(info, "shape")
        assert hasattr(info, "size")
        assert hasattr(info, "type")
        assert info.size > 0


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
        ttnn.add(input_a, input_b)
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
        ttnn.add(input_a, input_b)
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
        ttnn.multiply(x, x)

        captured_graph = ttnn.graph.end_graph_capture()

        peak_l1 = ttnn.graph.extract_peak_L1_memory_usage(captured_graph)
        assert isinstance(peak_l1, int)
        assert peak_l1 > 0, f"Expected non-zero peak L1 with chained ops, got {peak_l1}"


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
        ttnn.add(input_tensor, input_tensor)
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


def test_no_dispatch_vs_normal_mode_comparison():
    """Compare peak memory between NO_DISPATCH and NORMAL modes"""
    with ttnn.manage_device(device_id=0) as device:
        # Test same operations in both modes
        def run_operations(device):
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
            return ttnn.add(input_a, input_b)

        # NO_DISPATCH mode - theoretical allocation
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NO_DISPATCH)
        _ = run_operations(device)
        graph_no_dispatch = ttnn.graph.end_graph_capture()

        # NORMAL mode - actual allocation with possible fragmentation
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        _ = run_operations(device)
        graph_normal = ttnn.graph.end_graph_capture()

        peak_no_dispatch = ttnn.graph.extract_peak_L1_memory_usage(graph_no_dispatch)
        peak_normal = ttnn.graph.extract_peak_L1_memory_usage(graph_normal)

        print(f"\nMode Comparison:")
        print(f"  NO_DISPATCH peak: {peak_no_dispatch:,} bytes")
        print(f"  NORMAL peak:      {peak_normal:,} bytes")
        print(f"  Difference:       {abs(peak_normal - peak_no_dispatch):,} bytes")

        # Both should be non-zero
        assert peak_no_dispatch > 0, f"NO_DISPATCH should show memory usage, got {peak_no_dispatch}"
        assert peak_normal > 0, f"NORMAL should show memory usage, got {peak_normal}"

        # NORMAL mode may show different values due to fragmentation
        # But both modes should track the same operations
        assert isinstance(peak_no_dispatch, int)
        assert isinstance(peak_normal, int)


def test_normal_mode_shows_real_addresses():
    """Verify NORMAL mode captures real addresses while NO_DISPATCH uses placeholders"""
    with ttnn.manage_device(device_id=0) as device:
        # NO_DISPATCH - should have address 0 or placeholders
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NO_DISPATCH)
        input_tensor = ttnn.from_torch(
            torch.rand(1, 1, 32, 32, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        _ = ttnn.relu(input_tensor)
        graph_no_dispatch = ttnn.graph.end_graph_capture()

        # NORMAL - should have real addresses
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        input_tensor = ttnn.from_torch(
            torch.rand(1, 1, 32, 32, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        _ = ttnn.relu(input_tensor)
        graph_normal = ttnn.graph.end_graph_capture()

        # Check for buffer_allocate nodes and their addresses
        no_dispatch_addresses = []
        normal_addresses = []

        for node in graph_no_dispatch:
            if node.get("node_type") == "buffer_allocate":
                addr = node.get("params", {}).get("address", "0")
                no_dispatch_addresses.append(int(addr))

        for node in graph_normal:
            if node.get("node_type") == "buffer_allocate":
                addr = node.get("params", {}).get("address", "0")
                normal_addresses.append(int(addr))

        print(f"\nAddress Comparison:")
        print(f"  NO_DISPATCH addresses: {no_dispatch_addresses}")
        print(f"  NORMAL addresses:      {normal_addresses}")

        # NO_DISPATCH typically has 0 or placeholder addresses
        # NORMAL should have real non-zero addresses
        if normal_addresses:
            # At least some addresses in NORMAL mode should be non-zero
            has_real_address = any(addr > 0 for addr in normal_addresses)
            assert has_real_address, "NORMAL mode should have real non-zero addresses"


def test_extract_resource_usage_per_core():
    """Test per-core resource usage extraction from graph trace"""
    with ttnn.manage_device(device_id=0) as device:
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NO_DISPATCH)

        # Create tensors with known sizes
        input_a = ttnn.from_torch(
            torch.rand(1, 1, 32, 32, dtype=torch.bfloat16),
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
        ttnn.add(input_a, input_b)
        captured_graph = ttnn.graph.end_graph_capture()

        # Test the new function
        usage = ttnn.graph.extract_resource_usage_per_core(captured_graph)

        # Verify the struct has correct attributes
        assert hasattr(usage, "peak_cb"), "PeakMemoryUsagePerCore should have peak_cb attribute"
        assert hasattr(usage, "peak_l1"), "PeakMemoryUsagePerCore should have peak_l1 attribute"
        assert hasattr(usage, "peak_total"), "PeakMemoryUsagePerCore should have peak_total attribute"

        # Verify types and values
        assert isinstance(usage.peak_cb, int), "peak_cb should be an integer"
        assert isinstance(usage.peak_l1, int), "peak_l1 should be an integer"
        assert isinstance(usage.peak_total, int), "peak_total should be an integer"

        assert usage.peak_cb >= 0, "peak_cb should be non-negative"
        assert usage.peak_l1 >= 0, "peak_l1 should be non-negative"
        assert usage.peak_total >= 0, "peak_total should be non-negative"

        # Verify relationship: total should be sum of CB and L1
        assert (
            usage.peak_total == usage.peak_cb + usage.peak_l1
        ), f"peak_total ({usage.peak_total}) should equal peak_cb ({usage.peak_cb}) + peak_l1 ({usage.peak_l1})"

        # Should have some memory usage for this operation
        assert usage.peak_total > 0, "Expected non-zero memory usage for add operation"

        print(f"\nPer-core resource usage:")
        print(f"  Peak CB:    {usage.peak_cb:,} bytes")
        print(f"  Peak L1:    {usage.peak_l1:,} bytes")
        print(f"  Peak Total: {usage.peak_total:,} bytes")


def test_extract_resource_usage_per_core_repr():
    """Test __repr__ and __str__ methods of PeakMemoryUsagePerCore"""
    with ttnn.manage_device(device_id=0) as device:
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NO_DISPATCH)
        input_tensor = ttnn.from_torch(
            torch.rand(1, 1, 32, 32, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.relu(input_tensor)
        captured_graph = ttnn.graph.end_graph_capture()

        usage = ttnn.graph.extract_resource_usage_per_core(captured_graph)

        # Test __repr__ - should return a valid Python representation
        repr_str = repr(usage)
        assert "PeakMemoryUsagePerCore" in repr_str, "__repr__ should contain class name"
        assert "peak_cb=" in repr_str, "__repr__ should contain peak_cb"
        assert "peak_l1=" in repr_str, "__repr__ should contain peak_l1"
        assert "peak_total=" in repr_str, "__repr__ should contain peak_total"

        # Test __str__ - should return a nicely formatted string
        str_repr = str(usage)
        assert "Peak Memory Usage Per Core" in str_repr, "__str__ should have header"
        assert "CB:" in str_repr, "__str__ should show CB usage"
        assert "L1:" in str_repr, "__str__ should show L1 usage"
        assert "Total:" in str_repr, "__str__ should show total usage"
        assert "bytes" in str_repr, "__str__ should include unit"

        print(f"\n__repr__ output: {repr_str}")
        print(f"\n__str__ output:\n{str_repr}")


def test_extract_resource_usage_per_core_empty():
    """Test per-core resource usage with empty trace"""
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NO_DISPATCH)
    captured_graph = ttnn.graph.end_graph_capture()

    usage = ttnn.graph.extract_resource_usage_per_core(captured_graph)

    # Empty trace should have zero usage
    assert usage.peak_cb == 0, "Empty trace should have zero CB usage"
    assert usage.peak_l1 == 0, "Empty trace should have zero L1 usage"
    assert usage.peak_total == 0, "Empty trace should have zero total usage"


def test_metal2_dataflow_buffers_reach_resource_usage_per_core():
    """A Metal 2.0 op's L1 scratch must show up in the resource usage, under its own kind.

    Ported ops allocate scratch as dataflow buffers and never call CreateCircularBuffer, so
    peak_cb came out 0 for all of them and nothing else accounted for the bytes. #51674
    """
    with ttnn.manage_device(device_id=0) as device:
        input_tensor = ttnn.from_torch(
            torch.rand(1, 1, 64, 128, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            # Row major so repeat lands on the interleaved-RM factory, which builds two DFBs.
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NO_DISPATCH)
        # The forced-native entry is load-bearing: ttnn.repeat routes this case to the codegen
        # path, which uses circular buffers and reports a non-zero peak whether or not DFBs are
        # recorded.
        ttnn._ttnn.operations.data_movement.repeat_force_native(
            input_tensor, [1, 1, 2, 1], memory_config=ttnn.L1_MEMORY_CONFIG
        )
        captured_graph = ttnn.graph.end_graph_capture()

        # Each DFB is (2 * READ_ALIGNMENT) + page_size = 128 + 256 bytes, and neither is borrowed,
        # so both are owned L1 that the peak math must count exactly once.
        dfb_nodes = [node for node in captured_graph if node["node_type"] == "dataflow_buffer_allocate"]
        assert [int(node["params"]["size"]) for node in dfb_nodes] == [384, 384]
        assert all(int(node["params"]["borrows_memory"]) == 0 for node in dfb_nodes)
        # A dataflow buffer is not a circular buffer, and must not be reported as one.
        assert [node for node in captured_graph if node["node_type"] == "circular_buffer_allocate"] == []

        usage = ttnn.graph.extract_resource_usage_per_core(captured_graph)

        assert (
            usage.peak_dataflow_buffer == 768
        ), f"dataflow buffers did not reach the peak, got {usage.peak_dataflow_buffer}"
        assert usage.peak_cb == 0, f"a ported op has no circular buffers, got {usage.peak_cb}"
        assert usage.peak_scratchpad == 0
        # peak_total is the number an L1 budget check has to use.
        assert usage.peak_total == usage.peak_dataflow_buffer + usage.peak_l1


# Metal 2.0 program-scope L1 comes in three kinds and each is reported under its own node type.
# These traces are hand-built because the accounting has to be covered for kinds no Wormhole-reachable
# op produces: kernel scratchpads only appear in Quasar factories today.


def _l1_node(counter, node_type, size, **params):
    all_params = {"size": str(size), "address": "0", "core_range_set": "{[0-0 - 0-0]}", "device_id": "0"}
    all_params.update({key: str(value) for key, value in params.items()})
    return {"counter": counter, "node_type": node_type, "params": all_params, "connections": []}


def _cb(counter, size):
    return _l1_node(counter, "circular_buffer_allocate", size, globally_allocated=0)


def _dfb(counter, size, borrows_memory=0):
    return _l1_node(counter, "dataflow_buffer_allocate", size, borrows_memory=borrows_memory)


def _scratchpad(counter, size):
    return _l1_node(counter, "scratchpad_allocate", size)


def _dealloc_all(counter):
    return {"counter": counter, "node_type": "circular_buffer_deallocate_all", "params": {}, "connections": []}


def test_resource_usage_reports_each_program_l1_kind_separately():
    """Each kind gets its own peak, and every kind lands in the total."""
    trace = [_cb(0, 1024), _dfb(1, 2048), _scratchpad(2, 512)]

    usage = ttnn.graph.extract_resource_usage_per_core(trace)

    assert usage.peak_cb == 1024
    assert usage.peak_dataflow_buffer == 2048
    assert usage.peak_scratchpad == 512
    # peak_total is what an L1 budget check reads, so it must not miss a kind.
    assert usage.peak_total == 3584


def test_resource_usage_excludes_borrowed_dataflow_buffer():
    """A borrowed buffer is a view onto a tensor's L1, which the tensor already reports."""
    trace = [_dfb(0, 4096, borrows_memory=1), _dfb(1, 256)]

    usage = ttnn.graph.extract_resource_usage_per_core(trace)

    assert usage.peak_dataflow_buffer == 256
    assert usage.peak_total == 256


def test_resource_usage_releases_every_l1_kind_between_programs():
    """All three kinds are program-scope and are released together, so two programs do not stack."""
    one_program = [_cb(0, 1024), _dfb(1, 2048), _scratchpad(2, 512)]
    trace = one_program + [_dealloc_all(3), _cb(4, 1024), _dfb(5, 2048), _scratchpad(6, 512)]

    usage = ttnn.graph.extract_resource_usage_per_core(trace)

    assert usage.peak_cb == 1024
    assert usage.peak_dataflow_buffer == 2048
    assert usage.peak_scratchpad == 512
    assert usage.peak_total == 3584


def test_peak_l1_memory_usage_includes_every_program_l1_kind():
    """The single-figure legacy API has to count the new kinds too, or it under-reports."""
    trace = [_dfb(0, 2048), _scratchpad(1, 512)]

    assert ttnn.graph.extract_peak_L1_memory_usage(trace) == 2560


def test_extract_resource_usage_per_core_deprecated_kwarg():
    """The legacy interleaved_storage_cores kwarg must still be accepted (with a
    DeprecationWarning) until the removal date 2026-06-07."""
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NO_DISPATCH)
    captured_graph = ttnn.graph.end_graph_capture()

    baseline = ttnn.graph.extract_resource_usage_per_core(captured_graph)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy = ttnn.graph.extract_resource_usage_per_core(captured_graph, interleaved_storage_cores=64)

    assert any(
        issubclass(w.category, DeprecationWarning) and "interleaved_storage_cores" in str(w.message) for w in caught
    ), f"Expected DeprecationWarning mentioning interleaved_storage_cores, got {[str(w.message) for w in caught]}"
    assert legacy.peak_cb == baseline.peak_cb
    assert legacy.peak_l1 == baseline.peak_l1
    assert legacy.peak_total == baseline.peak_total
