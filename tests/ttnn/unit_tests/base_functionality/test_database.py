# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import sqlite3

import torch

import ttnn
import ttnn.database


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("height", [1024])
@pytest.mark.parametrize("width", [1024])
def test_enable_logging(device, height, width):
    torch.manual_seed(0)

    with ttnn.manage_config("enable_logging", True):
        torch_input_tensor = torch.rand(
            (height, width),
            dtype=torch.bfloat16,
        )

        input_tensor_a = ttnn.from_torch(
            torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )

        input_tensor_b = ttnn.from_torch(
            torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(input_tensor_b)
        ttnn.to_torch(output_tensor)

    sqlite_connection = sqlite3.connect(ttnn.CONFIG.report_path / ttnn.database.SQLITE_DB_PATH)
    cursor = sqlite_connection.cursor()
    cursor.execute("SELECT * FROM operations")
    operations = []
    for row in cursor.fetchall():
        operation = ttnn.database.Operation(*row)
        operations.append(operation)

    assert len(operations) == 5


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("height", [1024])
@pytest.mark.parametrize("width", [1024])
def test_enable_logging_and_enable_graph_report(device, height, width):
    torch.manual_seed(0)

    with ttnn.manage_config("enable_logging", True), ttnn.manage_config("enable_graph_report", True):
        torch_input_tensor = torch.rand(
            (height, width),
            dtype=torch.bfloat16,
        )

        input_tensor_a = ttnn.from_torch(
            torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )

        input_tensor_b = ttnn.from_torch(
            torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.to_torch(output_tensor)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("height", [1024])
@pytest.mark.parametrize("width", [1024])
def test_enable_logging_and_enable_detailed_buffer_report(device, height, width):
    torch.manual_seed(0)

    with ttnn.manage_config("enable_logging", True), ttnn.manage_config("enable_detailed_buffer_report", True):
        torch_input_tensor = torch.rand(
            (height, width),
            dtype=torch.bfloat16,
        )

        input_tensor_a = ttnn.from_torch(
            torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )

        input_tensor_b = ttnn.from_torch(
            torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(input_tensor_b)
        ttnn.to_torch(output_tensor)

    sqlite_connection = sqlite3.connect(ttnn.CONFIG.report_path / ttnn.database.SQLITE_DB_PATH)
    cursor = sqlite_connection.cursor()

    cursor.execute("SELECT * FROM buffers")
    buffers = []
    for row in cursor.fetchall():
        buffer = ttnn.database.Buffer(*row)
        buffers.append(buffer)
    assert len(buffers) > 0

    cursor.execute("SELECT * FROM buffer_pages")
    buffer_pages = []
    for row in cursor.fetchall():
        buffer_page = ttnn.database.BufferPage(*row)
        buffer_pages.append(buffer_page)
    assert len(buffer_pages) > 0


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("height", [1024])
@pytest.mark.parametrize("width", [1024])
def test_enable_logging_and_enable_comparison_mode(device, height, width):
    torch.manual_seed(0)

    with ttnn.manage_config("enable_logging", True), ttnn.manage_config("enable_comparison_mode", True):
        torch_input_tensor = torch.rand(
            (height, width),
            dtype=torch.bfloat16,
        )

        input_tensor_a = ttnn.from_torch(
            torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )

        input_tensor_b = ttnn.from_torch(
            torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.to_torch(output_tensor)

    sqlite_connection = sqlite3.connect(ttnn.CONFIG.report_path / ttnn.database.SQLITE_DB_PATH)
    cursor = sqlite_connection.cursor()
    cursor.execute("SELECT * FROM operations")
    operations = []
    for row in cursor.fetchall():
        operation = ttnn.database.Operation(*row)
        operations.append(operation)

    assert len(operations) > 0


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("height", [1024])
@pytest.mark.parametrize("width", [1024])
def test_enable_logging_and_enable_detailed_tensor_report(device, height, width):
    torch.manual_seed(0)

    with ttnn.manage_config("enable_logging", True), ttnn.manage_config("enable_detailed_tensor_report", True):
        torch_input_tensor = torch.rand(
            (height, width),
            dtype=torch.bfloat16,
        )

        input_tensor_a = ttnn.from_torch(
            torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )

        input_tensor_b = ttnn.from_torch(
            torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.to_torch(output_tensor)


@pytest.mark.requires_fast_runtime_mode_off
def test_sample_data_for_visualizer(device):
    torch.manual_seed(0)

    with ttnn.manage_config("enable_logging", True):
        num_layers = 10

        torch_input_tensor = torch.rand(2048, 2048, dtype=torch.float32)
        input_tensor = ttnn.from_torch(
            torch_input_tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        torch_mask = torch.rand(2048, 2048, dtype=torch.float32)
        mask = ttnn.from_torch(
            torch_mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        weights = [
            ttnn.from_torch(
                torch.rand(2048, 2048, dtype=torch.float32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
            for _ in range(num_layers)
        ]

        for layer_index in range(num_layers):
            intermediate_tensor = ttnn.multiply(input_tensor, mask, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(input_tensor)
            output_tensors = (
                ttnn.silu(intermediate_tensor),
                ttnn.gelu(intermediate_tensor),
                ttnn.relu(intermediate_tensor),
            )
            ttnn.deallocate(intermediate_tensor)
            intermediate_tensor = ttnn.softmax(sum(output_tensors), dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for tensor in output_tensors:
                ttnn.deallocate(tensor)
            output_tensor = ttnn.matmul(intermediate_tensor, weights[layer_index], memory_config=ttnn.L1_MEMORY_CONFIG)
            output_tensor = ttnn.exp(output_tensor)
            input_tensor = output_tensor

        ttnn.to_torch(output_tensor)
        ttnn.deallocate(output_tensor)


@pytest.mark.requires_fast_runtime_mode_off
def test_error_details_saved(device):
    torch.manual_seed(0)

    with ttnn.manage_config("enable_logging", True):
        with pytest.raises(Exception):
            # Get device memory information to calculate appropriate tensor sizes
            device_info = ttnn._ttnn.reports.get_device_info(device)
            l1_memory_per_core = device_info.worker_l1_size
            num_cores = device_info.num_compute_cores
            total_l1_memory = device_info.total_l1_memory

            # Calculate tensor size that should definitely exceed available L1 memory
            # Use a tensor size that's larger than total L1 memory across all cores
            bfloat16_size = 2  # bytes per bfloat16
            target_memory_usage = int(total_l1_memory * 1.5)  # 150% of total L1 memory
            elements_needed = target_memory_usage // bfloat16_size

            # Calculate dimensions for a square tensor with this many elements
            # Ensure dimensions are multiples of 32 for TILE_LAYOUT compatibility
            side_length = int((elements_needed**0.5))
            side_length = ((side_length + 31) // 32) * 32  # Round up to nearest multiple of 32

            # Try to create a tensor that should definitely exceed memory limits
            huge_tensor = ttnn.from_torch(
                torch.rand((side_length, side_length), dtype=torch.bfloat16),
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.synchronize_device(device)

            # If that somehow succeeds, try an operation that requires even more memory
            result = ttnn.exp(huge_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.synchronize_device(device)

            # Clean up if we somehow get here
            ttnn.deallocate(huge_tensor)
            ttnn.deallocate(result)

    sqlite_connection = sqlite3.connect(ttnn.CONFIG.report_path / ttnn.database.SQLITE_DB_PATH)
    cursor = sqlite_connection.cursor()
    cursor.execute("SELECT * FROM errors")
    error_records = []
    for row in cursor.fetchall():
        error_record = ttnn.database.ErrorRecord(*row)
        error_records.append(error_record)

    assert len(error_records) > 0

    error_record = error_records[0]
    assert error_record.operation_id is not None
    assert error_record.operation_name == "ttnn.from_torch"
    assert error_record.error_type == "RuntimeError"
    assert "Out of Memory: Not enough space to allocate" in error_record.error_message
    assert error_record.stack_trace.startswith("Traceback (most recent call last):")
    assert error_record.timestamp is not None
