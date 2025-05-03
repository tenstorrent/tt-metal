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
