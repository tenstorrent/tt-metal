# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

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
