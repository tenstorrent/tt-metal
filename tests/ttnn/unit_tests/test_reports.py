# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn


@pytest.mark.skip(reason="This test is flaky")
@pytest.mark.parametrize("height", [1024 * 5])
@pytest.mark.parametrize("width", [1024 * 2])
def test_enable_logging(height, width):
    ttnn.CONFIG.enable_logging = True

    torch.manual_seed(0)

    device = ttnn.open_device(device_id=0)

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

    device = ttnn.close_device(device)

    ttnn.CONFIG.enable_logging = False

    sqlite_connection = ttnn.database.get_or_create_sqlite_db()
    cursor = sqlite_connection.cursor()
    cursor.execute("SELECT * FROM operations")
    operations = []
    for row in cursor.fetchall():
        operation = ttnn.database.Operation(*row)
        operations.append(operation)

    assert len(operations) == 5
    for operation in operations:
        assert operation.desired_pcc is not None
        assert operation.actual_pcc is None
        assert operation.matches_golden is None


@pytest.mark.skip(reason="This test is flaky")
@pytest.mark.parametrize("height", [1024 * 5])
@pytest.mark.parametrize("width", [1024 * 2])
def test_enable_logging_and_enable_detailed_buffer_report(height, width):
    ttnn.CONFIG.enable_logging = True
    ttnn.CONFIG.enable_detailed_buffer_report = True

    torch.manual_seed(0)

    device = ttnn.open_device(device_id=0)

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

    device = ttnn.close_device(device)

    ttnn.CONFIG.enable_logging = False
    ttnn.CONFIG.enable_detailed_buffer_report = False

    sqlite_connection = ttnn.database.get_or_create_sqlite_db()
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


@pytest.mark.skip(reason="This test is flaky")
@pytest.mark.parametrize("height", [1024])
@pytest.mark.parametrize("width", [1024])
def test_enable_logging_and_enable_comparison_mode(height, width):
    ttnn.CONFIG.enable_logging = True
    ttnn.CONFIG.enable_comparison_mode = True

    torch.manual_seed(0)

    device = ttnn.open_device(device_id=0)

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

    device = ttnn.close_device(device)

    ttnn.CONFIG.enable_logging = False
    ttnn.CONFIG.enable_comparison_mode = False

    sqlite_connection = ttnn.database.get_or_create_sqlite_db()
    cursor = sqlite_connection.cursor()
    cursor.execute("SELECT * FROM operations")
    operations = []
    for row in cursor.fetchall():
        operation = ttnn.database.Operation(*row)
        operations.append(operation)

    assert len(operations) > 0
    num_compared_operations = 0
    for operation in operations:
        assert operation.desired_pcc is not None
        if operation.name == "ttnn.add":
            assert operation.actual_pcc is not None
            assert operation.matches_golden is not None
            num_compared_operations += 1
    assert num_compared_operations == 1  # Only one operation is compared (ttnn.add)


@pytest.mark.skip(reason="This test is flaky")
@pytest.mark.parametrize("height", [1024])
@pytest.mark.parametrize("width", [1024])
def test_enable_logging_and_enable_tensor_report(height, width):
    ttnn.CONFIG.enable_logging = True
    ttnn.CONFIG.enable_tensor_report = True

    torch.manual_seed(0)

    device = ttnn.open_device(device_id=0)

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

    device = ttnn.close_device(device)

    ttnn.CONFIG.enable_logging = False
    ttnn.CONFIG.enable_tensor_report = False
