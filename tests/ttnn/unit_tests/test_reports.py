# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn


@pytest.mark.parametrize("height", [1024 * 5])
@pytest.mark.parametrize("width", [1024 * 2])
def test_enable_buffer_report(tmp_path, height, width):
    ttnn.ENABLE_LOGGING = True
    ttnn.ENABLE_BUFFER_REPORT = True

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

    ttnn.ENABLE_LOGGING = False
    ttnn.ENABLE_BUFFER_REPORT = False

    sqlite_connection = ttnn.database.get_or_create_sqlite_db(ttnn.database.DATABASE_FILE)
    cursor = sqlite_connection.cursor()
    cursor.execute("SELECT * FROM buffer_pages")
    buffer_pages = []
    for row in cursor.fetchall():
        buffer_pages.append(row)
    assert len(buffer_pages) > 0


@pytest.mark.parametrize("height", [1024])
@pytest.mark.parametrize("width", [1024])
def test_enable_tensor_report(tmp_path, height, width):
    ttnn.ENABLE_LOGGING = True
    ttnn.ENABLE_TENSOR_REPORT = True

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

    ttnn.ENABLE_LOGGING = False
    ttnn.ENABLE_TENSOR_REPORT = False
