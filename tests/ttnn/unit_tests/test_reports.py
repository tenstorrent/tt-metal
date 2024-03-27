# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn


@pytest.mark.parametrize("height", [64])
@pytest.mark.parametrize("width", [64])
def test_print_l1_buffers_of_add_operation(tmp_path, height, width):
    ttnn.ENABLE_LOGGING = True
    torch.manual_seed(0)

    device = ttnn.open_device(device_id=0)

    torch_input_tensor = torch.rand(
        (height, width),
        dtype=torch.bfloat16,
    )

    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    output_tensor = ttnn.add(input_tensor, input_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)

    ttnn.to_torch(output_tensor)

    ttnn.print_l1_buffers(str(tmp_path / "l1_buffer_state.txt"))

    device = ttnn.close_device(device)

    with open(tmp_path / "l1_buffer_state.txt") as f:
        l1_buffer_report = f.read()

        GOLDEN_L1_BUFFER_REPORT = """L1 Buffers:
Device: 0
Core: (x=1,y=7)
  Address   1044480:	Buffer   0	Page    1	Page Size      2048
  Address   1046528:	Buffer   1	Page    1	Page Size      2048

Core: (x=2,y=9)
  Address    520192:	Buffer   0	Page    3	Page Size      2048
  Address    522240:	Buffer   1	Page    3	Page Size      2048

Core: (x=9,y=2)
  Address   1044480:	Buffer   0	Page    0	Page Size      2048
  Address   1046528:	Buffer   1	Page    0	Page Size      2048

Core: (x=10,y=9)
  Address   1044480:	Buffer   0	Page    2	Page Size      2048
  Address   1046528:	Buffer   1	Page    2	Page Size      2048


"""
    # assert l1_buffer_report == GOLDEN_L1_BUFFER_REPORT
    ttnn.ENABLE_LOGGING = False


@pytest.mark.parametrize("height", [64])
@pytest.mark.parametrize("width", [64])
def test_enable_l1_buffers_logging(tmp_path, height, width):
    ttnn.ENABLE_LOGGING = True

    torch.manual_seed(0)

    device = ttnn.open_device(device_id=0)

    torch_input_tensor = torch.rand(
        (height, width),
        dtype=torch.bfloat16,
    )

    with ttnn.manage_sqlite_db():
        input_tensor = ttnn.from_torch(
            torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        output_tensor = ttnn.add(input_tensor, input_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.to_torch(output_tensor)

    device = ttnn.close_device(device)

    ttnn.ENABLE_LOGGING = False
