# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

DRAM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
L1 = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)


MOVE_OPS = [ttnn.move, ttnn.experimental.quasar.move]
MOVE_OP_IDS = ["ttnn", "quasar"]


@pytest.mark.parametrize("move_op", MOVE_OPS, ids=MOVE_OP_IDS)
@pytest.mark.parametrize("mem_config", [DRAM, L1], ids=["DRAM", "L1"])
@pytest.mark.parametrize("width", [8, 16, 24, 32, 40, 72])
@pytest.mark.parametrize("extra_rows", [1, 3])
def test_move_overlap_row_major_multi_page_per_core(device, move_op, mem_config, width, extra_rows):
    # More rows than cores puts 2+ row-major pages on some cores, so the CB holds several pages back to back.
    grid = device.compute_with_storage_grid_size()
    rows = grid.x * grid.y + extra_rows
    torch.manual_seed(0)
    torch_input = torch.randn(1, 1, rows, width).to(torch.bfloat16)

    # A small freed buffer just below the input makes the output overlap the input (overlap path).
    dummy = ttnn.from_torch(
        torch.zeros(1, 1, 1, 8, dtype=torch.bfloat16),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=mem_config,
    )
    tensor = ttnn.from_torch(torch_input, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config)
    ttnn.deallocate(dummy)

    output = move_op(tensor, memory_config=mem_config)
    assert torch.equal(ttnn.to_torch(output).view(torch.int16), torch_input.view(torch.int16))
