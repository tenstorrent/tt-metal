# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize("rank", [1, 2])
@pytest.mark.parametrize("first_width", [32, 40])
@pytest.mark.parametrize("memories", [("L1", "DRAM"), ("DRAM", "DRAM"), ("L1", "L1"), ("DRAM", "L1")])
def test_concat_mixed_memory_row_alignment(device, rank, first_width, memories):
    shape_a = (first_width,) if rank == 1 else (4, first_width)
    shape_b = (32,) if rank == 1 else (4, 32)
    a = torch.arange(torch.Size(shape_a).numel(), dtype=torch.float32).reshape(shape_a)
    b = torch.arange(torch.Size(shape_b).numel(), dtype=torch.float32).reshape(shape_b) + 1000
    configs = {"L1": ttnn.L1_MEMORY_CONFIG, "DRAM": ttnn.DRAM_MEMORY_CONFIG}
    inputs = [
        ttnn.from_torch(
            x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=configs[memory]
        )
        for x, memory in zip((a, b), memories)
    ]
    output = ttnn.concat(inputs, dim=-1)
    assert torch.equal(ttnn.to_torch(output), torch.cat((a, b), dim=-1))
