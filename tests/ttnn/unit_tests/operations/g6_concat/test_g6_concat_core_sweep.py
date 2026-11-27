# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Core-count sweep of the row_gather production case (Qwen 79) for bottleneck classification.

Run under `scripts/run_safe_pytest.sh --profile`: a wall that stays flat as cores drop means a shared
resource (DRAM / NoC transaction rate) binds; a wall that scales with 1/cores means per-core issue.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_concat import g6_concat
from ttnn.operations.g6_concat import g6_concat_program_descriptor as program_descriptor


@pytest.mark.parametrize("max_cores", [None, 55, 28])
def test_qwen79_core_sweep(max_cores, device):
    shapes = [(1, 1, 1, 62080)] * 16
    torch_inputs = [torch.randn(s, dtype=torch.float32).to(torch.bfloat16) for s in shapes]
    tt_inputs = [ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) for t in torch_inputs]
    saved = program_descriptor.MAX_CORES
    program_descriptor.MAX_CORES = max_cores
    try:
        output = g6_concat(tt_inputs, 2)
    finally:
        program_descriptor.MAX_CORES = saved
    got = ttnn.to_torch(output)
    assert torch.equal(got.view(torch.int16), torch.cat(torch_inputs, 2).view(torch.int16))
