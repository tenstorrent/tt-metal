# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic debug tests for g6_repeat: face-row placement and the plan report.

Poisoned-padding coverage lives in test_g6_repeat.py.
"""

import torch
import ttnn

from ttnn.operations.g6_repeat import g6_repeat, plan_report


def test_tile_position_encoding(device):
    """Each element of x's row encodes its column: a misplaced face row shows its source."""
    w, r = 128, 20
    row = torch.arange(w, dtype=torch.float32).reshape(1, 1, 1, w).to(torch.bfloat16)
    x = ttnn.from_torch(
        row, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    got = ttnn.to_torch(g6_repeat(x, [1, 1, r, 1]))
    assert torch.equal(got.view(torch.int16), row.repeat(1, 1, r, 1).view(torch.int16))


def test_plan_report_p1(device):
    x = ttnn.from_torch(
        torch.zeros((1, 1, 1, 256), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    plan = plan_report(x, [1, 1, 16, 1])
    assert plan["regime"] == "column_split_row_broadcast"
    assert plan["num_cores"] == 8
    assert plan["block_width_tiles"] == 1
    assert plan["num_blocks"] == 1
    assert [c["start_tile_col"] for c in plan["cores"]] == list(range(8))
