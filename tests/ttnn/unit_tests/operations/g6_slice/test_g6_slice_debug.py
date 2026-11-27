# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Coverage of the narrow-last-column path (LEFT_HALF_LAST_COL) that the acceptance test only hits with
one tile column: several tile columns per row, the last holding <= 16 logical columns, in page_copy and
row_runs, with whole and ragged blocks. Position-encoded input so a misplaced face shows where it went.
"""

import importlib

import pytest
import torch
import ttnn

from ttnn.operations.g6_slice import describe_plan, g6_slice

g6_module = importlib.import_module("ttnn.operations.g6_slice.g6_slice")


CASES = {
    # (shape, starts, ends, regime)
    "page_copy_7_tile_cols_last_4": ((2, 64, 256), (0, 0, 0), (2, 64, 196), "page_copy"),
    "row_runs_3_tile_cols_last_16": ((1, 1, 96, 128), (0, 0, 5, 32), (1, 1, 70, 112), "row_runs"),
}


@pytest.mark.parametrize("block_pages_cap", [None, 3], ids=["default_block", "block3"])
@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_narrow_last_column(device, monkeypatch, case, block_pages_cap):
    if block_pages_cap is not None:
        monkeypatch.setattr(g6_module, "BLOCK_PAGES_CAP", block_pages_cap)
    shape, starts, ends, regime = CASES[case]
    numel = 1
    for d in shape:
        numel *= d
    # bf16 holds integers exactly up to 256; wrap so every value is exact and nearby positions differ.
    torch_x = (torch.arange(numel, dtype=torch.float32) % 251).reshape(shape).to(torch.bfloat16)
    x = ttnn.from_torch(
        torch_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    assert describe_plan(x, list(starts), list(ends))["regime"] == regime
    out = ttnn.to_torch(g6_slice(x, list(starts), list(ends)))
    expected = torch_x[tuple(slice(b, e) for b, e in zip(starts, ends))]
    assert torch.equal(out.view(torch.int16), expected.contiguous().view(torch.int16))
