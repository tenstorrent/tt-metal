# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1: the cut along W (dim -1), below validate().

Every dim = -1 entry of the specification's INPUTS (all ranks and chunk counts), in both layouts, DRAM and L1,
bfloat16 and float32, through `_dispatch`, each chunk compared bit-exactly with torch.split. The regimes reached:
tile_col_shift (TILE, split_size % 32 != 0), page_remap (TILE at a tile multiple) and rm_substick (ROW_MAJOR),
whose last-chunk page may be shorter than the others (the writer's sized explicit writes).
"""

import pytest
import torch

import ttnn
from eval.specs.split.spec import INPUTS
from ttnn.operations.split.split import _dispatch
from ttnn.operations.split.split_program_artifacts import (
    PAGE_REMAP,
    RM_SUBSTICK,
    TILE_COL_SHIFT,
    derive_geometry,
    select_regime,
)

MEMORY = {"DRAM": ttnn.DRAM_MEMORY_CONFIG, "L1": ttnn.L1_MEMORY_CONFIG}
CUT_W = [(shape, s, d) for shape, s, d in INPUTS if (d if d < 0 else d - len(shape)) == -1]


def _id(entry):
    shape, s, d = entry
    return "x".join(map(str, shape)) + f"_s{s}_dim{d}"


@pytest.mark.parametrize("memory", ["DRAM", "L1"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["TILE", "RM"])
@pytest.mark.parametrize("entry", CUT_W, ids=_id)
def test_cut_w_below_validate(device, entry, layout, dtype, memory):
    shape, split_size, dim = entry
    torch.manual_seed(13)
    host = torch.randn(shape, dtype=torch.float32)
    tensor = ttnn.from_torch(host, dtype=dtype, layout=layout, device=device, memory_config=MEMORY[memory])
    stored = ttnn.to_torch(tensor)  # the values as the device holds them (bfloat16 rounds once, on the way in)
    chunks = _dispatch(tensor, split_size, -1, MEMORY[memory])
    expected = torch.split(stored, split_size, dim=-1)
    assert isinstance(chunks, list) and len(chunks) == len(expected)
    for k, (chunk, want) in enumerate(zip(chunks, expected)):
        assert list(chunk.shape) == list(want.shape), (k, chunk.shape, want.shape)
        assert chunk.dtype == dtype and chunk.layout == layout, k
        assert chunk.memory_config().buffer_type == MEMORY[memory].buffer_type, k
        got = ttnn.to_torch(chunk)
        assert torch.equal(got, want), f"chunk {k}: max diff {(got.float() - want.float()).abs().max()}"


def test_cut_w_entries_reach_both_new_regimes():
    """The dim -1 entries above reach tile_col_shift, rm_substick and page_remap, and each has a geometry."""
    regimes = set()
    for shape, split_size, _ in CUT_W:
        for layout in (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT):
            tensor = ttnn.from_torch(torch.zeros(shape, dtype=torch.bfloat16), layout=layout)
            regime, n_chunks, num_units, _ = derive_geometry(tensor, split_size, -1)
            assert regime == select_regime(layout, -1, split_size)
            assert num_units > 0 and 1 <= n_chunks <= 4
            regimes.add(regime)
    assert regimes == {TILE_COL_SHIFT, RM_SUBSTICK, PAGE_REMAP}
