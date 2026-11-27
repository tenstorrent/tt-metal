# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2: pin every broadcast-build path bit-exactly (incl. x padding poisoned), including the
geometry the production shapes never reach — many columns per core (single block, split replicate)
and more than COL_BLOCK_CAP columns per core (two blocks, depth 2 → split gated off)."""

import pytest
import torch
import ttnn

from ttnn.operations.g6_expand import g6_expand
import ttnn.operations.g6_expand.g6_expand_program_descriptor as pd

_KNOBS = (
    "MAX_CORES",
    "HALF_TILE",
    "REPLICATE_CPU_ROWS",
    "REPLICATE_SPLIT",
    "ROW_FETCH_REQUESTS",
    "READER_WRITE_EIGHTHS",
)

# (half_tile, cpu_rows, split, fetch_requests, eighths)
_PATHS = [
    (1, 16, 1, 2, 4),  # default: half tile, pure RISC-V, split across NCRISC/BRISC
    (1, 16, 1, 2, 0),  # split, writer-only writes
    (0, 32, 1, 2, 4),  # split, whole 2 KiB tile (faces 2/3 by each RISC)
    (1, 8, 0, 1, 4),  # half tile, RISC-V + NoC doubling, one fetch request
    (0, 8, 0, 2, 4),  # Phase-0/R1 path
]


@pytest.mark.parametrize("path", _PATHS, ids=lambda p: "h{}c{}s{}f{}e{}".format(*p))
@pytest.mark.parametrize(
    "in_shape, out_h, max_cores",
    [
        ((1, 1, 1, 512), 512, 80),  # production W=512
        ((1, 1, 1, 207), 207, 80),  # w_non_aligned
        ((1, 1, 1, 512), 64, 1),  # one core, 16 columns: single block, split over many slots
        ((1, 1, 1, 2112), 64, 1),  # one core, 66 columns > COL_BLOCK_CAP: two blocks, split gated off
    ],
)
def test_g6_expand_replicate_paths(device, path, in_shape, out_h, max_cores):
    half, cpu_rows, split, fetch_reqs, eighths = path
    saved = {k: getattr(pd, k) for k in _KNOBS}
    pd.MAX_CORES, pd.HALF_TILE, pd.REPLICATE_CPU_ROWS = max_cores, half, cpu_rows
    pd.REPLICATE_SPLIT, pd.ROW_FETCH_REQUESTS, pd.READER_WRITE_EIGHTHS = split, fetch_reqs, eighths
    pd._cache.clear()
    try:
        torch.manual_seed(0)
        x = torch.randn(in_shape, dtype=torch.float32).to(torch.bfloat16)
        # Poison the tile padding of x: only row 0 may reach the output.
        padded = torch.full((1, 1, 32, -(-in_shape[-1] // 32) * 32), float("nan"), dtype=torch.bfloat16)
        padded[..., :1, : in_shape[-1]] = x
        tt_x = ttnn.from_torch(
            padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        tt_x = ttnn.reshape(tt_x, ttnn.Shape(in_shape), ttnn.Shape(list(padded.shape)))
        sizes = [1, 1, out_h, in_shape[-1]]
        for _ in range(2):  # second call exercises the cached descriptor
            got = ttnn.to_torch(g6_expand(tt_x, sizes))
            assert torch.equal(got.view(torch.int16), x.expand(*sizes).contiguous().view(torch.int16))
        plan = dict(next(iter(pd._cache.values()))[1])
        assert plan["replicate_split"] == int(
            bool(split) and cpu_rows >= pd.FACE_ROWS and plan["col_block"] == plan["max_cols_per_core"]
        )
    finally:
        for k, v in saved.items():
            setattr(pd, k, v)
        pd._cache.clear()
