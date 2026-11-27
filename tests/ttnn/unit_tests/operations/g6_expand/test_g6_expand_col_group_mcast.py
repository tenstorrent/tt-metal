# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 3: regime R2 (column-group placement + multicast of the broadcast slot).

Forces the descriptor knobs COL_GROUP_SPLIT / COL_GROUP_MCAST and checks bit-exactness on every
production width plus a multi-image geometry, so the R2 path stays green whatever the shipped
default is. (split, mcast) = (1, 0) is the diagnostic placement-only variant.
With G6E_PERF=1 under `--profile`, each case runs G6E_REPS calls (default 3) for device timing.
"""

import os

import pytest
import torch
import ttnn

from ttnn.operations.g6_expand import g6_expand
import ttnn.operations.g6_expand.g6_expand_program_descriptor as pd

_REPS = int(os.environ.get("G6E_REPS", "1"))


@pytest.mark.parametrize("split,mcast,row_wise", [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1)])
@pytest.mark.parametrize(
    "shape", [(1, 1, 1, 512), (1, 1, 1, 256), (1, 1, 1, 196), (1, 1, 1, 185), (1, 1, 1, 207), (2, 1, 1, 96)]
)
def test_g6_expand_col_group_mcast(device, split, mcast, row_wise, shape):
    saved = tuple(getattr(pd, n, None) for n in ("COL_GROUP_SPLIT", "COL_GROUP_MCAST", "COL_GROUP_ROW_WISE"))
    pd.COL_GROUP_SPLIT, pd.COL_GROUP_MCAST, pd.COL_GROUP_ROW_WISE = split, mcast, bool(row_wise)
    pd._cache.clear()
    try:
        torch.manual_seed(0)
        x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
        w = shape[-1]
        sizes = [shape[0], shape[1], w, w]
        tt_x = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        for _ in range(_REPS):
            out = g6_expand(tt_x, sizes)
        got = ttnn.to_torch(out)
        plan = dict(pd._cache[next(iter(pd._cache))][1])
        print(f"G6E_R2 split={split} mcast={mcast} row_wise={row_wise} shape={shape} plan={plan}")
        if split and mcast and plan.get("col_group_size", 0) >= 2:
            assert plan["regime"] == pd.REGIME_R2
        assert torch.equal(got.view(torch.int16), x.expand(*sizes).contiguous().view(torch.int16))
    finally:
        pd.COL_GROUP_SPLIT, pd.COL_GROUP_MCAST, pd.COL_GROUP_ROW_WISE = saved
        pd._cache.clear()
