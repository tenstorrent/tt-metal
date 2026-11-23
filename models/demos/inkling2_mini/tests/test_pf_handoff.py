# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Prefill -> decode handoff (tt/fused/pf_handoff.py): x_hist rows and dense fp32 ring vectors equal the source
rows (bf16 copied, or widened to fp32), for two cache sets with rows in several tile rows; rows without a source
stay zero."""

import torch

import ttnn
from models.demos.inkling2_mini.tt.fused import common as F
from models.demos.inkling2_mini.tt.fused.pf_handoff import PfHandoff

C = 2304


def test_pf_handoff(device):
    g = torch.Generator().manual_seed(0)
    dram = ttnn.DRAM_MEMORY_CONFIG
    bf = lambda a: ttnn.from_torch(
        a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
    )  # noqa: E731
    jobs, want = [], []
    for rows_x, rows_v in (([None, (0, 127), (1, 31), (0, 64)], [(1, 63), (0, 0), (1, 32)]), ([(1, 5)] * 9, [(0, 96)])):
        srcs = [bf(torch.randn(1, 1, 128, C, generator=g)), bf(torch.randn(1, 1, 64, C, generator=g))]
        host = [ttnn.to_torch(s).float()[0, 0] for s in srcs]
        xh = bf(torch.ones(1, 1, 32, C))
        rings = [F.dense_tensor(device, torch.ones(1, C)) for _ in rows_v]
        jobs.append((xh, [None if x is None else (srcs[x[0]], x[1]) for x in rows_x], [(r, srcs[i], row) for r, (i, row) in zip(rings, rows_v)]))  # fmt: skip
        wx = torch.zeros(32, C)
        for d, x in enumerate(rows_x):
            if x is not None:
                wx[d] = host[x[0]][x[1]]
        want.append((xh, wx, rings, [host[i][row] for i, row in rows_v]))
    PfHandoff(device)(jobs)
    for xh, wx, rings, wv in want:
        assert torch.equal(ttnn.to_torch(xh).float()[0, 0], wx)
        for r, w in zip(rings, wv):
            assert torch.equal(F.from_dense_matrix(ttnn.to_torch(r), C)[0], w)
