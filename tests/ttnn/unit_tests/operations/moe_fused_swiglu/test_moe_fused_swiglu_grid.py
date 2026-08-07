# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Correctness coverage for multiple worker-grid geometries.

Everything about this op's shape is derived from (HGROUPS, KGROUPS): the hidden split across
COLUMNS, the emb contraction split across ROWS, the reduce-scatter's column height, the h
all-gather's round count, and `cb_w_down`'s slot cycle. A grid change moves all of them at once,
and the failure mode is not a wrong number — it is a hang, because the collectives only agree
while every core computes the same plan.

8x8 is the interesting alternate cell: HGROUPS 8 rather than 11 changes `hn_pad` (8, not 6), the
chunk count (2, not 3), the reduce column height, and the number of all-gather rounds. 11x10 is the
full grid on the test device and exercises the operation's default geometry. 11x8 exercises a
smaller explicit override.

WHAT IS NOT TESTED HERE, and why. Grids much smaller than these do not fit L1 at any SUPPORTED
emb: fewer cores means a larger `kr_pad` and `ec_max` per core, so a 4x2 grid needs ~7.5 MB of CBs
against ~1.4 MB available. The op reports that with the computed numbers rather than hanging, and
`test_grid_too_small_reports_l1` pins that it is a clean refusal.
"""

import os

import pytest
import torch

import ttnn

from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu
from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu_geometry as geo

TILE = 32
NUM_GLOBAL_EXPERTS, NUM_LOCAL_EXPERTS, LOCAL_EXPERT_ID, GLOBAL_EXPERT_ID = 256, 8, 3, 137
PCC_GATE = 0.975  # the bfp4 format floor; same gate the golden suite uses

#: (hgroups, kgroups, hidden) — every combination that fits L1 at a supported emb on this device.
GRIDS = [
    (11, 8, 2048, "an explicit 88-core override"),
    (11, 10, 2048, "the full grid used by default on the test device"),
    (8, 8, 1024, "HGROUPS 8: different hn_pad, chunk count, column height and round count"),
]


def _pcc(a, b):
    a, b = a.flatten().to(torch.float32), b.flatten().to(torch.float32)
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def _reference(x, wg, wu, wd, count):
    xs = x[0, 0, :count].to(torch.float32)
    h = torch.nn.functional.silu(xs @ wg.to(torch.float32)) * (xs @ wu.to(torch.float32))
    return h @ wd.to(torch.float32)


@pytest.mark.parametrize("hgroups, kgroups, hidden, why", GRIDS)
def test_grid(device, hgroups, kgroups, hidden, why):
    grid = device.compute_with_storage_grid_size()
    if hgroups > int(grid.x) or kgroups > int(grid.y):
        pytest.skip(f"device grid is {grid.x}x{grid.y}, smaller than {hgroups}x{kgroups}")

    emb, capacity, count = 6144, 1024, 256
    torch.manual_seed(42)
    x = torch.randn((1, 1, capacity, emb), dtype=torch.float32)
    x[:, :, count:, :] = 100.0  # the phantom-row sentinel, as everywhere else
    xb = x.to(torch.bfloat16)
    wg, wu = (torch.randn((emb, hidden), dtype=torch.bfloat16) for _ in range(2))
    wd = torch.randn((hidden, emb), dtype=torch.bfloat16)

    dev = lambda t, d, l: ttnn.from_torch(t, dtype=d, layout=l, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_x = dev(xb, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    tt_w = [dev(w, ttnn.bfloat4_b, ttnn.TILE_LAYOUT) for w in (wg, wu, wd)]
    counts = torch.zeros(NUM_GLOBAL_EXPERTS, dtype=torch.int32)
    counts[GLOBAL_EXPERT_ID] = count
    idx = torch.tensor([(11 + 37 * i) % NUM_GLOBAL_EXPERTS for i in range(NUM_LOCAL_EXPERTS)], dtype=torch.int32)
    idx[LOCAL_EXPERT_ID] = GLOBAL_EXPERT_ID
    tt_counts = dev(counts, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    tt_idx = dev(idx, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

    out = moe_fused_swiglu(
        tt_x, tt_w[0], tt_w[1], tt_w[2], tt_counts, tt_idx, LOCAL_EXPERT_ID, core_grid=(hgroups, kgroups)
    )
    assert list(out.shape) == [1, 1, capacity, emb]
    got = ttnn.to_torch(out)[0, 0, :count]

    ref = _reference(xb, wg, wu, wd, count)
    pcc = _pcc(got, ref)
    blk = geo.Blocking(hgroups, kgroups, emb, hidden, capacity // TILE)
    assert torch.isfinite(got).all(), f"{hgroups}x{kgroups} ({why}) produced non-finite output"
    assert got.abs().max() > 0, f"{hgroups}x{kgroups} produced all zeros"
    assert pcc >= PCC_GATE, f"{hgroups}x{kgroups} ({why}): pcc {pcc:.6f} < {PCC_GATE}\n  {blk.describe()}"
    print(f"[grid] {hgroups}x{kgroups} hidden={hidden} pcc={pcc:.6f}  {blk.describe()}")


def test_grid_too_small_reports_l1(device, expect_error):
    """A grid that cannot hold the working set must REFUSE, with the numbers, not hang.

    Which limit it hits first depends on the shape. At 4x2 the `down` sub-block (ec_max = emb
    tiles / cores) busts the DEST budget before L1 does, so both refusals are accepted here — what
    is being pinned is that the op says WHICH resource ran out and by how much, not which one.
    """
    emb, hidden, capacity, count = 6144, 2048, 1024, 256
    torch.manual_seed(0)
    dev = lambda t, d, l: ttnn.from_torch(t, dtype=d, layout=l, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_x = dev(torch.randn((1, 1, capacity, emb), dtype=torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    tt_w = [
        dev(torch.randn(s, dtype=torch.bfloat16), ttnn.bfloat4_b, ttnn.TILE_LAYOUT)
        for s in ((emb, hidden), (emb, hidden), (hidden, emb))
    ]
    counts = torch.zeros(NUM_GLOBAL_EXPERTS, dtype=torch.int32)
    counts[GLOBAL_EXPERT_ID] = count
    idx = torch.zeros(NUM_LOCAL_EXPERTS, dtype=torch.int32)
    idx[LOCAL_EXPERT_ID] = GLOBAL_EXPERT_ID

    with expect_error((RuntimeError, ValueError), r"(?i)L1 per core|DEST (?:budget|limit)"):
        moe_fused_swiglu(
            tt_x,
            tt_w[0],
            tt_w[1],
            tt_w[2],
            dev(counts, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            dev(idx, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            LOCAL_EXPERT_ID,
            core_grid=(4, 2),
        )


def test_single_row_grid_is_refused(device, expect_error):
    """KGROUPS == 1 has no cross-column reduce to scatter, so it must be refused, not attempted."""
    with expect_error(ValueError, r"(?i)2 rows tall|cross-column"):
        geo.Blocking(11, 1, 6144, 2048, 32)
