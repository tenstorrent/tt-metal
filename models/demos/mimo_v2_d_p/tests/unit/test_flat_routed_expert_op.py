# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The C++ op ttnn.experimental.deepseek_prefill.flat_routed_expert against the Python builder it was ported from
(tt/flat_expert.py FlatExpert) on one chip: the same plan (role cores, split, arena), bit-identical y on the same
inputs (same kernels, same layout), and the quantized-weight reference per active expert. Also launches the op again
with a moved arena (a live L1 buffer in between) and different counts: the cache-hit address patch."""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mimo_v2_d_p.tt.flat_expert import FlatExpert, FlatRoutedExpert

# (tag, H, I, experts, capacity m, global experts): plan families: MiMo 1 subgrid (64 experts), K2 reader tails,
# TP4 2 subgrids, TP2 3 subgrids (packer L1 accumulation)
CASES = [
    ("mimo", 4096, 2048, 64, 2048, 256),
    ("k2", 7168, 2048, 8, 4096, 32),
    ("tp4", 7168, 512, 8, 4096, 32),
    ("tp2", 7168, 1024, 8, 4096, 32),
]


def _counts(E, m, seed):
    g = torch.Generator().manual_seed(seed)
    c = torch.randint(0, 90, (E,), generator=g)
    c[torch.randperm(E, generator=g)[: E // 3]] = 0
    c[seed % E] = m  # one expert at the full capacity (pinned)
    c[(seed + 3) % E] = m // 2 + 7
    return [int(v) for v in c]


def _inputs(device, E, H, NG, gids, counts):
    offs = [sum(-(-c // 32) * 32 for c in counts[:e]) for e in range(E)]
    rows = offs[-1] + -(-counts[-1] // 32) * 32 + 64
    x = torch.zeros(rows, H)
    c_ = torch.zeros(1, NG, dtype=torch.int32)
    r_ = torch.zeros(1, NG, dtype=torch.int32)
    for e, gid in enumerate(gids):
        x[offs[e] : offs[e] + counts[e]] = torch.randn(counts[e], H)
        c_[0, gid], r_[0, gid] = counts[e], offs[e]
    rm = lambda t, dt: ttnn.from_torch(
        t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return x, offs, rm(x, ttnn.bfloat16), rm(c_, ttnn.uint32), rm(r_, ttnn.uint32)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_flat_routed_expert_op(device, case):
    tag, H, I, E, m, NG = case
    torch.manual_seed(0)
    weights = [[(torch.randn(H, I) * 0.02, torch.randn(H, I) * 0.02, torch.randn(I, H) * 0.02) for _ in range(E)]]
    gids = [[(4 * e + 1) % NG if E * 4 <= NG else (2 * e + 1) % NG for e in range(E)]]
    assert len(set(gids[0])) == E
    op = FlatRoutedExpert(device, weights, m=m, H=H, I=I, gids=gids, n_global=NG, pin=1)
    ref_b = FlatExpert(device, weights, m=m, H=H, I=I, gids=gids, n_global=NG, pin=1)
    plan, lay = op.plan, ref_b.layout
    for k in ("readers", "gu", "relays", "down", "pcds", "col0s", "rg", "np", "g", "mt", "x_slots", "hbuf"):
        got = [tuple(v) for v in plan[k]] if k in ("readers", "gu", "relays", "down") else plan[k]
        assert got == lay[k], f"{tag}: plan {k}: C++ {got} vs Python {lay[k]}"
    for k in ("gu_rp", "arena_tiles", "n_rd_sg", "nsg", "rdown"):
        assert plan[k] == lay[k], f"{tag}: plan {k}: C++ {plan[k]} vs Python {lay[k]}"
    logger.info(f"{tag}: plans match (nsg {plan['nsg']}, np {plan['np']} g {plan['g']}, rdown {plan['rdown']})")

    q = lambda w: ttnn.to_torch(ttnn.from_torch(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()
    pinned = []
    for launch, seed in enumerate((1, 2)):
        counts = _counts(E, m, seed)
        x, offs, x_dev, c_dev, r_dev = _inputs(device, E, H, NG, gids[0], counts)
        y_op = ttnn.to_torch(op(x_dev, c_dev, r_dev)).float()
        y_py = ttnn.to_torch(ref_b(x_dev, c_dev, r_dev)).float()
        worst = 1.0
        for e in range(E):
            c = counts[e]
            if not c:
                continue
            o = offs[e]
            assert torch.equal(
                y_op[o : o + c], y_py[o : o + c]
            ), f"{tag} launch {launch}: expert {e} differs from builder"
            rows = list(range(0, c, max(1, c // 48)))[:48] + [c - 1]
            xe = x[o : o + c][rows]
            Wg, Wu, Wd = (q(w_) for w_ in weights[0][e])
            ref = (torch.nn.functional.silu(xe @ Wg) * (xe @ Wu)) @ Wd
            ok, pcc = comp_pcc(ref, y_op[o : o + c][rows], 0.99)
            worst = min(worst, float(pcc))
            assert ok, (tag, launch, e, c, pcc)
        logger.info(f"{tag} launch {launch}: bit-identical to the builder, min PCC {worst:.5f}, counts {counts}")
        pinned.append(  # the next launch's arena then sits lower: the cache hit must patch every address
            ttnn.from_torch(
                torch.zeros(32 * 110 // 16, 32),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        )
