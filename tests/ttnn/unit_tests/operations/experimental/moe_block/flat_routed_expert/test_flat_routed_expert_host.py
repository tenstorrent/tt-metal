# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the flat routed expert's Python side (no device): the weight bank layout places every weight
tile exactly once where the kernels read it, and FlatLayoutOptions keys / compares as the plan cache expects."""

import importlib.util
from pathlib import Path

import pytest
import torch

import ttnn

moe_block_utils = ttnn._ttnn.operations.experimental.moe_block

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[6]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


weights_py = _load(
    "flat_routed_expert_weights",
    REPO / "ttnn/cpp/ttnn/operations/experimental/moe_block/flat_routed_expert/flat_routed_expert_weights.py",
)


def _tagged(rows, cols, base):
    """[rows, cols] whose 32 x 32 tile (i, j) is filled with base + i * (cols / 32) + j (a unique tag per tile)."""
    tr, tc = rows // 32, cols // 32
    tags = base + torch.arange(tr * tc, dtype=torch.float64).reshape(tr, tc)
    return tags.repeat_interleave(32, 0).repeat_interleave(32, 1)


def _rotate(nd, banks):
    """Every down core's blocks rotated over all banks (block k of core d in bank (d + k) % banks), one slot a core."""
    return [[(d + j) % banks for j in range(banks)] for d in range(nd)], list(range(nd)), nd


def _layout(E, H, I, banks, nd, np_, nu, patterns=None):
    """A plan's layout facts for a small synthetic split (the fields flat_weight_regions reads). patterns: the down
    cores' (bank patterns, slots, slot count) - by default rotated over all banks."""
    it, ht = I // 32, H // 32
    ngrp = 2 * it // nu if nu else it // np_
    pcds = [ht // nd + (d < ht % nd) for d in range(nd)]
    col0s = [sum(pcds[:d]) for d in range(nd)]
    sub = max(
        -(-E * (it // weights_py.down_k_block_tiles(p, it)) // banks) * weights_py.down_k_block_tiles(p, it) * p
        for p in pcds
    )
    pats, slots, slot_count = patterns or _rotate(nd, banks)
    return dict(
        down_bank_patterns=pats,
        down_weight_slots=slots,
        down_weight_slot_count=slot_count,
        gate_up_weight_slots=banks,
        column_groups_per_weight_slot=ngrp // banks,
        gate_up_pairs_per_core=np_,
        units_per_core=nu,
        gate_up_k_blocks=ht // weights_py.KBLK,
        down_core_count=nd,
        down_columns_per_core=pcds,
        down_first_columns=col0s,
        dram_banks=banks,
        down_weight_region_tiles=sub,
    )


@pytest.mark.parametrize("units", [False, True], ids=["pairs", "units"])
def test_flat_weight_regions_places_every_tile(units):
    E, H, I, banks = 2, 256, 1024 if not units else 1536, 8
    lay = _layout(E, H, I, banks, nd=4, np_=0 if units else 1, nu=3 if units else 0)
    W = [
        (_tagged(H, I, 1e6 * (3 * e + 1)), _tagged(H, I, 1e6 * (3 * e + 2)), _tagged(I, H, 1e6 * (3 * e + 3)))
        for e in range(E)
    ]
    regions, dregs = weights_py.flat_weight_regions(W, lay)
    assert len(regions) == lay["gate_up_weight_slots"] and len({r.shape for r in regions}) == 1
    assert len(dregs) == lay["down_weight_slot_count"] * banks and len({r.shape for r in dregs}) == 1
    if not units:  # whole tiles: every gate / up tile once, every tile uniform
        got = torch.cat(regions)
        assert bool((got == got[:, :1, :1]).all())
        want = torch.cat([torch.cat([w[0], w[1]], 1)[::32, ::32].flatten() for w in W])
        assert sorted(got[:, 0, 0].tolist()) == sorted(want.tolist())
    else:  # 16-column units: every gate / up column half once (faces 16 wide)
        got = torch.cat(regions)
        halves = torch.cat([got[:, 0, :16][:, :1], got[:, 0, 16:][:, :1]]).flatten()
        want = torch.cat([torch.cat([w[0][::32, ::16], w[1][::32, ::16]], 1).flatten() for w in W])
        assert sorted(halves.tolist()) == sorted(want.tolist())
    # down: every Wd tile once over the banks (the rest is zero padding), block k of core d in bank (d + k) % banks
    dn = torch.cat(dregs)[:, 0, 0]
    nz = dn[dn != 0]
    want = torch.cat([w[2][::32, ::32].flatten() for w in W])
    assert sorted(nz.tolist()) == sorted(want.tolist())


# Down cores' bank patterns (8 banks; H 256: 8 tile columns): (patterns, slots, slot count) as the plan gives them -
# each core's dram_banks positions (block k at position k % banks) in one bank, or split over two (contiguous or
# interleaved positions), cores stacked in one bank at consecutive slots.
DOWN_PATTERNS = {
    "rotate": (4, _rotate(4, 8)),
    "one-bank": (8, ([[d] * 8 for d in range(8)], [0] * 8, 8)),
    "two-banks": (4, ([[2 * d] * 4 + [2 * d + 1] * 4 for d in range(4)], [0] * 4, 4)),
    "mixed": (
        6,
        (
            [[0] * 8, [1] * 8, [2] * 4 + [3] * 4, [2] * 4 + [3] * 4, [4] * 8, [5, 6] * 4],
            [0, 0, 0, 4, 0, 0],
            8,
        ),
    ),
}


@pytest.mark.parametrize("name", DOWN_PATTERNS)
def test_down_weight_regions_match_the_reader(name):
    """Every down core's blocks, read where down_weight_reader.cpp reads them - block k at position j = k % banks of
    the core's pattern, in that bank (k / banks) x (the bank's positions) + (its positions before j) blocks from the
    core's slot - are its K-blocks of Wd in order (expert, K-block), and no region holds two cores' blocks."""
    E, H, I, banks = 3, 256, 1024, 8
    nd, patterns = DOWN_PATTERNS[name]
    lay = _layout(E, H, I, banks, nd=nd, np_=1, nu=0, patterns=patterns)
    W = [
        (_tagged(H, I, 1e6 * (3 * e + 1)), _tagged(H, I, 1e6 * (3 * e + 2)), _tagged(I, H, 1e6 * (3 * e + 3)))
        for e in range(E)
    ]
    _, dregs = weights_py.flat_weight_regions(W, lay)
    sub, it = lay["down_weight_region_tiles"], I // 32
    assert len(dregs) == lay["down_weight_slot_count"] * banks and all(r.shape[0] == sub for r in dregs)
    bank_tiles = [torch.cat(dregs[b::banks])[:, 0, 0] for b in range(banks)]  # (bank b: its regions by slot)
    pats, slots = lay["down_bank_patterns"], lay["down_weight_slots"]
    for d in range(nd):
        p, c0 = lay["down_columns_per_core"][d], lay["down_first_columns"][d]
        kb = weights_py.down_k_block_tiles(p, it)
        want = [w[2][::32, ::32][c * kb : (c + 1) * kb, c0 : c0 + p].flatten() for w in W for c in range(it // kb)]
        pat = pats[d]
        for k, block in enumerate(want):
            j = k % banks
            b = pat[j]
            at = slots[d] * sub + ((k // banks) * pat.count(b) + pat[:j].count(b)) * kb * p
            got = bank_tiles[b][at : at + kb * p]
            assert torch.equal(got, block), f"{name}: down core {d} block {k} (bank {b}, tile {at})"
    nz = torch.cat(bank_tiles)
    assert sorted(nz[nz != 0].tolist()) == sorted(torch.cat([w[2][::32, ::32].flatten() for w in W]).tolist())


def test_down_weight_regions_reject_overlap():
    """Two down cores given the same (slot, bank) region is a plan error, not a silent overwrite."""
    lay = _layout(3, 256, 1024, 8, nd=8, np_=1, nu=0, patterns=([[0] * 8] * 8, [0] * 8, 8))
    W = [(torch.ones(256, 1024), torch.ones(256, 1024), torch.ones(1024, 256))] * 3
    with pytest.raises(ValueError, match="another core's"):
        weights_py.flat_weight_regions(W, lay)


def test_flat_layout_options():
    a = moe_block_utils.FlatLayoutOptions(arena_bytes=40960)
    assert a.arena_bytes == 40960
    assert a == weights_py.flat_layout(dict(arena_bytes=40960))
    # (arena_bytes, x_bf16, h_bf16)
    assert list(a.key()) == [40960, 0, 0]
    b = moe_block_utils.FlatLayoutOptions(arena_bytes=40960, x_bf16=True, h_bf16=True)
    assert (b.x_bf16, b.h_bf16) == (True, True) and list(b.key()) == [40960, 1, 1]
    assert b != a and b == weights_py.flat_layout(a, x_bf16=True, h_bf16=True)
