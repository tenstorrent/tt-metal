# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# Refinement 3 — interleaved width-axis work-split (perf).
# DO NOT DELETE — documents the width-split correctness contract.
#
# The width-split engages for wide, short tensors (small nt_h, large Wt) where the
# height-only split would leave the grid under-filled (nt_h * 4 < grid_cores and
# > 1 column-chunk). Correctness invariant: tilize is identity of element values
# regardless of how work is distributed. These shapes ENGAGE the 2D path; the
# non-engage shapes confirm the height-only path is untouched. Identity must hold
# exactly (bf16 bit-exact, fp32/uint32 exact) for every rank.

import pytest
import torch
import ttnn

from ttnn.operations.tilize import tilize as gen_tilize

# (shape, engages_width_split) — engage when nt_h*4 < grid_cores(=64) and C>=2.
WIDTH_SPLIT_SHAPES = [
    ([1, 1, 32, 16384], True),  # nt_h=1, Wt=512 — the DeepSeek MLA single-core collapse
    ([8, 1, 32, 7168], True),  # nt_h=8, Wt=224 — under-parallelized regime
    ([1, 1, 96, 2048], True),  # nt_h=3, Wt=64  — rank-4, multi-row-per-unit split
    ([2, 3, 32, 4096], True),  # nt_h=6, Wt=128 — rank-4 with folded batch
    ([1, 1, 32, 512], True),  # nt_h=1, Wt=16  — minimal engage (C=2)
    ([64, 4096], True),  # nt_h=2, Wt=128 — rank-2 engage
    ([1, 1, 2048, 2048], False),  # nt_h=64 — height path (grid full)
    ([512, 512], False),  # nt_h=16 — height path
]


def _to_torch_dtype(dt):
    return {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}[dt]


@pytest.mark.parametrize("shape,engages", WIDTH_SPLIT_SHAPES)
@pytest.mark.parametrize("dt", [ttnn.bfloat16, ttnn.float32])
def test_width_split_identity_float(device, shape, engages, dt):
    """Float identity across the width-split (and non-engage) shapes."""
    torch.manual_seed(0)
    t = torch.randn(shape, dtype=torch.float32).to(_to_torch_dtype(dt))
    tin = ttnn.from_torch(
        t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = gen_tilize(tin)
    assert out.layout == ttnn.TILE_LAYOUT
    r = ttnn.to_torch(out)
    # tilize is a bit-exact layout reshuffle: torch.equal must hold for both dtypes.
    assert torch.equal(r, t), f"shape={shape} dt={dt} max_diff={(r.float()-t.float()).abs().max()}"


@pytest.mark.parametrize("shape,engages", WIDTH_SPLIT_SHAPES)
def test_width_split_identity_uint32(device, shape, engages):
    """Integer passthrough identity across the width-split path."""
    torch.manual_seed(0)
    t = torch.randint(0, 1 << 30, shape, dtype=torch.int32)
    tin = ttnn.from_torch(
        t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = gen_tilize(tin)
    r = ttnn.to_torch(out)
    assert torch.equal(r.to(torch.int64), t.to(torch.int64)), f"shape={shape}"


def test_width_split_l1_output(device):
    """Width-split with an L1 output memory config (l1_to_l1)."""
    torch.manual_seed(0)
    shape = [1, 1, 32, 8192]  # nt_h=1, Wt=256 -> engage
    t = torch.randn(shape, dtype=torch.bfloat16)
    tin = ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    out = gen_tilize(tin, ttnn.L1_MEMORY_CONFIG)
    r = ttnn.to_torch(out)
    assert torch.equal(r, t)


def test_width_split_matches_single_core(device):
    """The 2D multicore split must produce byte-identical output to single-core."""
    torch.manual_seed(0)
    shape = [1, 1, 32, 16384]
    t = torch.randn(shape, dtype=torch.bfloat16)
    tin = ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    mc = ttnn.to_torch(gen_tilize(tin, use_multicore=True))
    sc = ttnn.to_torch(gen_tilize(tin, use_multicore=False))
    assert torch.equal(mc, sc)
