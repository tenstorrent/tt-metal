# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm — deterministic debugging tests.  DO NOT DELETE.

These are hand-calculable inputs used to isolate the three real bugs found
while bringing the op up.  Each test documents the bug it pins.

Run:
    scripts/run_safe_pytest.sh --run-all \
        tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_debug.py

The device comes from the conftest fixture (module-scoped); never open one here.
"""

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm


def _to_device(t, device, dtype, layout):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


LAYOUTS = [
    pytest.param(ttnn.TILE_LAYOUT, id="tile"),
    pytest.param(ttnn.ROW_MAJOR_LAYOUT, id="row_major"),
]


# ---------------------------------------------------------------------------
# 1. All ones.  Every intermediate is hand-calculable:
#       Sum(x^2) = W,  mean = 1,  rsqrt(1 + 1e-6) ~= 1,  out ~= 1
#    A wrong 1/W (e.g. using the PADDED tile width 32*Wt instead of the true W)
#    shows up immediately as a uniform scale != 1.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("shape", [(32, 32), (32, 64), (32, 40), (64, 96)], ids=str)
def test_all_ones_is_all_ones(device, layout, shape):
    x = torch.ones(shape, dtype=torch.bfloat16)
    out = ttnn.to_torch(rms_norm(_to_device(x, device, ttnn.bfloat16, layout))).to(torch.float32)
    assert torch.allclose(out, torch.ones(shape), rtol=2e-2, atol=2e-2), f"max diff {(out - 1.0).abs().max()}"


# ---------------------------------------------------------------------------
# 2. Gamma index probe.  x == ones makes the normalization the identity, so
#    out[r][c] must be exactly gamma[c].  This is THE test that caught
#    `apply_gamma_block` being silently dropped on the ROW_MAJOR path: with a
#    constant gamma the bug is invisible (any permutation of a constant is that
#    constant), so gamma must be an arange.
#
#    Root cause pinned: the in-place PackTile needs TileOffset::Set.  With
#    TileOffset::Unset the chain emits pack_tile<out_of_order_output=false>,
#    whose LLK path ignores the tile index and uses an internal running counter
#    that is only rewound by a pack reconfig — and the chain's reconfig fold
#    elides that reconfig when two consecutive chains pack to the SAME CB.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("width", [32, 64, 128], ids=str)
def test_gamma_lands_at_the_right_column(device, layout, width):
    x = torch.ones((64, width), dtype=torch.bfloat16)
    gamma = torch.arange(width, dtype=torch.float32).reshape(1, 1, 1, -1).to(torch.bfloat16)

    out = ttnn.to_torch(
        rms_norm(
            _to_device(x, device, ttnn.bfloat16, layout),
            gamma=_to_device(gamma, device, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        )
    ).to(torch.float32)

    expected = gamma.reshape(1, -1).expand(64, width).to(torch.float32)
    assert torch.allclose(
        out, expected, rtol=2e-2, atol=2e-2
    ), f"gamma mis-applied: row0 got {out[0, :8].tolist()} want {expected[0, :8].tolist()}"


# ---------------------------------------------------------------------------
# 3. Single-tile blocks.  BLOCK_ROWS * SLICE_HIDDEN_TILES == 1 leaves the
#    dst-sync window no slack, so an unsynchronized in-place handoff (chain N's
#    pack vs chain N+1's unpack of the same tile) corrupts the result.  These
#    shapes are exactly the ones that passed under --dev (watcher polling hides
#    the race) and failed in production before `sync_pack_to_unpack()` was added
#    between the in-place stages.  Run this file WITHOUT --dev to keep it honest.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("shape", [(32, 17), (256, 32), (32, 50), (17, 64)], ids=str)
def test_single_tile_block_inplace_race(device, layout, shape):
    torch.manual_seed(7)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    gamma = torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32).to(torch.bfloat16)

    out = ttnn.to_torch(
        rms_norm(
            _to_device(x, device, ttnn.bfloat16, layout),
            gamma=_to_device(gamma, device, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        )
    ).to(torch.float32)

    xf = x.to(torch.float32)
    expected = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * gamma.to(torch.float32).reshape(-1)

    a, b = out.flatten(), expected.flatten()
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    assert pcc > 0.995, f"PCC {pcc} — in-place handoff not ordered?"


# ---------------------------------------------------------------------------
# 4. W-mask probe.  Poison the tile padding a caller can never see: if the
#    kernel folds it into Sum(x^2) the answer is off by ~sqrt(W_padded/W), an
#    almost-uniform scale error that PCC alone is nearly blind to — so this
#    test checks the ABSOLUTE value against the true-W reference.
#    ROW_MAJOR has no reachable padding (the reader zero-fills), so this is a
#    TILE-layout test.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", [40, 50, 17, 72], ids=str)
def test_w_padding_is_not_folded_into_the_denominator(device, width):
    x = torch.ones((32, width), dtype=torch.bfloat16)
    out = ttnn.to_torch(rms_norm(_to_device(x, device, ttnn.bfloat16, ttnn.TILE_LAYOUT))).to(torch.float32)
    # mean(1^2) over the TRUE W is 1 -> out == 1.  Folding the pad would give
    # sqrt(32*ceil(W/32) / W) > 1 (e.g. 1.265 at W=40).
    assert torch.allclose(out, torch.ones((32, width)), rtol=2e-2, atol=2e-2), (
        f"W={width}: got {out[0, 0].item()}, want 1.0 "
        f"(pad-folded would be {(32 * ((width + 31) // 32) / width) ** 0.5:.4f})"
    )


# ---------------------------------------------------------------------------
# 5. Cross-core combine probe.  A wide, short row forces num_hidden_slices > 1,
#    so the row's Sum(x^2) is assembled from per-core partials gathered on the
#    row-group root and multicast back.  x == ones makes the whole pipeline
#    hand-calculable: any lost or double-counted contributor shows up as a
#    uniform scale of sqrt(s/s') != 1.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", [4096, 8192], ids=str)
def test_cross_core_combine_counts_every_contributor(device, width):
    x = torch.ones((1, 1, 32, width), dtype=torch.bfloat16)
    out = ttnn.to_torch(rms_norm(_to_device(x, device, ttnn.bfloat16, ttnn.TILE_LAYOUT))).to(torch.float32)
    assert torch.allclose(out, torch.ones_like(out), rtol=2e-2, atol=2e-2), (
        f"W={width}: got {out[0, 0, 0, :4].tolist()}, want 1.0 — a dropped or "
        "double-counted contributor scales the whole row"
    )
