# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic debugging tests for rms_norm. DO NOT DELETE.

Documents the bugs found while bringing the op up, each with a
hand-calculable input so DEVICE_PRINT values can be compared against exact
expectations.

Bug log
-------
1. fp32 + w_non_aligned produced a wrong RMS denominator (golden suite: 82
   failures, ALL at dtype=float32 AND alignment=w_non_aligned; every bfloat16
   w_non_aligned cell passed).

   Root cause: `cb_scaler` carries the partial-W 0/1 mask tile.
   `reduce_accumulate_via_add`'s `fold_partial_last` reads it through srcB via
   `llk_unpack_AB<ROW>(input_dfb, scaler_dfb)` and does NOT reconfigure srcB —
   reduce entry already issued `reconfig_data_format(input_dfb, input_dfb)`, so
   srcB is programmed at the INPUT format. A Float16_b mask tile under an fp32
   input is therefore reinterpreted as fp32 (adjacent bf16 lanes pair up into
   junk floats), so the padded lanes are not zeroed and the sum is wrong.

   Fix: `cb_scaler`'s data_format is the input dtype (op_design.md R4's fixed
   Float16_b is correct for the ReduceTile datapath, not for AccumulateViaAdd's
   mask). `prepare_reduce_mask` supports Float16_b and Float32 and deduces the
   format from the CB, so both cells are covered.

   The all-ones tests below pin the arithmetic exactly: with x == 1 everywhere,
   sum(x^2) over the VALID lanes is W and mean is W/W == 1 regardless of W, so
   rsqrt(1 + eps) ~= 1 and out ~= 1. If padding lanes leak into the sum the
   mean becomes (W + junk)/W and the output scales away from 1 — a mask bug
   shows up as a constant offset, not as noise.
"""

import pytest
import torch

import ttnn

from ttnn.operations.rms_norm import rms_norm


# --------------------------------------------------------------------------- #
# 1. All-ones: every intermediate is hand-calculable.
#
#    mean(x^2) = 1.0 exactly (for ANY W, aligned or not)
#    rsqrt(1.0 + 1e-6) = 0.9999995
#    out = 1 * 0.9999995 = 0.9999995
#
#    A partial-W mask bug makes mean(x^2) = (W + leaked)/W != 1, so the whole
#    output shifts by a constant factor. Tolerance is tight on purpose.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 32),  # aligned, single tile
        (1, 1, 32, 17),  # W non-aligned, mask covers 17 of 32 lanes
        (1, 1, 32, 47),  # W non-aligned, 2 W-tiles
        (32, 17),  # 2D, W non-aligned  (a golden-suite failing cell)
        (4, 128, 47),  # 3D, W non-aligned (a golden-suite failing cell)
        (1, 1, 32, 4050),  # W non-aligned AND NW > 1 (mask on the final chunk)
    ],
    ids=lambda s: "x".join(map(str, s)),
)
def test_all_ones_mean_is_exactly_one(device, shape, layout, dtype):
    """mean(ones^2) == 1 for any W — isolates the partial-W mask."""
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    torch_x = torch.ones(shape, dtype=torch_dtype)

    tt_x = ttnn.from_torch(torch_x, dtype=dtype, layout=layout, device=device)
    actual = ttnn.to_torch(rms_norm(tt_x, epsilon=1e-6)).to(torch.float32)

    expected = torch.full(shape, 1.0 / (1.0 + 1e-6) ** 0.5, dtype=torch.float32)
    max_diff = (actual - expected).abs().max().item()
    # If the mask leaked padding lanes into the sum, the error is a CONSTANT
    # factor of order (32/W) — far above this bound.
    assert max_diff < 2e-2, (
        f"max diff {max_diff} — mean(ones^2) != 1, so the partial-W mask is wrong. "
        f"actual[0,...,:4]={actual.reshape(-1)[:4].tolist()}"
    )


# --------------------------------------------------------------------------- #
# 2. Monotonic input: unique values expose any tile reordering / chunk-offset
#    mistake (a wrong TileOffset base in the resident regime, or a chunk read
#    at the wrong DRAM tile, shows up as swapped values rather than as scale).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 128),  # NW == 1
        (1, 1, 32, 1024),  # NW > 1 for fp32, == 1 for bf16
        (1, 1, 32, 4096),  # NW > 1 (chunked accumulate)
    ],
    ids=lambda s: "x".join(map(str, s)),
)
def test_monotonic_ratio_is_position_independent(device, shape, layout):
    """out/x must be one constant per row — catches chunk/offset mixups."""
    W = shape[-1]
    torch_x = (torch.arange(W, dtype=torch.float32) + 1.0).expand(shape).contiguous()

    tt_x = ttnn.from_torch(torch_x, dtype=ttnn.float32, layout=layout, device=device)
    actual = ttnn.to_torch(rms_norm(tt_x, epsilon=1e-6)).to(torch.float32)

    ratio = actual.reshape(-1, W) / torch_x.reshape(-1, W)
    spread = (ratio.max(dim=-1).values - ratio.min(dim=-1).values).max().item()
    assert spread < 2e-3, f"out/x varies by {spread} along W — a per-tile scale leaked in"


# --------------------------------------------------------------------------- #
# 3. NH_core > 1: more tile-rows per core than HT_BLOCK, so the per-core
#    row-block loop runs more than once AND its last row-block is short.
#    Every acceptance-test shape lands 1 row-block per core on a >=64-core grid,
#    so this is the only cover for that loop.
# --------------------------------------------------------------------------- #


def test_many_row_blocks_per_core(device):
    """Tall, narrow tensor: HT_BLOCK is large and rows-per-core exceeds it."""
    shape = (1, 1, 65536, 32)  # ht_total = 2048 tile-rows, Wt = 1
    torch_x = torch.ones(shape, dtype=torch.bfloat16)

    tt_x = ttnn.from_torch(torch_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(rms_norm(tt_x, epsilon=1e-6)).to(torch.float32)

    expected = 1.0 / (1.0 + 1e-6) ** 0.5
    max_diff = (actual - expected).abs().max().item()
    assert max_diff < 2e-2, f"max diff {max_diff} across {shape[-2] // 32} tile-rows"
