# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Precision baseline for tilize (Phase 0 SUPPORTED corner: bf16 -> bf16, no cast).

tilize performs no arithmetic, so the expected result is bit-identical output. The test
records PCC, max/mean abs error, relative RMS error, max ULP distance and the got/true ratio
spread (scale-bug detector), and asserts bit-exactness.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.tilize import tilize


def _ulp_bf16(a: torch.Tensor, b: torch.Tensor) -> int:
    ia = a.contiguous().view(torch.int16).to(torch.int32)
    ib = b.contiguous().view(torch.int16).to(torch.int32)
    # monotonic integer mapping of bf16 bit patterns
    ia = torch.where(ia < 0, -32768 - ia, ia)
    ib = torch.where(ib < 0, -32768 - ib, ib)
    return int((ia - ib).abs().max())


@pytest.mark.parametrize(
    "shape, tile_h",
    [
        ((1, 1, 32, 32), 32),  # single tile
        ((1, 1, 256, 128), 32),  # small multi-tile
        ((2, 3, 640, 160), 32),  # leading-dim fold
        ((1, 1, 16384, 64), 32),  # perf-focus shape (coalesced reader path)
        ((1, 1, 4096, 64), 16),  # tiny tile (slow tilize_block path)
    ],
    ids=["single_tile", "small", "folded", "perf_focus", "tiny16"],
)
def test_tilize_precision_baseline(device, shape, tile_h):
    torch.manual_seed(1234)
    x = torch.randn(shape, dtype=torch.bfloat16) * 4
    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = ttnn.to_torch(tilize(tt_in, tile=ttnn.Tile([tile_h, 32])))

    got = out.float()
    ref = x.float()
    diff = (got - ref).abs()
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    rel_rms = float(torch.sqrt(((got - ref) ** 2).mean()) / torch.sqrt((ref**2).mean()))
    ulp = _ulp_bf16(out, x)
    mask = torch.isfinite(ref) & (ref != 0)
    r = got[mask] / ref[mask]
    r_med = float(r.median())
    r_p5, r_p95 = (float(v) for v in torch.quantile(r[: 1 << 20], torch.tensor([0.05, 0.95])))
    passing, allclose_msg = comp_allclose(out, x, rtol=0, atol=0)
    print(
        f"\nPRECISION shape={shape} tile_h={tile_h} max_abs={max_abs} mean_abs={mean_abs} "
        f"rel_rms={rel_rms} max_ulp={ulp} ratio_median={r_med} ratio_p5={r_p5} ratio_p95={r_p95} "
        f"allclose={passing} ({allclose_msg})"
    )

    assert_with_pcc(x, out, 0.99999)
    assert passing
    assert torch.equal(out, x), "tilize is a pure re-lay: output must be bit-identical"
