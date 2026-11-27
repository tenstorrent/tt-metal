# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for g6_permute (rows_tile_scatter, bfloat16 TILE DRAM, dims (0, 2, 1, 3)).

A permute moves values and computes none, so the gate is bit-exact (helpers.TOLERANCES[bfloat16]
= ("exact", None)): ULP distance 0 everywhere. PCC, abs error, relative RMS and the got/true ratio
spread are recorded alongside so the baseline reads like every other op's.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.g6_permute import g6_permute

ROWS = (0, 2, 1, 3)

SHAPES = [
    (1, 1, 16, 64),  # smallest production shape: 2 input tiles
    (1, 6, 16, 64),  # production, 12 input tiles
    (2, 34, 47, 96),  # every axis past one tile
    (1, 64, 100, 256),  # 2048 input tiles, ~19 per core
]


def _bf16_ulp_distance(actual, expected):
    """|ordinal(actual) - ordinal(expected)| on the bfloat16 bit patterns (sign-magnitude to ordinal)."""
    a = actual.to(torch.bfloat16).contiguous().view(torch.int16).to(torch.int32)
    e = expected.to(torch.bfloat16).contiguous().view(torch.int16).to(torch.int32)

    def ordinal(bits):
        magnitude = bits & 0x7FFF
        return torch.where(bits < 0, -magnitude, magnitude)

    return (ordinal(a) - ordinal(e)).abs()


@pytest.mark.parametrize("shape", SHAPES, ids=["x".join(str(d) for d in s) for s in SHAPES])
def test_precision_baseline(shape, device):
    torch.manual_seed(1234)
    torch_input = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    expected = torch.permute(torch_input, ROWS).contiguous()
    actual = ttnn.to_torch(g6_permute(ttnn_input, ROWS))
    assert list(actual.shape) == list(expected.shape)

    _, pcc = comp_pcc(expected.float(), actual.float(), pcc=0.0)
    allclose, allclose_detail = comp_allclose(actual.float(), expected.float(), rtol=0.0, atol=0.0)
    # comp_allclose reports the max abs error; the mean and the relative RMS are not in its output.
    diff = (actual.float() - expected.float()).abs()
    rel_rms = float(torch.sqrt((diff**2).mean()) / expected.float().std())
    ulp = _bf16_ulp_distance(actual, expected)

    nonzero = expected.float() != 0
    ratio = actual.float()[nonzero] / expected.float()[nonzero]
    ratio_median = float(ratio.median())
    ratio_p5, ratio_p95 = (float(q) for q in torch.quantile(ratio, torch.tensor([0.05, 0.95])))

    print(
        f"\nPRECISION {shape}: pcc={pcc} [{allclose_detail}] mean_abs={float(diff.mean()):.3e} "
        f"rel_rms={rel_rms:.3e} max_ulp={int(ulp.max())} ratio_median={ratio_median} "
        f"ratio_p5={ratio_p5} ratio_p95={ratio_p95}"
    )

    assert int(ulp.max()) == 0, f"{int((ulp > 0).sum())} values differ from the input value they name"
    assert allclose, allclose_detail
    assert pcc == 1.0
    assert rel_rms == 0.0
