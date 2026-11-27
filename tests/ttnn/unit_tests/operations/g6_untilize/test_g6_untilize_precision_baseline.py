# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for g6_untilize on the production case.

The op is a byte move, so the only acceptable result is bit-identical (helpers.TOLERANCES[uint32] = exact).
The spec limits tests to the production case (BB-22), so the baseline varies the input distribution, not the
shape: each distribution targets a different narrow-path failure (float32 detour above 2^24, sign bit, all-ones).
PCC, abs error and relative RMS are computed on the unsigned values in float64 (exact below 2^53) so the
numbers are comparable with float ops; the gate is bit equality.
"""

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.g6_untilize import g6_untilize

SHAPE = (1, 1, 32, 128)


def _uint32(patterns):
    v = patterns.to(torch.int64) & 0xFFFFFFFF
    v = torch.where(v >= 2**31, v - 2**32, v)
    return v.to(torch.int32).view(torch.uint32)


def _full_range(gen):
    return torch.randint(-(2**31), 2**31, SHAPE, dtype=torch.int64, generator=gen)


def _above_2_24(gen):
    # Every value is odd and above 2^24: none survives a float32 detour.
    return torch.randint(2**23, 2**31, SHAPE, dtype=torch.int64, generator=gen) * 2 + 1 + 2**24


def _high_bit(gen):
    return torch.randint(0, 2**31, SHAPE, dtype=torch.int64, generator=gen) | 2**31


def _index(gen):
    return torch.arange(SHAPE[2] * SHAPE[3], dtype=torch.int64).reshape(SHAPE)


DISTRIBUTIONS = {"full_range": _full_range, "above_2_24": _above_2_24, "high_bit": _high_bit, "index": _index}


def _unsigned(t):
    return t.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF


@pytest.mark.parametrize("dist", list(DISTRIBUTIONS))
def test_g6_untilize_precision_baseline(device, dist):
    x = _uint32(DISTRIBUTIONS[dist](torch.Generator().manual_seed(0)))
    x_dev = ttnn.from_torch(
        x, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    got = ttnn.to_torch(g6_untilize(x_dev))

    a, e = _unsigned(got), _unsigned(x)
    mismatches = int((a != e).sum().item())
    max_ulp = int((a - e).abs().max().item())

    af, ef = a.to(torch.float64), e.to(torch.float64)
    _, pcc = comp_pcc(ef, af)
    _, allclose_msg = comp_allclose(ef, af)
    err = af - ef
    max_abs = err.abs().max().item()
    mean_abs = err.abs().mean().item()
    rel_rms = (err.pow(2).mean().sqrt() / ef.std()).item()
    nz = ef != 0
    ratio = af[nz] / ef[nz]
    r_med = ratio.median().item()
    r_spread = (ratio.quantile(0.95) / ratio.quantile(0.05)).item() if ratio.numel() > 1 else 1.0

    logger.info(
        f"g6_untilize precision [{dist}] shape={SHAPE}: mismatches={mismatches}/{e.numel()} max_ulp={max_ulp} "
        f"PCC={pcc:.6f} max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} rel_rms={rel_rms:.3e} "
        f"ratio median={r_med:.6f} p95/p5={r_spread:.6f} ({allclose_msg})"
    )
    assert mismatches == 0, f"{mismatches} elements differ from the input bits"
    assert max_ulp == 0 and pcc == 1.0 and rel_rms == 0.0
