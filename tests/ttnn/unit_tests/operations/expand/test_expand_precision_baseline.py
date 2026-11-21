# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of expand (Phase 0: bfloat16, tile layout, DRAM, rank 2, rows repeated).

The op moves bytes and computes nothing, so the specification's numeric requirement for every cell is `exact`
(NUMERIC_RESOLVED, class `movement`; rule R1). The baseline measures PCC, max / mean absolute error, relative
RMS (RMS error over the reference's std, as eval/metrics.py's check_output), the error in bfloat16 ULPs and the
got/true ratio spread (the scale-bug detector), and requires all of them at their exact values: PCC 1, every
error 0, the ratio exactly 1 everywhere, and the output bit-identical to torch's expand. A second test passes the
special values (NaN, +-inf, -0.0, subnormals, the largest finite values) and requires them bit-identical too.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from eval.specs.expand.spec import reference
from ttnn.operations.expand import expand

SHAPES = [
    ((1, 64), (32, -1)),  # small: one output tile row, two tile columns
    ((1, 50), (100, -1)),  # W and H non-aligned
    ((1, 2560), (1330, -1)),  # the traced Qwen3-VL call: 3,360 output tiles on the whole grid
    ((1, 8192), (17, -1)),  # wide: 256 sources, one output tile row
]


def _bits(t):
    return t.contiguous().view(torch.int16)


def _metrics(got, expected):
    """PCC, max / mean abs error, relative RMS, bfloat16 ULP error, got/true ratio quantiles (p5, p50, p95)."""
    got64 = got.double()
    diff = got64 - expected
    _, pcc = comp_pcc(expected, got64)
    _, allclose_msg = comp_allclose(got64, expected, rtol=0.0, atol=0.0)
    std = expected.std().item()
    abs_rms = diff.pow(2).mean().sqrt().item()
    rel_rms = abs_rms / std if std > 1e-12 else abs_rms
    ulp = (_bits(got).int() - _bits(expected.to(torch.bfloat16)).int()).abs().max().item()
    mask = torch.isfinite(expected) & (expected != 0)
    ratio = (got64[mask] / expected[mask]).float()
    q = torch.quantile(ratio[:1_000_000], torch.tensor([0.05, 0.5, 0.95])).tolist()
    return {
        "pcc": pcc,
        "max_abs": diff.abs().max().item(),
        "mean_abs": diff.abs().mean().item(),
        "rel_rms": rel_rms,
        "ulp": ulp,
        "ratio_p5_p50_p95": q,
        "allclose": allclose_msg,
    }


@pytest.mark.parametrize("shape,asked", SHAPES, ids=lambda v: "x".join(str(d) for d in v))
def test_precision_baseline(device, shape, asked, record_property):
    torch.manual_seed(0)
    t = torch.randn(shape, dtype=torch.bfloat16)  # NUMERIC["inputs"]: standard normal
    x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(expand(x, asked))
    expected = reference(t, asked)  # float64, as the specification compares

    m = _metrics(got, expected)
    for name, value in m.items():
        record_property(name, value)
    assert m["pcc"] == pytest.approx(1.0, abs=1e-12)
    assert m["max_abs"] == 0.0 and m["mean_abs"] == 0.0 and m["rel_rms"] == 0.0 and m["ulp"] == 0
    assert m["ratio_p5_p50_p95"] == [1.0, 1.0, 1.0]
    assert torch.equal(_bits(got), _bits(t.expand(*asked)))


def test_special_values_bit_exact(device):
    """NaN (with its payload), +-inf, -0.0, subnormals and the largest finite values are copied, not computed."""
    special = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), -0.0, 0.0, 9.2e-41, -9.2e-41, 1e-39, 3.38e38, -3.38e38, 1.1754944e-38],
        dtype=torch.bfloat16,
    )
    torch.manual_seed(1)
    t = torch.cat([special, torch.randn(64 - special.numel(), dtype=torch.bfloat16)]).reshape(1, 64)
    x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(expand(x, (40, -1)))
    assert torch.equal(_bits(got), _bits(t.expand(40, 64)))
