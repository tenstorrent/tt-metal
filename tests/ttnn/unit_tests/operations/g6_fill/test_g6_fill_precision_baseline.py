# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for g6_fill on the supported production shapes (float32 TILE).

A fill has one reference value, so the reference std is 0: PCC is undefined (comp_pcc falls
back to allclose) and relative RMS is reported against |e| instead of std. The gate is bit
equality of every logical element (helpers.TOLERANCES: exact); the other figures are recorded
to show that it holds with zero error and a got/true ratio of exactly 1.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose, comp_pcc, comp_ulp
from ttnn.operations.g6_fill import g6_fill

SHAPES = [(185,), (207,), (256,), (512,)]
# 1.0 is the production value; 1/3 needs all 24 mantissa bits; 1e-40 is a float32 subnormal.
VALUES = [1.0, 1 / 3, 1e-40]


def _metrics(got, want):
    _, pcc = comp_pcc(want, got)
    allclose_ok, allclose_str = comp_allclose(want, got)
    ulp_ok, ulp_str = comp_ulp(want, got, ulp_threshold=0)
    err = (got.double() - want.double()).abs()
    rms = err.pow(2).mean().sqrt().item()
    scale = want.double().abs().max().item()
    finite = torch.isfinite(want) & (want != 0)
    ratio = got[finite].double() / want[finite].double()
    return {
        "pcc": pcc,
        "allclose": (allclose_ok, allclose_str),
        "ulp": (ulp_ok, ulp_str),
        "max_abs": err.max().item(),
        "mean_abs": err.mean().item(),
        "rel_rms": rms / scale if scale else rms,
        "ratio_median": ratio.median().item(),
        "ratio_p5": ratio.quantile(0.05).item(),
        "ratio_p95": ratio.quantile(0.95).item(),
        "bit_equal": torch.equal(got.view(torch.int32), want.view(torch.int32)),
    }


@pytest.mark.parametrize("value", VALUES, ids=["one", "third", "subnormal"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"N{s[0]}")
def test_precision_baseline(device, shape, value):
    torch.manual_seed(0)
    x = ttnn.from_torch(
        torch.randn(shape) + 10.0,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.from_torch(
        torch.randn(shape) + 10.0,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    g6_fill(x, value, output_tensor=out)
    got = ttnn.to_torch(out)
    want = torch.tensor(value, dtype=torch.float64).to(torch.float32).expand(shape).contiguous()

    m = _metrics(got, want)
    print(
        f"\nPRECISION g6_fill N={shape[0]} v={value!r}: pcc={m['pcc']} max_abs={m['max_abs']:.3e} "
        f"mean_abs={m['mean_abs']:.3e} rel_rms={m['rel_rms']:.3e} ulp={m['ulp'][1]} "
        f"ratio median={m['ratio_median']!r} p5={m['ratio_p5']!r} p95={m['ratio_p95']!r} "
        f"{m['allclose'][1]} bit_equal={m['bit_equal']}"
    )

    assert m["bit_equal"], "logical elements differ from cast_float32(fill_value)"
    assert m["pcc"] >= 0.9999 and m["rel_rms"] == 0.0
    assert m["ulp"][0], m["ulp"][1]
    assert m["ratio_median"] == m["ratio_p5"] == m["ratio_p95"] == 1.0
