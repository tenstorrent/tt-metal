# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of scatter (verifier, Phase 0).

The op only moves values (rule R1), so the expectation is a bitwise-exact result: PCC 1, every error 0, 0 ULP,
got/true ratio exactly 1. The gate is the shared one (PCC and relative RMS together, as `check_output` in
eval/metrics.py) at the specification's exact tolerance, plus bitwise equality.

The measurements of each shape are written to `$SCATTER_PRECISION_DIR/<id>.json` (default
/tmp/scatter_precision_baseline): a test's own output is not shown under the simulator's forked workers.
"""

import json
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.scatter import scatter

OUT_DIR = Path(os.environ.get("SCATTER_PRECISION_DIR", "/tmp/scatter_precision_baseline"))

# (input shape, dim, source rows along dim): small, the traced Qwen3-VL call, a partial tile of source rows,
# tall, wide.
SHAPES = [
    ((32, 64), 0, 32),
    ((70, 512), 0, 64),
    ((128, 512), -2, 17),
    ((1024, 64), 0, 512),
    ((64, 8192), -2, 32),
]


def _ulp_bf16(got, expected):
    """Largest distance in bfloat16 units in the last place (monotone integer mapping of the bit patterns)."""

    def ordered(t):
        bits = t.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(bits < 0, -(bits & 0x7FFF), bits)

    return int((ordered(got) - ordered(expected)).abs().max().item())


@pytest.mark.parametrize("shape,dim,count", SHAPES, ids=lambda v: str(v).replace(" ", ""))
def test_scatter_precision_baseline(device, shape, dim, count, record_property):
    g = torch.Generator().manual_seed(1234)
    along = dim % len(shape)
    src_shape = list(shape)
    src_shape[along] = count
    tin = torch.randn(shape, generator=g).to(torch.bfloat16)
    tsrc = torch.randn(src_shape, generator=g).to(torch.bfloat16)
    tidx = torch.rand(shape, generator=g).argsort(dim=along).narrow(along, 0, count).contiguous()

    a = ttnn.from_torch(tin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    i = ttnn.from_torch(tidx.to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    s = ttnn.from_torch(tsrc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(scatter(a, dim, i, s))
    expected = tin.clone().scatter_(along, tidx, tsrc)

    _, pcc = comp_pcc(expected.float(), got.float(), pcc=0.9999)
    _, allclose = comp_allclose(got.float(), expected.float())
    diff = (got.double() - expected.double()).abs()
    rel_rms = float(torch.sqrt((diff**2).mean()) / expected.double().std())
    ulp = _ulp_bf16(got, expected)
    nonzero = expected.double() != 0
    ratio = got.double()[nonzero] / expected.double()[nonzero]
    result = {
        "shape": list(shape),
        "dim": dim,
        "source_rows": count,
        "pcc": float(pcc),
        "max_abs_err": float(diff.max()),
        "mean_abs_err": float(diff.mean()),
        "rel_rms": rel_rms,
        "max_ulp": ulp,
        "ratio_median": float(ratio.median()),
        "ratio_p5": float(ratio.quantile(0.05)) if ratio.numel() <= 16_000_000 else None,
        "ratio_p95": float(ratio.quantile(0.95)) if ratio.numel() <= 16_000_000 else None,
        "bitwise_mismatches": int((got.view(torch.int16) != expected.view(torch.int16)).sum()),
        "allclose": allclose,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = "x".join(str(d) for d in shape) + f"_dim{dim}_{count}"
    (OUT_DIR / f"{name}.json").write_text(json.dumps(result))
    for key, value in result.items():
        record_property(key, value)

    # The shared gate at the specification's exact tolerance (PCC >= 0.9999 and relative RMS 0) ...
    assert pcc >= 0.9999 and rel_rms == 0.0, result
    # ... and bitwise: the op moves values unchanged (R1).
    assert result["bitwise_mismatches"] == 0 and ulp == 0, result
