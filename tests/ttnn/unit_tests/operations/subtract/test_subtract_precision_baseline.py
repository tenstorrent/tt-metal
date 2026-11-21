# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of subtract on its supported cells (Phase 0: bfloat16 TILE, fp32 DEST).

Per shape: PCC and abs errors (comp_pcc / comp_allclose), relative RMS (RMS error / std of the reference,
as eval/metrics.check_output), the bfloat16 ULP distance to the reference rounded once to bfloat16 (the
specification's grade: 1 ulp, NUMERIC_RESOLVED ('bfloat16', 'default')), and the got/true ratio spread
(the scale-bug detector: a tight cluster at a constant != 1 is a structural bug, not rounding).

Output is not shown under workers: set SUBTRACT_PRECISION_OUT=<file> to get one JSON line per shape.
"""

import json
import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.subtract import subtract

SHAPES = [
    (1, 1, 32, 64),  # 2 tiles, tile-aligned
    (1, 1, 17, 50),  # both dims non-aligned
    (1, 2, 8400),  # YOLOv8 dist2bbox, 263 tiles over 32 clusters
    (2, 512, 1024),  # 1,024 tiles, 32 per cluster
]


def _ordered_bf16(x: torch.Tensor) -> torch.Tensor:
    """bfloat16 bit patterns mapped to a monotone integer line (so ULP distance is a subtraction)."""
    bits = x.to(torch.bfloat16).view(torch.int16).to(torch.int32)
    return torch.where(bits < 0, -(bits & 0x7FFF), bits)


@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["DRAM", "L1"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_precision_baseline(device, shape, memory_config, record_property):
    torch.manual_seed(0)
    ta = torch.randn(*shape, dtype=torch.float32).to(torch.bfloat16)
    tb = torch.randn(*shape, dtype=torch.float32).to(torch.bfloat16)
    put = lambda t: ttnn.from_torch(  # noqa: E731
        t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=memory_config
    )
    got = ttnn.to_torch(subtract(put(ta), put(tb))).to(torch.float64)
    ref = ta.double() - tb.double()  # the specification's reference: float64 on the stored inputs

    pcc_ok, pcc = comp_pcc(ref, got, 0.9999)
    _, allclose_msg = comp_allclose(ref, got)
    err = got - ref
    max_abs = float(err.abs().max())
    mean_abs = float(err.abs().mean())
    rel_rms = float(err.pow(2).mean().sqrt() / ref.std())
    ulp = (_ordered_bf16(got) - _ordered_bf16(ref)).abs()
    ulp_max = int(ulp.max())
    ulp_gt0 = float((ulp > 0).double().mean())

    nz = ref.abs() > 1e-3
    ratio = got[nz] / ref[nz]
    q = torch.quantile(ratio, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64))
    row = {
        "shape": list(shape),
        "memory": "L1" if memory_config == ttnn.L1_MEMORY_CONFIG else "DRAM",
        "pcc": float(pcc),
        "max_abs_err": max_abs,
        "mean_abs_err": mean_abs,
        "rel_rms": rel_rms,
        "ulp_max": ulp_max,
        "frac_ulp_gt0": ulp_gt0,
        "ratio_p05": float(q[0]),
        "ratio_median": float(q[1]),
        "ratio_p95": float(q[2]),
        "allclose": allclose_msg,
    }
    for k, v in row.items():
        record_property(k, v)
    out = os.environ.get("SUBTRACT_PRECISION_OUT")
    if out:
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")

    # Gates: PCC and relative RMS together (as check_output), and the specification's 1-ulp grade.
    assert pcc_ok, row
    assert rel_rms <= 0.01, row
    assert ulp_max <= 1, row
    # Scale-bug detector: the ratio must centre on 1.
    assert abs(row["ratio_median"] - 1.0) < 1e-3, row
