# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of split (verifier, Phase 0).

split moves values and computes nothing: every cell's numeric requirement is `exact` (spec NUMERIC_RESOLVED), so
the gate is PCC == 1 and relative RMS == 0 on every chunk, and every element bit-identical. The test also records
max / mean abs error, the max ULP distance and the got/true ratio spread (the scale-bug detector), per shape, for
both regimes and both layouts and placements of the first phase, in bfloat16 and (Refinement 2) float32, whose
values do not survive a bfloat16 buffer (spec R1).

Output under workers is not shown: each case appends one JSON line to $SPLIT_PRECISION_OUT when that is set.
"""

import json
import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.split import split

MEMORY = {"DRAM": ttnn.DRAM_MEMORY_CONFIG, "L1": ttnn.L1_MEMORY_CONFIG}

# (shape, split_size, dim): small (the traced small-grid call), the traced 640x640 call, a wide and a tall one.
CASES = [
    ((1, 4, 400), 2, 1),
    ((1, 4, 8400), 2, 1),
    ((1, 34, 4096), 17, -2),
    ((4, 128, 512), 64, 1),
]


def _ulp_max(got, want):
    """Largest distance in units in the last place of the tensors' own dtype (bit patterns as ordered integers)."""
    view, sign_mask = {torch.bfloat16: (torch.int16, 0x7FFF), torch.float32: (torch.int32, 0x7FFFFFFF)}[got.dtype]

    def ordered(t):
        bits = t.contiguous().view(view).to(torch.int64)
        return torch.where(bits < 0, -(bits & sign_mask), bits)

    return int((ordered(got) - ordered(want)).abs().max())


DTYPES = {"bf16": (ttnn.bfloat16, torch.bfloat16), "fp32": (ttnn.float32, torch.float32)}


@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("memory", ["DRAM", "L1"])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["TILE", "RM"])
@pytest.mark.parametrize("shape,split_size,dim", CASES, ids=lambda v: "x".join(map(str, v)) if isinstance(v, tuple) else str(v))
def test_split_precision_baseline(device, shape, split_size, dim, layout, memory, dtype, record_property):
    ttnn_dtype, torch_dtype = DTYPES[dtype]
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch_dtype)
    chunks = split(
        ttnn.from_torch(torch_input, dtype=ttnn_dtype, layout=layout, device=device, memory_config=MEMORY[memory]),
        split_size,
        dim,
    )
    expected = torch.split(torch_input, split_size, dim=dim)
    assert len(chunks) == len(expected)

    worst = {"pcc": 1.0, "max_abs": 0.0, "mean_abs": 0.0, "rel_rms": 0.0, "ulp_max": 0, "ratio_median": 1.0,
             "ratio_p5": 1.0, "ratio_p95": 1.0}
    for k, (chunk, want) in enumerate(zip(chunks, expected)):
        got = ttnn.to_torch(chunk)
        assert got.dtype == torch_dtype and list(got.shape) == list(want.shape), k
        _, pcc = comp_pcc(want.float(), got.float(), pcc=1.0)
        _, allclose_msg = comp_allclose(want.float(), got.float(), rtol=0.0, atol=0.0)
        diff = (got.double() - want.double()).abs()
        rel_rms = float(diff.pow(2).mean().sqrt() / want.double().std())
        nonzero = want.double() != 0
        ratio = got.double()[nonzero] / want.double()[nonzero]
        q = torch.quantile(ratio, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64))
        worst["pcc"] = min(worst["pcc"], float(pcc))
        worst["max_abs"] = max(worst["max_abs"], float(diff.max()))
        worst["mean_abs"] = max(worst["mean_abs"], float(diff.mean()))
        worst["rel_rms"] = max(worst["rel_rms"], rel_rms)
        worst["ulp_max"] = max(worst["ulp_max"], _ulp_max(got, want))
        worst["ratio_p5"], worst["ratio_median"], worst["ratio_p95"] = (float(v) for v in q)
        # exact: bit-identical, which also makes PCC 1 and relative RMS 0 (spec R1, NUMERIC exact)
        assert torch.equal(got, want), f"chunk {k}: {allclose_msg}"

    for key, value in worst.items():
        record_property(key, value)
    out = os.environ.get("SPLIT_PRECISION_OUT")
    if out:
        with open(out, "a") as f:
            f.write(json.dumps({"shape": shape, "split_size": split_size, "dim": dim, "layout": str(layout),
                                "memory": memory, "dtype": dtype, **worst}) + "\n")
    assert worst["pcc"] >= 1.0 - 1e-12 and worst["rel_rms"] == 0.0 and worst["ulp_max"] == 0
