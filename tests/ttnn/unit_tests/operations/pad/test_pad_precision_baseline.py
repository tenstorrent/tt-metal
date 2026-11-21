# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of pad on its supported cells (bfloat16, row-major, DRAM, value 0.0, padding after).

pad moves values and computes nothing, and its numeric requirement is `exact` for every cell (the
specification's NUMERIC_RESOLVED). So the gates are exactness, plus the shared PCC / relative-RMS pair as a
cross-check:
- 0 ULP, 0 max abs error and relative RMS 0, against the float64 reference on the stored input;
- the got/true ratio over the finite non-zero reference elements sits at exactly 1.0 (a ratio clustered at a
  constant other than 1.0 would be a scale bug, not rounding);
- rule R4: the output sits in the input's buffer, interleaved.

Measurements go to PAD_BASELINE_OUT (default /tmp/pad_precision_baseline.jsonl) and to record_property,
because a test's own output is not shown under workers on the simulator.
"""

import json
import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.pad import pad

OUT = os.environ.get("PAD_BASELINE_OUT", "/tmp/pad_precision_baseline.jsonl")

SHAPES = [
    ((1, 3, 20, 20), ((0, 0), (0, 13), (0, 0), (0, 0))),  # small: YOLOv8 stem
    ((4, 8, 47, 256), ((0, 2), (0, 3), (0, 1), (0, 4))),  # medium: every dimension
    ((1, 1, 2048, 256), ((0, 0), (0, 0), (0, 48), (0, 0))),  # larger: tall
    ((1, 1, 32, 4096), ((0, 0), (0, 0), (0, 0), (0, 4096))),  # larger: wide
]


def _ordered_bits(t: torch.Tensor) -> torch.Tensor:
    """bfloat16 bit patterns mapped to integers whose order is the float order (for a ULP distance)."""
    bits = t.contiguous().view(torch.int16).to(torch.int32)
    return torch.where(bits < 0, -(bits & 0x7FFF), bits)


def _id(case):
    shape, padding = case
    return "x".join(map(str, shape)) + "_p" + "_".join(f"{b}+{a}" for b, a in padding)


@pytest.mark.parametrize("case", SHAPES, ids=_id)
def test_pad_precision_baseline(device, case, record_property):
    shape, padding = case
    torch.manual_seed(0)
    t = torch.randn(shape, dtype=torch.bfloat16)
    x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    y = pad(x, padding, 0.0)

    # R4: the output lives where the input does, interleaved.
    mc = y.memory_config()
    assert mc.buffer_type == x.memory_config().buffer_type == ttnn.BufferType.DRAM
    assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert y.dtype == ttnn.bfloat16 and y.layout == ttnn.ROW_MAJOR_LAYOUT

    got = ttnn.to_torch(y)
    flat = [a for pair in reversed(padding) for a in pair]
    ref64 = torch.nn.functional.pad(t.double(), flat, mode="constant", value=0.0)
    ref = ref64.to(torch.bfloat16)  # exact: every value came from a bfloat16 input or is 0.0
    assert got.shape == ref.shape

    _, pcc = comp_pcc(ref64, got.double())
    _, allclose = comp_allclose(got.double(), ref64)
    err = got.double() - ref64
    max_abs = err.abs().max().item()
    mean_abs = err.abs().mean().item()
    rel_rms = (err.pow(2).mean().sqrt() / ref64.std()).item()
    max_ulp = (_ordered_bits(got) - _ordered_bits(ref)).abs().max().item()
    nz = (ref64 != 0) & torch.isfinite(ref64)
    ratio = got.double()[nz] / ref64[nz]
    result = {
        "shape": list(shape), "padding": [list(p) for p in padding], "pcc": float(pcc), "max_abs": max_abs,
        "mean_abs": mean_abs, "rel_rms": rel_rms, "max_ulp": max_ulp, "allclose": allclose,
        "ratio_median": ratio.median().item(), "ratio_p5": ratio.quantile(0.05).item(),
        "ratio_p95": ratio.quantile(0.95).item(),
    }
    for k, v in result.items():
        record_property(k, v)
    with open(OUT, "a") as f:
        f.write(json.dumps(result) + "\n")

    assert pcc >= 0.9999, result
    assert max_ulp == 0 and max_abs == 0.0 and rel_rms == 0.0, result
    assert result["ratio_p5"] == result["ratio_p95"] == 1.0, result
