# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of multiply on its Phase 0 surface (bf16 operands, TILE, fp32 DEST, HiFi4).

Per shape: PCC and abs errors (comp_pcc / comp_allclose), relative RMS (RMS error / std of the reference, as
eval/metrics.check_output), bf16 ULP distance (max and p99) against the float64 product rounded once into
bf16, the share of elements that match bit for bit, and the got/true ratio spread (the scale-bug detector: a
tight cluster around a constant other than 1.0 is a structural bug, not rounding).

The specification grades every cell at 0 ulp (NUMERIC_RESOLVED, `default` mode), so the gate is 0 ulp plus
the usual PCC / relative-RMS pair. Under workers a test's output is hidden: each row is appended as one JSON
line to $MULTIPLY_PRECISION_OUT (default /tmp/multiply_precision_baseline.jsonl) and recorded as a property.
"""

import json
import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.multiply import multiply

OUT_PATH = os.environ.get("MULTIPLY_PRECISION_OUT", "/tmp/multiply_precision_baseline.jsonl")

SHAPES = [
    # (a, b, memory)            regime
    ((32, 17), (1, 17), "DRAM"),  # row_bcast_strips, smallest, non-aligned
    ((1, 1, 32, 64), (1, 1, 32, 64), "DRAM"),  # flat_stream, one block
    ((1, 4, 400), (1, 1, 400), "L1"),  # YOLO small-grid call, L1 placement
    ((1, 4, 8400), (1, 1, 8400), "DRAM"),  # YOLO detect head, 640x640
    ((2, 4, 128, 512), (2, 4, 1, 512), "DRAM"),  # row_bcast_strips, 4x4 blocks
    ((2, 512, 1024), (2, 512, 1024), "DRAM"),  # flat_stream, large (2,048 tiles)
]


def _ordered_bf16_bits(t):
    bits = t.to(torch.bfloat16).view(torch.int16).to(torch.int64)
    return torch.where(bits < 0, -(bits + (1 << 15)), bits)


def _metrics(got, ref64):
    exp_bf16 = ref64.to(torch.bfloat16)
    got_f = got.to(torch.bfloat16)
    _, pcc = comp_pcc(ref64.float(), got_f.float())
    allclose_ok, allclose_msg = comp_allclose(ref64.float(), got_f.float())
    err = got_f.double() - ref64
    rel_rms = (err.pow(2).mean().sqrt() / ref64.std()).item()
    ulp = (_ordered_bf16_bits(got_f) - _ordered_bf16_bits(exp_bf16)).abs().float().flatten()
    k = max(1, int(0.99 * ulp.numel() + 0.999999))
    nz = (ref64.abs() > 1e-30) & torch.isfinite(got_f.double())
    r = (got_f.double()[nz] / ref64[nz]).float()
    return {
        "pcc": float(pcc),
        "max_abs_err": err.abs().max().item(),
        "mean_abs_err": err.abs().mean().item(),
        "rel_rms": rel_rms,
        "ulp_max": ulp.max().item(),
        "ulp_p99": ulp.kthvalue(k).values.item(),
        "exact_frac": (got_f == exp_bf16).float().mean().item(),
        "ratio_median": r.median().item(),
        "ratio_p5": r.kthvalue(max(1, int(0.05 * r.numel()))).values.item(),
        "ratio_p95": r.kthvalue(max(1, int(0.95 * r.numel()))).values.item(),
        "allclose": allclose_msg,
    }


@pytest.mark.parametrize("a_shape,b_shape,memory", SHAPES, ids=lambda v: "x".join(map(str, v)) if isinstance(v, tuple) else v)
def test_multiply_precision_baseline(device, record_property, a_shape, b_shape, memory):
    torch.manual_seed(1234)
    ta = torch.randn(a_shape).to(torch.bfloat16)
    tb = torch.randn(b_shape).to(torch.bfloat16)
    mem = ttnn.L1_MEMORY_CONFIG if memory == "L1" else ttnn.DRAM_MEMORY_CONFIG
    a = ttnn.from_torch(ta, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    b = ttnn.from_torch(tb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    got = ttnn.to_torch(multiply(a, b))
    ref64 = ta.double() * tb.double()
    assert got.shape == ref64.shape

    m = _metrics(got, ref64)
    row = {"a": list(a_shape), "b": list(b_shape), "memory": memory, **m}
    with open(OUT_PATH, "a") as f:
        f.write(json.dumps(row) + "\n")
    for k, v in m.items():
        record_property(k, v)

    assert m["pcc"] >= 0.9999, row
    assert m["rel_rms"] <= 0.01, row
    assert m["ulp_max"] == 0, row  # the spec's 0-ulp requirement for bf16 x bf16 -> bf16
