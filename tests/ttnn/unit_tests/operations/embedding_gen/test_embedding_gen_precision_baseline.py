# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of embedding_gen on supported cells (verification_report.md, Precision Baseline).

The op moves bytes and computes nothing, so the numeric requirement of every cell is `exact` (the specification's
NUMERIC_RESOLVED). Each case covers one (table layout, output layout) pair on a first-phase shape and records
PCC, max / mean absolute error, relative RMS (RMS error / std of the reference, as eval.metrics.check_output),
the bfloat16 ULP distance, and the got/true ratio spread (the scale-bug detector).

Test output is hidden under the simulator's forked workers, so each case appends its metrics as one JSON line to
$EMBEDDING_GEN_PRECISION_OUT (default /tmp/embedding_gen_precision_baseline.jsonl) and records them with
record_property.
"""

import json
import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.embedding_gen import embedding_gen

OUT = os.environ.get("EMBEDDING_GEN_PRECISION_OUT", "/tmp/embedding_gen_precision_baseline.jsonl")
LAYOUTS = {"TILE": ttnn.TILE_LAYOUT, "RM": ttnn.ROW_MAJOR_LAYOUT}

# (indices, table, table layout, output layout): small, the Llama call, a batch, wide D.
CASES = [
    ((1, 1, 1, 32), (1, 1, 64, 128), "RM", "TILE"),
    ((1, 32), (1, 1, 8192, 64), "TILE", "TILE"),
    ((4, 1, 1, 128), (1, 1, 2048, 512), "TILE", "RM"),
    ((2, 32), (1, 1, 64, 4096), "RM", "RM"),
]


def _bf16_ulp(got, expected):
    """ULP distance between two bfloat16 tensors: difference of their sign-magnitude ordered bit patterns."""

    def ordered(t):
        bits = t.to(torch.bfloat16).view(torch.int16).to(torch.int32)
        return torch.where(bits < 0, -(bits & 0x7FFF), bits)

    return (ordered(got) - ordered(expected)).abs()


def _ratio_spread(got, expected):
    """Median and p95/p5 of got/true over finite elements whose reference is non-zero."""
    mask = (expected != 0) & torch.isfinite(expected) & torch.isfinite(got)
    r = (got[mask] / expected[mask]).double()
    q = torch.quantile(r, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64))
    return float(q[1]), float(q[0]), float(q[2])


@pytest.mark.parametrize("indices_shape,table_shape,table_layout,out_layout", CASES,
                         ids=lambda p: "x".join(map(str, p)) if isinstance(p, tuple) else p)
def test_precision_baseline(device, record_property, indices_shape, table_shape, table_layout, out_layout):
    torch.manual_seed(0)
    V = table_shape[-2]
    idx = torch.randint(0, V, indices_shape, dtype=torch.int32)
    table = torch.randn(table_shape, dtype=torch.float32).to(torch.bfloat16)
    t_idx = ttnn.from_torch(idx, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    t_table = ttnn.from_torch(table, dtype=ttnn.bfloat16, layout=LAYOUTS[table_layout], device=device)

    out = embedding_gen(t_idx, t_table, layout=LAYOUTS[out_layout], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    got = ttnn.to_torch(out).to(torch.float64)

    rows = idx.long().reshape(idx.shape[0], idx.shape[-1])
    expected = torch.nn.functional.embedding(rows, table.double().reshape(V, -1))
    assert list(got.shape) == list(expected.shape)

    _, pcc = comp_pcc(expected, got, pcc=0.9999)
    _, allclose_msg = comp_allclose(expected, got)
    err = (got - expected).abs()
    rel_rms = float(torch.sqrt(((got - expected) ** 2).mean()) / expected.std())
    ulp = _bf16_ulp(got, expected)
    median_r, p5_r, p95_r = _ratio_spread(got, expected)
    metrics = {
        "case": f"{list(indices_shape)}x{list(table_shape)} {table_layout}->{out_layout}",
        "pcc": float(pcc),
        "max_abs": float(err.max()),
        "mean_abs": float(err.mean()),
        "rel_rms": rel_rms,
        "ulp_max": int(ulp.max()),
        "ratio_median": median_r,
        "ratio_p5": p5_r,
        "ratio_p95": p95_r,
        "allclose": allclose_msg,
    }
    for k, v in metrics.items():
        record_property(k, v)
    with open(OUT, "a") as f:
        f.write(json.dumps(metrics) + "\n")

    # `exact`: every output element is a table element, unchanged (R1). PCC and relative RMS are gated together
    # like eval.metrics.check_output, at the specification's legacy_default (0.9999, 0.0).
    assert pcc >= 0.9999 and rel_rms <= 0.0, metrics
    assert metrics["max_abs"] == 0.0 and metrics["ulp_max"] == 0, metrics
