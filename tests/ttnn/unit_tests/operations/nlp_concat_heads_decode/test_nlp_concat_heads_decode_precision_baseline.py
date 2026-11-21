# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of nlp_concat_heads_decode (verifier, Phase 0).

The op moves values and computes nothing (spec R1, NUMERIC class `movement`, metric `exact`), so the baseline is
the statement that every defined output element equals its input element bit for bit: PCC 1, every abs error 0,
relative RMS 0, ULP p99 0, and the got/true ratio pinned at 1.0 with zero spread (the scale-bug detector).

Each case also appends its metrics as one JSON line to $NCHD_PRECISION_OUT (default
/tmp/nchd_precision_baseline.jsonl): under xdist/forked workers a test's stdout is not shown.
"""

import json
import os

import pytest
import torch

import ttnn
from eval.metrics import compute_metrics_torch
from eval.sharding import auto_shard_config
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.nlp_concat_heads_decode import nlp_concat_heads_decode

OUT = os.environ.get("NCHD_PRECISION_OUT", "/tmp/nchd_precision_baseline.jsonl")

# (shape, num_heads): small (one tile), medium (the Qwen small-grid call), the model's 32-head call, and the
# widest result (8192 columns on 32 cores).
CASES = [
    ((1, 1, 32, 32), 1),
    ((1, 1, 32, 128), 2),
    ((1, 1, 32, 128), 32),
    ((1, 1, 32, 256), 32),
]


def _record(row):
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")


@pytest.mark.parametrize("shape,num_heads", CASES, ids=lambda v: "x".join(map(str, v)) if isinstance(v, tuple) else f"h{v}")
def test_precision_baseline(device, record_property, shape, num_heads):
    dtype, torch_dtype = ttnn.bfloat16, torch.bfloat16
    torch.manual_seed(1234)
    host = torch.randn(*shape, dtype=torch_dtype)
    mc = auto_shard_config(shape, ttnn.ShardStrategy.HEIGHT, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
    x = ttnn.from_torch(host, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)

    out = nlp_concat_heads_decode(x, num_heads=num_heads)

    users, head_dim = shape[1], shape[3]
    got = ttnn.to_torch(out)[:, :, :users, :]  # rows users..31 are padding (R3): not compared
    expected = host[:, :, :num_heads, :].reshape(1, 1, users, num_heads * head_dim)

    pcc_ok, pcc = comp_pcc(expected, got, 0.9999)
    _, allclose_msg = comp_allclose(expected, got)
    m = compute_metrics_torch(got, expected.double(), readback_dtype=torch_dtype)
    abs_err = (got.double() - expected.double()).abs()

    # got/true ratio over finite, non-zero reference elements: a tight cluster at a constant != 1 is a scale bug.
    ref = expected.double().flatten()
    act = got.double().flatten()
    keep = torch.isfinite(ref) & (ref != 0)
    ratio = act[keep] / ref[keep]
    r_med = ratio.median().item()
    r_p5, r_p95 = torch.quantile(ratio, torch.tensor([0.05, 0.95], dtype=torch.float64)).tolist()

    row = {
        "shape": list(shape),
        "num_heads": num_heads,
        "dtype": "bfloat16",
        "pcc": float(pcc),
        "max_abs_err": abs_err.max().item(),
        "mean_abs_err": abs_err.mean().item(),
        "rel_rms": m.rms,
        "ulp_p99": m.ulp_p99,
        "ratio_median": r_med,
        "ratio_p5": r_p5,
        "ratio_p95": r_p95,
        "allclose": allclose_msg,
    }
    _record(row)
    for k, v in row.items():
        record_property(k, v)

    # Gate on PCC and relative RMS together, as eval.metrics.check_output does; for a movement op both are exact.
    assert pcc_ok and float(pcc) >= 0.9999, row
    assert m.rms == 0.0, row
    assert torch.equal(got, expected), row
    assert r_med == 1.0 and r_p5 == 1.0 and r_p95 == 1.0, row
