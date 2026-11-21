# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of rotary_embedding_llama in its supported cells (bfloat16, 32-bit DEST, TILE,
L1 height-sharded, decode mode).

Per shape: PCC (comp_pcc), max / mean abs error (comp_allclose + torch), relative RMS (RMS error over the
reference's std, as eval/metrics.check_output), max / mean bfloat16 ULP error, and the got/true ratio spread
(median, p5, p95 over elements whose reference is not near zero), the scale-bug detector: a tight cluster
around a constant other than 1.0 is a structural bug, a spread centred on 1.0 is rounding.

Two references are measured against:
  * `exact`: the specification's float64 reference on the inputs as the device stores them;
  * `bf16_rotated`: the same, with `x @ T` rounded to bfloat16 before the multiply by sin, which is what the
    op's intermediate ring (`rotated`, kept in the input dtype on Quasar; l1_ledger.md) does. The gap between
    the two shows what that rounding costs.

Under pytest-xdist / --forked the test's output is not shown, so every row is written to the JSON file named by
ROTARY_PRECISION_OUT (default /tmp/rotary_embedding_llama_precision_baseline.json) and recorded as a property.
"""

import json
import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.rotary_embedding_llama import rotary_embedding_llama

TILE = 32
PCC_FLOOR = 0.9997  # NUMERIC_RESOLVED entry (bfloat16, fast): the grade of this cell
REL_RMS_CEILING = 0.04  # DEFAULT_TOLERANCES[(bfloat16, True)]
OUT = os.environ.get("ROTARY_PRECISION_OUT", "/tmp/rotary_embedding_llama_precision_baseline.json")


def _shard(device, torch_tensor):
    config = ttnn.create_sharded_memory_config(
        shape=(TILE, torch_tensor.shape[-1]),
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=config)


def _references(x, cos, sin, t):
    x64 = x.double()
    rotated = (x64.reshape(*x.shape[:-1], -1, TILE) @ t.double().reshape(TILE, TILE)).reshape(x.shape)
    exact = x64 * cos.double() + rotated * sin.double()
    bf16_rotated = x64 * cos.double() + rotated.bfloat16().double() * sin.double()
    return exact, bf16_rotated


def _bf16_ulp(ref):
    """The bfloat16 ULP at each reference value (8 significant bits)."""
    mag = ref.abs().clamp_min(torch.finfo(torch.float32).tiny)
    return torch.pow(2.0, torch.floor(torch.log2(mag)) - 7)


def _metrics(got, ref):
    got64, ref64 = got.double(), ref.double()
    err = got64 - ref64
    _, pcc = comp_pcc(ref64, got64)
    _, allclose_msg = comp_allclose(ref64, got64)
    ulp = err.abs() / _bf16_ulp(ref64)
    significant = ref64.abs() > 0.1 * ref64.std()
    ratio = got64[significant] / ref64[significant]
    return {
        "pcc": float(pcc),
        "max_abs_err": float(err.abs().max()),
        "mean_abs_err": float(err.abs().mean()),
        "rel_rms": float(err.pow(2).mean().sqrt() / ref64.std()),
        "max_ulp": float(ulp.max()),
        "mean_ulp": float(ulp.mean()),
        "ratio_median": float(ratio.median()),
        "ratio_p5": float(ratio.quantile(0.05)),
        "ratio_p95": float(ratio.quantile(0.95)),
        "allclose": allclose_msg,
    }


def _record(record_property, row):
    record_property("precision", json.dumps(row))
    rows = []
    if os.path.exists(OUT):
        with open(OUT) as f:
            rows = json.load(f)
    rows = [r for r in rows if r["shape"] != row["shape"]] + [row]
    with open(OUT, "w") as f:
        json.dump(sorted(rows, key=lambda r: r["shape"]), f, indent=1)


@pytest.mark.parametrize(
    "heads, head_dim",
    [(1, 32), (8, 64), (12, 96), (8, 128), (32, 128)],
    ids=lambda v: str(v),
)
def test_precision_baseline(device, record_property, heads, head_dim):
    torch.manual_seed(1234)
    host = (
        torch.randn(1, 1, heads, head_dim).bfloat16(),
        torch.randn(1, 1, 1, head_dim).bfloat16(),
        torch.randn(1, 1, 1, head_dim).bfloat16(),
        torch.randn(1, 1, TILE, TILE).bfloat16(),
    )
    out = rotary_embedding_llama(*(_shard(device, h) for h in host), is_decode_mode=True)
    got = ttnn.to_torch(out).float()
    exact, bf16_rotated = _references(*host)

    row = {
        "shape": [1, 1, heads, head_dim],
        "exact": _metrics(got, exact),
        "bf16_rotated": _metrics(got, bf16_rotated),
    }
    _record(record_property, row)

    m = row["exact"]
    assert m["pcc"] >= PCC_FLOOR, row
    assert m["rel_rms"] <= REL_RMS_CEILING, row
    # Scale-bug detector: the ratio of non-negligible elements is centred on 1.
    assert abs(m["ratio_median"] - 1.0) < 0.01, row
