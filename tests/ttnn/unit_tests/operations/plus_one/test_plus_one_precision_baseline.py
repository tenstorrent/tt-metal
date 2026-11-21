# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of plus_one over its supported cells (rank 1, DRAM, int32 / uint32, both flag values).

plus_one is graded `exact` (spec NUMERIC, class `integer`): the gate is integer equality, i.e. a max ULP
(integer distance, modulo 2^32) of 0. PCC, max / mean abs error, relative RMS and the got/true ratio spread
are recorded alongside, as the verifier protocol asks, and must read as an exact result (PCC 1, errors 0,
ratio median 1 with zero spread).

Inputs as in the golden suite: integers over the whole 32-bit range of the dtype, with the edge values the
rules turn on (wrap at INT32_MAX / 2^32 - 1, the [0, INT32_MAX) window of the flag) at the start.

Under the simulator's forked workers a test's output is hidden: each measurement is recorded with
`record_property` and, when PLUS_ONE_BASELINE_OUT names a file, appended to it as one JSON line.
"""

import json
import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.plus_one import plus_one

MASK = 0xFFFFFFFF
INT32_MAX = 2**31 - 1
EDGES = {
    ttnn.int32: [-1, 0, INT32_MAX, -(2**31), INT32_MAX - 1, 1],
    ttnn.uint32: [0, 2**32 - 1, INT32_MAX, 2**31, INT32_MAX - 1, 1],
}
SHAPES = [(1,), (17,), (100,), (4096,)]


def _values(shape, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 2**32, shape, generator=g, dtype=torch.int64)
    flat = bits.view(-1)
    edges = EDGES[dtype][: max(0, min(flat.numel() - 1, len(EDGES[dtype])))]
    for i, v in enumerate(edges):
        flat[i] = v & MASK
    if dtype == ttnn.int32:
        return torch.where(bits >= 2**31, bits - 2**32, bits).to(torch.int32)
    return bits  # uint32 values held in int64 on the host


def _reference(x, dtype, skip):
    """The specification's reference in float64: wraps at the dtype's ends; the flag tests the int32 view."""
    v = x.to(torch.float64)
    low, high = (-(2.0**31), 2.0**31 - 1) if dtype == ttnn.int32 else (0.0, 2.0**32 - 1)
    result = torch.where(v == high, torch.full_like(v, low), v + 1)
    if skip:
        result = torch.where((v >= 0) & (v < INT32_MAX), result, v)
    return result


@pytest.mark.parametrize("skip", [False, True], ids=["noskip", "skip"])
@pytest.mark.parametrize("dtype", [ttnn.int32, ttnn.uint32], ids=["int32", "uint32"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_precision_baseline(device, shape, dtype, skip, record_property):
    x = _values(shape, dtype, seed=hash((shape, skip)) & 0xFFFF)
    t = ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = plus_one(t, skip_negative_entries=skip)
    assert out is t, "plus_one must return the tensor it was given (R3)"

    got = ttnn.to_torch(out).to(torch.int64).reshape(shape)
    if dtype == ttnn.uint32:
        got = got & MASK  # a uint32 read back may come as a negative int32 view
    got64 = got.to(torch.float64)
    expected = _reference(x, dtype, skip)

    # ULP of an integer result: the distance between the two 32-bit words, modulo 2^32.
    diff = (got.to(torch.int64) - expected.to(torch.int64)) & MASK
    ulp = torch.minimum(diff, (2**32 - diff) & MASK)
    max_ulp = int(ulp.max())

    _, pcc = comp_pcc(expected, got64, pcc=0.9999)
    _, allclose_msg = comp_allclose(expected, got64)
    err = (got64 - expected).abs()
    ref_std = float(expected.std()) if expected.numel() > 1 else 0.0
    rms = float(err.pow(2).mean().sqrt())
    rel_rms = rms / ref_std if ref_std > 0 else (0.0 if rms == 0 else float("inf"))

    # got/true ratio spread (scale-bug detector), over the non-zero reference elements.
    nz = expected != 0
    ratio = got64[nz] / expected[nz]
    ratio_median = float(ratio.median()) if ratio.numel() else 1.0
    ratio_p5, ratio_p95 = (
        (float(torch.quantile(ratio, 0.05)), float(torch.quantile(ratio, 0.95))) if ratio.numel() else (1.0, 1.0)
    )

    row = {
        "shape": list(shape),
        "dtype": "int32" if dtype == ttnn.int32 else "uint32",
        "skip_negative_entries": skip,
        "pcc": float(pcc),
        "max_abs_err": float(err.max()),
        "mean_abs_err": float(err.mean()),
        "rel_rms": rel_rms,
        "max_ulp": max_ulp,
        "mismatches": int((ulp != 0).sum()),
        "ratio_median": ratio_median,
        "ratio_p5": ratio_p5,
        "ratio_p95": ratio_p95,
        "allclose": allclose_msg,
    }
    for key, value in row.items():
        record_property(key, value)
    out_path = os.environ.get("PLUS_ONE_BASELINE_OUT")
    if out_path:
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    # The gate: exact (spec NUMERIC: integer class, `exact`).
    assert max_ulp == 0, row
    assert float(pcc) >= 0.9999 and rel_rms == 0.0, row
