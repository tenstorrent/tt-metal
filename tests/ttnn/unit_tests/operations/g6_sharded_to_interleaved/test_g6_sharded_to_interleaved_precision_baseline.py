# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for g6_sharded_to_interleaved (exact-copy op).

Scope (BB-22): the two production cases, each on its production shard spec and on the
"other" shard spec the spec's rules require. The op does no arithmetic, so the expected result
is bit-exact; PCC / abs error / relative RMS / ULP / got-true ratio are recorded anyway.

p1 cases (bfloat8_b, height sharded, L1 output) are outside the Phase 0 SUPPORTED. They are run
with SUPPORTED widened for the duration of the test (``monkeypatch``), not through a gate bypass in
the op, so Refinement 1 starts from a measured result. The ``path`` column of the printed table
says which cells were claimed.
"""

from __future__ import annotations

import math
import sys

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.g6_sharded_to_interleaved import g6_sharded_to_interleaved

# The package re-exports the function under the module's own name, so fetch the module object itself.
op_module = sys.modules[g6_sharded_to_interleaved.__module__]

TILE = 32

_P2 = dict(
    shape=(1, 5120),
    dtype=ttnn.bfloat16,
    scheme=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    shard=(32, 160),
    grid=(((0, 0), (7, 3)),),
    orient=ttnn.ShardOrientation.ROW_MAJOR,
    out=ttnn.DRAM_MEMORY_CONFIG,
)
_P1 = dict(
    shape=(1, 128, 256, 32),
    dtype=ttnn.bfloat8_b,
    scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    shard=(320, 32),
    grid=(((0, 0), (10, 8)), ((0, 9), (3, 9))),
    orient=ttnn.ShardOrientation.ROW_MAJOR,
    out=ttnn.L1_MEMORY_CONFIG,
)
CASES = {
    "p2_production": _P2,
    "p2_other_spec": {**_P2, "shard": (32, 96), "grid": (((0, 0), (8, 5)),), "orient": ttnn.ShardOrientation.COL_MAJOR},
    "p1_production": _P1,
    "p1_other_spec": {
        **_P1,
        "shard": (352, 32),
        "grid": (((0, 0), (8, 9)), ((9, 0), (9, 3))),
        "orient": ttnn.ShardOrientation.COL_MAJOR,
    },
}


def _make_input(shape):
    torch.manual_seed(0)
    rows, cols = math.prod(shape[:-1]), shape[-1]
    x = torch.randn(rows, cols, dtype=torch.float32)
    tile_cols = -(-cols // TILE)
    tile_idx = (torch.arange(rows) // TILE)[:, None] * tile_cols + (torch.arange(cols) // TILE)[None, :]
    x = x * torch.pow(2.0, (tile_idx % 16 - 8).to(torch.float32))
    return x.reshape(shape).to(torch.bfloat16)


def _input_mc(device, case):
    size = device.compute_with_storage_grid_size()
    for _, (x1, y1) in case["grid"]:
        if x1 >= size.x or y1 >= size.y:
            pytest.skip(f"shard grid end ({x1},{y1}) outside worker grid")
    grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for (x0, y0), (x1, y1) in case["grid"]]
    )
    return ttnn.MemoryConfig(case["scheme"], ttnn.BufferType.L1, ttnn.ShardSpec(grid, case["shard"], case["orient"]))


def _metrics(actual, expected):
    a, e = actual.to(torch.float32).flatten(), expected.to(torch.float32).flatten()
    _, pcc = comp_pcc(expected, actual, pcc=0.0)
    _, allclose_msg = comp_allclose(expected, actual)
    finite = torch.isfinite(a) & torch.isfinite(e)
    err = (a - e)[finite]
    max_abs = float(err.abs().max()) if err.numel() else 0.0
    mean_abs = float(err.abs().mean()) if err.numel() else 0.0
    std = float(e[finite].std())
    rel_rms = float(err.pow(2).mean().sqrt()) / std if std > 0 else 0.0
    ulp = (
        actual.contiguous().view(torch.int16).to(torch.int32) - expected.contiguous().view(torch.int16).to(torch.int32)
    ).abs()
    nz = finite & (e != 0)
    r = a[nz] / e[nz]
    return dict(
        pcc=float(pcc),
        max_abs=max_abs,
        mean_abs=mean_abs,
        rel_rms=rel_rms,
        max_ulp=int(ulp.max()),
        bit_mismatches=int((ulp != 0).sum()),
        ratio_median=float(r.median()),
        ratio_p5=float(torch.quantile(r[:1_000_000], 0.05)),
        ratio_p95=float(torch.quantile(r[:1_000_000], 0.95)),
        allclose=allclose_msg,
    )


@pytest.mark.parametrize("case_id", list(CASES))
def test_precision_baseline(device, case_id, monkeypatch):
    case = CASES[case_id]
    claimed = case["dtype"] in op_module.SUPPORTED["dtype"]
    if not claimed:
        # Measure the not-yet-claimed production cell through the real entry point.
        widened = {k: list(v) for k, v in op_module.SUPPORTED.items()}
        widened["dtype"].append(ttnn.bfloat8_b)
        widened["in_scheme"].append("height_sharded")
        widened["buffer"].append("l1")
        monkeypatch.setattr(op_module, "SUPPORTED", widened)

    torch_input = _make_input(case["shape"])
    tt_input = ttnn.from_torch(
        torch_input, dtype=case["dtype"], layout=ttnn.TILE_LAYOUT, device=device, memory_config=_input_mc(device, case)
    )
    expected = torch_input if case["dtype"] == ttnn.bfloat16 else ttnn.to_torch(tt_input)
    out = g6_sharded_to_interleaved(tt_input, case["out"])
    actual = ttnn.to_torch(out)

    m = _metrics(actual, expected)
    path = "entry point (SUPPORTED)" if claimed else "entry point (SUPPORTED widened, pre-claim)"
    print(
        f"\nPRECISION {case_id} shape={case['shape']} dtype={case['dtype']} path={path} "
        f"pcc={m['pcc']:.6f} max_abs={m['max_abs']} mean_abs={m['mean_abs']} rel_rms={m['rel_rms']} "
        f"max_ulp={m['max_ulp']} bit_mismatches={m['bit_mismatches']} "
        f"ratio median={m['ratio_median']} p5={m['ratio_p5']} p95={m['ratio_p95']}"
    )
    # Exact-copy op: TOLERANCES are ("exact", None).
    assert m["bit_mismatches"] == 0, f"{case_id}: {m}"
    for t in (out, tt_input):
        ttnn.deallocate(t)
