# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for g6_sharded_to_interleaved.

The op is an exact copy, so the baseline is a bit-exactness record: PCC, abs errors, relative RMS,
max ULP distance and the got/true ratio spread over the two production shapes, each on its
production shard spec and on the other spec the contract requires (BB-22: no other shapes).

Cells inside SUPPORTED go through the public entry point. A cell outside it is measured through
`_dispatch`, which skips only the support gate, so a refinement that claims it starts from a measured
baseline.

`test_g6_sharded_to_interleaved_precision_matrix` crosses every SUPPORTED dtype with the same four
geometries and both input distributions. The golden suite pairs bfloat8_b only with height -> L1 and
bfloat16 only with width -> DRAM; the matrix covers the other pairings SUPPORTED claims. There is no
compute configuration axis: the op does no arithmetic and its signature takes none.
"""

from __future__ import annotations

import math

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.g6_sharded_to_interleaved import SUPPORTED, g6_sharded_to_interleaved
from ttnn.operations.g6_sharded_to_interleaved.g6_sharded_to_interleaved import _dispatch

TILE = 32

P2_FLUX = {
    "shape": (1, 5120),
    "dtype": ttnn.bfloat16,
    "in_scheme": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    "shard_shape": (32, 160),
    "grid": (((0, 0), (7, 3)),),
    "orientation": ttnn.ShardOrientation.ROW_MAJOR,
    "out": ttnn.DRAM_MEMORY_CONFIG,
}
P1_PANOPTIC = {
    "shape": (1, 128, 256, 32),
    "dtype": ttnn.bfloat8_b,
    "in_scheme": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    "shard_shape": (320, 32),
    "grid": (((0, 0), (10, 8)), ((0, 9), (3, 9))),
    "orientation": ttnn.ShardOrientation.ROW_MAJOR,
    "out": ttnn.L1_MEMORY_CONFIG,
}
CASES = {
    "p2_flux_160t": P2_FLUX,
    "p2_other_spec_160t": {
        **P2_FLUX,
        "shard_shape": (32, 96),
        "grid": (((0, 0), (8, 5)),),
        "orientation": ttnn.ShardOrientation.COL_MAJOR,
    },
    "p1_panoptic_1024t": P1_PANOPTIC,
    "p1_other_spec_1024t": {
        **P1_PANOPTIC,
        "shard_shape": (352, 32),
        "grid": (((0, 0), (8, 9)), ((9, 0), (9, 3))),
        "orientation": ttnn.ShardOrientation.COL_MAJOR,
    },
}
_SCHEME_TAG = {
    ttnn.TensorMemoryLayout.WIDTH_SHARDED: "width_sharded",
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED: "height_sharded",
}
_BUFFER_TAG = {ttnn.BufferType.DRAM: "dram", ttnn.BufferType.L1: "l1"}


def _in_supported(case):
    cell = {
        "dtype": case["dtype"],
        "layout": ttnn.TILE_LAYOUT,
        "in_scheme": _SCHEME_TAG[case["in_scheme"]],
        "buffer": _BUFFER_TAG[case["out"].buffer_type],
    }
    return all(value in SUPPORTED[axis] for axis, value in cell.items())


def _make_input(device, case, distribution="randn"):
    size = device.compute_with_storage_grid_size()
    for _, (x1, y1) in case["grid"]:
        if x1 >= size.x or y1 >= size.y:
            pytest.skip(f"shard grid end ({x1},{y1}) exceeds the device worker grid ({size.x},{size.y})")
    grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for (x0, y0), (x1, y1) in case["grid"]]
    )
    memory_config = ttnn.MemoryConfig(
        case["in_scheme"], ttnn.BufferType.L1, ttnn.ShardSpec(grid, case["shard_shape"], case["orientation"])
    )
    shape = case["shape"]
    rows, cols = math.prod(shape[:-1]), shape[-1]
    # Per-tile power-of-two scale: neighbouring tiles differ, so a misplaced tile cannot hide.
    tile_index = (torch.arange(rows) // TILE)[:, None] * math.ceil(cols / TILE) + (torch.arange(cols) // TILE)[None, :]
    sample = torch.randn if distribution == "randn" else torch.rand
    x = sample(rows, cols) * torch.pow(2.0, (tile_index % 16 - 8).float())
    x = x.reshape(shape).to(torch.bfloat16)
    tt_input = ttnn.from_torch(
        x, dtype=case["dtype"], layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    # bfloat8_b is stored as the host packed it: its reference is the readback of the input as stored.
    expected = x if case["dtype"] == ttnn.bfloat16 else ttnn.to_torch(tt_input)
    return tt_input, expected


# Readback dtype -> (same-width int view, its minimum). bfloat8_b reads back as float32.
_BIT_VIEW = {torch.bfloat16: (torch.int16, -(1 << 15)), torch.float32: (torch.int32, -(1 << 31))}


def _bits(t):
    return t.contiguous().view(_BIT_VIEW[t.dtype][0])


def _ordered(t):
    """Bit patterns mapped to a monotonic integer line, so |a - b| is the ULP distance in t's format."""
    bits = _bits(t).to(torch.int64)
    return torch.where(bits < 0, _BIT_VIEW[t.dtype][1] - bits, bits)


def _metrics(actual, expected):
    _, pcc = comp_pcc(expected, actual, pcc=1.0)
    _, allclose = comp_allclose(expected, actual, rtol=0.0, atol=0.0)
    a, e = actual.double(), expected.double()
    finite = torch.isfinite(a) & torch.isfinite(e)
    err = (a - e)[finite]
    rel_rms = float(err.pow(2).mean().sqrt() / e[finite].std())
    ulp_max = int((_ordered(actual) - _ordered(expected))[finite].abs().max())
    assert actual.dtype == expected.dtype, f"readback dtype {actual.dtype} != {expected.dtype}"
    bit_mismatches = int((_bits(actual) != _bits(expected)).sum())
    nonzero = finite & (e != 0)
    ratio = (a[nonzero] / e[nonzero]).float()
    q = torch.quantile(ratio[: 1 << 24], torch.tensor([0.05, 0.5, 0.95]))
    return {
        "pcc": float(pcc),
        "max_abs": float(err.abs().max()),
        "mean_abs": float(err.abs().mean()),
        "rel_rms": rel_rms,
        "ulp_max": ulp_max,
        "bit_mismatches": bit_mismatches,
        "ratio_p5_p50_p95": tuple(float(v) for v in q),
        "allclose": allclose,
    }


@pytest.mark.parametrize("name", list(CASES), ids=list(CASES))
def test_precision_baseline(device, name):
    _run_case(device, name, CASES[name], "randn")


@pytest.mark.parametrize("distribution", ["rand", "randn"], ids=["uniform", "normal"])
@pytest.mark.parametrize("dtype", SUPPORTED["dtype"], ids=lambda d: str(d).split(".")[-1].lower())
@pytest.mark.parametrize("name", list(CASES), ids=list(CASES))
def test_g6_sharded_to_interleaved_precision_matrix(device, name, dtype, distribution):
    _run_case(device, f"{name}-{dtype}-{distribution}", {**CASES[name], "dtype": dtype}, distribution)


def _run_case(device, name, case, distribution):
    torch.manual_seed(0)
    tt_input, expected = _make_input(device, case, distribution)
    supported = _in_supported(case)
    try:
        if supported:
            tt_output = g6_sharded_to_interleaved(tt_input, case["out"])
        else:
            tt_output = _dispatch(tt_input, case["out"])
        actual = ttnn.to_torch(tt_output)
        m = _metrics(actual, expected)
        path = "entry point" if supported else "_dispatch (outside SUPPORTED)"
        print(
            f"\nPRECISION {name} {case['shape']} {case['dtype']} [{path}]: "
            f"pcc={m['pcc']:.6f} max_abs={m['max_abs']:.3g} mean_abs={m['mean_abs']:.3g} "
            f"rel_rms={m['rel_rms']:.3g} ulp_max={m['ulp_max']} bit_mismatches={m['bit_mismatches']} "
            f"ratio(p5,p50,p95)={m['ratio_p5_p50_p95']} | {m['allclose']}"
        )
        # Exact-copy gate (helpers.TOLERANCES: "exact" for both dtypes).
        assert m["bit_mismatches"] == 0
        assert m["ulp_max"] == 0 and m["max_abs"] == 0.0 and m["rel_rms"] == 0.0
        assert m["pcc"] == pytest.approx(1.0, abs=1e-12)
        ttnn.deallocate(tt_output)
    finally:
        ttnn.deallocate(tt_input)
