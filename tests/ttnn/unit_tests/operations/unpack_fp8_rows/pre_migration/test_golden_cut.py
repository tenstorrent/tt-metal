# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Golden coverage cut of unpack_fp8_rows (golden spec: ttnn/ttnn/bringup/fp8_rows_ttnn/tests/unit/test_unpack_fp8_rows.py).

The golden suite has no TARGET / SUPPORTED table; its axes are the rows' memory (ND-sharded 16-row records,
interleaved), ``whole_grid`` (off / on), ``common_values`` (off / on), ranges / batches / an offset output row, and its
inputs (per chip): B3 R512 M640 (the smallest decoded), B2 R2048, B2 R16384 and B1 R65536 (the largest: every core
many bands). Every axis value appears at least once (combined freely); the smallest and the largest inputs are here,
on the whole box (each chip its own rows, the golden's topology), checked with the golden suite's own reference
(e4m3 x scale in fp32, rounded to nearest even onto bfloat8_b in float64) and its tolerance (exact equality).

Every call goes through the acceptance suite's ``unpack_fp8_rows_op`` fixture (the same route selection)."""

import pytest
import torch
import ttnn

import unpack_fp8_rows_pre_migration_helpers as H
from test_acceptance import unpack_fp8_rows_op  # noqa: F401  (the route-selecting fixture)


def _golden_rows(chips, b, r, seed):
    """[chips, b, 1, r, 192] rows, generated a chip at a time (bounded host memory), the golden's key pattern."""
    out = torch.empty(chips, b, 1, r, H.ROW, dtype=torch.uint8)
    for c in range(chips):
        out[c] = H.encode(H.keys((b, 1), r, seed * 1000 + c))
    return out


def _upload(mesh, rows_host, nd):
    return H.rows_to_device(mesh, rows_host, H.nd_config(mesh, 16) if nd else ttnn.DRAM_MEMORY_CONFIG)


def _out(mesh, chips, m, value=3.0):
    prior = torch.full((chips, 1, 1, m, H.VALUES), value)
    return prior, H.out_to_device(mesh, prior)


@pytest.mark.parametrize("nd", [True, False], ids=["nd_sharded", "interleaved"])
def test_golden_smallest(unpack_mesh, unpack_fp8_rows_op, nd):  # noqa: F811
    """The golden's main case (B3 R512 M640): batch 1 rows [64, 448) -> out rows [128, 512), then batch 2 [0, 128)
    at its default out row, in place; the rest of out kept."""
    mesh = unpack_mesh
    chips = H.num_chips(mesh)
    rows_host = _golden_rows(chips, 3, 512, seed=7)
    rows = _upload(mesh, rows_host, nd)
    prior, out = _out(mesh, chips, 640)
    grid = H.grid_cap(mesh)
    unpack_fp8_rows_op(rows, out, row_lo=64, row_hi=448, batch=1, out_row0=128, max_grid=grid)
    exp = H.expected_out(prior, rows_host, 64, 448, 1, 128)
    H.assert_exact(H.to_host(out), exp, "batch 1 [64, 448) -> 128")
    unpack_fp8_rows_op(rows, out, row_lo=0, row_hi=128, batch=2, max_grid=grid)
    exp = H.expected_out(exp, rows_host, 0, 128, 2, 0)
    got = H.to_host(out)
    H.assert_exact(got, exp, "batch 2 [0, 128)")
    assert torch.all(got[..., 7, :] == 0), "a zero row decodes to zero"


# (id, B, R (= M), ND-sharded, whole_grid, common_values, ranges (row_lo, bands, out_row0))
RANGES = [
    ("whole_grid_r2048_nd", 2, 2048, True, True, False, [(0, 1, 0), (96, 3, 64), (512, 37, 256)]),
    ("common_values_r2048_interleaved", 2, 2048, False, False, True, [(0, 1, 0), (96, 3, 64), (512, 37, 256)]),
    (
        "whole_grid_common_values_r16384_nd",
        2,
        16384,
        True,
        True,
        True,
        [(0, 1, 0), (96, 3, 64), (512, 37, 256), (1024, 300, 1536)],
    ),
]


@pytest.mark.parametrize("b, r, nd, whole_grid, common_values, ranges", [pytest.param(*c[1:], id=c[0]) for c in RANGES])
def test_golden_ranges(unpack_mesh, unpack_fp8_rows_op, b, r, nd, whole_grid, common_values, ranges):  # noqa: F811
    """The golden's whole_grid / common_values range sequences (1, 3, 37, 300 bands of batch 1), each to its own
    output rows of one out, exact against the reference."""
    mesh = unpack_mesh
    chips = H.num_chips(mesh)
    rows_host = _golden_rows(chips, b, r, seed=13 + r)
    rows = _upload(mesh, rows_host, nd)
    prior, out = _out(mesh, chips, r)
    exp = prior
    grid = H.grid_cap(mesh)
    for lo, nb, o0 in ranges:
        hi = lo + 32 * nb
        unpack_fp8_rows_op(
            rows,
            out,
            row_lo=lo,
            row_hi=hi,
            batch=1,
            out_row0=o0,
            max_grid=grid,
            whole_grid=whole_grid,
            common_values=common_values,
        )
        exp = H.expected_out(exp, rows_host, lo, hi, 1, o0)
    H.assert_exact(H.to_host(out), exp, "ranges")


def test_golden_largest_many_bands(unpack_mesh, unpack_fp8_rows_op):  # noqa: F811
    """65536 rows a chip (the golden's largest input), the whole range on the capped grid: every core many bands;
    exact against the reference."""
    mesh = unpack_mesh
    chips = H.num_chips(mesh)
    r = 65536
    rows_host = _golden_rows(chips, 1, r, seed=11)
    rows = _upload(mesh, rows_host, True)
    prior, out = _out(mesh, chips, r, value=0.0)
    unpack_fp8_rows_op(rows, out, row_lo=0, row_hi=r, max_grid=H.grid_cap(mesh))
    H.assert_exact(H.to_host(out), H.expected_out(prior, rows_host, 0, r, 0, 0), "65536 rows")
