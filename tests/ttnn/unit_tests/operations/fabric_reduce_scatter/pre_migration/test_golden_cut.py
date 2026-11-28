# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Golden coverage cut of fabric_reduce_scatter: every value of every axis the golden suite exercises (cluster_axis
0 / 1 / None; dim 2 / 3; bf16 FPU adds / fp32 exact adds; the default path, cache_descriptors, program_hash,
prepared), the golden suite's largest inputs (axis 0 [4096, 4096], whole mesh [5120, 4096]) and its smallest
([256, 512]), each checked with the golden suite's own reference and tolerance, on the op's unit-test mesh
(BRINGUP_TEST_MESH, else 2x2):

  * test_fabric_reduce_scatter.py: bf16 randn partials vs the float torch sum; max abs error <= max(0.25, 2 x the
    error of its comparison path against the same torch sum): ttnn.reduce_scatter for an axis, the op's own axis 0
    then axis 1 two-stage form for the whole mesh (as the golden test does);
  * test_fabric_rs_exact_fp32.py: a group of 2 bit-identical to torch's fp32 a + b; a group of G within G - 1 fp32
    roundings of the float64 sum (|got - sum| <= (G - 1) * 2^-24 * sum |x_i|; the golden test's "3 roundings" is
    G - 1 at its 2x2 whole mesh); the whole-mesh column count is widened to G tiles where 512 does not split;
  * test_fabric_rs_program_hash.py / test_fabric_rs_prepared.py: bit-identical to the default path, and 2 x the input
    in fresh buffers gives exactly 2 x the output.

Every call of the op goes through test_acceptance.py's ``fabric_reduce_scatter_op`` route fixture (loaded from that
file)."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import torch
import ttnn

from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology

_spec = importlib.util.spec_from_file_location(
    "_fabric_rs_pre_migration_acceptance", Path(__file__).with_name("test_acceptance.py")
)
_acceptance = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_acceptance)
fabric_reduce_scatter_op = _acceptance.fabric_reduce_scatter_op  # the route-selected entry (source / native)

_MESH = tuple(int(v) for v in (os.environ.get("BRINGUP_TEST_MESH") or "2x2").split("x"))
MESH = pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param(_MESH, fabric2d_device_params(), id="x".join(map(str, _MESH)))],
    indirect=["mesh_device", "device_params"],
)

BF16_FLOOR = 0.25
H = 4096


def _bf16_parts(rows, cols, T, seed):
    torch.manual_seed(seed)
    return torch.randn(rows, cols, T, H).bfloat16().float()


@MESH
@pytest.mark.parametrize("cluster_axis", [0, 1], ids=["axis0", "axis1"])
def test_bf16_axis_golden(mesh_device, device_params, fabric_reduce_scatter_op, rs, cluster_axis):
    """test_fabric_rs: T = 4096 rows on axis 0, 4096 // rows on axis 1, H = 4096, 2 links."""
    rows, cols = tuple(mesh_device.shape)
    T = 4096 if cluster_axis == 0 else 4096 // rows
    parts = _bf16_parts(rows, cols, T, cluster_axis)
    x = rs.upload(mesh_device, parts, ttnn.bfloat16)
    g = rs.group_size(mesh_device, cluster_axis)
    want = rs.expected(parts, cluster_axis, 2)
    out = fabric_reduce_scatter_op(x, cluster_axis=cluster_axis, num_links=2)
    rs.assert_output_meta(out, [1, 1, T // g, H], ttnn.bfloat16)
    got = rs.per_chip(out)
    stock = ttnn.reduce_scatter(
        x,
        dim=2,
        cluster_axis=cluster_axis,
        topology=per_axis_topology()[cluster_axis],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    stock_err = rs.assert_max_abs(rs.per_chip(stock), want, float("inf"), "ttnn.reduce_scatter")
    tol = max(BF16_FLOOR, 2 * stock_err)
    err = rs.assert_max_abs(got, want, tol, f"axis {cluster_axis} bf16 sum")
    print(f"golden bf16 axis {cluster_axis} [{T}, {H}]: max abs err {err:.3g}, ttnn.reduce_scatter {stock_err:.3g}")


@MESH
def test_bf16_full_mesh_golden(mesh_device, device_params, fabric_reduce_scatter_op, rs):
    """test_fabric_rs_full_mesh: T = 5120, H = 4096, 2 links, cluster_axis=None vs the axis 0 then axis 1 form."""
    rows, cols = tuple(mesh_device.shape)
    if rows < 2 or cols < 2:
        pytest.skip("needs a 2D mesh")
    T = 5120
    parts = _bf16_parts(rows, cols, T, 7)
    x = rs.upload(mesh_device, parts, ttnn.bfloat16)
    want = rs.expected(parts, None, 2)
    out = fabric_reduce_scatter_op(x, cluster_axis=None, num_links=2)
    rs.assert_output_meta(out, [1, 1, T // (rows * cols), H], ttnn.bfloat16)
    got = rs.per_chip(out)
    ttnn.deallocate(out)
    a = fabric_reduce_scatter_op(x, cluster_axis=0, num_links=2)
    b = fabric_reduce_scatter_op(a, cluster_axis=1, num_links=2)
    two_err = rs.assert_max_abs(rs.per_chip(b), want, float("inf"), "two-stage")
    tol = max(BF16_FLOOR, 2 * two_err)
    err = rs.assert_max_abs(got, want, tol, "whole-mesh bf16 sum")
    print(f"golden bf16 full mesh [{T}, {H}]: max abs err {err:.3g}, two-stage {two_err:.3g}")


ROWS, COLS = 1024, 512  # test_fabric_rs_exact_fp32


def _check_exact(rs, got, parts, cluster_axis, dim, g):
    if g == 2:
        rs.assert_exact(got, rs.expected(parts, cluster_axis, dim, acc=torch.float32), "exact pair vs fp32 a + b")
    want64 = rs.expected(parts, cluster_axis, dim)
    bound = rs.expected(parts.abs(), cluster_axis, dim)
    worst = max(
        float(((t.double() - w).abs() / (max(g - 1, 1) * 2.0**-24 * b)).max()) for t, w, b in zip(got, want64, bound)
    )
    assert worst <= 1.0, f"exact adds: |err| / ({g - 1}-rounding bound) = {worst:.3f}"
    return worst


@MESH
@pytest.mark.parametrize(
    ("cluster_axis", "dim"),
    [pytest.param(0, 2, id="axis0_rows"), pytest.param(1, 3, id="axis1_cols")],
)
def test_exact_axis_golden(mesh_device, device_params, fabric_reduce_scatter_op, rs, cluster_axis, dim):
    """test_exact_pair: fp32 spread partials [1024, 512], 2 links, exact_fp32_add."""
    rows, cols = tuple(mesh_device.shape)
    g = rs.group_size(mesh_device, cluster_axis)
    parts = rs.fp32_spread_parts(rows, cols, ROWS, COLS, seed=11 + cluster_axis + 2 * dim)
    x = rs.upload(mesh_device, parts, ttnn.float32)
    out = fabric_reduce_scatter_op(x, cluster_axis=cluster_axis, num_links=2, dim=dim, exact_fp32_add=True)
    rs.assert_output_meta(out, rs.out_shape([1, 1, ROWS, COLS], g, dim), ttnn.float32)
    _check_exact(rs, rs.per_chip(out), parts, cluster_axis, dim, g)


@MESH
@pytest.mark.parametrize("dim", [pytest.param(2, id="rows"), pytest.param(3, id="cols")])
def test_exact_full_mesh_golden(mesh_device, device_params, fabric_reduce_scatter_op, rs, dim):
    """test_exact_full_mesh: cluster_axis=None, exact on (within the bound) and off (misses it)."""
    rows, cols = tuple(mesh_device.shape)
    if rows < 2 or cols < 2:
        pytest.skip("needs a 2D mesh")
    n = rows * cols
    W = max(COLS, 32 * n) if dim == 3 else COLS
    parts = rs.fp32_spread_parts(rows, cols, ROWS, W, seed=5 + dim)
    x = rs.upload(mesh_device, parts, ttnn.float32)
    out = fabric_reduce_scatter_op(x, cluster_axis=None, num_links=2, dim=dim, exact_fp32_add=True)
    rs.assert_output_meta(out, rs.out_shape([1, 1, ROWS, W], n, dim), ttnn.float32)
    _check_exact(rs, rs.per_chip(out), parts, None, dim, n)
    off = rs.per_chip(fabric_reduce_scatter_op(x, cluster_axis=None, num_links=2, dim=dim))
    want64, bound = rs.expected(parts, None, dim), rs.expected(parts.abs(), None, dim)
    worst_off = max(
        float(((t.double() - w).abs() / ((n - 1) * 2.0**-24 * b)).max()) for t, w, b in zip(off, want64, bound)
    )
    assert worst_off > 1.0, "control: the FPU path should miss the exact-add bound"


SMALL_R, SMALL_W = 256, 512  # test_fabric_rs_program_hash / test_fabric_rs_prepared


@MESH
@pytest.mark.parametrize(
    ("mode", "cluster_axis", "dim", "exact"),
    [
        pytest.param("program_hash", 1, 3, True, id="program_hash_axis1_cols_fp32_exact"),
        pytest.param("program_hash", 0, 2, False, id="program_hash_axis0_rows_bf16"),
        pytest.param("prepared", 0, 2, False, id="prepared_axis0_rows_bf16"),
        pytest.param("prepared", 1, 3, True, id="prepared_axis1_cols_fp32_exact"),
    ],
)
def test_host_modes_golden(mesh_device, device_params, fabric_reduce_scatter_op, rs, mode, cluster_axis, dim, exact):
    op = fabric_reduce_scatter_op
    rows, cols = tuple(mesh_device.shape)
    g = rs.group_size(mesh_device, cluster_axis)
    dtype = ttnn.float32 if exact else ttnn.bfloat16
    gen = torch.Generator().manual_seed(7 + cluster_axis + 2 * dim + 4 * int(exact))
    W = SMALL_W * g // 2 if mode == "prepared" else SMALL_W
    parts = torch.randn(rows, cols, SMALL_R, W, generator=gen)
    if not exact:
        parts = parts.bfloat16().float()
    x = rs.upload(mesh_device, parts, dtype)
    kw = dict(cluster_axis=cluster_axis, num_links=2, dim=dim, exact_fp32_add=exact)
    fast = dict(cache_descriptors=True, program_hash=True, prepared=mode == "prepared")
    plain = rs.per_chip(op(x, **kw))
    if exact:
        _check_exact(rs, plain, parts, cluster_axis, dim, g)
    else:
        rs.assert_max_abs(plain, rs.expected(parts, cluster_axis, dim), BF16_FLOOR, "default path")
    first_out = op(x, **kw, **fast)
    rs.assert_exact(rs.per_chip(first_out), plain, f"{mode} vs the default path")
    rs.assert_exact(rs.per_chip(op(x, output=first_out, **kw, **fast)), plain, f"{mode} repeat")
    keep = [
        ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 64, 64]), dtype, ttnn.TILE_LAYOUT, mesh_device)
        for _ in range(3)
    ]
    x2 = rs.upload(mesh_device, 2 * parts, dtype)
    assert x2.buffer_address() != x.buffer_address()
    rs.assert_exact(rs.per_chip(op(x2, **kw, **fast)), [2 * p for p in plain], f"{mode} fresh buffers")
    rs.assert_exact(rs.per_chip(op(x, output=first_out, **kw, **fast)), plain, f"{mode} first buffers again")
    for t in keep:
        ttnn.deallocate(t)
