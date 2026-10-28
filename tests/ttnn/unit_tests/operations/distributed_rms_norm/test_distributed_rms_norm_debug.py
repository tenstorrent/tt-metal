# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Debug tests for distributed_rms_norm — DO NOT DELETE (documents the debugging process).

1. test_golden_ring_mock_repro: the golden test_ring_mock case (seeded inputs) with per-device / per-tile-row / per
   W-slice error localization printed (golden failed at PCC ~0.93 on 640x1792 while the acceptance test passed).
2. test_row_scaled: x[p][row] = (1 + row % 13) * randn — every row has a distinct sum of squares, so any
   row <-> statistic mix-up (wrong record, wrong wave, wrong owner slot) shows up as a large per-row error.
   Found: the ragged last row-group (640 rows / 7 groups) got stale inv rows — SEM_INV counted cumulatively over waves
   while the owner of an empty wave announced at once (fixed: wave-ordered announces); 8192x256 (G_w 2, 5 waves)
   covers owners that own several waves (SEM_GATHER per-owned-wave bit fields).
"""

from __future__ import annotations

import os

import pytest
import torch
import ttnn

from eval.golden_tests.distributed_rms_norm.conftest import _FabricMeshCache
from eval.golden_tests.distributed_rms_norm.helpers import (
    create_ttnn_gamma,
    create_ttnn_input_tensor,
    pytorch_distributed_rms_norm,
    random_stacked,
)
from ttnn.operations.distributed_rms_norm import distributed_rms_norm, plan_summary


# Driver / debug file, not a correctness gate: run it with `scripts/run_safe_pytest.sh --no-precompile`. Under the
# up-front precompile collector (UP_FRONT_COLLECT=1) the real pass of these modules reads their ttnn.from_torch inputs
# back as zeros (observed with INSTATS prints: the device input itself is zero, before the op runs); the acceptance and
# golden suites are unaffected.
pytestmark = pytest.mark.skipif(os.environ.get("UP_FRONT_COLLECT") == "1", reason="run with --no-precompile")


@pytest.fixture(scope="module")
def mesh_cache():
    cache = _FabricMeshCache()
    yield cache
    cache.close()


def _localize(out, expected, mesh_shape):
    rows, cols = mesh_shape
    worst = 0.0
    for idx, d in enumerate(ttnn.get_device_tensors(out)):
        r, c = divmod(idx, cols)
        got = ttnn.to_torch(d).float()
        exp = expected[r, c].float()
        err = (got - exp).abs()
        rel = err / (exp.abs() + 1e-2)
        M, W = got.shape[-2], got.shape[-1]
        T = (M + 31) // 32
        bad_rows = []
        for t in range(T):
            e = rel[..., t * 32 : (t + 1) * 32, :].mean().item()
            if e > 0.05:
                bad_rows.append((t, round(e, 3)))
        bad_cols = []
        for s in range(0, (W + 31) // 32):
            e = rel[..., :, s * 32 : (s + 1) * 32].mean().item()
            if e > 0.05:
                bad_cols.append(s)
        ratio = (got / exp).flatten()
        print(
            f"DBG dev ({r},{c}) max_err {err.max().item():.4f} mean_rel {rel.mean().item():.4f} "
            f"bad tile-rows {bad_rows[:12]}{'...' if len(bad_rows) > 12 else ''} bad tile-cols {bad_cols[:20]} "
            f"ratio median {ratio.median().item():.4f}"
        )
        worst = max(worst, rel.mean().item())
    return worst


@pytest.mark.parametrize("mesh_shape,cluster_axis", [((1, 4), 1)])
@pytest.mark.parametrize("shape", [(1, 1, 640, 1792), (1, 1, 96, 512)])
def test_golden_ring_mock_repro(mesh_cache, mesh_shape, cluster_axis, shape):
    mesh = mesh_cache.get(mesh_shape)
    x = random_stacked(mesh, shape, torch.bfloat16, seed=11)
    gamma = random_stacked(mesh, (1, 1, 1, shape[-1]), torch.bfloat16, seed=12)
    expected = pytorch_distributed_rms_norm(x, gamma, cluster_axis, 1e-6)
    tx = create_ttnn_input_tensor(x, mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tg = create_ttnn_gamma(gamma, mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    print("DBG plan", plan_summary(tx, cluster_axis=cluster_axis, gamma=tg, topology=ttnn.Topology.Ring, num_links=1))
    out = distributed_rms_norm(
        tx, cluster_axis=cluster_axis, gamma=tg, epsilon=1e-6, topology=ttnn.Topology.Ring, num_links=1
    )
    assert _localize(out, expected, mesh_shape) < 0.02


@pytest.mark.parametrize("mesh_shape,cluster_axis", [((1, 4), 1)])
@pytest.mark.parametrize("shape", [(1, 1, 640, 1792), (1, 1, 96, 512), (1, 1, 8192, 256), (1, 1, 2048, 1280)])
def test_row_scaled(mesh_cache, mesh_shape, cluster_axis, shape):
    mesh = mesh_cache.get(mesh_shape)
    torch.manual_seed(0)
    x = torch.randn((*mesh_shape, *shape))
    rows = torch.arange(shape[-2]).float()
    x = (x * (1 + rows % 13).reshape(-1, 1)).to(torch.bfloat16)
    expected = pytorch_distributed_rms_norm(x, None, cluster_axis, 1e-6)
    tx = create_ttnn_input_tensor(x, mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    out = distributed_rms_norm(tx, cluster_axis=cluster_axis, epsilon=1e-6, topology=ttnn.Topology.Ring, num_links=1)
    # implied inv per row on device 0 vs the true inv: which statistic did each bad row get?
    got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()[0, 0]
    xf = x[0, 0, 0, 0].float()
    full = torch.cat([x[0, c, 0, 0].float() for c in range(mesh_shape[1])], dim=-1)
    inv_true = torch.rsqrt(full.pow(2).mean(-1) + 1e-6)
    inv_got = (got / xf).median(dim=-1).values
    ss_true = full.pow(2).sum(-1)
    ss_got = shape[-1] * mesh_shape[1] * (1.0 / inv_got.pow(2) - 1e-6)
    for row in list(range(0, 4)) + list(range(574, 582)) + list(range(606, 610)) + list(range(636, 640)):
        if row < shape[-2]:
            print(
                f"DBG row {row} inv_true {inv_true[row]:.5f} inv_got {inv_got[row]:.5f} ss_true {ss_true[row]:.1f} ss_got {ss_got[row]:.1f} frac {ss_got[row] / ss_true[row]:.4f}"
            )
    assert _localize(out, expected, mesh_shape) < 0.02
