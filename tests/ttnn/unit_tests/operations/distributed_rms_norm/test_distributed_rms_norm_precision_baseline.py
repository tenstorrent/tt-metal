# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for distributed_rms_norm (verifier, Phase 0).

Runs the production geometry: the box's 4-chip snake ring (1 x N / N x 1), Ring, G = N, gamma TILE, eps 1e-5, HiFi4,
over a few shapes x {bf16 + fp32 DEST, bf16 + 16-bit DEST, fp32}. Per device it reports PCC (assert_with_pcc),
comp_allclose, max / mean abs error, relative RMS error, the output-dtype ULP distance against the reference rounded
to that dtype, and the got/true ratio spread (scale-bug detector: a tight cluster around a non-1.0 constant is a
structural bug, a broad spread around 1.0 is rounding noise). Prints one `PRECISION` line per (case, device).

Run with `--no-precompile` (as test_distributed_rms_norm_perf.py): under the up-front precompile collector the real
pass of this module reads its ttnn.from_torch inputs back as zeros (harness artifact, verification_report.md).

    scripts/run_safe_pytest.sh --no-precompile --run-all <this file> -s
"""

from __future__ import annotations

import os

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc
from eval.golden_tests.distributed_rms_norm.conftest import _FabricMeshCache
from eval.golden_tests.distributed_rms_norm.topology import ring_mock_meshes, system_mesh_shape, topology_feasible
from ttnn.operations.distributed_rms_norm import distributed_rms_norm

from tests.ttnn.unit_tests.operations.distributed_rms_norm.test_distributed_rms_norm import (
    _gamma_to_mesh,
    _reference,
    _stacked,
    _to_mesh,
)

torch.manual_seed(0)
EPS = 1e-5

_SYSTEM = system_mesh_shape()
_RINGS = [(_SYSTEM, ax) for ax in (0, 1) if topology_feasible(ttnn.Topology.Ring, ax, _SYSTEM) is None] or list(
    ring_mock_meshes()
)
pytestmark = [
    pytest.mark.skipif(not _RINGS, reason="needs a ring of >= 3 devices (real wrap axis or snake ring)"),
    pytest.mark.skipif(os.environ.get("UP_FRONT_COLLECT") == "1", reason="run with --no-precompile"),
]

SHAPES = [
    (1, 1, 32, 1792),  # decode (one tile-row)
    (1, 1, 64, 1000),  # W non-aligned
    (1, 1, 640, 1792),  # perf focus
    (1, 1, 2048, 1280),  # larger prefill chunk
]
CONFIGS = [
    # (x dtype, fp32_dest_acc_en, PCC floor)
    pytest.param(ttnn.bfloat16, True, 0.999, id="bf16-fp32acc"),
    pytest.param(ttnn.bfloat16, False, 0.999, id="bf16-16acc"),
    pytest.param(ttnn.float32, True, 0.9999, id="fp32-fp32acc"),
]
TORCH = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}


@pytest.fixture(scope="module")
def mesh_cache():
    cache = _FabricMeshCache()
    yield cache
    cache.close()


def _ulp_distance(got: torch.Tensor, ref: torch.Tensor, torch_dtype) -> torch.Tensor:
    """|ordered-int(got) - ordered-int(ref)| in units of the output dtype (ref rounded to that dtype first)."""
    if torch_dtype == torch.bfloat16:
        a = got.to(torch.bfloat16).view(torch.int16).to(torch.int64)
        b = ref.to(torch.bfloat16).view(torch.int16).to(torch.int64)
        sign = 1 << 15
    else:
        a = got.to(torch.float32).view(torch.int32).to(torch.int64)
        b = ref.to(torch.float32).view(torch.int32).to(torch.int64)
        sign = 1 << 31
    # map sign-magnitude to a monotonic integer line
    a = torch.where(a < 0, -(a + sign), a)
    b = torch.where(b < 0, -(b + sign), b)
    return (a - b).abs()


@pytest.mark.parametrize("dtype,fp32_acc,pcc_floor", CONFIGS)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_precision_baseline(mesh_cache, shape, dtype, fp32_acc, pcc_floor):
    mesh_shape, ax = _RINGS[0]
    mesh = mesh_cache.get(mesh_shape)
    x = _stacked(mesh_shape, shape, TORCH[dtype])
    gamma = _stacked(mesh_shape, (1, 1, 1, shape[-1]), torch.bfloat16)
    tx = _to_mesh(x, mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    tg = _gamma_to_mesh(gamma, mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32_acc, math_approx_mode=False
    )
    out = distributed_rms_norm(
        tx, cluster_axis=ax, gamma=tg, epsilon=EPS, topology=ttnn.Topology.Ring, compute_kernel_config=cfg
    )
    expected = _reference(x, gamma, ax, EPS)
    cols = mesh_shape[1]
    for idx, d in enumerate(ttnn.get_device_tensors(out)):
        r, c = divmod(idx, cols)
        got = ttnn.to_torch(d).to(torch.float32)
        ref = expected[r, c].to(torch.float32)
        diff = (got - ref).abs()
        rel_rms = float(((got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp_min(1e-30)))
        ulp = _ulp_distance(got, ref, TORCH[dtype])
        mask = torch.isfinite(ref) & (ref.abs() > 1e-3)
        ratio = (got[mask] / ref[mask]).double()
        q = torch.quantile(ratio, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64))
        _, allclose_msg = comp_allclose(ref, got)
        print(
            f"PRECISION shape={shape} cfg={dtype}/fp32acc={fp32_acc} dev=({r},{c}) "
            f"max_abs={float(diff.max()):.6g} mean_abs={float(diff.mean()):.6g} rel_rms={rel_rms:.6g} "
            f"ulp_max={int(ulp.max())} ulp_mean={float(ulp.float().mean()):.4f} "
            f"ratio_p05={float(q[0]):.6f} ratio_med={float(q[1]):.6f} ratio_p95={float(q[2]):.6f} | {allclose_msg}"
        )
        passed, pcc_msg = assert_with_pcc(ref, got, pcc_floor)
        print(f"PRECISION_PCC shape={shape} cfg={dtype}/fp32acc={fp32_acc} dev=({r},{c}) pcc={pcc_msg}")
