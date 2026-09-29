# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for high_bw_all_reduce (Phase 0: bf16, TILE, tile_aligned, Linear, axes 0/1).

For every device of the mesh, compares the op's output against
  * `ref_bf16` — the fp32 group sum rounded once to bf16 (the best a bf16 output can be), and
  * `ref_fp32` — the exact fp32 group sum (true value),
and records PCC, max/mean abs error, relative RMS error, bf16 ULP error vs `ref_bf16`, and the
got/true ratio spread (scale-bug detector: a tight cluster around a non-1.0 constant = scale bug).

Run: scripts/run_safe_pytest.sh --dev tests/ttnn/unit_tests/operations/high_bw_all_reduce/test_high_bw_all_reduce_precision_baseline.py -s
"""

from __future__ import annotations

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.high_bw_all_reduce import high_bw_all_reduce

PCC_BF16 = 0.995


def _system_mesh_shape():
    shape = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape()
    return tuple(shape[i] for i in range(shape.dims()))


@pytest.fixture(scope="module")
def mesh_device():
    rows, cols = _system_mesh_shape()
    if rows * cols < 2:
        pytest.skip("high_bw_all_reduce needs a multi-device mesh")
    from tests.scripts.common import get_updated_device_params

    fabric_config = ttnn.FabricConfig.FABRIC_2D
    ttnn.set_fabric_config(fabric_config, ttnn.FabricReliabilityMode.STRICT_INIT)
    device_params = get_updated_device_params({"fabric_config": fabric_config})
    device_params.pop("fabric_config")
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(rows, cols), **device_params)
    try:
        yield mesh
    finally:
        for submesh in mesh.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _to_mesh(stacked, mesh_device):
    rows, cols = stacked.shape[0], stacked.shape[1]
    global_tensor = torch.cat([torch.cat(list(stacked[r]), dim=1) for r in range(rows)], dim=0)
    return ttnn.from_torch(
        global_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(rows, cols)),
    )


def _bf16_ulp(x):
    """Spacing of bf16 values at |x| (7 explicit mantissa bits)."""
    ax = x.abs().clamp_min(torch.finfo(torch.bfloat16).tiny)
    return torch.exp2(torch.floor(torch.log2(ax)) - 7)


def _metrics(actual, ref_bf16, ref_fp32):
    a = actual.to(torch.float64)
    t = ref_fp32.to(torch.float64)
    b = ref_bf16.to(torch.float64)
    diff = a - t
    rel_rms = (diff.pow(2).mean().sqrt() / t.pow(2).mean().sqrt()).item()
    ulp = ((a - b).abs() / _bf16_ulp(b)).flatten()
    nz = t.abs() > 1e-3
    ratio = (a[nz] / t[nz]).flatten()
    q = torch.quantile(ratio[: min(ratio.numel(), 1 << 24)].float(), torch.tensor([0.05, 0.5, 0.95]))
    return {
        "max_abs": diff.abs().max().item(),
        "mean_abs": diff.abs().mean().item(),
        "rel_rms": rel_rms,
        "ulp_max": ulp.max().item(),
        "ulp_mean": ulp.mean().item(),
        "frac_exact_bf16": (a == b).double().mean().item(),
        "ratio_p5": q[0].item(),
        "ratio_median": q[1].item(),
        "ratio_p95": q[2].item(),
    }


SHAPES = [
    (1, 1, 32, 32),  # single tile
    (1, 1, 256, 512),  # small multi-chunk
    (4, 2048, 1024),  # 16 MB bf16, rank 3
    (1, 1, 4096, 4096),  # 32 MB bf16
]


@pytest.mark.parametrize("cluster_axis", [0, 1])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_precision_baseline(mesh_device, shape, cluster_axis):
    rows, cols = tuple(mesh_device.shape)
    if (rows, cols)[cluster_axis] < 2:
        pytest.skip(f"cluster_axis={cluster_axis} has size 1")
    torch.manual_seed(1234)
    stacked = torch.randn((rows, cols, *shape), dtype=torch.bfloat16)
    ref_fp32 = stacked.float().sum(dim=cluster_axis, keepdim=True).expand_as(stacked.float())
    ref_bf16 = ref_fp32.to(torch.bfloat16)

    out = high_bw_all_reduce(_to_mesh(stacked, mesh_device), cluster_axis=cluster_axis, num_links=None)

    worst = None
    for idx, dev_tensor in enumerate(ttnn.get_device_tensors(out)):
        r, c = divmod(idx, cols)
        actual = ttnn.to_torch(dev_tensor)
        assert list(actual.shape) == list(shape)
        _, pcc_msg = assert_with_pcc(ref_fp32[r, c], actual.float(), PCC_BF16)
        _, allclose_msg = comp_allclose(ref_fp32[r, c], actual.float())
        m = _metrics(actual, ref_bf16[r, c], ref_fp32[r, c])
        logger.info(
            f"PRECISION shape={shape} axis={cluster_axis} dev=({r},{c}) {pcc_msg} {allclose_msg} "
            + " ".join(f"{k}={v:.6g}" for k, v in m.items())
        )
        if worst is None or m["rel_rms"] > worst[1]["rel_rms"]:
            worst = ((r, c), m, pcc_msg)
    (r, c), m, pcc_msg = worst
    print(
        f"\nPRECISION_BASELINE shape={'x'.join(map(str, shape))} axis={cluster_axis} worst_dev=({r},{c}) "
        f"pcc=[{pcc_msg}] max_abs={m['max_abs']:.5g} mean_abs={m['mean_abs']:.5g} rel_rms={m['rel_rms']:.5g} "
        f"ulp_max={m['ulp_max']:.3g} ulp_mean={m['ulp_mean']:.3g} exact_bf16={m['frac_exact_bf16']:.6f} "
        f"ratio[p5,med,p95]=[{m['ratio_p5']:.5f},{m['ratio_median']:.5f},{m['ratio_p95']:.5f}]"
    )
    # The chain rounds the partial to bf16 once per hop (G-1 fp32-DEST adds), so the output is within
    # G-1 bf16 ULPs of the once-rounded reference (1 ULP on a 2x2, G=2).
    group = (rows, cols)[cluster_axis]
    assert m["ulp_max"] <= group - 1, f"ULP error {m['ulp_max']} exceeds {group - 1} bf16 ULP"
