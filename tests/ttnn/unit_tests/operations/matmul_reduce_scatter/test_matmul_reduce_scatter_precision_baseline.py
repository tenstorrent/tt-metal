# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for matmul_reduce_scatter (verifier).

Runs on the perf placement: the box's snake ring opened as 1 x N under FABRIC_2D_TORUS_XY, Ring along
cluster_axis 1 (a QuietBox 2 gives the 4-chip ring of the perf cases), with the production router payload.
A Galaxy uses its real torus axis instead. Per shape it records, over every device's output:
PCC, max / mean abs error, relative RMS error, ULP distance (bf16, vs the fp32 reference rounded to bf16),
and the got/true ratio spread (scale-bug detector: a tight cluster around a non-1.0 constant = scale bug).

Results are printed (`-s`) as one `PRECISION_BASELINE` line per case; asserts are the golden tolerances.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from eval.golden_tests.matmul_reduce_scatter.helpers import (
    TOLERANCES,
    create_ttnn_input_tensor,
    output_shape,
    pytorch_matmul_reduce_scatter,
    quantize_like_device,
    random_stacked,
)
from eval.golden_tests.matmul_reduce_scatter.topology import ring_mock_meshes, system_mesh_shape, wrap_axes
from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc, ulp_distance
from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter


def _placement():
    if wrap_axes():
        rows, cols = system_mesh_shape()
        axis = next((a for a in wrap_axes() if (rows, cols)[a] >= 3), None)
        return None if axis is None else ((rows, cols), axis)
    mocks = ring_mock_meshes()
    if not mocks:
        return None
    shape, axis, _ = mocks[0]
    return shape, axis


@pytest.fixture(scope="module")
def ring_mesh():
    placement = _placement()
    if placement is None:
        pytest.skip("this cluster cannot close a ring")
    shape, axis = placement
    from tests.scripts.common import get_updated_device_params

    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = (14 if "blackhole" in ttnn.get_arch_name() else 7) * 1024 + 64
    fc = ttnn.FabricConfig.FABRIC_2D_TORUS_XY
    ttnn.set_fabric_config(
        fc,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        cfg,
    )
    params = get_updated_device_params({"fabric_config": fc})
    params.pop("fabric_config")
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*shape), **params)
    mesh.ring_cluster_axis = axis
    yield mesh
    for submesh in mesh.get_submeshes():
        ttnn.close_mesh_device(submesh)
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# (id, A shape, W shape, scatter_dim, weight dtype, fp32_dest_acc_en floor)
CASES = [
    ("small", (1, 1, 128, 256), (256, 512), -1, ttnn.bfloat16, True),
    ("medium", (1, 1, 640, 1024), (1024, 2048), -2, ttnn.bfloat16, True),
    ("focus_kimi_oproj", (1, 1, 640, 2048), (2048, 7168), -1, ttnn.bfloat8_b, False),
    ("mimo_oproj_rows", (1, 1, 2048, 2048), (2048, 4096), -2, ttnn.bfloat8_b, True),
]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_precision_baseline(ring_mesh, case):
    name, a_shape, w_shape, scatter_dim, weight_dtype, fp32_dest = case
    cluster_axis = ring_mesh.ring_cluster_axis
    g = tuple(ring_mesh.shape)[cluster_axis]
    a = random_stacked(ring_mesh, a_shape, torch.bfloat16, seed=0)
    w = random_stacked(ring_mesh, w_shape, torch.bfloat16, seed=1, scale=w_shape[0] ** -0.5)
    w = quantize_like_device(w, weight_dtype)
    expected = pytorch_matmul_reduce_scatter(a.float(), w.float(), cluster_axis, scatter_dim)  # fp32 reference

    out = matmul_reduce_scatter(
        create_ttnn_input_tensor(a, ring_mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        create_ttnn_input_tensor(w, ring_mesh, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT),
        cluster_axis=cluster_axis,
        scatter_dim=scatter_dim,
        topology=ttnn.Topology.Ring,
        num_links=None,
        compute_kernel_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=fp32_dest
        ),
    )
    assert list(out.shape) == output_shape(a_shape, w_shape, scatter_dim, g)

    cols = tuple(ring_mesh.shape)[1]
    got = torch.stack(
        [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(out)]
    )  # (devices, *out_shape), device order r*cols + c
    ref = torch.stack([expected[i // cols, i % cols] for i in range(got.shape[0])]).float()

    pcc_target, rms_target = TOLERANCES[weight_dtype]
    assert_with_pcc(ref, got, pcc=pcc_target)
    _, allclose_msg = comp_allclose(ref, got)
    err = (got - ref).abs()
    rel_rms = ((got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    # bf16 ULP vs the fp32 reference rounded to bf16. Near-zero references (cancellation in the K-sum) inflate
    # ULP without being large errors, so the tail is also reported on |ref| >= 0.1 * rms(ref) ("nonsmall").
    ulp = ulp_distance(got.to(torch.bfloat16), ref.to(torch.bfloat16)).float().flatten()
    nonsmall = (ref.abs() >= 0.1 * ref.pow(2).mean().sqrt()).flatten()
    sample = lambda t: t[torch.randperm(t.numel())[: 1 << 20]]
    ulp_q = torch.quantile(sample(ulp), torch.tensor([0.5, 0.99]))
    ulp_ns = torch.quantile(sample(ulp[nonsmall]), torch.tensor([0.5, 0.99]))
    nz = (ref.abs() > 1e-3) & torch.isfinite(ref)
    ratio = got[nz] / ref[nz]
    q = torch.quantile(sample(ratio), torch.tensor([0.05, 0.5, 0.95]))
    centered_pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
    print(
        f"\nPRECISION_BASELINE {name}: pcc={centered_pcc:.6f} max_abs={err.max().item():.5f} "
        f"mean_abs={err.mean().item():.6f} rel_rms={rel_rms:.5f} ulp_median={ulp_q[0].item():.0f} "
        f"ulp_p99={ulp_q[1].item():.0f} ulp_nonsmall_median={ulp_ns[0].item():.0f} "
        f"ulp_nonsmall_p99={ulp_ns[1].item():.0f} ulp_max={ulp.max().item():.0f} "
        f"ratio_p5={q[0].item():.4f} ratio_median={q[1].item():.4f} ratio_p95={q[2].item():.4f} | {allclose_msg}"
    )
    assert rel_rms <= rms_target, f"rel_rms {rel_rms} > {rms_target}"
