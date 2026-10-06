# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf harness for matmul_reduce_scatter (run under scripts/run_safe_pytest.sh --profile).

N back-to-back calls of a LOOSE_CASES geometry; read DEVICE KERNEL DURATION of the later calls from the profiler CSV
(the first call carries the cross-chip launch skew at the ready fence)."""

import os

import pytest
import torch
import ttnn

from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter

CASES = {
    "focus": ((1, 1, 640, 2048), (2048, 7168), 1, -1, ttnn.bfloat8_b, False),
    "glm": ((1, 1, 640, 4096), (4096, 6144), 1, -1, ttnn.bfloat8_b, False),
    "mimo": ((1, 1, 2048, 2048), (2048, 4096), 1, -2, ttnn.bfloat8_b, True),
}


def _system_mesh_shape():
    shape = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape()
    return tuple(shape[i] for i in range(shape.dims()))


def _to_mesh(stacked, mesh_device, dtype):
    rows = stacked.shape[0]
    glob = torch.cat([torch.cat(list(stacked[r]), dim=1) for r in range(rows)], dim=0)
    return ttnn.from_torch(
        glob,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=tuple(mesh_device.shape)),
    )


@pytest.mark.parametrize("mesh_device", [_system_mesh_shape()], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}], indirect=True)
@pytest.mark.parametrize("case", os.environ.get("MMRS_CASES", "focus").split(","))
def test_perf(mesh_device, case):
    a_shape, w_shape, axis, sd, wdt, fp32 = CASES[case]
    rows, cols = tuple(mesh_device.shape)
    torch.manual_seed(0)
    a = _to_mesh(torch.randn((rows, cols, *a_shape)).to(torch.bfloat16), mesh_device, ttnn.bfloat16)
    w = _to_mesh((torch.randn((rows, cols, *w_shape)) * w_shape[0] ** -0.5), mesh_device, wdt)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=fp32)
    for _ in range(int(os.environ.get("MMRS_CALLS", "5"))):
        matmul_reduce_scatter(a, w, cluster_axis=axis, scatter_dim=sd, num_links=2, compute_kernel_config=cfg)
    ttnn.synchronize_device(mesh_device)
