# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gate P2.1: open the 1x4 Blackhole mesh and check the CCLs the model will use."""

import os

import pytest
import torch

import ttnn
from models.demos.ernie45_d_p.bringup import metrics
from models.demos.ernie45_d_p.reference.ernie_ref import pcc

TASK = os.environ.get("ERNIE_BRINGUP_TASK", "P2.1")
MESH_PARAMS = [pytest.param((1, 4), id="1x4")]
DEVICE_PARAMS = [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}]


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", MESH_PARAMS, indirect=True)
def test_mesh_info(mesh_device):
    n = mesh_device.get_num_devices()
    grid = mesh_device.compute_with_storage_grid_size()
    dram = mesh_device.dram_grid_size()
    print(
        f"mesh shape={tuple(mesh_device.shape)} devices={n} arch={mesh_device.arch()} grid={grid.x}x{grid.y} dram={dram.x}"
    )
    metrics.record(TASK, "num_devices", n)
    metrics.record(TASK, "worker_grid", f"{grid.x}x{grid.y}")
    metrics.record(TASK, "dram_banks", dram.x)


def _to_mesh(t, mesh, dim):
    mapper = ttnn.ShardTensorToMesh(mesh, dim=dim) if dim is not None else ttnn.ReplicateTensorToMesh(mesh)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=mapper)


def _per_device(t, mesh):
    return [ttnn.to_torch(x) for x in ttnn.get_device_tensors(t)]


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", MESH_PARAMS, indirect=True)
@pytest.mark.parametrize("seq", [2048])
def test_ccl(mesh_device, seq):
    torch.manual_seed(0)
    H = 2560
    n = mesh_device.get_num_devices()

    # all_gather along hidden: each device holds H/4 columns -> full H everywhere
    full = torch.randn(1, 1, seq, H)
    ag = ttnn.all_gather(_to_mesh(full, mesh_device, 3), dim=3, cluster_axis=1)
    p_ag = min(pcc(x, full) for x in _per_device(ag, mesh_device))

    # reduce_scatter: every device holds a partial [S, H]; result = sum, scattered along H
    parts = torch.randn(n, 1, seq, H)
    rs = ttnn.reduce_scatter(_to_mesh(parts, mesh_device, 0), dim=3, cluster_axis=1)
    got = torch.cat(_per_device(rs, mesh_device), dim=3)
    p_rs = pcc(got, parts.sum(0, keepdim=True))

    # all_reduce: partials -> full sum on every device
    ar = ttnn.all_reduce(_to_mesh(parts, mesh_device, 0), cluster_axis=1)
    p_ar = min(pcc(x, parts.sum(0, keepdim=True)) for x in _per_device(ar, mesh_device))

    print(f"all_gather pcc={p_ag:.6f} reduce_scatter pcc={p_rs:.6f} all_reduce pcc={p_ar:.6f}")
    metrics.record(TASK, "pcc_all_gather", p_ag)
    metrics.record(TASK, "pcc_reduce_scatter", p_rs)
    metrics.record(TASK, "pcc_all_reduce", p_ar)
    assert min(p_ag, p_rs, p_ar) > 0.9999
