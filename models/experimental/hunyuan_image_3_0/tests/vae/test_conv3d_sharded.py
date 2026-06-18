# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Spatially-sharded conv3d == replicated conv3d (same weights), on a 2x2 mesh.

Validates VAE spatial parallelism Phase 1: the conv neighbor-pads its H/W shard
boundary across the mesh (halo = padding) and runs with internal H/W padding off.
H is sharded on axis 0, W on axis 1.

Run:
  python_env/bin/python -m pytest \
    models/experimental/hunyuan_image_3_0/tests/vae/test_conv3d_sharded.py -v -s --timeout=600
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.tt_dit.parallel.manager import CCLManager
from models.experimental.hunyuan_image_3_0.tt.vae.conv3d import HunyuanSymmetricConv3d


@pytest.fixture(scope="function")
def device_params(request):
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_conv3d_sharded_vs_replicated(mesh_device):
    mesh_device.enable_program_cache()
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    B, T, H, W, Cin, Cout = 1, 4, 64, 64, 32, 32  # 2x2 -> local 32x32

    w = torch.randn(Cout, Cin, 3, 3, 3) * 0.05
    b = torch.randn(Cout) * 0.05
    state = {"weight": w, "bias": b}

    ref = HunyuanSymmetricConv3d(Cin, Cout, kernel_size=3, stride=1, padding=1, mesh_device=mesh_device, t=T, h=H, w=W)
    ref.load_torch_state_dict({k: v.clone() for k, v in state.items()})
    shd = HunyuanSymmetricConv3d(
        Cin,
        Cout,
        kernel_size=3,
        stride=1,
        padding=1,
        mesh_device=mesh_device,
        t=T,
        h=H,
        w=W,
        ccl_manager=ccl,
        h_mesh_axis=0,
        w_mesh_axis=1,
    )
    shd.load_torch_state_dict({k: v.clone() for k, v in state.items()})

    torch.manual_seed(0)
    x = torch.randn(B, T, H, W, Cin) * 0.1

    # Reference: replicated input -> replicated conv -> device-0 slice.
    x_rep = ttnn.from_torch(
        x,
        dtype=ref.dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    y_ref = ttnn.to_torch(ref(x_rep), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:B].float()

    # Sharded: H on axis 0 (tensor dim 2), W on axis 1 (tensor dim 3).
    x_shd = ttnn.from_torch(
        x,
        dtype=shd.dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[2, 3]),
    )
    out_shd = shd(x_shd)
    y_shd = ttnn.to_torch(
        out_shd, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[2, 3])
    ).float()

    assert tuple(y_shd.shape) == tuple(y_ref.shape), f"{y_shd.shape} vs {y_ref.shape}"
    passing, pcc = comp_pcc(y_ref, y_shd, 0.999)
    logger.info(f"conv3d sharded(2x2 HxW) vs replicated PCC: {pcc:.6f}  shape={tuple(y_shd.shape)}")
    assert passing, f"PCC {pcc:.6f} < 0.999 — neighbor-pad halo / sharded conv is wrong"
