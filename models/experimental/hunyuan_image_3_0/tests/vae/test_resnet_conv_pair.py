# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""ResnetBlock conv pair PCC on device."""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.hunyuan_image_3_0.ref.vae.decoder import LATENT_H, LATENT_T, LATENT_W, MID_CHANNELS, load_mid
from models.experimental.hunyuan_image_3_0.tt.vae.decoder import ResnetBlockTTNN
from models.experimental.hunyuan_image_3_0.tt.vae.decoder_weights import load_resnet_block
from models.experimental.hunyuan_image_3_0.tt.vae.spatial import enable_vae_spatial, gather_hw
from models.tt_dit.parallel.manager import CCLManager


@pytest.fixture(scope="function")
def device_params(request):
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_resnet_conv_pair_replicated_vs_pytorch(mesh_device):
    mesh_device.enable_program_cache()
    ref = load_mid().block_1
    x = torch.randn(1, MID_CHANNELS, LATENT_T, LATENT_H, LATENT_W) * 0.1
    with torch.no_grad():
        pt_out = ref(x)

    tt = ResnetBlockTTNN(MID_CHANNELS, MID_CHANNELS, mesh_device, t=LATENT_T, h=LATENT_H, w=LATENT_W)
    load_resnet_block(tt, ref)
    x_tt = ttnn.from_torch(
        x.permute(0, 2, 3, 4, 1).contiguous(),
        dtype=tt.dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    y = tt(x_tt)
    tt_out = (
        ttnn.to_torch(y, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1].permute(0, 4, 1, 2, 3).float()
    )
    passing, pcc = comp_pcc(pt_out, tt_out, 0.999)
    logger.info(f"ResnetConvPair replicated PCC: {pcc:.6f}")
    assert passing


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_resnet_conv_pair_sharded_vs_pytorch(mesh_device):
    mesh_device.enable_program_cache()
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    ref = load_mid().block_1
    x = torch.randn(1, MID_CHANNELS, LATENT_T, LATENT_H, LATENT_W) * 0.1
    with torch.no_grad():
        pt_out = ref(x)

    tt = ResnetBlockTTNN(MID_CHANNELS, MID_CHANNELS, mesh_device, t=LATENT_T, h=LATENT_H, w=LATENT_W)
    load_resnet_block(tt, ref)
    enable_vae_spatial(tt, ccl, h_mesh_axis=0, w_mesh_axis=1)

    x_tt = ttnn.from_torch(
        x.permute(0, 2, 3, 4, 1).contiguous(),
        dtype=tt.dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[2, 3]),
    )
    y = tt(x_tt)
    y_full = gather_hw(ccl, y, h_mesh_axis=0, w_mesh_axis=1)
    tt_out = (
        ttnn.to_torch(y_full, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1]
        .permute(0, 4, 1, 2, 3)
        .float()
    )
    passing, pcc = comp_pcc(pt_out, tt_out, 0.999)
    logger.info(f"ResnetConvPair sharded PCC: {pcc:.6f}")
    assert passing
