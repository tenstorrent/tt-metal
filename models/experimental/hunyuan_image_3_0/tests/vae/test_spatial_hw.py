# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""VAE spatial-parallel helpers (Phase V2 mechanism) on a 2x2 mesh.

gather_hw / partition_hw are the only new ops behind norm_sharded (and the attention
wrapper in V3): GroupNorm/attention themselves are unchanged and already validated,
so validating that gather_hw reconstructs the full tensor and partition_hw is its exact
inverse proves gather->op->partition is correct by composition.

Run:
  python_env/bin/python -m pytest \
    models/experimental/hunyuan_image_3_0/tests/vae/test_spatial_hw.py -v -s --timeout=300
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.tt_dit.parallel.manager import CCLManager
from models.experimental.hunyuan_image_3_0.tt.vae.spatial import gather_hw, partition_hw


@pytest.fixture(scope="function")
def device_params(request):
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_gather_partition_hw(mesh_device):
    mesh_device.enable_program_cache()
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    B, T, H, W, C = 1, 2, 64, 64, 64  # 2x2 -> local 32x32
    HA, WA = 0, 1  # H on axis 0, W on axis 1

    torch.manual_seed(0)
    x = torch.randn(B, T, H, W, C)

    # Upload sharded: H -> tensor dim 2 / axis 0, W -> tensor dim 3 / axis 1.
    x_shd = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[2, 3]),
    )

    # (a) gather_hw must reconstruct the full spatial tensor (== original on every device).
    full = gather_hw(ccl, x_shd, h_mesh_axis=HA, w_mesh_axis=WA)
    g = ttnn.to_torch(full, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:B].float()
    ok_a, pcc_a = comp_pcc(x, g, 0.999)
    logger.info(f"gather_hw full-spatial PCC: {pcc_a:.6f}  shape={tuple(g.shape)}")
    assert ok_a, f"gather_hw wrong: PCC {pcc_a:.6f}"

    # (b) partition_hw(gather_hw(x)) must equal the original sharding (round-trip identity).
    rt = partition_hw(full, h_mesh_axis=HA, w_mesh_axis=WA)
    r = ttnn.to_torch(
        rt, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[2, 3])
    ).float()
    ok_b, pcc_b = comp_pcc(x, r, 0.999)
    logger.info(f"partition_hw(gather_hw) round-trip PCC: {pcc_b:.6f}  shape={tuple(r.shape)}")
    assert ok_b, f"partition_hw not inverse of gather_hw: PCC {pcc_b:.6f}"
