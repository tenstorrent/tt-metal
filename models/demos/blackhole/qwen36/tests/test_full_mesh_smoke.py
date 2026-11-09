# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device smoke test for QWEN36_TEST_FULL_MESH=1: the fixture opens the full Galaxy with the ring fabric and hands
back a (1, 8) submesh; one fabric collective on it must complete and match torch.

Run (dev Galaxy, no server running):
    QWEN36_TEST_FULL_MESH=1 MESH_DEVICE=P150x8 pytest models/demos/blackhole/qwen36/tests/test_full_mesh_smoke.py -s
"""
import os

import pytest
import torch
from loguru import logger

import ttnn


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1 << 20}], indirect=True
)
def test_full_mesh_submesh_all_gather(mesh_device):
    assert tuple(mesh_device.shape) == (1, 8), tuple(mesh_device.shape)
    logger.info(f"submesh shape {tuple(mesh_device.shape)}, full={os.environ.get('QWEN36_TEST_FULL_MESH')}")
    n = mesh_device.get_num_devices()
    torch.manual_seed(0)
    x = torch.randn(1, 1, 32, 32 * n, dtype=torch.bfloat16)
    tt = ttnn.from_torch(
        x,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )
    out = ttnn.all_gather(tt, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # same call the model makes in-trace
    got = ttnn.to_torch(ttnn.get_device_tensors(out)[0])
    assert torch.equal(got, x), "all_gather on the submesh did not reproduce the input"
    logger.info("full-mesh submesh all_gather OK")
