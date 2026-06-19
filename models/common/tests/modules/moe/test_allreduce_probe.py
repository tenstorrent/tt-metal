# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Scratch diagnostic: bare ttnn.all_reduce on a (1,8) mesh — no model, no checkpoint, no weights.

Isolates the exact collective that the MoE1D / Gemma4 forwards wedge on, to tell a flaky-ring box
apart from module code. Healthy ring: completes in ms. Bad ring: hangs (timeout-kill). Delete after.
"""

import pytest
import torch
from loguru import logger

import ttnn


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 8)], ids=["1x8"], indirect=True)
@pytest.mark.parametrize("topology", [ttnn.Topology.Ring, ttnn.Topology.Linear], ids=["ring", "linear"])
@pytest.mark.parametrize(
    "shape", [(1, 1, 32, 4096), (1, 1, 32, 256), (1, 1, 1, 256)], ids=["w4096", "w256", "decode256"]
)
def test_allreduce_probe(ttnn_mesh_device: ttnn.MeshDevice, topology, shape):
    md = ttnn_mesh_device
    logger.info(f"all_reduce probe: {md.get_num_devices()} devices, topology={topology}, shape={shape}")
    t = ttnn.from_torch(
        torch.randn(*shape),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    logger.info("issuing all_reduce ...")
    out = ttnn.all_reduce(
        t,
        cluster_axis=1,
        num_links=1,
        topology=topology,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.synchronize_device(md)
    logger.info(f"all_reduce DONE shape={tuple(out.shape)}")
    assert out is not None
