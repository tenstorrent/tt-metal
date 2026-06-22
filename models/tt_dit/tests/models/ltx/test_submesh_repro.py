# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Minimal repro for the overlapping-submesh fabric deadlock behind LTX warm 1x4.

A parent-mesh fabric CCL hangs when an overlapping child submesh has run its own CCL
first on the shared physical chips: the child leaves the shared EDM router channel
mid-connection and the parent's next CCL can't reconcile it. Mirrors warmup_buffers,
where the 1x4 audio submesh all_gather poisons the following 4x8 Gemma-encoder
reduce_scatter (the shape/axis here match that op). No model, two CCL ops — minutes
per iteration. Baseline hangs in the final synchronize_device; the fabric teardown fix
must make it return.
"""
import pytest
import torch

import ttnn
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.test import ring_params


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [{**ring_params, "num_command_queues": 2}], indirect=True)
@pytest.mark.timeout(240)
def test_parent_ccl_after_child_submesh_ccl(mesh_device):
    parent = mesh_device.create_submesh(ttnn.MeshShape(4, 8))
    parent_ccl = CCLManager(parent, num_links=2, topology=ttnn.Topology.Ring)

    child = parent.create_submesh(ttnn.MeshShape(1, 4))
    child_ccl = CCLManager(child, num_links=2, topology=ttnn.Topology.Linear)

    ct = ttnn.from_torch(
        torch.randn(1, 1, 32, 128 * 4),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=child,
        mesh_mapper=ttnn.ShardTensor2dMesh(child, dims=(None, 3), mesh_shape=(1, 4)),
    )
    child_ccl.all_gather(ct, dim=3, mesh_axis=1, use_hyperparams=False)
    ttnn.synchronize_device(child)

    pt = ttnn.from_torch(
        torch.randn(1, 1, 1024, 128 * 8),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=parent,
        mesh_mapper=ttnn.ShardTensor2dMesh(parent, dims=(None, 3), mesh_shape=(4, 8)),
    )
    parent_ccl.reduce_scatter(pt, dim=3, mesh_axis=1)
    ttnn.synchronize_device(parent)
