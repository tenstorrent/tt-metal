# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MeshConfig.allgather on the TP axis: bit-exact on both sides of the large-gather worker threshold.

The norm all-gathers (layer._gather_emb) go through MeshConfig.allgather, which raises
num_workers_per_link to 8 on Blackhole once a gather moves >= 4 MB per link. ``rows`` is per chip, so
rows 512 stays on the op default and rows 2048 / 4096 (W=4096 / 8192 at SP=2) take the 8-worker path.
"""

import pytest
import torch

import ttnn
from models.demos.minimax_m3.config import MeshConfig, _allgather_workers_per_link
from models.demos.minimax_m3.tt.ccl import CCLManager
from models.demos.minimax_m3.utils.general_utils import get_default_num_links

from ..test_factory import parametrize_mesh_with_fabric

HIDDEN = 6144


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
@pytest.mark.parametrize("rows", [512, 2048, 4096], ids=["rows512", "rows2048", "rows4096"])
def test_mesh_config_allgather_tp(mesh_device, device_params, rows, reset_seeds):
    sp, tp = tuple(mesh_device.shape)
    mesh_config = MeshConfig((sp, tp), tp=tp)
    ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)

    x = torch.randn(1, 1, rows * sp, HIDDEN).bfloat16()
    tt_x = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=(2, 3)),
    )
    workers = _allgather_workers_per_link(tt_x, tp, ccl.num_links, ccl.topology)
    if ttnn.get_arch_name() == "blackhole":
        assert workers == (None if rows == 512 else 8), f"rows {rows}: unexpected worker choice {workers}"

    out = mesh_config.allgather(tt_x, ccl, axis=mesh_config.tp_axis, dim=3)
    for idx, shard in enumerate(ttnn.get_device_tensors(out)):
        r = idx // tp
        expected = x[:, :, r * rows : (r + 1) * rows, :]
        assert torch.equal(ttnn.to_torch(shard), expected), f"chip {idx} (row {r}) all-gather mismatch"
