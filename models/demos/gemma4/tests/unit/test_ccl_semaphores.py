# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Collective semaphores must not land in worker L1 after the model is built.

The generic collectives allocate their global semaphores once per cached program, on the first call
of a new shape. A shape first seen after the decode trace was captured (the first served prefill)
would put them in worker L1, top-down -- the addresses the trace's L1 intermediates used -- and the
next replay overwrites them (wrong first token or a hang on the prefill after the first traced
decode). With an L1_SMALL region every first-time collective grows L1_SMALL and leaves L1 alone."""
import torch

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.ccl import (
    CCLManager,
    ccl_allreduce,
    ccl_lane_gather_rows,
    ccl_lane_scatter_rows,
    ccl_semaphores_in_l1_small,
)

from ...tests.test_factory import parametrize_mesh_with_fabric

L1_SMALL_SIZE = 65536


def _l1_allocated(mesh_device, buffer_type):
    ttnn.synchronize_device(mesh_device)
    return int(ttnn.get_memory_view(mesh_device, buffer_type).total_bytes_allocated_per_bank)


def _lane_tensor(mesh_device, mesh_config, rows, width):
    mapper = mesh_config.lane_shard_mapper(mesh_device, tensor_dim=2)
    hidden = torch.randn(1, 1, mesh_device.shape[0] * rows, width, dtype=torch.bfloat16)
    return ttnn.from_torch(hidden, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=mapper)


@parametrize_mesh_with_fabric(mesh_shapes=[(2, 4)], device_params_extra={"l1_small_size": L1_SMALL_SIZE})
def test_first_time_collectives_keep_worker_l1(mesh_device, reset_seeds):
    tp, ep = mesh_device.shape[1], mesh_device.shape[0]
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp, ep=ep), prefill=ModeConfig(tp=tp, ep=ep))
    mesh_config.lane_sharded = True
    ccl_manager = CCLManager(mesh_device, num_links=1)
    assert ccl_semaphores_in_l1_small(mesh_device)

    # Model built (manager semaphores allocated); from here on nothing may grow worker L1 persistently.
    l1_before = _l1_allocated(mesh_device, ttnn.BufferType.L1)
    small_before = _l1_allocated(mesh_device, ttnn.BufferType.L1_SMALL)

    for rows, width in ((32, 1024), (64, 1024), (128, 2048)):  # three never-seen shapes = fresh programs
        x = _lane_tensor(mesh_device, mesh_config, rows, width)
        gathered = ccl_lane_gather_rows(x, mesh_config)
        scattered = ccl_lane_scatter_rows(gathered, mesh_config, ccl_manager=ccl_manager)
        reduced = ccl_allreduce(scattered, mesh_config, ccl_manager)
        reduced.deallocate(True)

    l1_after = _l1_allocated(mesh_device, ttnn.BufferType.L1)
    small_after = _l1_allocated(mesh_device, ttnn.BufferType.L1_SMALL)
    assert l1_after == l1_before, f"first-time collectives left {l1_after - l1_before} B per bank in worker L1"
    assert small_after > small_before, "collective semaphores were expected in L1_SMALL"
