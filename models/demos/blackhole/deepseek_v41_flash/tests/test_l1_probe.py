# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Probe: which operations leave persistent L1 allocations that fragment the L1 allocator (and clash with the
static dataflow buffers of the 5120-wide rms_norm)?  Writes /mnt/tt-data/ssinghal/dsv4-logs/l1_probe.txt."""

import pytest
import torch

import ttnn
from models.demos.gpt_oss.tt.ccl import CCLManager

OUT = "/mnt/tt-data/ssinghal/dsv4-logs/l1_probe.txt"


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, id="ring")],
    indirect=True,
)
def test_l1_probe(mesh_device):
    log = open(OUT, "w")

    def view(tag):
        ttnn.synchronize_device(mesh_device)
        mv = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1)
        log.write(
            f"{tag:46s} allocated/bank {mv.total_bytes_allocated_per_bank:7d}  largest_free {mv.largest_contiguous_bytes_free_per_bank:8d}\n"
        )
        log.flush()

    w = ttnn.from_torch(
        torch.ones(1, 1, 160, 32),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    x = ttnn.from_torch(
        torch.randn(1, 1, 4, 5120),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def norm(tag):
        try:
            ttnn.synchronize_device(mesh_device)
            y = ttnn.rms_norm(x, weight=w, epsilon=1e-20)
            ttnn.synchronize_device(mesh_device)
            log.write(f"   rms_norm OK after: {tag}\n")
            ttnn.deallocate(y)
        except Exception as e:
            log.write(f"   rms_norm FAILED after: {tag}: {str(e)[:120]}\n")
        log.flush()

    view("start")
    norm("start")
    ccl = CCLManager(mesh_device, num_links=2, topology=ttnn.Topology.Ring)
    view("after CCLManager init")
    norm("CCLManager init")
    sems = [ttnn.create_global_semaphore(mesh_device, ccl.ccl_cores, 0) for _ in range(5)]
    view("after 5 extra global semaphores")
    norm("5 extra global semaphores")

    t = ttnn.from_torch(
        torch.randn(1, 1, 4, 5120),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    for i in range(2):
        try:
            r = ttnn.reduce_scatter(t, dim=3, cluster_axis=1, num_links=2, topology=ttnn.Topology.Linear)
            view(f"after generic ttnn.reduce_scatter #{i}")
            norm(f"generic ttnn.reduce_scatter #{i}")
            ttnn.deallocate(r)
        except Exception as e:
            log.write(f"reduce_scatter #{i} error: {str(e)[:200]}\n")
    view("end")
    log.close()
