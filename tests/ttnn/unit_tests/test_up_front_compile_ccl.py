# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Up-front precompile on a CCL op (all_gather_async) across a multi-device mesh.

A CCL op builds a MeshWorkload with multiple programs per device, unlike the
homogeneous single-program mesh case. This drives all_gather_async through
collect -> parallel compile -> warm run to confirm up-front precompile captures
and warms the CCL program set, and that the warm run is correct.

Needs fabric; parametrized for a 1x2 mesh (skips on fewer devices). Internal
hardware test (mesh CCL goes over fabric / eth) — run under the device lock:
    scripts/run_safe_pytest.sh tests/ttnn/unit_tests/test_up_front_compile_ccl.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

OUTPUT_SHAPE = [1, 1, 32, 256]
GATHER_DIM = 3
NUM_LINKS = 1


def _shard_input(mesh_device, num_devices, full):
    """Shard `full` along GATHER_DIM across the mesh — the all_gather input."""
    return ttnn.from_torch(
        full,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh_device,
            ttnn.MeshMapperConfig(
                [ttnn.PlacementReplicate(), ttnn.PlacementShard(GATHER_DIM)], ttnn.MeshShape(1, num_devices)
            ),
        ),
    )


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_all_gather_up_front_compile(mesh_device):
    num_devices = mesh_device.get_num_devices()

    # CCL needs a worker sub-device + global semaphores.
    grid = mesh_device.compute_with_storage_grid_size()
    ccl_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    worker_sub_device = ttnn.SubDevice([ccl_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])

    # Two semaphores per all_gather_async call: one pair for the collect-build, one for the warm run.
    sems = [ttnn.create_global_semaphore(mesh_device, ccl_crs, 0) for _ in range(4)]

    torch.manual_seed(0)
    full = torch.rand(OUTPUT_SHAPE).bfloat16()

    def all_gather(inp, s0, s1):
        return ttnn.experimental.all_gather_async(
            inp,
            GATHER_DIM,
            multi_device_global_semaphore=[s0, s1],
            num_links=NUM_LINKS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
            subdevice_id=worker_sub_device_id,
        )

    # Collect (NO_DISPATCH): capture the all_gather MeshWorkload (multi-program per device).
    ttnn.graph.up_front_clear()
    ttnn.graph.up_front_begin_collect()
    try:
        all_gather(_shard_input(mesh_device, num_devices, full), sems[0], sems[1])
    finally:
        ttnn.graph.up_front_end_collect()
    n_unique = ttnn.graph.up_front_num_unique()
    print(f"\nCCL collect: {ttnn.graph.up_front_num_collected()} ops -> {n_unique} unique programs")
    assert ttnn.graph.up_front_num_collected() >= 1, "collect captured no CCL programs"

    # Parallel compile warms the cache, error-free. A CCL op's MeshWorkload holds multiple
    # programs (sender/receiver), so compile builds >= the unique workload count — unlike a
    # homogeneous op where one program covers the whole mesh.
    num_programs, num_errors, _, wall = ttnn.graph.up_front_compile(mesh_device, 4)
    print(
        f"CCL parallel compile: {num_programs} programs in {wall:.2f}s (errors={num_errors}, unique workloads={n_unique})"
    )
    assert num_errors == 0, "parallel compile reported errors"
    assert num_programs >= n_unique >= 1, f"expected >= {n_unique} programs compiled, got {num_programs}"

    # Warm run + correctness: all_gather reconstructs the full tensor on every device.
    out = all_gather(_shard_input(mesh_device, num_devices, full), sems[2], sems[3])
    ttnn.synchronize_device(mesh_device, sub_device_ids=[worker_sub_device_id])

    for t in ttnn.get_device_tensors(out):
        got = t.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        assert_with_pcc(full, got, 0.99)

    mesh_device.reset_sub_device_stall_group()
