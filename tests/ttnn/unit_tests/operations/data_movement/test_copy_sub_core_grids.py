# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "worker_l1_size": 1345000, "trace_region_size": 1000000}],
    indirect=True,
)
def test_copy_on_disjoint_sub_device_cores(mesh_device, device_params, expect_error):
    """Exercise all copy factories, uneven work splits and cached trace replay."""
    grids = [
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 1)),
                ttnn.CoreRange(ttnn.CoreCoord(3, 0), ttnn.CoreCoord(3, 2)),
            ]
        ),
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(3, 0), ttnn.CoreCoord(3, 1))]),
    ]
    manager = mesh_device.create_sub_device_manager([ttnn.SubDevice([grids[0]])], 0)
    mesh_device.load_sub_device_manager(manager)
    traces = []
    resident = None
    try:
        resident = ttnn.allocate_tensor_on_device(
            (1, 1, 5 * 384, 1024),
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
            mesh_device,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(grids[0], (384, 1024), ttnn.ShardOrientation.ROW_MAJOR),
            ),
        )
        cases = []
        for layout, memory_config, shape in (
            (ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG, (1, 1, 19, 128)),
            (ttnn.ROW_MAJOR_LAYOUT, ttnn.L1_MEMORY_CONFIG, (1, 1, 19, 128)),
            (ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG, (1, 1, 128, 224)),
            (ttnn.TILE_LAYOUT, ttnn.L1_MEMORY_CONFIG, (1, 1, 128, 224)),
            (ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG, (1, 1, 19, 131072)),
        ):
            host = torch.arange(torch.tensor(shape).prod().item()).reshape(shape).remainder(127).to(torch.bfloat16)
            source = ttnn.from_torch(
                host, device=mesh_device, layout=layout, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
            )
            destination = ttnn.empty_like(source, memory_config=memory_config)
            before = mesh_device.num_program_cache_entries()
            for grid in grids:
                ttnn.copy(source, destination, sub_core_grids=grid)
                torch.testing.assert_close(ttnn.to_torch(ttnn.get_device_tensors(destination)[0]), host, rtol=0, atol=0)
            assert mesh_device.num_program_cache_entries() == before + len(grids)
            cases.append((source, destination, host))
        entries = mesh_device.num_program_cache_entries()
        for source, destination, host in cases:
            for grid in grids:
                trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                ttnn.copy(source, destination, sub_core_grids=grid)
                ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
                traces.append(trace_id)
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
                torch.testing.assert_close(ttnn.to_torch(ttnn.get_device_tensors(destination)[0]), host, rtol=0, atol=0)
        assert mesh_device.num_program_cache_entries() == entries
        with expect_error(RuntimeError, "must not be empty"):
            ttnn.copy(cases[0][0], cases[0][1], sub_core_grids=ttnn.CoreRangeSet([]))
    finally:
        for trace_id in traces:
            ttnn.release_trace(mesh_device, trace_id)
        if resident is not None:
            ttnn.deallocate(resident)
        mesh_device.clear_loaded_sub_device_manager()
        mesh_device.remove_sub_device_manager(manager)
