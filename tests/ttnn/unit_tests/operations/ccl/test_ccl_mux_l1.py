# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CCL ops keep their fabric mux in program-local L1, so they share cores with persistent L1 (#61572)."""

import pytest
import torch
import ttnn

from tests.nightly.t3000.ccl.test_all_gather import run_all_gather_impl
from tests.nightly.t3000.ccl.test_minimal_reduce_scatter_async import run_reduce_scatter_impl
from tests.nightly.t3000.ccl.test_strided_all_gather_async import run_strided_all_gather_impl

# More than one worker per direction, so each link gets a fabric mux core.
NUM_WORKERS_PER_LINK = 2
# Larger than any mux L1 map below, so a map at the old fixed address, the base of the persistent arena, would land
# inside the ring.
RING_SIZE = 128 * 1024

DEVICE_PARAMS = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 1540000}


def _run_all_gather_async(mesh_device, enable_trace, num_buffers_per_channel=8):
    # FabricMuxConfig, ProgramDescriptor factory.
    run_all_gather_impl(
        mesh_device,
        ag_output_shape=[1, 1, 256, 2048],
        dim=3,
        ag_input_dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mem_config_input=ttnn.DRAM_MEMORY_CONFIG,
        mem_config_ag=ttnn.DRAM_MEMORY_CONFIG,
        all_gather_topology=ttnn.Topology.Linear,
        num_iters=3,
        enable_trace=enable_trace,
        num_workers_per_link=NUM_WORKERS_PER_LINK,
        num_buffers_per_channel=num_buffers_per_channel,
        all_gather_function=ttnn.experimental.all_gather_async,
    )


def _run_reduce_scatter(mesh_device, enable_trace, use_new):
    # FabricMuxConfig, line builder: Program (reduce_scatter_minimal_async) or ProgramDescriptor (ttnn.reduce_scatter).
    run_reduce_scatter_impl(
        mesh_device,
        mesh_device.get_num_devices(),
        [1, 1, 256, 2048],
        3,
        1,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.DRAM_MEMORY_CONFIG,
        rs_topology=ttnn.Topology.Linear,
        enable_trace=enable_trace,
        num_iters=2,
        num_workers_per_link=NUM_WORKERS_PER_LINK,
        num_buffers_per_channel=2,
        use_new=use_new,
    )


def _run_strided_all_gather(mesh_device, enable_trace):
    # FabricMuxV2Config, Program factory. On two devices the op runs Ring as Linear.
    run_strided_all_gather_impl(
        mesh_device,
        mesh_device.get_num_devices(),
        [1, 1, 32, 256],
        3,
        2,
        1,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.DRAM_MEMORY_CONFIG,
        all_gather_topology=ttnn.Topology.Ring,
        mm_cores_y=1,
        mm_block_h=32,
        mm_block_w=32,
        enable_trace=enable_trace,
        num_workers_per_link=NUM_WORKERS_PER_LINK,
    )


CCL_RUNNERS = {
    "all_gather_async": _run_all_gather_async,
    "reduce_scatter_minimal_async": lambda mesh_device, enable_trace: _run_reduce_scatter(
        mesh_device, enable_trace, use_new=False
    ),
    "reduce_scatter": lambda mesh_device, enable_trace: _run_reduce_scatter(mesh_device, enable_trace, use_new=True),
    "strided_all_gather_async": _run_strided_all_gather,
}


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("enable_trace", [False, True], ids=["eager", "trace"])
@pytest.mark.parametrize("num_buffers_per_channel", [2, 8])
def test_all_gather_async_mux(mesh_device, enable_trace, num_buffers_per_channel):
    _run_all_gather_async(mesh_device, enable_trace, num_buffers_per_channel)


def _pattern(num_bytes, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (num_bytes,), dtype=torch.uint8, generator=generator).numpy().tobytes()


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("enable_trace", [False, True], ids=["eager", "trace"])
@pytest.mark.parametrize("ccl_op", list(CCL_RUNNERS))
def test_ccl_mux_keeps_persistent_l1(mesh_device, enable_trace, ccl_op):
    # The CCL ops take their mux and worker cores from the top of the grid, starting at (0, 0).
    grid = mesh_device.compute_with_storage_grid_size()
    space = ttnn.experimental.create_prefetcher_pipe_space(
        mesh_device,
        sender_cores=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(0, 2))}),
        receiver_domain=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, 1))}),
        ring_size=RING_SIZE,
        max_receivers_per_pipe=1,
    )
    regions = [(space.buffer_address(), space.ring_size()), (space.config_address(), space.config_page_size())]

    expected = {}
    for device_id in mesh_device.get_device_ids():
        for core in ttnn.corerange_to_cores(space.reservation_cores()):
            virtual_core = mesh_device.worker_core_from_logical_core(core)
            for address, size in regions:
                key = (device_id, virtual_core.x, virtual_core.y, address)
                expected[key] = _pattern(size, seed=len(expected))
                ttnn.cluster.write_to_core(device_id, virtual_core.x, virtual_core.y, address, expected[key])

    CCL_RUNNERS[ccl_op](mesh_device, enable_trace)

    clobbered = [key for key, data in expected.items() if ttnn.cluster.read_from_core(*key, len(data)) != data]
    assert not clobbered, f"{ccl_op} overwrote persistent L1 (device, x, y, address): {clobbered}"
