# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""all_gather_async's fabric mux lives in program-local L1, so it shares cores with persistent L1 (#61572)."""

import pytest
import torch
import ttnn

from tests.nightly.t3000.ccl.test_all_gather import run_all_gather_impl

# More than one worker per direction, so each link gets a fabric mux core.
NUM_WORKERS_PER_LINK = 2
# Larger than the mux's L1 map at the largest num_buffers_per_channel below, so the old fixed-address map at the
# base of the persistent arena would land inside the ring.
RING_SIZE = 128 * 1024


def _run_mux_all_gather(mesh_device, enable_trace, num_buffers_per_channel):
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


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}],
    indirect=True,
)
@pytest.mark.parametrize("enable_trace", [False, True], ids=["eager", "trace"])
@pytest.mark.parametrize("num_buffers_per_channel", [2, 8])
def test_all_gather_async_mux(mesh_device, enable_trace, num_buffers_per_channel):
    _run_mux_all_gather(mesh_device, enable_trace, num_buffers_per_channel)


def _pattern(num_bytes, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (num_bytes,), dtype=torch.uint8, generator=generator).numpy().tobytes()


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}],
    indirect=True,
)
@pytest.mark.parametrize("enable_trace", [False, True], ids=["eager", "trace"])
def test_all_gather_async_mux_keeps_persistent_l1(mesh_device, enable_trace):
    # The CCL takes its mux and worker cores from the top of the grid, starting at (0, 0).
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

    _run_mux_all_gather(mesh_device, enable_trace, num_buffers_per_channel=8)

    clobbered = [key for key, data in expected.items() if ttnn.cluster.read_from_core(*key, len(data)) != data]
    assert not clobbered, f"all_gather_async overwrote persistent L1 (device, x, y, address): {clobbered}"
