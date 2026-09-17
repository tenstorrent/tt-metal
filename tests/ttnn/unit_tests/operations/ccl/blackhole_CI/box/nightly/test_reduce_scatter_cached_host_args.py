# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize("mesh_device", [(8, 1)], indirect=True)
@pytest.mark.parametrize(
    "device_params,topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 2097152}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 2097152}, ttnn.Topology.Linear),
    ],
    ids=["ring", "linear"],
    indirect=["device_params"],
)
@pytest.mark.parametrize("dim,persistent", [(0, False), (0, True), (3, False), (3, True)])
def test_reduce_scatter_cached_host_args(mesh_device, topology, dim, persistent):
    """Fresh buffers/semaphores, auto/explicit links, and retained trace across eager rebinding."""
    torch.manual_seed(56258)
    mesh_device.enable_program_cache()
    grid = mesh_device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    shape = [8, 1, 32, 256] if dim == 0 else [1, 1, 32, 256]
    mapper = ttnn.ReplicateTensorToMesh(mesh_device)

    def upload(value):
        return ttnn.from_torch(
            value,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    # Keep all allocations alive: updates must refresh genuinely different addresses.
    hosts = [torch.randn(shape, dtype=torch.bfloat16) for _ in range(6)]
    inputs = [upload(value) for value in hosts]
    semaphores = [[ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(3)] for _ in hosts]
    barriers = [ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in hosts]
    buffers = []
    output_shape = shape.copy()
    output_shape[dim] //= 8
    for tensor in inputs:
        if not persistent:
            buffers.append(None)
        elif topology == ttnn.Topology.Ring and dim != 0:
            # This helper rejects a fallback to Linear, so this case proves actual Ring dispatch.
            interm, penult = ttnn.experimental.reduce_scatter_minimal_async_create_intermediate_buffer(
                tensor,
                dim=dim,
                topology=topology,
            )
            buffers.append([interm, upload(torch.zeros(output_shape)), penult])
        else:
            interm_shape = ([2] + shape) if topology == ttnn.Topology.Linear else shape
            buffers.append([upload(torch.zeros(interm_shape)), upload(torch.zeros(output_shape))])

    def run(i):
        return ttnn.experimental.reduce_scatter_minimal_async(
            inputs[i],
            persistent_output_buffers=buffers[i],
            dim=dim,
            multi_device_global_semaphore=semaphores[i],
            barrier_semaphore=barriers[i],
            num_links=[None, 1, 2][i % 3],
            topology=topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def check(output, i):
        expected = torch.chunk(hosts[i].float() * 8, 8, dim=dim)
        for tensor, golden in zip(ttnn.get_device_tensors(output), expected):
            actual = ttnn.to_torch(tensor).float()
            torch.testing.assert_close(actual, golden, rtol=0.03, atol=0.125)

    entries_before = mesh_device.num_program_cache_entries()
    outputs = [run(i) for i in range(3)]
    warmed = mesh_device.num_program_cache_entries()
    # The resolved count is keyed: auto may reuse one of the explicit-link programs.
    assert entries_before + 2 <= warmed <= entries_before + 3
    for i, output in enumerate(outputs):
        check(output, i)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced = run(0)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        check(traced, 0)
        for i in range(3, 6):
            outputs.append(run(i))
            check(outputs[-1], i)
            assert mesh_device.num_program_cache_entries() == warmed
        poison = ttnn.from_torch(torch.zeros(output_shape), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        ttnn.copy_host_to_device_tensor(poison, traced)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        check(traced, 0)
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize("mesh_device", [(8, 1)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 2097152}],
    indirect=True,
)
def test_public_reduce_scatter_cached_host_args(mesh_device):
    from tests.nightly.t3000.ccl.test_minimal_reduce_scatter_async import run_reduce_scatter_impl

    run_reduce_scatter_impl(
        mesh_device,
        8,
        [1, 1, 32, 256],
        3,
        1,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.Topology.Ring,
        num_iters=3,
        enable_trace=True,
        use_new=True,
        use_persistent_buffers=False,
    )
