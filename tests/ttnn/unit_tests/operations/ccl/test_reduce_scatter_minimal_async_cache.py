# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from tests.nightly.t3000.ccl.test_minimal_reduce_scatter_async import run_reduce_scatter_impl


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params, topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1540000}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 1540000}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["ring", "line"],
)
@pytest.mark.parametrize("dim", [0, 3])
@pytest.mark.parametrize("enable_trace", [False, True], ids=["eager", "trace"])
@pytest.mark.parametrize(
    "persistent, contiguous_staging",
    [(False, None), (True, None), (True, False)],
    ids=["allocated", "persistent", "persistent_tiled"],
)
def test_reduce_scatter_minimal_async_cache_arguments(
    mesh_device, topology, dim, enable_trace, persistent, contiguous_staging, monkeypatch
):
    if contiguous_staging is False and (topology != ttnn.Topology.Ring or dim == 0):
        pytest.skip("Tiled staging is already covered by the default persistent case")
    # The driver keeps three distinct inputs, semaphore sets, and (when requested)
    # persistent outputs/staging buffers live. Every invocation checks a different
    # random input against torch, so stale cached addresses cannot pass unnoticed.
    cache_entries = []
    operation = ttnn.experimental.reduce_scatter_minimal_async

    def record_cache(*args, **kwargs):
        result = operation(*args, **kwargs)
        cache_entries.append(mesh_device.num_program_cache_entries())
        return result

    monkeypatch.setattr(ttnn.experimental, "reduce_scatter_minimal_async", record_cache)
    shape = [8, 1, 32, 256] if dim == 0 else [1, 1, 128, 2048]
    run_reduce_scatter_impl(
        mesh_device,
        8,
        shape,
        dim,
        1,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.DRAM_MEMORY_CONFIG,
        rs_topology=topology,
        num_iters=3,
        enable_trace=enable_trace,
        use_barrier=True,
        use_persistent_buffers=persistent,
        cluster_axis=1,
        contiguous_staging=contiguous_staging,
    )
    assert len(cache_entries) >= 3
    assert cache_entries[0] > 0
    assert all(count == cache_entries[0] for count in cache_entries), cache_entries


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY, "trace_region_size": 1540000}],
    indirect=True,
)
@pytest.mark.parametrize("dim", [0, 3])
@pytest.mark.parametrize(
    "persistent, contiguous_staging",
    [(False, True), (True, True), (True, False)],
    ids=["allocated", "persistent_contiguous", "persistent_tiled"],
)
@pytest.mark.parametrize("use_barrier", [False, True], ids=["no_barrier", "barrier"])
def test_reduce_scatter_minimal_async_cache_arguments_galaxy(
    mesh_device, dim, persistent, contiguous_staging, use_barrier
):
    if dim == 0 and not contiguous_staging:
        pytest.skip("dim0 uses tiled staging in every case")
    mesh_device.quiesce_devices()
    mesh_device.clear_program_cache()
    device = mesh_device.create_submesh(ttnn.MeshShape(8, 1))
    try:
        _run_reduce_scatter_galaxy_cache(device, dim, persistent, contiguous_staging, use_barrier)
    finally:
        mesh_device.quiesce_devices()
        device.clear_program_cache()


def _run_reduce_scatter_galaxy_cache(device, dim, persistent, contiguous_staging, use_barrier):
    import torch

    # The certified Galaxy topology is 8x4; use its complete axis-0 ring.
    grid = device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    torch.manual_seed(0)
    shape = [8, 1, 32, 256] if dim == 0 else [1, 1, 128, 2048]
    # Small integer inputs make the replicated 8-way BF16 reduction exact.
    goldens = [torch.randint(-4, 5, shape).to(torch.bfloat16) for _ in range(3)]

    def upload(value):
        return ttnn.from_torch(
            value,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

    inputs = [upload(value) for value in goldens]
    semaphores = [[ttnn.create_global_semaphore(device, cores, 0) for _ in range(3)] for _ in inputs]
    barriers = [ttnn.create_global_semaphore(device, cores, 0) for _ in inputs]
    buffers = []
    for value in inputs:
        if not persistent:
            buffers.append(None)
            continue
        if dim == 0 or not contiguous_staging:
            intermediate, penult = upload(torch.zeros(shape, dtype=torch.bfloat16)), None
        else:
            intermediate, penult = ttnn.experimental.reduce_scatter_minimal_async_create_intermediate_buffer(
                value, dim=dim, topology=ttnn.Topology.Ring, cluster_axis=0
            )
        output_shape = shape.copy()
        output_shape[dim] //= 8
        output = upload(torch.zeros(output_shape, dtype=torch.bfloat16))
        buffers.append([intermediate, output, penult] if penult is not None else [intermediate, output])

    def run(index):
        return ttnn.experimental.reduce_scatter_minimal_async(
            inputs[index],
            dim=dim,
            cluster_axis=0,
            topology=ttnn.Topology.Ring,
            num_links=1,
            multi_device_global_semaphore=semaphores[index],
            barrier_semaphore=barriers[index] if use_barrier else None,
            persistent_output_buffers=buffers[index],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def check(outputs):
        ttnn.synchronize_device(device)
        for output, golden in zip(outputs, goldens):
            actual = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=dim))
            assert torch.equal(actual, golden * 8)

    outputs = []
    entries = []
    for index in range(len(inputs)):
        outputs.append(run(index))
        entries.append(device.num_program_cache_entries())
    check(outputs)
    assert entries[0] > 0 and all(count == entries[0] for count in entries), entries

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    traced_outputs = [run(index) for index in range(len(inputs))]
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    try:
        for _ in range(2):
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            check(traced_outputs)
        assert device.num_program_cache_entries() == entries[0]
    finally:
        ttnn.release_trace(device, trace_id)


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY, "trace_region_size": 1540000}],
    indirect=True,
)
@pytest.mark.parametrize("dim", [0, 3])
@pytest.mark.parametrize("persistent", [False, True], ids=["allocated", "persistent"])
@pytest.mark.parametrize("use_barrier", [False, True], ids=["no_barrier", "barrier"])
def test_reduce_scatter_minimal_async_tp_linear_cache(mesh_device, dim, persistent, use_barrier):
    # Match GLM's TP collective: an axis-1 four-device line on the certified Galaxy mesh.
    mesh_device.quiesce_devices()
    mesh_device.clear_program_cache()
    device = mesh_device.create_submesh(ttnn.MeshShape(1, 4))
    try:
        _run_reduce_scatter_tp_linear_cache(device, dim, persistent, use_barrier)
    finally:
        mesh_device.quiesce_devices()
        device.clear_program_cache()


def _run_reduce_scatter_tp_linear_cache(device, dim, persistent, use_barrier):
    import torch

    grid = device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    torch.manual_seed(0)
    shape = [4, 1, 32, 256] if dim == 0 else [1, 1, 128, 2048]
    # Small integer inputs make the replicated 4-way BF16 reduction exact.
    goldens = [torch.randint(-4, 5, shape).to(torch.bfloat16) for _ in range(3)]

    def upload(value):
        return ttnn.from_torch(
            value,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

    inputs = [upload(value) for value in goldens]
    semaphores = [[ttnn.create_global_semaphore(device, cores, 0) for _ in range(3)] for _ in inputs]
    barriers = [ttnn.create_global_semaphore(device, cores, 0) for _ in inputs]
    buffers = []
    for value in inputs:
        if not persistent:
            buffers.append(None)
            continue
        intermediate, penult = upload(torch.zeros([2] + shape, dtype=torch.bfloat16)), None
        output_shape = shape.copy()
        output_shape[dim] //= 4
        output = upload(torch.zeros(output_shape, dtype=torch.bfloat16))
        buffers.append([intermediate, output, penult] if penult is not None else [intermediate, output])

    def run(index):
        return ttnn.experimental.reduce_scatter_minimal_async(
            inputs[index],
            dim=dim,
            cluster_axis=1,
            topology=ttnn.Topology.Linear,
            num_links=1,
            multi_device_global_semaphore=semaphores[index],
            barrier_semaphore=barriers[index] if use_barrier else None,
            persistent_output_buffers=buffers[index],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def check(outputs):
        ttnn.synchronize_device(device)
        for output, golden in zip(outputs, goldens):
            actual = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=dim))
            assert torch.equal(actual, golden * 4)

    outputs = []
    entries = []
    for index in range(len(inputs)):
        outputs.append(run(index))
        entries.append(device.num_program_cache_entries())
    check(outputs)
    assert entries[0] > 0 and all(count == entries[0] for count in entries), entries

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    traced_outputs = [run(index) for index in range(len(inputs))]
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    try:
        for _ in range(2):
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            check(traced_outputs)
        assert device.num_program_cache_entries() == entries[0]
    finally:
        ttnn.release_trace(device, trace_id)
