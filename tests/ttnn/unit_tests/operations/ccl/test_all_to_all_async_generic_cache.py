# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


@pytest.mark.parametrize(
    "mesh_device, device_params, topology",
    [
        ((1, 8), {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1540000}, ttnn.Topology.Ring),
        ((1, 8), {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 1540000}, ttnn.Topology.Linear),
        (
            (8, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY, "trace_region_size": 1540000},
            ttnn.Topology.Ring,
        ),
        (
            (8, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY, "trace_region_size": 1540000},
            ttnn.Topology.Linear,
        ),
    ],
    indirect=["mesh_device", "device_params"],
    ids=["ring", "line", "galaxy_ring", "galaxy_tp_linear"],
)
@pytest.mark.parametrize(
    "in_dim, out_dim",
    [(2, 3), (3, 2), (1, 2), (2, 1)],
    ids=["height_to_width", "width_to_height", "heads_to_sequence", "sequence_to_heads"],
)
@pytest.mark.parametrize("num_links", [1, 2])
@pytest.mark.parametrize("persistent", [False, True], ids=["allocated", "persistent"])
def test_all_to_all_async_generic_cache_arguments(mesh_device, topology, in_dim, out_dim, num_links, persistent):
    parent_mesh_device = mesh_device
    parent_mesh_device.quiesce_devices()
    parent_mesh_device.clear_program_cache()
    if tuple(mesh_device.shape) == (8, 4):
        mesh_device = mesh_device.create_submesh(
            ttnn.MeshShape(1, 4) if topology == ttnn.Topology.Linear else ttnn.MeshShape(8, 1)
        )
    cluster_axis = 0 if tuple(mesh_device.shape)[1] == 1 else 1
    try:
        _run_all_to_all_cache_arguments(mesh_device, topology, in_dim, out_dim, num_links, persistent, cluster_axis)
    finally:
        # Cached workloads own global semaphores on the submesh. Release them before
        # the next parameterized case opens another child of the same parent mesh.
        parent_mesh_device.quiesce_devices()
        mesh_device.clear_program_cache()


def _run_all_to_all_cache_arguments(mesh_device, topology, in_dim, out_dim, num_links, persistent, cluster_axis):
    torch.manual_seed(0)
    # Keep all inputs and outputs alive to force different addresses on cache hits.
    shape = [1, 8 if 1 in (in_dim, out_dim) else 1, 256, 512]
    goldens = [torch.rand(shape, dtype=torch.bfloat16) for _ in range(3)]
    inputs = [
        ttnn.from_torch(
            value,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=in_dim),
        )
        for value in goldens
    ]
    assert len({tensor.buffer_address() for tensor in inputs}) == len(inputs)
    output_shape = list(goldens[0].shape)
    output_shape[out_dim] //= mesh_device.get_num_devices()
    persistent_outputs = (
        [
            ttnn.from_torch(
                torch.zeros(output_shape, dtype=torch.bfloat16),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            for _ in goldens
        ]
        if persistent
        else [None] * len(goldens)
    )

    def run(index):
        return ttnn.experimental.all_to_all_async_generic(
            inputs[index],
            in_dim=in_dim,
            out_dim=out_dim,
            num_links=num_links,
            topology=topology,
            cluster_axis=cluster_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            persistent_output_buffer=persistent_outputs[index],
        )

    def check(outputs):
        ttnn.synchronize_device(mesh_device)
        for output, golden in zip(outputs, goldens):
            actual = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=out_dim))
            assert torch.equal(actual, golden)

    outputs = []
    cache_entries = []
    for index in range(len(inputs)):
        outputs.append(run(index))
        cache_entries.append(mesh_device.num_program_cache_entries())
    check(outputs)
    assert cache_entries[0] > 0
    assert all(count == cache_entries[0] for count in cache_entries), cache_entries

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_outputs = [run(index) for index in range(len(inputs))]
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        # Replay twice: cached addresses and internal barriers must remain valid.
        for _ in range(2):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            check(traced_outputs)
        assert mesh_device.num_program_cache_entries() == cache_entries[0]
    finally:
        ttnn.release_trace(mesh_device, trace_id)
