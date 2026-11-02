# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Program-cache regression test for reduce_scatter_minimal_async (Ring and Line) on a 4 chip Blackhole box.

Only the first call builds the program. Every later call is a cache hit, and each one passes new tensors
(input and, where used, persistent intermediate and output buffers) and a new set of GlobalSemaphores. On a
hit, the op has to re-point every tensor address and re-apply the semaphores, which are not part of the
program hash. After each call its semaphores are set to a large value. A program that still pointed at
them would see every semaphore wait already satisfied and read the intermediate before the other chips
wrote it, so the output check fails instead of passing by accident.

The aliased-miss case passes the input as the persistent intermediate on the call that builds the program,
so the tensor bindings resolved on that call cannot tell the two apart; the later calls use distinct
tensors and must still be correct. Only those later calls are checked.
"""
import pytest
import torch

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

NUM_CALLS = 3
POISON = 1 << 20

RING = {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}
LINE = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params, topology, rs_input_shape, dim, use_persistent_buffers, alias_input_on_miss",
    [
        # Tiled layout: the caller's input-shaped intermediate.
        (RING, ttnn.Topology.Ring, [1, 1, 256, 1024], 3, True, False),
        # Contiguous layout: the op allocates the intermediate and the penult intermediate on every call.
        (RING, ttnn.Topology.Ring, [1, 1, 256, 1024], 3, False, False),
        (RING, ttnn.Topology.Ring, [4, 1, 64, 256], 0, True, False),
        (LINE, ttnn.Topology.Linear, [1, 1, 256, 1024], 3, True, False),
        (LINE, ttnn.Topology.Linear, [1, 1, 256, 1024], 3, False, False),
        (RING, ttnn.Topology.Ring, [1, 1, 256, 1024], 3, True, True),
    ],
    ids=[
        "ring_tiled",
        "ring_contiguous",
        "ring_dim0",
        "line_persistent",
        "line_non_persistent",
        "ring_aliased_miss",
    ],
    indirect=["device_params"],
)
def test_reduce_scatter_minimal_async_cache_hit(
    mesh_device, topology, rs_input_shape, dim, use_persistent_buffers, alias_input_on_miss
):
    torch.manual_seed(0)
    nd = mesh_device.get_num_devices()
    dtype = ttnn.bfloat16
    mem_config = ttnn.DRAM_MEMORY_CONFIG

    grid_size = mesh_device.compute_with_storage_grid_size()
    sem_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))}
    )
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    shard_input = ttnn.create_mesh_mapper(
        mesh_device,
        ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementShard(dim)], ttnn.MeshShape(1, nd)),
    )

    def to_device(t, mapper):
        return ttnn.from_torch(
            t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=mem_config, mesh_mapper=mapper
        )

    global_input_shape = rs_input_shape[:]
    global_input_shape[dim] *= nd
    intermediate_shape = rs_input_shape[:]
    if topology == ttnn.Topology.Linear:
        # Line RS stages the forward and backward partials separately.
        intermediate_shape.insert(0, 2)
    output_shape = rs_input_shape[:]
    output_shape[dim] //= nd

    # Everything every call needs is created up front and kept alive, so each call's tensors and semaphores
    # live at addresses no other call uses.
    calls = []
    for _ in range(NUM_CALLS):
        global_input = torch.rand(global_input_shape).bfloat16().float()
        calls.append(
            {
                "input": to_device(global_input, shard_input),
                "interm": to_device(torch.zeros(intermediate_shape), replicate) if use_persistent_buffers else None,
                "out_buf": to_device(torch.zeros(output_shape), replicate) if use_persistent_buffers else None,
                "rs_sems": [ttnn.create_global_semaphore(mesh_device, sem_crs, 0) for _ in range(3)],
                "barrier_sem": ttnn.create_global_semaphore(mesh_device, sem_crs, 0),
                "ref": torch.stack(torch.chunk(global_input, nd, dim)).sum(0),
            }
        )

    sem_addrs = [
        tuple(ttnn.get_global_semaphore_address(s) for s in call["rs_sems"] + [call["barrier_sem"]]) for call in calls
    ]
    assert len(set(sem_addrs)) == NUM_CALLS, f"the calls must not share semaphore addresses: {sem_addrs}"

    # Only the op runs between the cache-entry readings, so any extra entry is the op missing the cache.
    outputs = []
    cache_entries = []
    for i, call in enumerate(calls):
        persistent_output_buffers = None
        if use_persistent_buffers:
            interm = call["input"] if (alias_input_on_miss and i == 0) else call["interm"]
            persistent_output_buffers = [interm, call["out_buf"]]
        out = ttnn.experimental.reduce_scatter_minimal_async(
            call["input"],
            persistent_output_buffers=persistent_output_buffers,
            dim=dim,
            multi_device_global_semaphore=call["rs_sems"],
            barrier_semaphore=call["barrier_sem"],
            num_links=1,
            memory_config=mem_config,
            intermediate_memory_config=mem_config,
            topology=topology,
            subdevice_id=None,
            cluster_axis=None,
        )
        ttnn.synchronize_device(mesh_device)
        cache_entries.append(mesh_device.num_program_cache_entries())
        for sem in call["rs_sems"] + [call["barrier_sem"]]:
            ttnn.reset_global_semaphore_value(sem, POISON)
        outputs.append(out)

    assert all(
        n == cache_entries[0] for n in cache_entries
    ), f"program cache entries per call: {cache_entries}; the calls after the first must be cache hits"

    for i, (call, out) in enumerate(zip(calls, outputs)):
        if alias_input_on_miss and i == 0:
            # The input was also the intermediate, so the other chips overwrote it mid-op.
            continue
        what = f"call {i} (cache {'miss' if i == 0 else 'hit'})"
        result = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=dim)).float()
        passed, pcc = comp_pcc(call["ref"], result, 0.999)
        assert passed, f"{what}: {pcc}"
