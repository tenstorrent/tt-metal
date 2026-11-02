# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Program-cache regression test for the unicast all_gather factory (Linear and Ring) on a 4 chip Blackhole box.

Only the first call builds the program. Every later call is a cache hit and passes a new input tensor (and,
where used, a new persistent output tensor) holding different data. The op's semaphores belong to the cached
workload, so on a hit only the tensor addresses have to be re-pointed. A program that still pointed at an
earlier call's tensors would gather that call's data, so each call is checked against its own golden.

Validation requires the output to be num_devices times the input along the gather dim, so the input can never
share a buffer with the persistent output and there is no aliased case to cover.
"""
import pytest
import torch

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal

NUM_CALLS = 3

LINE = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}
RING = {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}

# Both shapes route to the unicast factory through the Blackhole branch of
# AllGatherDeviceOperation::select_program_factory: unicast is chosen when
#   per_link_bytes = per-device bytes * num_devices / num_links >= min(4 MB, page^2 * 1 MB / 2048^2).
# The factory's Blackhole workers_per_dir heuristic then picks the connection mode, for any num_links in 1..4:
#   mux:    tile bf16, 2 KB pages, 4 MiB per device -> per_link_bytes >= 4 MiB -> unicast, 8 workers/direction,
#           so the workers share a fabric mux.
#   direct: row-major bf16, 128 B pages, 128 KiB per device -> ~3.9 KB ceiling <= per_link_bytes <= 512 KiB
#           (< 750 KB) -> unicast, 1 worker/direction connected straight to the router.
MUX = ([1, 1, 1024, 2048], 3, ttnn.TILE_LAYOUT)
DIRECT = ([1, 1, 1024, 64], 2, ttnn.ROW_MAJOR_LAYOUT)


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
# all_gather takes its Linear/Ring topology from the fabric config.
@pytest.mark.parametrize("device_params", [LINE, RING], ids=["line", "ring"], indirect=True)
@pytest.mark.parametrize(
    "per_device_shape, dim, layout",
    [MUX, DIRECT],
    ids=["mux", "direct"],
)
@pytest.mark.parametrize("use_persistent_output", [False, True], ids=["new_output", "persistent_output"])
def test_all_gather_unicast_cache_hit(mesh_device, per_device_shape, dim, layout, use_persistent_output):
    torch.manual_seed(0)
    nd = mesh_device.get_num_devices()
    dtype = ttnn.bfloat16
    mem_config = ttnn.DRAM_MEMORY_CONFIG

    shard_input = ttnn.create_mesh_mapper(
        mesh_device,
        ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementShard(dim)], ttnn.MeshShape(1, nd)),
    )
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)

    def to_device(t, mapper):
        return ttnn.from_torch(
            t, dtype=dtype, layout=layout, device=mesh_device, memory_config=mem_config, mesh_mapper=mapper
        )

    gathered_shape = per_device_shape[:]
    gathered_shape[dim] *= nd

    # Everything every call needs is created up front and kept alive, so each call's tensors live at addresses
    # no other call uses.
    calls = []
    for _ in range(NUM_CALLS):
        golden = torch.rand(gathered_shape).bfloat16()
        calls.append(
            {
                "input": to_device(golden, shard_input),
                "output": to_device(torch.zeros(gathered_shape), replicate) if use_persistent_output else None,
                "golden": golden,
            }
        )

    # Only the op runs between the cache-entry readings, so any extra entry is the op missing the cache.
    outputs = []
    cache_entries = []
    for call in calls:
        out = ttnn.all_gather(
            call["input"],
            dim,
            memory_config=mem_config,
            output_tensor=call["output"],
        )
        ttnn.synchronize_device(mesh_device)
        cache_entries.append(mesh_device.num_program_cache_entries())
        outputs.append(out)

    assert all(
        n == cache_entries[0] for n in cache_entries
    ), f"program cache entries per call: {cache_entries}; the calls after the first must be cache hits"

    for i, (call, out) in enumerate(zip(calls, outputs)):
        what = f"call {i} (cache {'miss' if i == 0 else 'hit'})"
        result = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        expected = call["golden"].repeat(nd, *([1] * (len(gathered_shape) - 1)))
        passed, message = comp_equal(expected, result)
        assert passed, f"{what}: {message}"
