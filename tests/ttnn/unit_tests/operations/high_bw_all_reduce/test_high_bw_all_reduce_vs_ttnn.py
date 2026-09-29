# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""high_bw_all_reduce vs the production ttnn.all_reduce (reduce_scatter_minimal_async + all_gather_async
per mesh axis) on the same 2x2 mesh, same tensors, same instrument. Run with `-s` for the CMP lines.

device_us = sum over every program the call dispatched of its duration (max over chips), from the
realtime profiler. wall_us = host time for call + synchronize (includes Python descriptor building and
dispatch). Both are medians of 5 after a warm-up call."""

import statistics
import time

import pytest
import torch
import ttnn

from ttnn.operations.high_bw_all_reduce import high_bw_all_reduce
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program


def _router(max_payload):
    config = ttnn.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload
    return config


def _params(fabric, big_packets):
    p = {"fabric_config": fabric}
    if big_packets:
        p["fabric_router_config"] = _router(14 * 1024)
        p["reliability_mode"] = ttnn.FabricReliabilityMode.RELAXED_INIT
    return p


FABRICS = [
    pytest.param(_params(ttnn.FabricConfig.FABRIC_2D, False), id="2d-4k"),
    pytest.param(_params(ttnn.FabricConfig.FABRIC_2D, True), id="2d-14k"),
    pytest.param(_params(ttnn.FabricConfig.FABRIC_1D, False), id="1d-4k"),
    pytest.param(_params(ttnn.FabricConfig.FABRIC_1D, True), id="1d-14k"),
]

# (impl, cluster_axis, topology)
CASES = [
    pytest.param("ours", 0, ttnn.Topology.Linear, id="ours-axis0"),
    pytest.param("ours", None, ttnn.Topology.Linear, id="ours-mesh-snake"),
    pytest.param("ours", None, ttnn.Topology.Ring, id="ours-mesh-ring"),
    pytest.param("ttnn", 0, ttnn.Topology.Linear, id="ttnn-axis0"),
    pytest.param("ttnn", None, None, id="ttnn-mesh"),
]


def _device_ns(mesh_device, run):
    _, records = profile_realtime_program(mesh_device, run, collect_all=True, record_timeout_seconds=5.0)
    programs = {}
    for record in records:
        programs[record["runtime_id"]] = max(programs.get(record["runtime_id"], 0.0), record["duration_ns"])
    assert programs, "realtime profiler returned no program"
    return sum(programs.values()), len(programs)


@pytest.mark.parametrize("device_params", FABRICS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
@pytest.mark.parametrize("num_links", [1, 2], ids=["links1", "links2"])
@pytest.mark.parametrize("impl,cluster_axis,topology", CASES)
@pytest.mark.parametrize("shape", [(1, 1, 2048, 2048), (1, 1, 4096, 4096)], ids=["8MB", "32MB"])
def test_cmp(mesh_device, shape, impl, cluster_axis, topology, num_links):
    fabric = ttnn.get_fabric_config()
    is_2d = fabric == ttnn.FabricConfig.FABRIC_2D
    if impl == "ours" and not is_2d:
        pytest.skip("high_bw_all_reduce requires FABRIC_2D")

    torch.manual_seed(0)
    rows, cols = tuple(mesh_device.shape)
    stacked = torch.randn((rows, cols, *shape), dtype=torch.bfloat16)
    dims = (0, 1) if cluster_axis is None else (cluster_axis,)
    expected = stacked.float().sum(dim=dims, keepdim=True).expand_as(stacked)
    glob = torch.cat([torch.cat(list(stacked[r]), dim=1) for r in range(rows)], dim=0)
    t = ttnn.from_torch(
        glob,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(rows, cols)),
    )

    if impl == "ours":
        run = lambda: high_bw_all_reduce(t, cluster_axis=cluster_axis, topology=topology, num_links=num_links)
    else:
        kw = {"cluster_axis": cluster_axis, "num_links": num_links}
        if topology is not None:
            kw["topology"] = topology
        run = lambda: ttnn.all_reduce(t, **kw)

    out = run()
    ttnn.synchronize_device(mesh_device)
    for idx, dt in enumerate(ttnn.get_device_tensors(out)):
        r, c = divmod(idx, cols)
        assert torch.allclose(ttnn.to_torch(dt).float(), expected[r, c], rtol=0.02, atol=0.1)

    _device_ns(mesh_device, run)
    dev = [_device_ns(mesh_device, run) for _ in range(5)]
    device_ns = statistics.median(d[0] for d in dev)
    n_programs = dev[-1][1]

    walls = []
    for _ in range(5):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        run()
        ttnn.synchronize_device(mesh_device)
        walls.append(time.perf_counter() - t0)
    wall_ns = statistics.median(walls) * 1e9

    size = torch.Size(shape).numel() * 2
    print(
        f"CMP fabric={fabric.name} payload={ttnn.get_tt_fabric_max_payload_size_bytes()} impl={impl} "
        f"axis={cluster_axis} topo={topology.name if topology else 'default'} links={num_links} "
        f"size={size >> 20}MB programs={n_programs} device_us={device_ns / 1e3:.1f} wall_us={wall_ns / 1e3:.1f} "
        f"algbw_dev={size / device_ns:.2f}GB/s algbw_wall={size / wall_ns:.2f}GB/s"
    )
