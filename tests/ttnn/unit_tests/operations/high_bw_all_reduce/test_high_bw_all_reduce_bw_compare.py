# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-link bandwidth of high_bw_all_reduce, measured with the same realtime-profiler instrument as
test_high_bw_all_gather.py, under the default fabric router and under high_bw_all_gather's 14 KiB
max-packet-payload router config. Run with `-s` to see the HIGH_BW_ALL_REDUCE lines."""

import statistics

import pytest
import torch
import ttnn

from ttnn.operations.high_bw_all_reduce import high_bw_all_reduce
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program


def _router(max_payload):
    config = ttnn.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload
    return config


def _profile(mesh_device, run):
    _, records = profile_realtime_program(mesh_device, run, collect_all=True, record_timeout_seconds=5.0)
    programs = {}
    for record in records:
        if not any("high_bw_all_reduce" in s for s in record["kernel_sources"]):
            continue
        programs[record["runtime_id"]] = max(programs.get(record["runtime_id"], 0.0), record["duration_ns"])
    assert programs, "realtime profiler returned no high_bw_all_reduce program"
    return sum(programs.values())


@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_2D},
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": _router(14 * 1024),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        },
    ],
    ids=["router_default", "router_14k"],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
# None = every usable link (4 per neighbour pair on the QuietBox); skips the SUPPORTED num_links gate.
@pytest.mark.parametrize("num_links", [1, 2, None], ids=["links1", "links2", "links_all"])
@pytest.mark.parametrize(
    "cluster_axis,topology",
    [(0, ttnn.Topology.Linear), (None, ttnn.Topology.Ring)],
    ids=["axis0-linear", "none-ring"],
)
@pytest.mark.parametrize("shape", [(1, 1, 4096, 4096), (1, 1, 8192, 4096)], ids=lambda s: "x".join(map(str, s)))
def test_bw(mesh_device, shape, cluster_axis, topology, num_links):
    torch.manual_seed(0)
    rows, cols = tuple(mesh_device.shape)
    stacked = torch.randn((rows, cols, *shape), dtype=torch.bfloat16)
    dims = (0, 1) if cluster_axis is None else (cluster_axis,)
    expected = stacked.float().sum(dim=dims, keepdim=True).expand_as(stacked).bfloat16()
    glob = torch.cat([torch.cat(list(stacked[r]), dim=1) for r in range(rows)], dim=0)
    t = ttnn.from_torch(
        glob,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(rows, cols)),
    )
    run = lambda: high_bw_all_reduce(t, cluster_axis=cluster_axis, topology=topology, num_links=num_links)
    out = run()
    ttnn.synchronize_device(mesh_device)
    for idx, dt in enumerate(ttnn.get_device_tensors(out)):
        r, c = divmod(idx, cols)
        assert torch.allclose(ttnn.to_torch(dt).float(), expected[r, c].float(), rtol=0.02, atol=0.1)

    _profile(mesh_device, run)  # warm
    median_ns = statistics.median(_profile(mesh_device, run) for _ in range(5))

    size_bytes = torch.Size(shape).numel() * 2
    group = rows * cols if cluster_axis is None else tuple(mesh_device.shape)[cluster_axis]
    # Bytes each link direction carries: (G-1)/G of the tensor per direction on the rotated ring, the
    # whole tensor each way on a line (partials forward, finals back), split over num_links lanes.
    per_dir = size_bytes * ((group - 1) / group if topology == ttnn.Topology.Ring else 1.0)
    lanes = num_links or 4
    per_link_dir_gbps = per_dir / lanes / median_ns
    print(
        f"HIGH_BW_ALL_REDUCE payload={ttnn.get_tt_fabric_max_payload_size_bytes()}B shape={shape} "
        f"axis={cluster_axis} topo={topology} links={lanes} median={median_ns / 1e3:.1f}us "
        f"algbw={size_bytes / median_ns:.2f}GB/s per_link_dir={per_link_dir_gbps:.2f}GB/s"
    )
