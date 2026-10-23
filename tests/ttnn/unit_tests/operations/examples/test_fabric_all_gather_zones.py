# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-zone capture of fabric_all_gather send modes on a ring, under --profile (ZONES_FABRIC=1d_ring|2d_torus_xy).

Per case: trace [filler (~113 us ttnn.add), all-gather] x ROUNDS, replay REPLAYS times, read the device profiler.
Prints `FAG_ZONES <case> rids=<first>-<last>`: the host op IDs of the all-gather calls captured for that case, so the
analysis can pick them out of profile_log_device.csv (zones S_* / R_* from the example's kernels), and one `FAG_PLAN`
line per chip (device id, ring position, port cores, routes). FAG_MESH / FAG_AXIS as in test_fabric_all_gather_modes.py.
Tables: fag_zones_analyze.py <log> generated/profiler/.logs/profile_log_device.csv
"""

import os

import pytest
import torch
import ttnn

from ttnn.operations.examples.fabric_all_gather import fabric_all_gather, plan

VARIANTS = tuple(os.environ.get("ZONES_VARIANTS", "b,m,mb").split(","))
LINKS = tuple(int(x) for x in os.environ.get("ZONES_LINKS", "1,2").split(","))
SHAPES = [
    tuple(int(v) for v in s.split(","))
    for s in os.environ.get("ZONES_SHAPES", "128,128;512,512;1024,1024;2048,2048").split(";")
]
ROUNDS, REPLAYS = 4, 2
MESH = tuple(int(v) for v in os.environ.get("FAG_MESH", "1,4").split(","))
AXIS = int(os.environ.get("FAG_AXIS", "1"))
_FABRICS = {"1d_ring": "FABRIC_1D_RING", "2d_torus_xy": "FABRIC_2D_TORUS_XY"}
FABRIC = os.environ.get("ZONES_FABRIC", "1d_ring")
_KW = {
    "b": dict(mode="relay", balance=True),
    "m": dict(mode="mcast", balance=False),
    "mb": dict(mode="mcast", balance=True),
    "ms": dict(mode="mcast_split", balance=False),
    "mh": dict(mode="mcast_halves", balance=False),
}


def _expected(host, c):
    """Chip c's (row-major) gathered output: its ring's shards in ring-axis order."""
    r, col = divmod(c, MESH[1])
    shards = [host[k, col] for k in range(MESH[0])] if AXIS == 0 else [host[r, k] for k in range(MESH[1])]
    H, W = host.shape[-2:]
    return torch.cat([x.reshape(1, 1, H, W) for x in shards], dim=0)


def _router():
    cfg = ttnn.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = int(
        os.environ.get("FAG_PAYLOAD", "14336")
    )  # see test_fabric_all_gather_modes.py
    return cfg


def _op_id():
    return int(ttnn._ttnn.get_device_operation_id())


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": getattr(ttnn.FabricConfig, _FABRICS[FABRIC]),
            "fabric_router_config": _router(),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "trace_region_size": 64 * 1024 * 1024,
        }
    ],
    indirect=True,
    ids=[f"fabric_{FABRIC}"],
)
@pytest.mark.parametrize("mesh_device", [MESH], ids=[f"mesh{MESH[0]}x{MESH[1]}"], indirect=True)
def test_fabric_all_gather_zones(mesh_device):
    big = ttnn.from_torch(
        torch.randn(1, 1, 2048, 4096),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    for H, W in SHAPES:
        host = torch.randn((*MESH, H, W), dtype=torch.bfloat16)
        inp = ttnn.from_torch(
            host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=MESH),
        )
        for links in LINKS:
            for v in VARIANTS:
                name, _, place = v.partition("@")  # "@auto": per-chip placement, as in test_fabric_all_gather_modes.py
                kw = dict(
                    _KW[name],
                    placement=place or "uniform",
                    cluster_axis=AXIS,
                    topology=ttnn.Topology.Ring,
                    num_links=links,
                )
                out = fabric_all_gather(inp, **kw)
                for c, t in enumerate(ttnn.get_device_tensors(out)):
                    expected = _expected(host, c)
                    assert torch.equal(ttnn.to_torch(t).reshape(expected.shape), expected), f"{v} chip {c}"
                ttnn.add(big, big)
                ttnn.synchronize_device(mesh_device)
                keep, rids = [], []
                tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                for _ in range(ROUNDS):
                    keep.append(ttnn.add(big, big))
                    rids.append(_op_id())
                    fabric_all_gather(inp, output=out, **kw)
                ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
                for _ in range(REPLAYS):
                    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)
                ttnn.release_trace(mesh_device, tid)
                ttnn.ReadDeviceProfiler(mesh_device)
                chips, _ = plan(
                    mesh_device,
                    cluster_axis=AXIS,
                    topology=ttnn.Topology.Ring,
                    num_links=links,
                    placement=kw["placement"],
                    balance=kw["balance"],
                    mode=kw["mode"],
                )
                ids = mesh_device.get_device_ids()
                for coord, ch in chips.items():
                    rg = ch["rings"][0]
                    dev = ids[coord[0] * MESH[1] + coord[1]]
                    vc = lambda c: mesh_device.worker_core_from_logical_core(c)  # noqa: E731
                    ports = ";".join(f"{d}{l}@{vc(c).x}.{vc(c).y}" for (j, d, l), c in ch["ports"].items())
                    print(
                        f"FAG_PLAN links={links} {v} {H}x{W} dev={dev} p={rg['p']} ports={ports} "
                        f"fwd={rg.get('fwd_route')} bwd={rg.get('bwd_route')}",
                        flush=True,
                    )
                print(
                    f"FAG_ZONES links={links} {v} {H}x{W} tiles={(H // 32) * (W // 32)} rids={','.join(map(str, rids))}",
                    flush=True,
                )
                for t in keep:
                    ttnn.deallocate(t)
                ttnn.deallocate(out)
        ttnn.deallocate(inp)
