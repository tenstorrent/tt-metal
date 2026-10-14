# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Scratch: the sparse all-gather prototype (ttnn.bringup.fabric_all_gather_ttnn.sparse_all_gather_py) on the 8x1
TorusY LoudBox ring, against the Python dense all-gather (same placement / fence / connections), measured like the
dispatch worker. Routings as scratch_dispatch_loudbox.py: 256 experts, 32 per chip, top-8, 640 rows per chip.
Run with TT_MESH_GRAPH_DESC_PATH=.../flat_combine_tools/p150_x8_ring_8x1.textproto.
"""

import statistics

import pytest
import torch
from loguru import logger
from ttnn.bringup.fabric_all_gather_ttnn import fabric_all_gather as py_all_gather
from ttnn.bringup.fabric_all_gather_ttnn.sparse_all_gather_py import landed, route_rows, sparse_all_gather

import ttnn
from models.demos.deepseek_v3_d_p.tests.op_unit_tests.test_dispatch_fabric2d import PRODUCTION_ROUTING, _draw_indices
from models.demos.deepseek_v3_d_p.tests.pcc.mesh_configs import ALL_MESH_CONFIGS

S = 640
G = 8
E, K, EPC = 256, 8, 32
ITERATIONS = 10
LINK_GBPS = 25.0  # 200G per direction

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


def _slowest_chip_ns(mesh_device):
    """Device kernel time of the programs since the last read, slowest chip (as test_fabric_all_gather_py): needs
    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1. Not tracy: its
    host-side report chokes on the Python-built programs' zone volume."""
    ttnn.ReadDeviceProfiler(mesh_device)
    chip_ns = []
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        ns = [
            float(p.program_analyses_results[_DURATION_KEY].duration)
            for p in programs
            if _DURATION_KEY in (getattr(p, "program_analyses_results", None) or {})
        ]
        if ns:
            chip_ns.append(sum(ns))
    assert chip_ns, "profiler produced no kernel durations (TT_METAL_DEVICE_PROFILER=1, profiler build?)"
    return max(chip_ns)


_MESH_CONFIGS = [param for param in ALL_MESH_CONFIGS if param.id == "fabric2d-torus-y-8x1-2link"]


def _need(routing):
    torch.manual_seed(42)
    if routing == "dense":
        return torch.ones(G, S, G, dtype=torch.bool)
    if routing == "uniform":
        idx = torch.stack([torch.stack([torch.randperm(E)[:K] for _ in range(S)]) for _ in range(G)])
    elif routing == "hot":
        idx = _draw_indices(1, G, S, K, E, 1.0, PRODUCTION_ROUTING[1])[0]
    else:  # DeepSeek group-limited: top-4 of 8 groups of 32 experts (a group is one chip here), then top-8
        idx = torch.zeros(G, S, K, dtype=torch.int64)
        for o in range(G):
            for t in range(S):
                g = torch.randperm(8)[:4]
                cand = (g.view(-1, 1) * 32 + torch.arange(32)).reshape(-1)
                idx[o, t] = cand[torch.randperm(128)[:K]]
    need = torch.zeros(G, S, G, dtype=torch.bool)
    need.scatter_(2, idx // EPC, True)
    return need


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links", _MESH_CONFIGS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("emb", [7168], ids=lambda e: f"emb{e}")
@pytest.mark.parametrize("routing", ["cpp-ag", "py-ag", "dense", "uniform", "hot", "grouped"])
@pytest.mark.timeout(1800)
def test_sparse_ag_loudbox_perf(mesh_device, device_params, num_links, emb, routing):
    assert tuple(mesh_device.shape) == (G, 1)
    torch.manual_seed(7)
    x = torch.randn(G, 1, S, emb).to(torch.bfloat16)
    tt_x = ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(G, 1), dims=(0, None)),
    )
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, G * S, emb]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
    )
    want = x.reshape(G * S, emb)
    row_mb = emb * 2 / 1e6

    if routing == "cpp-ag":
        run = lambda: ttnn.bringup.fabric_all_gather(
            tt_x, dim=2, output_tensor=out, cluster_axis=0, num_links=num_links
        )
        reach = torch.ones(G, S, G, dtype=torch.bool)
        busiest = 3.5 * S / num_links
    elif routing == "py-ag":
        run = lambda: py_all_gather(
            tt_x, cluster_axis=0, topology=ttnn.Topology.Ring, num_links=num_links, dim=2, balance=True, output=out
        )
        reach = torch.ones(G, S, G, dtype=torch.bool)
        busiest = 3.5 * S / num_links
    else:
        need = _need(routing)
        routes = route_rows(need, num_links=num_links)
        fh, bh = routes[0], routes[1]
        reach = landed(fh, bh)
        assert bool((reach | ~need).all()), "a needed row is not routed to its chip"
        busiest = float(routes[4].max())
        hops = int(fh.sum() + bh.sum())
        logger.info(
            f"{routing}: needed (row, chip) pairs {int(need.sum()) - G * S} remote, rows landing {int(reach.sum()) - G * S}, "
            f"row-hops {hops} (dense all-gather {G * S * (G - 1)}), busiest link {busiest:.0f} rows"
        )
        run = lambda: sparse_all_gather(tt_x, need, cluster_axis=0, num_links=num_links, output=out, routes=routes)
    logger.info(
        f"{routing}: busiest link {busiest:.0f} rows = {busiest * row_mb:.1f} MB -> floor {busiest * row_mb / LINK_GBPS * 1e3:.0f} us"
    )

    run()
    ttnn.synchronize_device(mesh_device)
    for r, t in enumerate(ttnn.get_device_tensors(out)):
        got = ttnn.to_torch(t).reshape(G * S, emb)
        mask = reach[:, :, r].reshape(-1)
        assert torch.equal(got[mask], want[mask]), f"chip {r}: a landed row differs"
    logger.info(f"{routing}: every landed row byte-exact on all {G} chips")

    _slowest_chip_ns(mesh_device)  # drop the warm-up's records
    us = []
    for _ in range(ITERATIONS):
        run()
        ttnn.synchronize_device(mesh_device)
        us.append(_slowest_chip_ns(mesh_device) / 1e3)
    floor = busiest * row_mb / LINK_GBPS * 1e3
    med = statistics.median(us)
    logger.info(
        f"SPARSE_AG {routing:8s} emb{emb}: median {med:7.1f} us (min {min(us):.1f}, max {max(us):.1f}) | busiest link "
        f"{busiest:.0f} rows, floor {floor:.0f} us, {100 * floor / med:.0f}% of line rate"
    )
