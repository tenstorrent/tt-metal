"""CCL microbench on the real decode shapes (lever L07): MoE combine RS (stock ttnn.reduce_scatter vs reduce_scatter_minimal_async) and the RS+AG pairs,
swept over links / topology / chunks / workers / buffers; traced per-call time (chain_ms) and PCC vs the torch sum."""
import itertools
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_moe_stages import chain_ms
from models.demos.gpt_oss.tt.ccl import CCLManager

R = int(os.environ.get("CCL_ROWS", "16"))
D = 5120


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 300_000_000}],
    indirect=True,
)
@torch.no_grad()
def test_ccl_micro(mesh_device):
    md = mesh_device
    rows, cols = 4, 8
    ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(rows, cols, R, D, generator=g).to(torch.bfloat16)  # per-device partial sums
    t = ttnn.from_torch(
        x,
        device=md,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, 1), mesh_shape=(rows, cols)),
    )  # [1,1,R,D] per device
    ref = x.float().sum(1)  # [rows, R, D] sum over the 8 columns
    comp_cat = lambda y: ttnn.to_torch(
        y, mesh_composer=ttnn.ConcatMesh2dToTensor(md, dims=(0, 3), mesh_shape=(rows, cols))
    ).float()

    def rs_min(links=None, topo=ttnn.Topology.Ring, chunks=None, workers=None, bufs=None, mem=ttnn.DRAM_MEMORY_CONFIG):
        def f():
            return ttnn.experimental.reduce_scatter_minimal_async(
                t,
                dim=3,
                multi_device_global_semaphore=ccl.get_rs_ping_pong_semaphore(),
                num_links=links or 2,
                memory_config=mem,
                topology=topo,
                cluster_axis=1,
                barrier_semaphore=ccl.get_barrier_semaphore(),
                chunks_per_sync=chunks,
                num_workers_per_link=workers,
                num_buffers_per_channel=bufs,
            )

        return f

    def rs_stock():
        return ttnn.reduce_scatter(t, dim=3, cluster_axis=1)

    def ag(links=2, topo=ttnn.Topology.Ring, src=None, workers=None, bufs=None):
        def f():
            return ttnn.experimental.all_gather_async(
                src if src is not None else t,
                dim=3,
                cluster_axis=1,
                mesh_device=md,
                topology=topo,
                multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(),
                num_links=links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                barrier_semaphore=ccl.get_barrier_semaphore(),
                num_workers_per_link=workers,
                num_buffers_per_channel=bufs,
            )

        return f

    def pair(rs, ag_f):
        def f():
            return ag_f(rs())

        return f

    def show(name, fn, check=False):
        extra = ""
        if check:
            res = []
            fn()  # cold start of a CCL program is unreliable
            ttnn.synchronize_device(md)
            for _ in range(3):
                y = fn()
                ttnn.synchronize_device(md)
                if y.shape[-1] == D:
                    c = ttnn.to_torch(
                        y, mesh_composer=ttnn.ConcatMesh2dToTensor(md, dims=(0, 1), mesh_shape=(rows, cols))
                    ).float()[:, 0:1]
                else:
                    c = comp_cat(y)
                res.append(pcc(c.reshape(rows, R, -1), ref))
            extra = " pcc " + "/".join(f"{p_:.5f}" for p_ in res)
        us = chain_ms(md, fn) * 1e3
        print(f"CCL R={R} {name}: {us:.1f} us{extra}", flush=True)

    ttn = ttnn.Topology
    show("rs_stock", rs_stock, True)
    show("rs_min default", rs_min(), True)
    for links, workers, chunks in itertools.product((1, 2), (None, 1, 2), (None,)):
        for rep in range(2):
            show(f"rs_min links={links} workers={workers} rep{rep}", rs_min(links, ttn.Ring, chunks, workers), rep == 0)
    sc = rs_min()()
    ttnn.synchronize_device(md)
    for links, workers, bufs in itertools.product((1, 2), (None, 1, 2), (None, 2)):
        show(f"ag links={links} workers={workers} bufs={bufs}", ag(links, ttn.Ring, sc, workers, bufs))
    show("pair stock-rs + ag default", pair(rs_stock, lambda y: ag(src=y)()), True)
    show("pair min-rs default + ag default", pair(rs_min(), lambda y: ag(src=y)()), True)
    for rl, rw, al, aw in ((1, 1, 2, None), (1, 1, 2, 1), (1, 1, 1, None), (2, 1, 2, 1), (1, 1, 2, 2)):
        show(
            f"pair rs(l{rl},w{rw}) + ag(l{al},w{aw})",
            pair(rs_min(rl, ttn.Ring, None, rw), lambda y, al=al, aw=aw: ag(al, ttn.Ring, y, aw)()),
            True,
        )
    try:
        show(
            "ttnn.all_reduce",
            lambda: ttnn.all_reduce(t, cluster_axis=1, num_links=2, topology=ttnn.Topology.Ring),
            True,
        )
    except Exception as e:
        print(f"CCL ttnn.all_reduce FAIL {str(e)[:150]}", flush=True)
