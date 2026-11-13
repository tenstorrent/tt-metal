"""collective variants microbench (traced): q all_gather, o all_to_all"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dk_dev import timed_traced
from dm_block import *

import ttnn

full, md = open_row()


def T(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return to_mesh(md, (torch.randn(8 * shape[0], *shape[1:])), dtype, layout)


def bench(name, f):
    try:
        us = timed_traced(md, f)
    except Exception as e:
        try:
            ttnn.end_trace_capture(md, 0, cq_id=0)
        except Exception:
            pass
        print("RES " + json.dumps({"op": name, "error": str(e).strip().splitlines()[0][:200]}), flush=True)
        return
    print("RES " + json.dumps({"op": name, "us": us}), flush=True)


try:
    from models.demos.gpt_oss.tt.ccl import CCLManager

    S = int(os.environ.get("S", "1"))
    for topo_name, topo in (("ring", ttnn.Topology.Ring), ("linear", ttnn.Topology.Linear)):
        for nl in (1, 2):
            ccl = CCLManager(md, num_links=nl, topology=topo)

            def ag(x, dim, **kw):
                return ttnn.experimental.all_gather_async(
                    x,
                    dim=dim,
                    cluster_axis=1,
                    mesh_device=md,
                    topology=topo,
                    multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(),
                    num_links=nl,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    barrier_semaphore=ccl.get_barrier_semaphore(),
                    **kw,
                )

            q32 = T([1, 32, S, 576], layout=ttnn.ROW_MAJOR_LAYOUT)
            q8 = T([1, 8, S, 576], layout=ttnn.ROW_MAJOR_LAYOUT)
            q32t = T([1, 32, S, 576])
            q8t = T([1, 8, S, 576])
            bench(f"{topo_name} nl{nl} AG q32 RM dim2", lambda: ag(q32, 2))
            bench(f"{topo_name} nl{nl} AG q8 RM dim2", lambda: ag(q8, 2))
            bench(f"{topo_name} nl{nl} AG q8 RM dim1", lambda: ag(q8, 1))
            bench(f"{topo_name} nl{nl} AG q8 TILE dim2", lambda: ag(q8t, 2))
            bench(f"{topo_name} nl{nl} AG q8 TILE dim1", lambda: ag(q8t, 1))
            bench(f"{topo_name} nl{nl} AG q32 TILE dim2", lambda: ag(q32t, 2))
            bench(f"{topo_name} nl{nl} AG q8 RM dim2 workers1", lambda: ag(q8, 2, num_workers_per_link=1))
            o = T([8, S, 32, 608])
            bench(
                f"{topo_name} nl{nl} A2A o[8,S,32,608]",
                lambda: ttnn.experimental.all_to_all_async_generic(
                    o, in_dim=1, out_dim=0, num_links=nl, topology=topo, cluster_axis=1
                ),
            )
            o2 = T([8, S, 32, 576])
            bench(
                f"{topo_name} nl{nl} A2A o[8,S,32,576]",
                lambda: ttnn.experimental.all_to_all_async_generic(
                    o2, in_dim=1, out_dim=0, num_links=nl, topology=topo, cluster_axis=1
                ),
            )
            o3 = T([8, S, 32, 544])
            bench(
                f"{topo_name} nl{nl} A2A o[8,S,32,544]",
                lambda: ttnn.experimental.all_to_all_async_generic(
                    o3, in_dim=1, out_dim=0, num_links=nl, topology=topo, cluster_axis=1
                ),
            )
            ob = T([8, S, 32, 608], dtype=ttnn.bfloat8_b)
            bench(
                f"{topo_name} nl{nl} A2A bfp8 o[8,S,32,608]",
                lambda: ttnn.experimental.all_to_all_async_generic(
                    ob, in_dim=1, out_dim=0, num_links=nl, topology=topo, cluster_axis=1
                ),
            )
            o8 = T([8, S, 8, 608])
            bench(
                f"{topo_name} nl{nl} A2A o[8,S,8,608]",
                lambda: ttnn.experimental.all_to_all_async_generic(
                    o8, in_dim=1, out_dim=0, num_links=nl, topology=topo, cluster_axis=1
                ),
            )
finally:
    close_row(full, md)
