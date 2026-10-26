# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DevicePlan (plan.cpp, one program) against DevicePlanSplit (plan_slice.cpp + plan_gather.cpp) on Galaxy row 0.

For each token count: random top-8 routing (distinct experts per token, skewed like real routing), both plans must
give bit-identical row_token / row_weight / row_dest / team plan, then each is timed over ``--iters`` calls
(enqueued back to back, one device synchronize). Prints MOE_PLAN_BENCH lines. Under ``python -m tracy -r`` with
MIMO_PROFILE_ZONES=1 each timed loop is a zone (``profiled_chunk/<plan>_T<tokens>``) for parse_prefill_zones.py: the
device time without the host's program building.

    python models/experimental/mimo_v2_6_flash/tests/perf/moe_plan_bench.py [--tokens 1024,4096] [--iters 50]
"""

import argparse
import time
from types import SimpleNamespace

import torch
from loguru import logger

import ttnn
from models.experimental.mimo_v2_6_flash.tests.perf import moe_decode_common as common
from models.experimental.mimo_v2_6_flash.tt.common import MeshPlan
from models.experimental.mimo_v2_6_flash.tt.moe_prefill import DevicePlan, DevicePlanSplit, GroupedExperts
from models.experimental.mimo_v2_6_flash.tt.profiling import zone

E, K = 256, 8


def routing(T, seed):
    """Distinct top-K experts per token, drawn with a skew (a few hot experts), and normalized weights."""
    g = torch.Generator().manual_seed(seed)
    popularity = torch.rand(E, generator=g) ** 3 + 0.05
    ids = torch.multinomial(popularity.expand(T, E), K, replacement=False, generator=g).to(torch.int32)
    w = torch.rand(T, K, generator=g)
    return ids, w / w.sum(-1, keepdim=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="32,1056,4096")
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    mesh, row = common.open_row()
    try:
        moe = SimpleNamespace(num_experts=E, plan=SimpleNamespace(mesh_device=row))
        grouped = SimpleNamespace(num_teams=30, PLAN_WORDS=GroupedExperts.PLAN_WORDS)
        plans = {"single": DevicePlan(moe, grouped), "split": DevicePlanSplit(moe, grouped)}
        for T in (int(t) for t in args.tokens.split(",")):
            ids, w = routing(T, T)
            kw = dict(device=row, layout=ttnn.TILE_LAYOUT, mesh_mapper=MeshPlan(row).replicate())
            ids_tt = ttnn.from_torch(ids.reshape(1, 1, T, K), dtype=ttnn.uint32, **kw)
            w_tt = ttnn.from_torch(w.reshape(1, 1, T, K), dtype=ttnn.float32, **kw)
            outs = {}
            for name, plan in plans.items():
                res = plan(ids_tt, w_tt)
                outs[name] = [ttnn.to_torch(ttnn.get_device_tensors(t)[0]) for t in res]
                for t in res:
                    t.deallocate(True)
            labels = ("row_token", "row_weight", "row_dest", "plan")
            same = {
                lab: torch.equal(
                    a.view(torch.int16) if a.dtype == torch.bfloat16 else a,
                    b.view(torch.int16) if b.dtype == torch.bfloat16 else b,
                )
                for lab, a, b in zip(labels, outs["single"], outs["split"])
            }
            assert all(same.values()), f"T={T}: planner outputs differ: {same}"
            ms = {}
            for name, plan in plans.items():
                for _ in range(3):
                    for t in plan(ids_tt, w_tt):
                        t.deallocate(True)
                ttnn.synchronize_device(row)
                ttnn.ReadDeviceProfiler(row)
                t0 = time.time()
                with zone("profiled_chunk"), zone(f"{name}_T{T}"):
                    for _ in range(args.iters):
                        for t in plan(ids_tt, w_tt):
                            t.deallocate(True)
                ttnn.synchronize_device(row)
                ms[name] = (time.time() - t0) / args.iters * 1e3
                ttnn.ReadDeviceProfiler(row)
            logger.info(f"T={T}: identical {same}; single {ms['single']:.3f} ms, split {ms['split']:.3f} ms")
            print(
                f"MOE_PLAN_BENCH T={T} identical={all(same.values())} single_ms={ms['single']:.3f} "
                f"split_ms={ms['split']:.3f}",
                flush=True,
            )
            ids_tt.deallocate(True)
            w_tt.deallocate(True)
    finally:
        ttnn.close_mesh_device(row)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
