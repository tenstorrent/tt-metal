# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Scratch: the routing all-to-all + local planner (ttnn.bringup.combine_rf.routing_a2a) against today's path - the
routing all-gather (fabric_all_gather of the joint rows) and combine_plan's multi-core planner - on the LoudBox as a
2 x 4 FABRIC_2D mesh, at a Galaxy 8x4 per-chip load (home tokens and experts a chip). Every per-chip table bit for bit
against the current planner and its host model; with TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1
TT_METAL_PROFILER_CPP_POST_PROCESS=1 also both paths timed (slowest chip, median of RA_ITERS calls) and the new
path's phases per core."""

import os
import statistics
import types

import pytest
import torch
from loguru import logger
from ttnn.bringup.combine_rf.combine_plan import combine_planner_prepare, plan_reference, port_luts
from ttnn.bringup.combine_rf.dispatch_prep import dispatch_prep_prepare
from ttnn.bringup.combine_rf.fabric_all_gather_py import fabric_all_gather
from ttnn.bringup.combine_rf.routing_a2a import routing_a2a_planner_prepare, routing_a2a_sizes
from ttnn.bringup.combine_rf.sparse_all_gather_2d import route_words
from ttnn.bringup.flat_routed_expert_ttnn.closeout_models import CHUNKS, MODELS, STAGES, per_chip

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tests.perf.scratch_flat_combine_rf import _chip_ns, _ids

R, C = 2, 4
N = R * C
L = 2
TPP = 32  # routing rows per page (router_topk's layout)
A_OF = {"k3": 16, "glm53": 12}
PROFILE = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
ITERS = int(os.environ.get("RA_ITERS", "8"))

_CASES = [
    pytest.param(
        (R, C),
        fabric2d_device_params(fabric_payload_size=7232, l1_small_size=0),
        m,
        chunk,
        id=f"{m}-{chunk}",
    )
    for m in os.environ.get("RA_MODELS", "k3,glm53").split(",")
    for chunk in [int(x) for x in os.environ.get("RA_CHUNKS", "5120,16384").split(",")]
]


def _case(model, chunk):
    m = MODELS[model]
    pc = per_chip(model, "8x4", chunk)
    EPC, S, K = pc["epc"], pc["S"], m["K"]
    return types.SimpleNamespace(S=S, K=K, EPC=EPC, E=N * EPC, A=A_OF[model])


def _joint(ids, gates):
    """[N, S, K] ids / bf16 gates -> per home chip [S / TPP, TPP x 2 K] uint16 rows (ids, then weight bits)."""
    n, S, K = ids.shape
    wb = gates.view(torch.int16).to(torch.int64) & 0xFFFF
    j = torch.cat([ids.to(torch.int64), wb], -1)  # [N, S, 2 K]
    return j.reshape(n, S // TPP, TPP * 2 * K).to(torch.int32)


def _words(t):
    return [ttnn.to_torch(x).to(torch.int64).reshape(-1) & 0xFFFFFFFF for x in ttnn.get_device_tensors(t)]


@pytest.mark.parametrize("mesh_device, device_params, model, chunk", _CASES, indirect=["mesh_device", "device_params"])
@pytest.mark.timeout(3600)
def test_routing_a2a(mesh_device, device_params, model, chunk):
    c = _case(model, chunk)
    S, K, EPC, A = c.S, c.K, c.EPC, c.A
    luts, nps = port_luts(mesh_device, L)
    shard = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=(0, 1))
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    up = lambda t, mapper: ttnn.from_torch(  # noqa: E731
        t,
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )
    routings = os.environ.get("RA_ROUTINGS", "uniform,zipf_spread_1.2,zipf_packed_1.2").split(",")
    # one rows_cap for every routing (the largest), so both planners are built once
    cases = []
    rows_cap = 32
    for ri, routing in enumerate(routings):
        g = torch.Generator().manual_seed(ri)
        ids = _ids(routing, g, c)  # [N, S, K] global ids
        gates = torch.softmax(torch.randn(N, S, K, generator=g), -1).to(torch.bfloat16)
        for me in range(N):
            cnt = torch.bincount((ids[(ids // EPC) == me] - me * EPC).reshape(-1), minlength=EPC)
            rows_cap = max(rows_cap, int(((cnt + 31) // 32 * 32).sum()))
        cases.append((routing, ids, gates))
    rows_cap = (rows_cap + 31) // 32 * 32
    logger.info(
        f"{model} chunk {chunk}: S {S} home tokens a chip, K {K}, {EPC} experts a chip, A {A}, rows_cap {rows_cap}"
    )

    pcores = sorted({(i % 8, i // 8) for i in range(32)}, key=lambda x: (x[1], x[0]))
    old = None
    new = routing_a2a_planner_prepare(
        mesh_device,
        S=S,
        K=K,
        EPC=EPC,
        A=A,
        L=L,
        port_luts=luts,
        n_ports=nps,
        rows_cap=rows_cap,
        tokens_per_page=TPP,
        plan_cores=int(os.environ.get("RA_CORES", "0")) or None,
        # (probes: RA_DCACHE=0 the planner without its L1 data cache, RA_NOSTORE=1 no token-index / table stores)
        dcache=os.environ.get("RA_DCACHE", "1") == "1",
        probe_nostore=os.environ.get("RA_NOSTORE", "0") == "1",
    )
    for routing, ids, gates in cases:
        joint = _joint(ids, gates)  # [N, S / TPP, TPP 2 K]
        own_t = up(joint.reshape(R, C, S // TPP, TPP * 2 * K), shard)
        rep_t = up(joint.reshape(N * S // TPP, TPP * 2 * K), rep)
        if old is None:
            old = combine_planner_prepare(
                mesh_device,
                S=S,
                K=K,
                EPC=EPC,
                A=A,
                L=L,
                port_luts=luts,
                n_ports=nps,
                rows_cap=rows_cap,
                cores=pcores,
                row_pages=True,
                joint=True,
                tokens_per_page=TPP,
            )
        # the gathered routing equals the replicated one (today's path's input)
        gat = fabric_all_gather(own_t, cluster_axis=None, topology=ttnn.Topology.Linear, num_links=1, dim=0)
        for ci, t in enumerate(ttnn.get_device_tensors(gat)):
            got = ttnn.to_torch(t).to(torch.int64).reshape(N * S // TPP, -1) & 0xFFFF
            assert torch.equal(got, joint.reshape(N * S // TPP, -1).to(torch.int64) & 0xFFFF), ci
        o_outs = {k: _words(v) for k, v in old(gat, gat).items()}
        n_outs = {k: _words(v) for k, v in new(own_t).items()}
        ttnn.synchronize_device(mesh_device)
        wb = gates.view(torch.int16).to(torch.int64) & 0xFFFF
        for ci in range(N):
            ch = (ci // C, ci % C)
            ref = plan_reference(
                ids, wb, ci, R, C, S, K, EPC, N * EPC, A, L, nps[ch], luts[ch], rows_cap, token_major=True
            )
            keys = ("counts", "regions", "plan") if os.environ.get("RA_NOSTORE") == "1" else None
            for k in keys or ("counts", "regions", "token_index", "table", "plan"):
                a_, b_ = o_outs[k][ci], n_outs[k][ci]
                exp = torch.tensor(ref[k], dtype=torch.int64)
                if k in ("token_index", "table"):  # (rows past the plan's row count are never read: a previous
                    # routing's values may stay there, in both planners' outputs)
                    nw = min(ref["plan"][600], rows_cap) * (2 if k == "table" else 1)
                    a_, b_, exp = a_[:nw], b_[:nw], exp[:nw]
                bad = (b_[: exp.numel()] != exp).nonzero().flatten()
                assert bad.numel() == 0, (
                    f"{routing} chip {ci} {k}: new vs host model {bad.numel()} words differ, first {bad[:8].tolist()}"
                    f" new {b_[bad[:8]].tolist()} ref {exp[bad[:8]].tolist()}"
                )
                diff = (a_ != b_).nonzero().flatten()
                assert diff.numel() == 0, (
                    f"{routing} chip {ci} {k}: new vs current planner {diff.numel()} words differ, first "
                    f"{diff[:8].tolist()} new {b_[diff[:8]].tolist()} old {a_[diff[:8]].tolist()}"
                )
        logger.info(f"{model} {chunk} {routing}: all {N} chips bit-exact vs the current planner and its host model")
        # the dispatch's route words from each chip's own routing = from the replicated routing
        rws = []
        for jr, own_ in ((rep_t, False), (own_t, True)):
            dp = dispatch_prep_prepare(
                mesh_device, S=S, K=K, EPC=EPC, tokens_per_page=TPP, copy=False, own_routing=own_
            )
            xb = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, N * S, 32]),
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
                mesh_device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
            rt_ = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, 1, route_words(N) * S]),
                ttnn.uint32,
                ttnn.ROW_MAJOR_LAYOUT,
                mesh_device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
            dp(jr, None, xb, rt_)
            rws.append(_words(rt_))
        for ci in range(N):
            assert torch.equal(rws[0][ci], rws[1][ci]), f"{routing} chip {ci}: route words differ"
        logger.info(f"{model} {chunk} {routing}: dispatch route words from the own routing = from the replicated one")
        # the next call's parity: run it again and check once more (the double-buffered slots)
        n2 = {k: _words(v) for k, v in new(own_t).items()}
        for k in [k for k in n_outs if os.environ.get("RA_NOSTORE") != "1" or k in ("counts", "regions", "plan")]:
            for ci in range(N):
                assert torch.equal(n2[k][ci], n_outs[k][ci]), f"{routing} second call chip {ci} {k}"
        if PROFILE:  # (per chip, device time per call: each path alone, ITERS calls queued back to back)
            out = ttnn.allocate_tensor_on_device(gat.shape, gat.dtype, gat.layout, mesh_device, ttnn.DRAM_MEMORY_CONFIG)
            gather = lambda: fabric_all_gather(  # noqa: E731
                own_t, cluster_axis=None, topology=ttnn.Topology.Linear, num_links=1, dim=0, output=out
            )
            res = {}
            for name, fn in (
                ("gather", gather),
                ("planner", lambda: old(gat, gat)),
                ("gather+planner", lambda: (gather(), old(out, out))),
                ("a2a", lambda: new(own_t)),
            ):
                fn()
                ttnn.synchronize_device(mesh_device)
                _chip_ns(mesh_device)
                for _ in range(ITERS):
                    fn()
                ttnn.synchronize_device(mesh_device)
                per = [x / ITERS / 1e3 for x in _chip_ns(mesh_device)]
                res[name] = (max(per), statistics.median(per))
            logger.info(
                f"RA_TIME {model} {chunk} {routing} (us per call, slowest chip / median chip, {ITERS} queued): "
                + ", ".join(f"{k} {v[0]:.1f} / {v[1]:.1f}" for k, v in res.items())
            )
            ttnn.deallocate(out)
        if new.dbg is not None:  # (phases of the last call per chip, us from the chip's first core start)
            worst = None
            for ci, t in enumerate(ttnn.get_device_tensors(new.dbg)):
                d = (ttnn.to_torch(t).to(torch.int64).reshape(-1, 16) & 0xFFFFFFFF).tolist()
                pl_, sd_ = d[: new.M], [r for r in d[new.M :] if r[0]]
                assert all(r[5] == pl_[0][13] for r in sd_), "port cores' CB 0 base differs from the planner's"
                t0 = min(r[0] for r in pl_ + sd_)
                span = max([r[8] for r in pl_] + [r[3] for r in sd_]) - t0
                logger.info(
                    f"RA_CHIP {model} {chunk} {routing} chip {ci}: ports sent "
                    f"{max((r[3] - t0) / 1350 for r in sd_):.1f}, arrived {(pl_[0][4] - t0) / 1350:.1f}, "
                    f"end {(max(r[8] for r in pl_) - t0) / 1350:.1f} us (start {t0})"
                )
                if worst is None or span > worst[0]:
                    worst = (span, ci, pl_, sd_, t0)
            span, ci, pl_, sd_, t0 = worst
            us = lambda x: (x - t0) / 1350  # noqa: E731
            for si, r in enumerate(sd_):
                logger.info(
                    f"RA_PHASE {model} {chunk} {routing} chip {ci} port {si} ({r[4]} packets): start {us(r[0]):.1f} "
                    f"slices in {us(r[1]):.1f} headers {us(r[2]):.1f} sent {us(r[3]):.1f}"
                )
            for pi, r in enumerate(pl_):
                logger.info(
                    f"RA_PHASE {model} {chunk} {routing} chip {ci} planner {pi} ({r[12]} own entries, {r[9]} rows "
                    f"{r[10]}-{r[11]}): start {us(r[0]):.1f} counted {us(r[1]):.1f} exchanged {us(r[2]):.1f} "
                    f"sent {us(r[3]):.1f} arrived {us(r[4]):.1f} read {us(r[5]):.1f} sorted {us(r[6]):.1f} "
                    f"placed {us(r[7]):.1f} end {us(r[8]):.1f}"
                )
            logger.info(f"RA_PHASE {model} {chunk} {routing}: slowest chip {ci}, {span / 1350:.1f} us start to end")


# the stages' axes that wrap (Galaxy 8x4: a torus both ways; half Galaxy 4x4: TP wraps, SP a line; LoudBox 2x4: lines)
_WRAP = {"8x4": (True, True), "4x4": (True, False), "2x4": (False, False)}


def _port_dests(sp, tp, wrap, L=2):
    """The most destinations one port core serves: destination d on (first hop, link d % L), Y first, then X."""
    R_, C_ = sp, tp
    worst = 0
    for me in range(R_ * C_):
        serve = {}
        for d in range(R_ * C_):
            if d == me:
                continue
            sr, sc, dr, dc = me // C_, me % C_, d // C_, d % C_
            if sr != dr:
                h = (2 if 2 * ((dr - sr) % R_) < R_ else 3) if wrap[1] else (2 if dr > sr else 3)
            else:
                h = (0 if 2 * ((dc - sc) % C_) <= C_ else 1) if wrap[0] else (0 if dc > sc else 1)
            serve[(h, d % L)] = serve.get((h, d % L), 0) + 1
        worst = max(worst, max(serve.values()) + 1)  # (+ my own packet, on one of them)
    return worst


@pytest.mark.parametrize("stage", ["8x4", "4x4"])
@pytest.mark.parametrize("model", list(MODELS))
@pytest.mark.parametrize("chunk", CHUNKS)
def test_routing_a2a_sizes(stage, model, chunk):
    """Host only: the routing all-to-all's L1 / DRAM / local memory on the Galaxy stages (rows_cap for twice the
    uniform rows: skew headroom)."""
    if model == "k27" and chunk != 5120:
        pytest.skip("K2.7: chunk 5120 only")
    m = MODELS[model]
    pc = per_chip(model, stage, chunk)
    sp, tp = STAGES[stage]
    S, K, EPC = pc["S"], m["K"], pc["epc"]
    rows_cap = (2 * S * K + 32 * EPC + 31) // 32 * 32
    z = routing_a2a_sizes(
        N=sp * tp, S=S, K=K, EPC=EPC, A=16, rows_cap=rows_cap, port_dests=_port_dests(sp, tp, _WRAP[stage])
    )
    logger.info(
        f"RA_SIZE {stage} {model} {chunk}: S {S}, EPC {EPC}, M {z['M']}, planner L1 {z['plan_l1'] >> 10} KB, port L1 "
        f"{z['port_l1'] >> 10} KB, packet <= {z['packet'] >> 10} KB, slots {z['slots_dram'] >> 20} MB DRAM, local "
        f"{z['local']} B: {'fits' if z['fits'] else z['why']}"
    )
    assert z["fits"], z["why"]
