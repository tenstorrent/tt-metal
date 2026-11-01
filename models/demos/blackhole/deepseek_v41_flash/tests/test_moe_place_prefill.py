# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unified (deepseek_prefill) routed-expert prefill reading the decode ring weights in place, with a static expert placement (DSV41_MOE_PLACE) vs the sequential layout,
on REAL layer weights: the router ids are relabelled by pi (tt/moe_place.py::relabel) before bincount / dispatch.

Env: MPP_LAYER (4), MPP_PLACE (flat json), MPP_CACHE (rw dir for the placed packed weights), MPP_N (tokens per mesh row, 512; 512 uses the real post-norm FFN inputs of the S=128 dump,
else random), MPP_REF (16 tokens checked against torch), MPP_TIME (1).
Checks: moe_cols output (router + all-gather + unified MoE + reduce-scatter) of the placed block vs the sequential block on the same input: PCC / bit-identity; both vs the torch reference on
the first MPP_REF tokens; window time (chain_ms) of both.
"""
import json
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference.ref_layer import pcc
from models.demos.blackhole.deepseek_v41_flash.tests.test_moe_stages import chain_ms
from models.demos.blackhole.deepseek_v41_flash.tests.test_unified_moe import torch_ref
from models.demos.blackhole.deepseek_v41_flash.tt import moe_place
from models.demos.blackhole.deepseek_v41_flash.tt import moe_weights as MW
from models.demos.blackhole.deepseek_v41_flash.tt.moe_block import DSV41MoEBlock
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_unified_moe import COLS, ROWS, TOPK, DSV41UnifiedMoE, moe_cols
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

D = 5120


def xod_full(x, md, up):
    return up(x.reshape(ROWS, 1, x.shape[1], D), ttnn.bfloat16, ttnn.TILE_LAYOUT)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 400_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(5400)
@torch.no_grad()
def test_moe_place_prefill(mesh_device):
    md = mesh_device
    layer = int(os.environ.get("MPP_LAYER", "4"))
    N = int(os.environ.get("MPP_N", "512"))
    nref = int(os.environ.get("MPP_REF", "16"))
    place_file = os.environ["MPP_PLACE"]
    n_own = N // COLS
    if os.environ.get("MPP_REAL", "1") == "1" and N == 512:
        d_ = torch.load(f"/mnt/tt-data/ssinghal/dsv4-prefill-s128/layer_{layer}.pt", mmap=True)["prefill"]["ffn_in"]
        x = d_.float().reshape(ROWS, N, D).to(torch.bfloat16)
    else:
        torch.manual_seed(0)
        x = torch.randn(ROWS, N, D).to(torch.bfloat16)
    xo = x.reshape(ROWS, N // 256, COLS, 32, D).permute(0, 2, 1, 3, 4).reshape(ROWS, COLS, n_own, D)
    xod = ttnn.from_torch(
        xo,
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, 1), mesh_shape=(ROWS, COLS)),
    )
    mc, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    comp = ttnn.ConcatMesh2dToTensor(md, dims=(0, 1), mesh_shape=(ROWS, COLS))

    def natural(own):  # [ROWS, COLS, n_own, D] -> [ROWS, N, D] in token order
        t = ttnn.to_torch(own, mesh_composer=comp).float().reshape(ROWS, COLS, N // 256, 32, D)
        return t.permute(0, 2, 1, 3, 4).reshape(ROWS, N, D)

    def run(blk, um):
        o = moe_cols(um, blk.gate, xod, mc, ccl)
        ttnn.synchronize_device(md)
        y = natural(o)
        ttnn.deallocate(o)
        return y

    os.environ.pop("DSV41_MOE_PLACE", None)
    t0 = time.time()
    seq = DSV41MoEBlock(md, MW.load_moe_layer(layer), batch_per_device=32, gate_bias_shift=0.0)
    seq.warmup()
    um_seq = DSV41UnifiedMoE(md, layer, ring=(seq.decode.expert_state.tt_w0_w1, seq.decode.expert_state.tt_w2))
    assert um_seq.lut is None
    print(f"MPPLOG seq built {time.time() - t0:.0f} s", flush=True)
    y_seq = run(seq, um_seq)

    os.environ["DSV41_MOE_PLACE"] = place_file
    MW.WEIGHT_CACHE = os.environ["MPP_CACHE"]
    a = moe_place.assignment(layer)
    assert a is not None, "layer has no placement in the file"
    t0 = time.time()
    placed = DSV41MoEBlock(md, MW.load_moe_layer(layer), batch_per_device=32, gate_bias_shift=0.0)
    placed.warmup()
    um_pl = DSV41UnifiedMoE(
        md, layer, ring=(placed.decode.expert_state.tt_w0_w1, placed.decode.expert_state.tt_w2), gate=placed.gate
    )
    assert um_pl.lut is not None and um_pl.gate_view is not None  # permuted-router path
    um_lut = DSV41UnifiedMoE(md, layer, ring=(placed.decode.expert_state.tt_w0_w1, placed.decode.expert_state.tt_w2))
    assert um_lut.lut is not None and um_lut.gate_view is None  # LUT-gather fallback
    print(f"MPPLOG placed built {time.time() - t0:.0f} s samecol={moe_place.samecol(a)}", flush=True)
    y_pl = run(placed, um_pl)
    y_lut = run(placed, um_lut)
    print(
        f"MPPRES gate-view vs LUT-gather placed output: bit_identical {bool(torch.equal(y_pl, y_lut))} max_abs {float((y_pl - y_lut).abs().max()):.3e}",
        flush=True,
    )
    res = dict(
        samecol=moe_place.samecol(a),
        pcc_placed_vs_seq=[pcc(y_seq[r], y_pl[r]) for r in range(ROWS)],
        bit_identical=bool(torch.equal(y_seq, y_pl)),
        max_abs=float((y_seq - y_pl).abs().max()),
    )
    print(f"MPPRES placed-vs-seq {res}", flush=True)
    # the router ids (same gate in both blocks) -> torch reference on the first tokens
    if nref:
        xd = ttnn.from_torch(
            x.reshape(ROWS, 1, N, D),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(ROWS, COLS)),
        )
        from models.demos.blackhole.deepseek_v41_flash.tests.test_unified_moe import route

        sc, ix = route(placed.gate, xd, N)
        dev = lambda t, r: ttnn.to_torch(ttnn.get_device_tensors(t)[r * COLS])
        for r in (0, 3):
            ref = torch_ref(
                x[r, :nref],
                dev(ix, r).long().reshape(N, TOPK)[:nref],
                dev(sc, r).float().reshape(N, TOPK)[:nref],
                layer,
            )
            p_pl, p_sq = pcc(y_pl[r, :nref], ref), pcc(y_seq[r, :nref], ref)
            print(f"MPPRES row {r} vs torch: placed {p_pl:.6f} sequential {p_sq:.6f}", flush=True)
            res[f"torch_row{r}"] = [p_pl, p_sq]
    if os.environ.get("MPP_TIME", "1") == "1":
        # timings, alternating to cancel drift
        ts = {"seq": [], "pl": [], "lut": []}
        for _ in range(3):
            ts["seq"].append(chain_ms(md, lambda: ttnn.deallocate(moe_cols(um_seq, seq.gate, xod, mc, ccl))))
            ts["pl"].append(chain_ms(md, lambda: ttnn.deallocate(moe_cols(um_pl, placed.gate, xod, mc, ccl))))
            ts["lut"].append(chain_ms(md, lambda: ttnn.deallocate(moe_cols(um_lut, placed.gate, xod, mc, ccl))))
        res["ms"] = ts
        print(
            f"MPPRES window moe_cols N={N}: sequential {ts['seq']} placed(gate view) {ts['pl']} placed(LUT) {ts['lut']} ms",
            flush=True,
        )
    rp = os.environ.get(
        "MPP_ROUTES"
    )  # real decode routing ([step][layer][64,6]) as the id stream of a prefill window: realistic expert load of the window
    if rp:
        routes = torch.load(rp)
        pool = torch.cat([routes[st][layer].reshape(-1, TOPK) for st in range(len(routes))]).long()
        g = torch.Generator().manual_seed(11)
        ids = pool[torch.randint(0, pool.shape[0], (ROWS * N,), generator=g)].reshape(ROWS, N, TOPK)
        pi = torch.tensor(moe_place.relabel(layer))
        sch = torch.rand(ROWS, N, TOPK, generator=g).to(torch.bfloat16) + 0.1
        shard = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(ROWS, COLS))
        up = lambda t, dt, lay: ttnn.from_torch(
            t, device=md, dtype=dt, layout=lay, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=shard
        )
        mk = lambda i: up(i.to(torch.int32).reshape(ROWS * N, 1, 1, TOPK), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT)
        ix_s, ix_p = mk(ids), mk(pi[ids])
        sc_t = up(sch.reshape(ROWS * N, 1, 1, TOPK), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        x_rm = ttnn.reshape(ttnn.to_layout(xod_full(x, md, up), ttnn.ROW_MAJOR_LAYOUT), [1, N, D])
        f_seq = lambda: ttnn.deallocate(um_seq.forward(x_rm, sc_t, ix_s))
        f_lut = lambda: ttnn.deallocate(um_lut.forward(x_rm, sc_t, ix_s))
        f_gv = lambda: ttnn.deallocate(um_pl.forward(x_rm, sc_t, ix_p, relabelled=True))
        cnt = torch.bincount(ids.flatten(), minlength=384)
        seqc = torch.tensor(moe_place.sequential())
        for nm, lab in (("sequential", seqc), ("placed", torch.tensor(moe_place.assignment(layer)))):
            per = torch.zeros(32).scatter_add_(0, lab, cnt.float())
            print(
                f"MPPLOG {nm}: busiest chip {int(per.max())} of {int(per.sum())} selections (mean {per.mean():.0f})",
                flush=True,
            )
        r_ = {"seq": [], "lut": [], "gv": []}
        for _ in range(3):
            for nm, f in (("seq", f_seq), ("lut", f_lut), ("gv", f_gv)):
                r_[nm].append(chain_ms(md, f))
        res["routes_ms"] = r_
        print(
            f"MPPRES unified moe (real-routing ids, N={N}, no router/RS) sequential {r_['seq']} placed(LUT) {r_['lut']} placed(gate view) {r_['gv']} ms",
            flush=True,
        )
    json.dump(res, open(os.path.join(os.environ.get("MP_OUT", "/tmp"), "mpp.json"), "w"))
    assert min(res["pcc_placed_vs_seq"]) > 0.9999
