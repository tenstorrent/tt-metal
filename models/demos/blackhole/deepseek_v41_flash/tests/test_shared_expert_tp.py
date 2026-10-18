# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Candidate: shared expert split over the 8 mesh columns (tt/shared_expert_tp.py), partial folded into the routed partial BEFORE the MoE
block's reduce_scatter over axis 1 (so the all_gather after it carries routed + shared; no extra collective).

One device session, layer 3 weights, real FFN input of checkpoint layer 2 (post ffn_norm; layer 3 cannot run standalone in the reference):
  1. shared only: column-sharded partials summed over the 8 columns (bf16 sequential, like a reduce_scatter, and fp32) vs the replicated
     DSV41SharedExpertV2 (the shipped one) and vs a torch fp32 golden; T = 4 (B=16, 4 users/row) and T = 16.
  2. end to end on the real DSV41MoEBlock: ttnn.reduce_scatter is wrapped so that `red + partial` goes in (bf16 add + bf16 RS, and
     fp32 add + fp32 RS); result vs shipped (gathered routed + replicated shared fp32).
  3. traced microbenchmarks (REP calls per trace, median over N replays): replicated vs column-sharded (config sweep), the add, the RS with
     and without the extra add, DRAM per chip.
Env: SETP_CFGS / SETP_CUR_SWEEP (config sweeps), SETP_TS (default 4,16), SETP_N (default 25), SETP_SWEEP=0 to skip the config sweep, SETP_E2E=0 to skip part 2, SETP_LAYER (3).
Prints lines starting with SETP.
"""

import os
import statistics
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt.dsv41_model import GATE_CUTOFFS
from models.demos.blackhole.deepseek_v41_flash.tt.moe_block import DSV41MoEBlock
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import load_moe_layer
from models.demos.blackhole.deepseek_v41_flash.tt.shared_expert_tp import DSV41SharedExpertColTP
from models.demos.blackhole.deepseek_v41_flash.tt.shared_expert_v2 import DSV41SharedExpertV2

LAYER = int(os.environ.get("SETP_LAYER", "3"))
ACT_LAYER = int(
    os.environ.get("SETP_ACT_LAYER", "2")
)  # layer 3 cannot run standalone in the reference (reads the kv cache of its source layer): FFN inputs come from layer 2
TS = [int(t) for t in os.environ.get("SETP_TS", "4,16").split(",")]
NREP = int(os.environ.get("SETP_N", "25"))
REP = 20
D = 5120
LIM = 10.0


def p(msg):
    print("SETP " + msg, flush=True)


def stats(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    d = (a - b).abs()
    return dict(
        pcc=R.pcc(a, b),
        maxabs=float(d.max()),
        rel_l2=float((a - b).norm() / (b.norm() + 1e-30)),
        ref_rms=float(b.pow(2).mean().sqrt()),
        ref_max=float(b.abs().max()),
    )


def fmt(s):
    return f"pcc {s['pcc']:.6f} maxabs {s['maxabs']:.4g} relL2 {s['rel_l2']:.3e} (ref rms {s['ref_rms']:.3g} max {s['ref_max']:.3g})"


def traced_us(mesh, fn, n=None, rep=REP):
    """median us per call: fn() x rep inside one trace, n replays timed one by one."""
    n = n or NREP

    def body():
        for _ in range(rep):
            o = fn()
            if isinstance(o, ttnn.Tensor):
                ttnn.deallocate(o)

    body()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    body()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    for _ in range(3):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        ts.append((time.perf_counter() - t) / rep * 1e6)
    ttnn.release_trace(mesh, tid)
    return statistics.median(ts)


def dram_bytes(md):
    mv = ttnn.get_memory_view(md, ttnn.BufferType.DRAM)
    return mv.total_bytes_allocated_per_bank * getattr(mv, "num_banks", 1), mv.total_bytes_allocated_per_bank


def dev_t(t, r, c, cols):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[r * cols + c]).float()


def bf16_seq_sum(parts):
    acc = parts[0].to(torch.bfloat16)
    for x in parts[1:]:
        acc = (acc + x.to(torch.bfloat16)).to(torch.bfloat16)
    return acc.float()


def golden_shared(h, w0, w1, w2):
    """torch fp32: (silu(min(h@w0, L)) * clamp(h@w1, -L, L)) @ w2, h [n, D]."""
    g = (h.float() @ w0[0, 0].float()).clamp(max=LIM)
    u = (h.float() @ w1[0, 0].float()).clamp(-LIM, LIM)
    return (torch.nn.functional.silu(g) * u) @ w2[0, 0].float()


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 200_000_000,
            },
            id="fabric_1D_ring",
        ),
    ],
    indirect=True,
)
@pytest.mark.timeout(5400)
@torch.no_grad()
def test_shared_expert_tp(mesh_device):
    md = mesh_device
    torch.manual_seed(1234)
    rows, cols = tuple(md.shape)
    res = {}

    # ---------------- weights, real FFN inputs
    w = load_moe_layer(LAYER)
    sid = next(iter(w["shared_w0"]))
    w0, w1, w2 = w["shared_w0"][sid], w["shared_w1"][sid], w["shared_w2"][sid]
    maxT = max(TS)
    batch = rows * maxT
    blk = R.build_layer(ACT_LAYER, max_batch_size=2, max_seq_len=256)
    cap = {}
    blk.ffn.register_forward_hook(lambda m, i, o: cap.update(x=i[0].detach()))
    xs = []
    for _ in range(batch // 16):  # the reference attention takes short sequences only: 16 tokens (2 x 8) per call
        hh, pm = R.embed_tokens(torch.randint(1000, 100000, (2, 8)))
        blk(hh, 0, pm, None)
        xs.append(cap["x"].reshape(16, -1))
    xall = torch.cat(xs).to(torch.bfloat16)  # [batch, D] real FFN input (post ffn_norm)
    del blk
    p(
        f"layer {LAYER} weights, layer {ACT_LAYER} FFN input rms {xall.float().pow(2).mean().sqrt():.3f} max {xall.float().abs().max():.2f}"
    )

    d0, _ = dram_bytes(md)
    cur = DSV41SharedExpertV2(md, w0, w1, w2)  # the shipped one (layer.py: mode fused1d)
    ttnn.synchronize_device(md)
    d1, _ = dram_bytes(md)
    tp = DSV41SharedExpertColTP(md, w0, w1, w2)
    ttnn.synchronize_device(md)
    d2, _ = dram_bytes(md)
    p(
        f"DRAM per chip (all banks): replicated shared expert {(d1 - d0) / 2**20:.2f} MiB, column-sharded {(d2 - d1) / 2**20:.2f} MiB, saved/layer {((d1 - d0) - (d2 - d1)) / 2**20:.2f} MiB, x40 layers {((d1 - d0) - (d2 - d1)) * 40 / 2**30:.2f} GiB"
    )

    row_shard = lambda x, layout: ttnn.from_torch(  # per mesh row its own rows of x, replicated over the columns
        x,
        device=md,
        dtype=ttnn.bfloat16,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(2, None), mesh_shape=(rows, cols)),
    )

    # ---------------- 1. shared only
    for T in TS:
        x = xall[: rows * T].reshape(1, 1, rows * T, D)
        h = row_shard(x, ttnn.TILE_LAYOUT)
        o_cur, o_tp = cur.forward(h), tp.forward(h)
        ttnn.synchronize_device(md)
        for r in range(rows if T == 4 else 1):
            xr = x[0, 0, r * T : (r + 1) * T].float()
            gold = golden_shared(xr, w0, w1, w2)
            c_cur = dev_t(o_cur, r, 0, cols).reshape(T, D)
            parts = [dev_t(o_tp, r, c, cols).reshape(T, D) for c in range(cols)]
            s16, s32 = bf16_seq_sum(parts), sum(parts)
            if r == 0:
                p(f"T={T} row0 shared-only: replicated(shipped) vs fp32 golden: {fmt(stats(c_cur, gold))}")
                p(f"T={T} row0 shared-only: cols fp32-sum vs shipped: {fmt(stats(s32, c_cur))}")
                p(f"T={T} row0 shared-only: cols bf16-seq-sum vs shipped: {fmt(stats(s16, c_cur))}")
                p(
                    f"T={T} row0 shared-only: cols bf16-seq-sum vs golden: {fmt(stats(s16, gold))}   fp32-sum vs golden {fmt(stats(s32, gold))}"
                )
            else:
                p(
                    f"T={T} row{r} bf16-sum vs shipped pcc {stats(s16, c_cur)['pcc']:.6f} maxabs {stats(s16, c_cur)['maxabs']:.4g}"
                )
        ttnn.deallocate(o_cur)
        ttnn.deallocate(o_tp)

    # ---------------- 2. end to end through the real MoE block
    moe = None
    if os.environ.get("SETP_E2E", "1") != "0":
        K = GATE_CUTOFFS[str(LAYER)]  # the shipped per-layer cutoff (layer 3 cannot run standalone in the reference)
        moe4 = DSV41MoEBlock(md, w, topology=ttnn.Topology.Linear, batch_per_device=4, gate_bias_shift=K)
        orig_rs = ttnn.reduce_scatter
        mode = {"m": None, "part": None}

        def rs_wrap(red, **kw):
            m = mode["m"]
            if m is None:
                return orig_rs(red, **kw)
            if m == "bf16":
                s = ttnn.add(red, mode["part"], dtype=ttnn.bfloat16)
            else:  # fp32 add + fp32 reduce_scatter
                s = ttnn.add(red, mode["part"], dtype=ttnn.float32)
            o = orig_rs(s, **kw)
            ttnn.deallocate(s)
            return o

        ttnn.reduce_scatter = rs_wrap
        try:
            for T in TS:
                blk_moe = moe4 if T == 4 else moe4.for_batch(T)
                x = xall[: rows * T].reshape(rows * T, 1, 1, D)
                xg = xall[: rows * T].reshape(1, 1, rows * T, D)
                tt_tok = ttnn.from_torch(
                    x,
                    device=md,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols)),
                )
                h = row_shard(xg, ttnn.TILE_LAYOUT)
                comp = ttnn.ConcatMesh2dToTensor(md, dims=(2, 3), mesh_shape=(rows, cols))
                gather = (
                    lambda o: ttnn.to_torch(ttnn.to_memory_config(o, ttnn.DRAM_MEMORY_CONFIG), mesh_composer=comp)
                    .reshape(rows * T, D)
                    .float()
                )
                sh_cur = cur.forward(h)
                sh_cur_h = torch.cat([dev_t(sh_cur, r, 0, cols).reshape(T, D) for r in range(rows)])
                mode["m"] = None
                routed = gather(blk_moe.forward(h, tt_tok))
                ref_tot = routed + sh_cur_h  # shipped total (routed bf16 + shared fp32 added later in the expand)
                gold_sh = torch.cat(
                    [golden_shared(xg[0, 0, r * T : (r + 1) * T].float(), w0, w1, w2) for r in range(rows)]
                )
                ref_exact = routed + gold_sh
                p(
                    f"E2E T={T} routed rms {routed.pow(2).mean().sqrt():.3g}  shared rms {sh_cur_h.pow(2).mean().sqrt():.3g}  shipped total vs routed+golden shared: {fmt(stats(ref_tot, ref_exact))}"
                )
                for name, odt, m in (
                    ("bf16 add + bf16 RS", ttnn.bfloat16, "bf16"),
                    ("fp32 add + fp32 RS", ttnn.float32, "fp32"),
                ):
                    tpx = DSV41SharedExpertColTP(md, w0, w1, w2, out_dtype=odt)
                    mode.update(m=m, part=tpx.forward(h))
                    try:
                        got = gather(blk_moe.forward(h, tt_tok))
                        s_vs_ship = stats(got, ref_tot)
                        s_vs_exact = stats(got, ref_exact)
                        # the shipped layer feeds routed bf16 and shared fp32 into the expand: its own rounding vs ours
                        p(f"E2E T={T} {name}: vs shipped total {fmt(s_vs_ship)}")
                        p(
                            f"E2E T={T} {name}: vs routed+golden-shared {fmt(s_vs_exact)}  (shipped itself: maxabs {stats(ref_tot, ref_exact)['maxabs']:.4g})"
                        )
                    except Exception as e:  # noqa: BLE001
                        p(f"E2E T={T} {name}: FAILED {str(e)[:300]}")
                    mode["m"] = None
                    ttnn.deallocate(mode["part"])
                    del tpx
                ttnn.deallocate(sh_cur)
            moe = moe4
        finally:
            ttnn.reduce_scatter = orig_rs

    # ---------------- 3. micro benchmarks
    rs_kw = None
    if moe is not None:
        rs_kw = moe.decode.config.reduce_scatter.model_dump()
    for T in TS:
        x = xall[: rows * T].reshape(1, 1, rows * T, D)
        h = row_shard(x, ttnn.TILE_LAYOUT)
        us_cur = traced_us(md, lambda: cur.forward(h))
        p(f"BENCH T={T} replicated shared expert (shipped, fp32 out): {us_cur:.1f} us")
        configs = [("fused pn1=1 pn2=2 bw1=8 bw2=3", dict())]
        if os.environ.get("SETP_SWEEP", "1") != "0":
            configs += [
                ("fused pn1=2 pn2=2", dict(pn1=2)),
                ("fused pn1=1 pn2=1", dict(pn2=1)),
                ("fused pn1=1 pn2=4", dict(pn2=4)),
                ("fused pn1=1 pn2=5 bw2=9", dict(pn2=5, bw2=9)),
                ("fused pn1=1 pn2=2 bw1=4", dict(bw1=4)),
                ("fused pn1=1 pn2=2 bw1=16 bw2=9", dict(bw1=16, bw2=9)),
                ("auto (no program config)", dict(cfg=False)),
                ("fused LoFi", dict(fid="LoFi")),
            ]
        if os.environ.get("SETP_CFGS"):  # python list of (name, kwargs) replacing the default sweep
            configs = eval(os.environ["SETP_CFGS"])
        for ck in eval(
            os.environ.get("SETP_CUR_SWEEP", "[]")
        ):  # extra program-config variants of the REPLICATED expert (fair baseline)
            try:
                vc = DSV41SharedExpertV2(md, w0, w1, w2, mode="fused1d", **ck)
                p(f"BENCH T={T} replicated shared expert config {ck}: {traced_us(md, lambda: vc.forward(h)):.1f} us")
                del vc
            except Exception as e:  # noqa: BLE001
                p(f"BENCH T={T} replicated config {ck}: FAILED {str(e)[:150]}")
        best = None
        for name, kw in configs:
            for odt_name, odt in (("bf16", ttnn.bfloat16), ("fp32", ttnn.float32)):
                if odt_name == "fp32" and name != configs[0][0]:
                    continue
                try:
                    v = DSV41SharedExpertColTP(md, w0, w1, w2, out_dtype=odt, **kw)
                    us = traced_us(md, lambda: v.forward(h))
                    p(f"BENCH T={T} column-sharded [{name}] out={odt_name}: {us:.1f} us")
                    if odt_name == "bf16" and (best is None or us < best[0]):
                        best = (us, name, kw)
                    del v
                except Exception as e:  # noqa: BLE001
                    p(f"BENCH T={T} column-sharded [{name}] out={odt_name}: FAILED {str(e)[:200]}")
        us_tp, name_tp, kw_tp = best
        p(
            f"BENCH T={T} best column-sharded: [{name_tp}] {us_tp:.1f} us  -> saves {us_cur - us_tp:.1f} us/layer before the extra add"
        )
        v = DSV41SharedExpertColTP(md, w0, w1, w2, **kw_tp)
        part = v.forward(h)
        red = ttnn.from_torch(
            torch.randn(1, 1, T, D).to(torch.bfloat16),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        )
        us_add = traced_us(md, lambda: ttnn.add(red, part, dtype=ttnn.bfloat16))
        p(f"BENCH T={T} add(red bf16, partial bf16) -> bf16: {us_add:.1f} us")
        if rs_kw is not None:
            us_rs = traced_us(md, lambda: ttnn.reduce_scatter(red, **rs_kw))
            us_add_rs = traced_us(md, lambda: ttnn.reduce_scatter(ttnn.add(red, part, dtype=ttnn.bfloat16), **rs_kw))
            p(
                f"BENCH T={T} reduce_scatter alone {us_rs:.1f} us ; add+reduce_scatter {us_add_rs:.1f} us (extra {us_add_rs - us_rs:.1f} us)"
            )
            red32 = ttnn.typecast(red, ttnn.float32)
            part32 = DSV41SharedExpertColTP(md, w0, w1, w2, out_dtype=ttnn.float32, **kw_tp).forward(h)
            us_rs32 = traced_us(md, lambda: ttnn.reduce_scatter(red32, **rs_kw))
            us_add_rs32 = traced_us(md, lambda: ttnn.reduce_scatter(ttnn.add(red, part32, dtype=ttnn.float32), **rs_kw))
            p(
                f"BENCH T={T} fp32: reduce_scatter alone {us_rs32:.1f} us ; add+reduce_scatter {us_add_rs32:.1f} us (extra vs bf16 RS alone {us_add_rs32 - us_rs:.1f} us)"
            )
            net = us_tp + (us_add_rs - us_rs) - us_cur
            net32 = None
            p(
                f"BENCH T={T} NET per layer (bf16 fold): column-sharded {us_tp:.1f} + extra(add+RS) {us_add_rs - us_rs:.1f} - replicated {us_cur:.1f} = {net:+.1f} us ; x40 = {net * 40 / 1000:+.2f} ms/token"
            )
        else:
            p(f"BENCH T={T} NET (no RS measured) : {us_tp + us_add - us_cur:+.1f} us")
